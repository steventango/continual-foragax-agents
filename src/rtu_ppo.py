# Modified from esraaelelimy/continuing_ppo
import sys

import socket
import time
import logging
from jax.tree_util import tree_map
import numpy as np
from flax import struct
from utils.preempt import TimeoutHandler
from functools import partial
from typing import Sequence, NamedTuple, Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax_tqdm.scan_pbar import scan_tqdm
from jax_tqdm.base import PBar
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict
from flax import traverse_util
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.ml_instrumentation.Sampler import Mean
from utils.ml_instrumentation.utils import Last
from utils.preempt import TimeoutHandler
from algorithms.PPORegistry import getAgent

import optax
from flax.training.train_state import TrainState
import argparse
from foragax.registry import make

@struct.dataclass
class LogEnvState:
    returned_returns: float
    timestep: int

@struct.dataclass
class TrainConfig:
    # ---- STATIC (uniform across vmapped runs) ----
    d_hidden: int = struct.field(pytree_node=False)
    hidden_size: int = struct.field(pytree_node=False)
    agent_type: str = struct.field(pytree_node=False)
    rollout_steps: int = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)
    num_mini_batch: int = struct.field(pytree_node=False)
    gradient_clipping: bool = struct.field(pytree_node=False)
    num_updates: int = struct.field(pytree_node=False)
    env_id: str = struct.field(pytree_node=False)
    aperture_size: int = struct.field(pytree_node=False)
    # ---- DYNAMIC (may vary per idx; arithmetic only) ----
    max_grad_norm: float
    alpha_pi: float
    alpha_vf: float
    adam_eps_pi: float
    adam_eps_vf: float
    id: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    entropy_coef: float
    
class GymnaxEnvState(struct.PyTreeNode):
    env_step: Callable = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env_step, env_params, env_state, **kwargs):
        """Creates a new instance"""
        return cls(
            env_step=env_step,
            env_params=env_params,
            env_state=env_state,
            **kwargs,
        )

class Transition(NamedTuple):
    action: jnp.ndarray # a_t
    value: jnp.ndarray # v(o_t)
    reward: jnp.ndarray # r[t+1]
    log_prob: jnp.ndarray
    obs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] # o_t, a_{t-1}, r_{t-1}
    info: jnp.ndarray

class Interaction(NamedTuple):
    a: int
    r: bool
@jax.jit
def calculate_gae(traj_batch, last_val,gamma,gae_lambda):
    def _get_advantages(carry, transition):
        gae, next_value = carry
        value, reward = (
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value  - value
        gae = (
            delta
            + gamma * gae_lambda * gae
        )
        return (gae, value), gae
    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

@jax.jit
def calculate_average_reward_gae(traj_batch, last_val,gamma,gae_lambda):
    sample_avg_reward = jnp.mean(traj_batch.reward) #r_\pi
    
    def _get_advantages(next_value, transition):
        value, reward = (
            transition.value,
            transition.reward,
        )
        gae = reward - sample_avg_reward + next_value - value
        return value, gae
    
    _, advantages = jax.lax.scan(
        _get_advantages,
        last_val,
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

@partial(jax.jit, static_argnums=(1,))
def loss_fn(params,agent_fn, traj_batch, gae, targets,init_hstate,clip_eps,vf_coef,ent_coef):
    rnn_in = traj_batch.obs
    _,pi, value = agent_fn(params, init_hstate,rnn_in)
    log_prob = pi.log_prob(traj_batch.action)
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    total_loss = (
        loss_actor
        + vf_coef * value_loss
        - ent_coef * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

@jax.jit
def agent_step(last_obs,train_state,rng,hstate):
    last_obs, last_action_encoded, last_reward = last_obs
    rnn_in = (jnp.expand_dims(last_obs,0), jnp.expand_dims(last_action_encoded,0), jnp.expand_dims(last_reward,0))
    last_hidden,pi, value = train_state.apply_fn(train_state.params, hstate,rnn_in)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value, last_hidden
    
def env_step(runner_state,_):
    train_state,gymnax_state,log_env_state,last_obs,last_action, last_reward, rng,hstate = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    action_encoded = jnp.zeros((4,))
    action_encoded = action_encoded.at[last_action].set(1)
    last_reward_encoded = jnp.expand_dims(last_reward, 0)
    last_obs_encoded = (last_obs, action_encoded, last_reward_encoded)
                  
    action, log_prob, value, last_hidden = agent_step(last_obs_encoded,train_state,_rng,hstate)
    # STEP ENV
    obs, env_state, reward, done, info = gymnax_state.env_step(
        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )
    step = log_env_state.timestep + 1
    new_return = 0.999 * log_env_state.returned_returns + (1.0 - 0.999) * (reward)

    log_env_state = LogEnvState(returned_returns=new_return, timestep=step)
    info = {}
    info["reward"] = reward
    info["moving_average"] = new_return
    info["timestep"] = log_env_state.timestep
    info["pos"] = env_state.pos
    ### Create transition
    transition = Transition(action.squeeze(), value.squeeze(), reward, log_prob.squeeze(), last_obs_encoded, info)
    ### Update runner state
    gymnax_state = GymnaxEnvState.create(
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (train_state, gymnax_state,log_env_state, obs,action.squeeze(),reward, rng,last_hidden)
    return runner_state, (transition,hstate)

@jax.jit
def update_minbatch(carry_in, batch_info):
    train_state,config = carry_in
    minibatch,init_hstate = batch_info 
    # minibatch: (seq_len,minibatch_size, _)
    # init_hstate: (1, d_hidden)
    traj_batch, advantages, targets = minibatch
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params, train_state.apply_fn, traj_batch, advantages, targets, init_hstate,
        config.clip_eps, config.vf_coef, config.entropy_coef
    )
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state,config), total_loss

    
'''
Batch shape = (num_steps, _)
Divide the batch into n minibatches
each minibatch has the shape of (seq_len, minibatch_size, _)
minibatch_size = num_steps//n*seq_len

1. re-run the network through the batch and store hiddens states for positions (0,seq_len,2*seq_len,...)
2. Divide num_steps into sequences of length seq_len : number of sequences = num_steps//seq_len
3. Divide the sequences into n minibatches
4. shuffle the minibatches
output shape = (num_minibatches, seq_len, minibatch_size, _)
'''
@jax.jit
def create_minibaches(config: TrainConfig, hstate_batch, batch, rng, train_state):
    batch_hstate = jax.tree_util.tree_map(lambda y:jnp.squeeze(y,axis=1),hstate_batch)
    traj_batch, advantages, targets = batch
    batch = (batch_hstate,traj_batch, advantages, targets)
    
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, config.rollout_steps)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch)

    minibatch_size = config.rollout_steps // config.num_mini_batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_mini_batch, minibatch_size,) + x.shape[1:]), shuffled_batch)

    batch_hstate,traj_batch, advantages, targets = shuffled_batch 
    minibatches_info = ((traj_batch, advantages, targets ), batch_hstate)
    return minibatches_info,rng
    
@jax.jit
def update_epoch(update_state, unused):
    train_state,init_hstate, traj_batch,hstate_batch, advantages, targets, rng, config = update_state
    batch = (traj_batch, advantages, targets)
    # Prepare minibatches
    # minibatches: (num_minibatches, seq_len,minibatch_size, _)
    minibatches_info, rng = create_minibaches(config, hstate_batch, batch, rng, train_state)
    # Loop through minibatches
    carry_in = (train_state, config)
    carry_out, total_loss = jax.lax.scan(update_minbatch, carry_in, minibatches_info)
    train_state = carry_out[0]
    update_state = (train_state,init_hstate, traj_batch,hstate_batch, advantages, targets, rng, config)
    return update_state, total_loss


@jax.jit
def update_step(update_state):
    train_state, init_hstate, traj_batch, hstate_batch, advantages, targets, rng, config = update_state
    update_state, loss_info = jax.lax.scan(update_epoch, update_state, None, config.epochs)
    ## Update runner state
    train_state = update_state[0]
    rng = update_state[-2]
    return (train_state, rng), loss_info


def experiment(rng, config: TrainConfig):
    env = make(config.env_id, aperture_size=(config.aperture_size, config.aperture_size), observation_type="object")

    ### Initialize the environment states    
    log_env_state = LogEnvState(returned_returns=0,timestep=0)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env.default_params)
    gymnax_state = GymnaxEnvState.create(
        env_step=env.step, env_params=env.default_params, env_state=env_state
    )
    action_dim = 4
    
    agent = getAgent(config.agent_type)
    
    # Create and initialize the network.
    network = agent(
        action_dim=action_dim,
        activation='tanh',
        hidden_size=config.hidden_size,
        d_hidden=config.d_hidden,
        cont=False)
    
    
    rng, _rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, *obs.shape)), jnp.zeros((1, action_dim)), jnp.zeros((1, 1)))
    
    init_hstate = agent.initialize_memory(1, config.d_hidden, config.hidden_size)
    network_params = network.init(_rng, init_hstate, init_x)
    
    def make_label_tree(params):
        flat = traverse_util.flatten_dict(params, sep="/")

        def label_for_path(path_str):
            if "critic" in path_str:
                return "vf"
            elif "actor" in path_str:
                return "pi"
            return ""

        labels_flat = {k: label_for_path(k) for k in flat.keys()}
        return traverse_util.unflatten_dict(
            {tuple(k.split("/")): v for k, v in labels_flat.items()}
        )

    labels = make_label_tree(network_params)

    if config.gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.partition(
                {
                    "pi": optax.adam(config.alpha_pi, eps=config.adam_eps_pi),
                    "vf": optax.adam(config.alpha_vf, eps=config.adam_eps_vf),
                },
                labels,
            ),
        )
    else:
        tx = optax.partition(
            {
                "pi": optax.adam(config.alpha_pi, eps=config.adam_eps_pi),
                "vf": optax.adam(config.alpha_vf, eps=config.adam_eps_vf),
            },
            labels,
        )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    ### Experiment 
    env_step_state = (train_state, gymnax_state, log_env_state, obs, 0, 0, rng, init_hstate)

    @scan_tqdm(config.num_updates)
    def experiment_step(carry, _):
        env_step_state, train_state, rng = carry
        # Roll out for config.rollout_steps
        env_step_state, traj_hstate_batch = jax.lax.scan(
            env_step, env_step_state, length=config.rollout_steps
        )
        traj_batch, hstate_batch = traj_hstate_batch
        train_state, gymnax_state, log_env_state, last_obs, last_action, last_reward, rng, last_hstate = env_step_state

        # Build last observation with previous action encoding
        action_encoded = jnp.zeros((4,))
        action_encoded = action_encoded.at[last_action].set(1)
        last_reward_encoded = jnp.expand_dims(last_reward, 0)
        last_obs_encoded = (last_obs, action_encoded, last_reward_encoded)

        # Bootstrap value at last state
        _, _, last_value, _ = agent_step(last_obs_encoded, train_state, rng, last_hstate)
        last_val = last_value.squeeze()

        # Calculate GAE
        advantages, targets = calculate_gae(traj_batch, last_val, config.gamma, config.gae_lambda)

        # Update step
        rng, update_rng = jax.random.split(rng)
        update_state = (train_state, init_hstate, traj_batch, hstate_batch, advantages, targets, update_rng, config)
        (train_state, rng), loss_info = update_step(update_state)

        # Rebuild env_step_state for next iteration
        env_step_state = (train_state, gymnax_state, log_env_state, last_obs, last_action, last_reward, rng, last_hstate)

        # Collect a scalar reward summary for this iteration (mean reward over rollout)
        rewards = traj_batch.reward
        pos = traj_batch.info["pos"]

        # Optional lightweight debug
        return (env_step_state, train_state, rng), (rewards, pos)

    # Run training loop with lax.scan (collect per-iteration rewards)
    last_carry, info = jax.lax.scan(
        experiment_step,
        PBar(id=config.id, carry=(env_step_state, train_state, rng)),
        xs=jnp.arange(int(config.num_updates))
    )
    rewards, pos = info
    env_step_state, train_state, rng = last_carry.carry
    return rewards, pos
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-i", "--idxs", nargs="+", type=int, required=True)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)

    args = parser.parse_args()
    
    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")
    
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger = logging.getLogger("exp")
    prod = "cdr" in socket.gethostname() or args.silent
    if not prod:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        
    # ----------------------
    # -- Experiment Def'n --
    # ----------------------
    timeout_handler = TimeoutHandler()

    exp = ExperimentModel.load(args.exp)
    
    indices = args.idxs
    
    # --------------------
    # -- Batch Set-up --
    # --------------------
    start_time = time.time()

    collectors = []
    rngs = []
    chks = []
    configs = []
    for idx in indices:
        chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
        chk.load_if_exists()
        timeout_handler.before_cancel(chk.save)
        chks.append(chk)

        collector = chk.build(
            "collector",
            lambda: Collector(
                # specify which keys to actually store and ultimately save
                # Options are:
                #  - Identity() (save everything)
                #  - Window(n)  take a window average of size n
                #  - Subsample(n) save one of every n elements
                config={
                    "ewm_reward": Pipe(
                        MovingAverage(0.999),
                        Subsample(max(exp.total_steps // 1000, 1)),
                    ),
                    "mean_ewm_reward": Last(
                        MovingAverage(0.999),
                        Mean(),
                    ),
                },
                # by default, ignore keys that are not explicitly listed above
                default=Ignore(),
            ),
        )
        collector.set_experiment_id(idx)
        collectors.append(collector)

        hypers = exp.get_hypers(idx)

        seed = exp.getRun(idx) + hypers.get("seed_offset", 0)
        rng = jax.random.PRNGKey(seed)
        rngs.append(rng)

        # derive num_updates if not explicitly present
        num_updates = int(hypers['num_updates']) if 'num_updates' in hypers else exp.total_steps // int(hypers['rollout_steps'])
        config = TrainConfig(
            d_hidden=int(hypers['representation']['d_hidden']),
            agent_type=exp.agent,
            hidden_size=int(hypers['representation']['hidden']),
            rollout_steps=int(hypers['rollout_steps']),
            epochs=int(hypers['epochs']),
            num_mini_batch=int(hypers.get('num_mini_batch', hypers.get('num_minibatch', 1))),
            gradient_clipping=bool(hypers.get('gradient_clipping', False)),
            max_grad_norm=float(hypers.get('max_grad_norm', 0.0)),
            alpha_pi=float(hypers['optimizer_actor']['alpha']),
            alpha_vf=float(hypers['optimizer_critic']['alpha']),
            adam_eps_pi=float(hypers['optimizer_actor'].get('eps', 1e-5)),
            adam_eps_vf=float(hypers['optimizer_critic'].get('eps', 1e-5)),
            num_updates=num_updates,
            aperture_size=int(hypers["environment"]["aperture_size"]),
            env_id=hypers["environment"]["env_id"],
            gamma=float(hypers.get('gamma', 0.99)),
            gae_lambda=float(hypers.get('gae_lambda', 0.95)),
            clip_eps=float(hypers.get('clip_eps', 0.2)),
            vf_coef=float(hypers.get('vf_coef', 0.5)),
            entropy_coef=float(hypers.get('entropy_coef', 0.01)),
            id=idx,
        )
        configs.append(config)


    batch_experiment = jax.vmap(experiment, in_axes=(0, 0))
    rngs = jnp.stack(rngs)
    configs = tree_map(lambda *xs: jnp.stack(xs), *configs)
    results = batch_experiment(rngs, configs)
    rewards, pos = results
    rewards = rewards.reshape((rewards.shape[0], -1))
    pos = pos.reshape((pos.shape[0], -1, pos.shape[-1]))
    print(pos.shape)
    # --------------------
    # -- Saving --
    # --------------------
    total_collect_time = 0
    total_numpy_time = 0
    total_db_time = 0
    num_indices = len(indices)
    for i, idx in enumerate(indices):
        collector = collectors[i]
        chk = chks[i]

        # process rewards for this run
        run_rewards = rewards[i]
        run_pos = pos[i]

        start_time = time.time()
        for reward in run_rewards:
            collector.next_frame()
            collector.collect("ewm_reward", reward.item())
            collector.collect("mean_ewm_reward", reward.item())
        logger.debug(f"Mean rewards {run_rewards.mean()}")
        collector.reset()
        total_collect_time += time.time() - start_time

        # ------------
        # -- Saving --
        # ------------
        context = exp.buildSaveContext(idx, base=args.save_path)
        save_path = context.resolve("results.db")
        data_path = context.resolve(f"data/{idx}.npz")
        context.ensureExists(data_path, is_file=True)

        start_time = time.time()
        np.savez_compressed(data_path, rewards=run_rewards, pos=run_pos)
        total_numpy_time += time.time() - start_time

        meta = getParamsAsDict(exp, idx)
        meta |= {"seed": exp.getRun(idx)}
        attach_metadata(save_path, idx, meta)

        start_time = time.time()
        collector.merge(context.resolve("results.db"))
        total_db_time += time.time() - start_time

        collector.close()
        chk.delete()
    logger.debug("--- Saving Timings ---")
    logger.debug(
        f"Total collect time: {total_collect_time:.4f}s | Average: {total_collect_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
    )
    total_save_time = total_collect_time + total_numpy_time + total_db_time
    logger.debug(
        f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
    )

if __name__ == "__main__":
    main()  
