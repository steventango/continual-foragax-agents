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
from gymnasium.utils.save_video import save_video
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

PERIOD = 182500

@struct.dataclass
class LogEnvState:
    returned_returns: float
    timestep: int
    frames: Any

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
    render_mode: str = struct.field(pytree_node=False)
    observation_type: str = struct.field(pytree_node=False)
    repeat: int = struct.field(pytree_node=False)
    reward_delay: int = struct.field(pytree_node=False)
    use_sinusoidal_encoding: bool = struct.field(pytree_node=False)
    use_reward_trace: bool = struct.field(pytree_node=False)
    allocate_frames: bool = struct.field(pytree_node=False)
    # ---- DYNAMIC (may vary per idx; arithmetic only) ----
    max_grad_norm: float
    l2_reg_pi: float
    l2_reg_vf: float
    alpha_pi: float
    alpha_vf: float
    adam_eps_pi: float
    adam_eps_vf: float
    
    sparsity: float
    spectral_radius: float
    
    id: int
    reward_trace_decay: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    entropy_coef: float
    freeze_after_steps: int = -1
    
class GymnaxEnvState(struct.PyTreeNode):
    to_render: bool = struct.field(pytree_node=True)
    cond_render: Callable = struct.field(pytree_node=False)
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
    last_obs, last_action_encoded, last_reward, sine, cosine, reward_trace = last_obs
    rnn_in = (jnp.expand_dims(last_obs,0), jnp.expand_dims(last_action_encoded,0), jnp.expand_dims(last_reward,0), jnp.expand_dims(sine,0), jnp.expand_dims(cosine,0), jnp.expand_dims(reward_trace,0))
    last_hidden,pi, value = train_state.apply_fn(train_state.params, hstate,rnn_in)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value, last_hidden
    
def env_step(runner_state,_):
    train_state,gymnax_state,log_env_state,config, last_obs,last_action, last_reward, reward_trace, rng, hstate = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    action_encoded = jnp.zeros((4,))
    action_encoded = action_encoded.at[last_action].set(1)
    last_reward_encoded = jnp.expand_dims(last_reward, 0)
    sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    cosine = jnp.expand_dims(jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    reward_trace = config.reward_trace_decay * reward_trace + (1.0 - config.reward_trace_decay) * last_reward
    reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
    last_obs_encoded = (last_obs, action_encoded, last_reward_encoded, sine, cosine, reward_trace_encoded)
                  
    action, log_prob, value, last_hidden = agent_step(last_obs_encoded,train_state,_rng,hstate)
    # STEP ENV
    obs, env_state, reward, done, info = gymnax_state.env_step(
        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )
    step = log_env_state.timestep + 1
    new_return = 0.999 * log_env_state.returned_returns + (1.0 - 0.999) * (reward)
    
    frame = gymnax_state.cond_render(gymnax_state.to_render, gymnax_state.env_state)

    log_env_state = LogEnvState(returned_returns=new_return, timestep=step, frames=log_env_state.frames)

    info["reward"] = reward
    info["moving_average"] = new_return
    info["timestep"] = log_env_state.timestep
    info["pos"] = env_state.pos
    info["frame"] = frame

    ### Create transition
    transition = Transition(action.squeeze(), value.squeeze(), reward, log_prob.squeeze(), last_obs_encoded, info)
    ### Update runner state
    gymnax_state = GymnaxEnvState.create(
        to_render=gymnax_state.to_render,
        cond_render=gymnax_state.cond_render,
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (train_state, gymnax_state,log_env_state,config, obs,action.squeeze(),reward,reward_trace, rng,last_hidden)
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
    kwards = {}
    if config.observation_type is not None:
        kwards["observation_type"] = config.observation_type
    if config.repeat is not None:
        kwards["repeat"] = config.repeat
    if config.reward_delay is not None:
        kwards["reward_delay"] = config.reward_delay
        
    print(f"Creating env {config.env_id} with aperture size {config.aperture_size} and kwargs {kwards}")
        
    env = make(config.env_id, aperture_size=config.aperture_size, **kwards)

    ### Initialize the environment states
    if config.allocate_frames:
        frames = jnp.zeros((config.rollout_steps, env.size[0]*24, env.size[1]*24, 3), dtype=jnp.uint8)
    else:
        frames = jnp.zeros((0, env.size[0]*24, env.size[1]*24, 3), dtype=jnp.uint8)
    log_env_state = LogEnvState(returned_returns=0,timestep=0, frames=frames)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env.default_params)
    
    def real_render(env_state):
        return env.render(env_state, None, render_mode=config.render_mode).astype(
            jnp.uint8
        )
        
    def void_render(env_state):
        return jnp.zeros((env.size[0]*24, env.size[1]*24, 3), dtype=jnp.uint8)

    def render(cond, env_state):
        return jax.lax.cond(
            cond,
            real_render,
            void_render,
            env_state
        )
    
    gymnax_state = GymnaxEnvState.create(
        to_render=False,
        cond_render=render,
        env_step=env.step,
        env_params=env.default_params,
        env_state=env_state
    )
    action_dim = 4
    
    agent = getAgent(config.agent_type)
    
    kwards = {}
    if config.sparsity is not None:
        kwards["sparsity"] = config.sparsity
    if config.spectral_radius is not None:
        kwards["spectral_radius"] = config.spectral_radius
    
    # Create and initialize the network.
    network = agent(
        action_dim=action_dim,
        activation='tanh',
        hidden_size=config.hidden_size,
        d_hidden=config.d_hidden,
        cont=False,
        use_sinusoidal_encoding=config.use_sinusoidal_encoding,
        use_reward_trace=config.use_reward_trace,
        **kwards
    )
    
    
    rng, _rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, *obs.shape)), jnp.zeros((1, action_dim)), jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    
    init_hstate = agent.initialize_memory(1, config.d_hidden, config.hidden_size)
    network_params = network.init(_rng, init_hstate, init_x)
    
    def make_label_tree(params):
        flat = traverse_util.flatten_dict(params, sep="/")

        def label_for_path(path_str):
            if "critic" in path_str:
                return "vf"
            elif "actor" in path_str:
                return "pi"
            elif "frozen" in path_str:
                return "frozen"
            return ""

        labels_flat = {k: label_for_path(k) for k in flat.keys()}
        return traverse_util.unflatten_dict(
            {tuple(k.split("/")): v for k, v in labels_flat.items()}
        )

    labels = make_label_tree(network_params)

    if config.gradient_clipping:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(config.alpha_pi, eps=config.adam_eps_pi)
                ),
                "vf": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(config.alpha_vf, eps=config.adam_eps_vf)
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )
    else:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(config.alpha_pi, eps=config.adam_eps_pi)
                ),
                "vf": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(config.alpha_vf, eps=config.adam_eps_vf)
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    ### Experiment 
    def _zero_loss_info(config: TrainConfig):
        zeros = jnp.zeros((config.epochs, config.num_mini_batch), dtype=jnp.float32)
        return (zeros, (zeros, zeros, zeros))

    env_step_state = (train_state, gymnax_state, log_env_state, config, obs, 0, 0, 0, rng, init_hstate)

    @scan_tqdm(config.num_updates)
    def experiment_step(carry, iteration_idx):
        env_step_state, train_state, rng = carry
        train_state, gymnax_state, log_env_state, config, last_obs,last_action, last_reward, reward_trace, rng, hstate = env_step_state
        
        to_render = iteration_idx == (config.num_updates - 1) & config.allocate_frames
        
        gymnax_state = GymnaxEnvState.create(
            to_render=to_render,
            cond_render=gymnax_state.cond_render,
            env_step=gymnax_state.env_step,
            env_params=gymnax_state.env_params,
            env_state=gymnax_state.env_state,
        )
        
        env_step_state = train_state, gymnax_state, log_env_state, config, last_obs,last_action, last_reward, reward_trace, rng, hstate
        
        # Roll out for config.rollout_steps
        env_step_state, traj_hstate_batch = jax.lax.scan(
            env_step, env_step_state, length=config.rollout_steps
        )
        traj_batch, hstate_batch = traj_hstate_batch
        train_state, gymnax_state, log_env_state, config, last_obs, last_action, last_reward, reward_trace, rng, last_hstate = env_step_state

        # Build last observation with previous action encoding
        action_encoded = jnp.zeros((4,))
        action_encoded = action_encoded.at[last_action].set(1)
        last_reward_encoded = jnp.expand_dims(last_reward, 0)
        sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
        cosine = jnp.expand_dims(jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
        reward_trace = config.reward_trace_decay * reward_trace + (1.0 - config.reward_trace_decay) * last_reward
        reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
        last_obs_encoded = (last_obs, action_encoded, last_reward_encoded, sine, cosine, reward_trace_encoded)

        # Bootstrap value at last state
        _, _, last_value, _ = agent_step(last_obs_encoded, train_state, rng, last_hstate)
        last_val = last_value.squeeze()

        # Calculate GAE
        advantages, targets = calculate_gae(traj_batch, last_val, config.gamma, config.gae_lambda)

        # Conditionally perform the update based on how many env steps have elapsed.
        # If freeze_after_steps <= 0, updates are always performed.
        # Otherwise, once log_env_state.timestep exceeds freeze_after_steps, we stop updating.
        rng, update_rng = jax.random.split(rng)
        update_state = (train_state, init_hstate, traj_batch, hstate_batch, advantages, targets, update_rng, config)

        def skip_update(update_state):
            train_state, init_hstate, traj_batch, hstate_batch, advantages, targets, rng, config = update_state
            return (train_state, rng), _zero_loss_info(config)

        should_update = jnp.logical_or(config.freeze_after_steps <= 0,
                                       log_env_state.timestep <= config.freeze_after_steps)
        (train_state, rng), loss_info = jax.lax.cond(should_update,
                                                     update_step,
                                                     skip_update,
                                                     update_state)

        # Collect a scalar reward summary for this iteration (mean reward over rollout)
        rewards = traj_batch.reward
        pos = traj_batch.info["pos"]
        biome_id = traj_batch.info["biome_id"]
        object_collected_id = traj_batch.info["object_collected_id"]
        if config.allocate_frames:
            frames = traj_batch.info["frame"]
        else:
            frames = log_env_state.frames
        log_env_state = LogEnvState(returned_returns=log_env_state.returned_returns, timestep=log_env_state.timestep, frames=frames)
        
        # Rebuild env_step_state for next iteration
        env_step_state = (train_state, gymnax_state, log_env_state, config, last_obs, last_action, last_reward, reward_trace, rng, last_hstate)

        # Optional lightweight debug
        return (env_step_state, train_state, rng), (rewards, pos, loss_info, biome_id, object_collected_id)

    # Run training loop with lax.scan (collect per-iteration rewards)
    last_carry, info = jax.lax.scan(
        experiment_step,
        PBar(id=config.id, carry=(env_step_state, train_state, rng)),
        xs=jnp.arange(int(config.num_updates))
    )
    rewards, pos, loss_info, biome_id, object_collected_id = info
    rewards = rewards.reshape((-1))
    pos = pos.reshape((-1, pos.shape[-1]))
    total_loss = jnp.mean(loss_info[0], axis=(-1, -2))
    value_loss = jnp.mean(loss_info[1][0], axis=(-1, -2))
    policy_loss = jnp.mean(loss_info[1][1], axis=(-1, -2))
    entropy = jnp.mean(loss_info[1][2], axis=(-1, -2))
    biome_id = biome_id.reshape((-1))
    object_collected_id = object_collected_id.reshape((-1))
    env_step_state, train_state, rng = last_carry.carry
    frames = env_step_state[2].frames
    return rewards, pos, (total_loss, (value_loss, policy_loss, entropy)), biome_id, object_collected_id, frames
    
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
    allocate_frames = len(indices) <= 1

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
        num_updates = int(hypers['num_updates']) if 'num_updates' in hypers else (exp.total_steps // int(hypers['rollout_steps']) + 1)
        config = TrainConfig(
            d_hidden=int(hypers['representation']['d_hidden']),
            agent_type=exp.agent,
            hidden_size=int(hypers['representation']['hidden']),
            rollout_steps=int(hypers['rollout_steps']),
            epochs=int(hypers['epochs']),
            num_mini_batch=int(hypers['num_mini_batch']),
            gradient_clipping=bool(hypers['gradient_clipping']),
            max_grad_norm=float(hypers['max_grad_norm']),
            alpha_pi=float(hypers['optimizer_actor']['alpha']),
            alpha_vf=float(hypers['optimizer_critic'].get('alpha', hypers['optimizer_critic'].get('lr_scale', jnp.nan) * hypers['optimizer_actor']['alpha'])),
            adam_eps_pi=float(hypers['optimizer_actor']['eps']),
            adam_eps_vf=float(hypers['optimizer_critic']['eps']),
            l2_reg_pi=float(hypers.get('l2_reg_pi', hypers.get('l2_reg', 0.0))),
            l2_reg_vf=float(hypers.get('l2_reg_vf', hypers.get('l2_reg', 0.0))),
            
            sparsity=hypers['representation'].get('sparsity', None),
            spectral_radius=hypers['representation'].get('spectral_radius', None),
            
            use_sinusoidal_encoding=bool(hypers.get('use_sinusoidal_encoding', False)),
            use_reward_trace=bool(hypers.get('use_reward_trace', False)),
            reward_trace_decay=float(hypers.get('reward_trace_decay', 1.0)),
            num_updates=num_updates,
            aperture_size=int(hypers["environment"]["aperture_size"]),
            render_mode=hypers["environment"].get("render_mode", "world_reward"),
            observation_type=hypers["environment"].get("observation_type", None),
            repeat=hypers["environment"].get("repeat", None),
            reward_delay=hypers["environment"].get("reward_delay", None),
            env_id=hypers["environment"]["env_id"],
            gamma=float(hypers['gamma']),
            gae_lambda=float(hypers['gae_lambda']),
            clip_eps=float(hypers['clip_eps']),
            vf_coef=float(hypers['vf_coef']),
            entropy_coef=float(hypers['entropy_coef']),
            id=idx,
            freeze_after_steps=int(hypers.get('freeze_after_steps', -1)),
            allocate_frames=allocate_frames,
        )
        configs.append(config)

    batch_experiment = jax.vmap(experiment, in_axes=(0, 0))
    rngs = jnp.stack(rngs)
    configs_stacked = tree_map(lambda *xs: jnp.stack(xs), *configs)
    results = batch_experiment(rngs, configs_stacked)
    rewards, pos, (total_loss, (value_loss, policy_loss, entropy)), biome_id, object_collected_id, frames  = results

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
        config = configs[i]
        # process rewards for this run
        run_rewards = rewards[i]
        run_pos = pos[i]
        run_total_loss = total_loss[i]
        run_value_loss = value_loss[i]
        run_policy_loss = policy_loss[i]
        run_entropy = entropy[i]
        run_biome_id = biome_id[i]
        run_object_collected_id = object_collected_id[i]
        run_frames = frames[i]
        start_time = time.time()
        # for reward in run_rewards:
        #     collector.next_frame()
        #     collector.collect("ewm_reward", reward.item())
        #     collector.collect("mean_ewm_reward", reward.item())
        logger.debug(f"Mean rewards {run_rewards.mean()}")
        collector.reset()
        total_collect_time += time.time() - start_time

        # ------------
        # -- Saving --
        # ------------
        context = exp.buildSaveContext(idx, base=args.save_path)
        save_path = context.resolve("results.db")
        data_path = context.resolve(f"data/{idx}.npz")
        video_path = context.resolve(f"videos/{idx}")
        context.ensureExists(data_path, is_file=True)
        context.ensureExists(video_path, is_file=True)

        start_time = time.time()
        if config.allocate_frames:
            start_frame = config.num_updates * config.rollout_steps - run_frames.shape[0]
            end_frame = config.num_updates * config.rollout_steps
            save_video(
                list(run_frames),
                video_path,
                name_prefix=f"{start_frame}_{end_frame}",
                fps=8,
            )
        np.savez_compressed(
            data_path, 
            rewards=run_rewards, 
            pos=run_pos, 
            total_loss=run_total_loss, 
            value_loss=run_value_loss, 
            policy_loss=run_policy_loss, 
            entropy=run_entropy,
            biome_id = run_biome_id,
            object_collected_id = run_object_collected_id
        )
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
