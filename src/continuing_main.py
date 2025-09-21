import os
import sys

sys.path.append(os.getcwd())

import argparse
import logging
import lzma
import pickle
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax_tqdm.scan_pbar import scan_tqdm
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict

from environments.Foragax import Foragax
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.ml_instrumentation.Sampler import Mean
from utils.ml_instrumentation.utils import Last
from utils.preempt import TimeoutHandler
from utils.rlglue import RlGlue

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-i", "--idxs", nargs="+", type=int, required=True)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
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

Problem = getProblem(exp.problem)

# --------------------
# -- Batch Set-up --
# --------------------
start_time = time.time()

collectors = []
glues = []
chks = []
first_hypers = None
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
            config={},
            # by default, ignore keys that are not explicitly listed above
            default=Ignore(),
        ),
    )
    collector.set_experiment_id(idx)
    collectors.append(collector)

    run = exp.getRun(idx)

    # set random seeds accordingly
    hypers = exp.get_hypers(idx)
    if not first_hypers:
        first_hypers = hypers

    # validate that shape changing hypers are static.
    assert hypers.get("batch") == first_hypers.get("batch")
    assert hypers.get("buffer_size") == first_hypers.get("buffer_size")
    assert hypers.get("buffer_min_size") == first_hypers.get("buffer_min_size")
    assert hypers.get("environment", {}).get("aperture_size") == first_hypers.get(
        "environment", {}
    ).get("aperture_size")
    assert hypers.get("n_step") == first_hypers.get("n_step")
    assert hypers.get("optimizer", {}).get("name") == first_hypers.get(
        "optimizer", {}
    ).get("name")
    assert hypers.get("representation", {}).get("type") == first_hypers.get(
        "representation", {}
    ).get("type")
    assert hypers.get("representation", {}).get("hidden") == first_hypers.get(
        "representation", {}
    ).get("hidden")

    # build stateful things and attach to checkpoint
    problem = chk.build("p", lambda: Problem(exp, idx, collector))
    agent = chk.build("a", problem.getAgent)
    env = chk.build("e", problem.getEnvironment)

    glue = chk.build("glue", lambda: RlGlue(agent, env))
    glues.append(glue)

if len(glues) > 1:
    # vmap glue methods
    v_start = jax.vmap(glues[0]._start)
    v_step = jax.vmap(glues[0]._step)
else:
    v_start = glues[0]._start
    v_step = glues[0]._step

total_setup_time = time.time() - start_time
num_indices = len(indices)
logger.debug("--- Batch Set-up Timings ---")
logger.debug(
    f"Total setup time: {total_setup_time:.4f}s | Average: {total_setup_time / num_indices:.4f}s"
)

n = exp.total_steps

# render video of first env
if args.video:
    from gymnasium.utils.save_video import save_video
    first_glue = glues[0]
    first_idx = indices[0]
    first_state = first_glue._start(first_glue.state)[0]

    video_length = 1_000
    video_every = 100_000

    def video_step(state, _):
        frame = first_glue.environment.env.render(
            state.env_state.state, None, render_mode="world"
        )
        next_state, _ = first_glue._step(state)
        return next_state, frame

    def no_video_step(state, _):
        next_state, _ = first_glue._step(state)
        return next_state, None

    @scan_tqdm(n)
    def maybe_video_step(state, _):
        state, _ = jax.lax.scan(no_video_step, state, jnp.arange(video_every - video_length))
        state, frames = jax.lax.scan(video_step, state, jnp.arange(video_length))
        return state, frames

    _, framess = jax.lax.scan(maybe_video_step, first_state, jnp.arange(0, n, video_every))

    framess = np.asarray(framess)
    for i, frames in enumerate(framess):
        frames = [frames[i] for i in range(frames.shape[0])]

        context = exp.buildSaveContext(first_idx, base=args.save_path)
        start_frame = video_every * (i + 1) - video_length
        end_frame = start_frame + video_length
        video_path = context.resolve(f"videos/{first_idx}_{start_frame}_{end_frame}.mp4")
        context.ensureExists(video_path, is_file=True)
        save_video(
            frames,
            video_path,
            name_prefix="foragax",
            fps=8,
        )
    exit(0)

# --------------------
# -- Batch Execution --
# --------------------

start_step = None
save_every = 1_000_000
datas = {}
datas["rewards"] = np.empty((len(indices), n), dtype=np.float16)
datas["weight_change"] = np.empty((len(indices), n), dtype=np.float16)
datas["squared_td_error"] = np.empty((len(indices), n), dtype=np.float16)
datas["abs_td_error"] = np.empty((len(indices), n), dtype=np.float16)
datas["loss"] = np.empty((len(indices), n), dtype=np.float16)


def get_agent_metrics(agent_state, batch_shape):
    """Safely extract metrics from agent state, handling different agent types."""
    weight_change = jnp.zeros(batch_shape)
    squared_td_error = jnp.zeros(batch_shape)
    abs_td_error = jnp.zeros(batch_shape)
    loss = jnp.zeros(batch_shape)

    if hasattr(agent_state, "metrics"):
        metrics = agent_state.metrics
        if hasattr(metrics, "weight_change"):
            weight_change = metrics.weight_change
        if hasattr(metrics, "squared_td_error"):
            squared_td_error = metrics.squared_td_error
        if hasattr(metrics, "abs_td_error"):
            abs_td_error = metrics.abs_td_error
        if hasattr(metrics, "loss"):
            loss = metrics.loss

    return weight_change, squared_td_error, abs_td_error, loss


if isinstance(glues[0].environment, Foragax):
    datas["pos"] = np.empty((len(indices), n, 2), dtype=np.int32)
    def get_data(carry, interaction):
        weight_change, squared_td_error, abs_td_error, loss = get_agent_metrics(
            carry.agent_state, interaction.reward.shape
        )
        data = {
            "rewards": interaction.reward,
            "pos": carry.env_state.state.pos,
            "weight_change": weight_change,
            "squared_td_error": squared_td_error,
            "abs_td_error": abs_td_error,
            "loss": loss,
        }
        return data
else:
    def get_data(carry, interaction):
        weight_change, squared_td_error, abs_td_error, loss = get_agent_metrics(
            carry.agent_state, interaction.reward.shape
        )
        data = {
            "rewards": interaction.reward,
            "weight_change": weight_change,
            "squared_td_error": squared_td_error,
            "abs_td_error": abs_td_error,
            "loss": loss,
        }
        return data


for i, idx in enumerate(indices):
    context = exp.buildSaveContext(idx, base=args.checkpoint_path)
    step_path = context.resolve(f"{idx}/step.txt")
    glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
    data_path = context.resolve(f"{idx}/data.npz")
    if os.path.exists(step_path):
        with open(step_path, "r") as f:
            start_step_idx = int(f.read())
            if start_step is None:
                start_step = start_step_idx
            else:
                assert start_step == start_step_idx

        with lzma.open(glue_state_path, "rb") as f:
            glues[i].state = pickle.load(f)

        with np.load(data_path) as data_idx:
            for key in data_idx.keys():
                datas[key][i, : len(data_idx[key])] = data_idx[key]

if len(glues) > 1:
    # combine states
    glue_states = tree_map(lambda *leaves: jnp.stack(leaves), *[g.state for g in glues])
else:
    glue_states = glues[0].state

if start_step is None:
    glue_states, _ = v_start(glue_states)
    start_step = 0
else:
    logger.debug(f"Loaded checkpoints, resuming from step {start_step}")

for current_step in range(start_step, n, save_every):
    steps_in_iter = min(save_every, n - current_step)
    if steps_in_iter <= 0:
        break

    @scan_tqdm(n, print_rate=min(n // 20, 10000), initial=current_step)
    def step(carry, _):
        carry, interaction = v_step(carry)
        data = get_data(carry, interaction)
        return carry, data

    steps = jnp.arange(steps_in_iter)
    glue_states, data_chunk = jax.lax.scan(step, glue_states, steps, unroll=1)
    # data_chunk is dict of (steps, batch, ...)

    # checkpointing
    checkpoint_start_time = time.time()
    if len(glues) < 2:
        data_chunk = tree_map(lambda x: np.expand_dims(x, 1), data_chunk)

    for i, idx in enumerate(indices):
        data_idx = tree_map(lambda x: x[:, i], data_chunk)
        for key in datas:
            datas[key][i, current_step : current_step + steps_in_iter] = data_idx[key]

        if n < save_every:
            continue
        context = exp.buildSaveContext(idx, base=args.checkpoint_path)
        glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
        context.ensureExists(glue_state_path, is_file=True)
        if len(glues) > 1:
            glue_state_idx = tree_map(lambda x: x[i], glue_states)
        else:
            glue_state_idx = glue_states
        with lzma.open(glue_state_path, "wb") as f:
            pickle.dump(glue_state_idx, f)

        data_to_save = tree_map(
            lambda d: d[i, : current_step + steps_in_iter], datas
        )
        data_path = context.resolve(f"{idx}/data.npz")
        np.savez_compressed(data_path, **data_to_save)

        step_path = context.resolve(f"{idx}/step.txt")
        with open(step_path, "w") as f:
            f.write(str(current_step + steps_in_iter))
    checkpoint_time = time.time() - checkpoint_start_time
    logger.debug(
        f"Checkpointed at {current_step + steps_in_iter} in {checkpoint_time:.4f}s"
    )

# --------------------
# -- Saving --
# --------------------
total_numpy_time = 0
total_db_time = 0
num_indices = len(indices)
for i, idx in enumerate(indices):
    collector = collectors[i]

    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve("results.db")
    data_path = context.resolve(f"data/{idx}.npz")
    context.ensureExists(data_path, is_file=True)

    start_time = time.time()
    data_to_save = tree_map(lambda d: d[i], datas)
    np.savez_compressed(data_path, **data_to_save)
    total_numpy_time += time.time() - start_time

    meta = getParamsAsDict(exp, idx)
    meta |= {"seed": exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)

    start_time = time.time()
    collector.merge(context.resolve("results.db"))
    total_db_time += time.time() - start_time

    collector.close()

logger.debug("--- Saving Timings ---")
logger.debug(
    f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
)
logger.debug(
    f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
)
total_save_time = total_numpy_time + total_db_time
logger.debug(
    f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
)
