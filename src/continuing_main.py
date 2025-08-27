import os
import sys

sys.path.append(os.getcwd())

import json
import time
import socket
import logging
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from utils.rlglue import RlGlue
from problems.registry import getProblem
from PyExpUtils.results.tools import getParamsAsDict
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from utils.ml_instrumentation.Sampler import Last, Mean
from ml_instrumentation.utils import Pipe
from ml_instrumentation.metadata import attach_metadata
from jax_tqdm.scan_pbar import scan_tqdm

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-i", "--idxs", nargs="+", type=int, required=True)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
parser.add_argument("--silent", action="store_true", default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
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
for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
    chk.load_if_exists()
    timeout_handler.before_cancel(chk.save)

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
                    Subsample(exp.total_steps // 1000),
                ),
                "mean_ewm_reward": Pipe(
                    MovingAverage(0.999),
                    Mean(),
                    Last(),
                ),
            },
            # by default, ignore keys that are not explicitly listed above
            default=Ignore(),
        ),
    )
    collector.set_experiment_id(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly
    hypers = exp.get_hypers(idx)
    seed = run + hypers.get("experiment", {}).get("seed_offset", 0)

    # build stateful things and attach to checkpoint
    problem = chk.build("p", lambda: Problem(exp, idx, collector))
    agent = chk.build("a", problem.getAgent)
    env = chk.build("e", problem.getEnvironment)

    glue = chk.build("glue", lambda: RlGlue(agent, env))

    # Run the experiment
    start_time = time.time()

    # if we haven't started yet, then make the first interaction
    glue_state = glue.state
    if glue.state.total_steps == 0:
        glue_state, _ = glue._start(glue.state)

    n = int(exp.total_steps - glue_state.total_steps)
    unroll = (2 ** jnp.abs(jnp.log10(n) - 3)).astype(int).item()

    @scan_tqdm(n, print_rate=max(n // 10, 1))
    def step(carry, _):
        carry, interaction = glue._step(carry)
        return carry, interaction.reward

    glue_state, rewards = jax.lax.scan(step, glue_state, jnp.arange(n), unroll=unroll)

    for reward in rewards:
        collector.next_frame()
        collector.collect("ewm_reward", reward.item())
        collector.collect("mean_ewm_reward", reward.item())

    collector.reset()
    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve("results.db")
    data_path = context.resolve(f"data/{idx}.npz")
    context.ensureExists(data_path, is_file=True)
    start = time.time()
    np.savez_compressed(data_path, rewards=rewards)
    end = time.time()
    meta = getParamsAsDict(exp, idx)
    meta |= {"seed": exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)
    collector.merge(context.resolve("results.db"))
    collector.close()
    chk.delete()
