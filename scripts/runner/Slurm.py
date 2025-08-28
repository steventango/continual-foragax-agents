import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import PyExpUtils.runner.Slurm as Slurm


@dataclass
class SingleNodeOptions(Slurm.SingleNodeOptions):
    gpus: int | None = None

@dataclass
class MultiNodeOptions(Slurm.MultiNodeOptions):
    gpus: int | None = None


def fromFile(path: str):
    with open(path, "r") as f:
        d = json.load(f)

    assert "type" in d, "Need to specify scheduling strategy."
    t = d["type"]
    del d["type"]

    if t == "single_node":
        return SingleNodeOptions(**d)

    elif t == "multi_node":
        return MultiNodeOptions(**d)

    raise Exception("Unknown scheduling strategy")


def buildParallel(
    executable: str,
    tasks: Iterable[Any],
    opts: SingleNodeOptions | MultiNodeOptions,
    parallelOpts: Dict[str, Any] = {},
):
    threads = 1
    if isinstance(opts, SingleNodeOptions):
        threads = opts.threads_per_task

    cores = int(opts.cores / threads)

    gpu_str = f"--gpus-per-node={opts.gpus}" if opts.gpus else ""

    parallel_exec = f"srun -N1 -n{threads} --exclusive {gpu_str} {executable}"
    if isinstance(opts, SingleNodeOptions):
        parallel_exec = executable

    task_str = " ".join(map(str, tasks))
    return (
        f'run-parallel --parallel {cores} --exec "{parallel_exec}" --tasks {task_str}'
    )
