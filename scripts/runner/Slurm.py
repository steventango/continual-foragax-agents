import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.utils.cmdline import flagString


@dataclass
class SingleNodeOptions(Slurm.SingleNodeOptions):
    gpus: int | str | None = None
    tasks_per_core: int = 1


@dataclass
class MultiNodeOptions(Slurm.MultiNodeOptions):
    gpus: int | str | None = None
    tasks_per_core: int = 1


def check_account(account: str):
    assert (
        account.startswith("rrg-")
        or account.startswith("def-")
        or account.startswith("aip-")
    )
    assert not account.endswith("_cpu") and not account.endswith("_gpu")


def shared_validation(options: SingleNodeOptions | MultiNodeOptions):
    check_account(options.account)
    Slurm.check_time(options.time)
    options.mem_per_core = Slurm.normalize_memory(options.mem_per_core)


def single_validation(options: SingleNodeOptions):
    shared_validation(options)
    # TODO: validate that the current cluster has nodes that can handle the specified request


def multi_validation(options: MultiNodeOptions):
    shared_validation(options)


def validate(options: SingleNodeOptions | MultiNodeOptions):
    if isinstance(options, SingleNodeOptions):
        single_validation(options)
    elif isinstance(options, MultiNodeOptions):
        multi_validation(options)


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


def to_cmdline_flags(
    options: SingleNodeOptions | MultiNodeOptions,
    skip_validation: bool = False,
) -> str:
    if not skip_validation:
        validate(options)

    args = [
        ("--account", options.account),
        ("--time", options.time),
        ("--mem-per-cpu", options.mem_per_core),
        ("--output", options.log_path),
    ]

    if isinstance(options, SingleNodeOptions):
        args += [
            ("--ntasks", options.cores),
            ("--nodes", 1),
            ("--cpus-per-task", 1),
        ]

    elif isinstance(options, MultiNodeOptions):
        args += [
            ("--ntasks", options.cores),
            ("--cpus-per-task", 1),
        ]

    if options.gpus is not None:
        args += [
            ("--gpus-per-node", options.gpus),
        ]

    return flagString(args)


def buildParallel(
    executable: str,
    tasks: Iterable[Any],
    opts: SingleNodeOptions | MultiNodeOptions,
    parallelOpts: Dict[str, Any] = {},
):
    threads = 1
    tasks_per_core = 1
    if isinstance(opts, SingleNodeOptions):
        threads = opts.threads_per_task
        tasks_per_core = opts.tasks_per_core

    jobs = int(opts.cores / threads) * tasks_per_core

    parallel_exec = f"srun -N1 -n{threads} --exclusive {executable}"
    if isinstance(opts, SingleNodeOptions):
        parallel_exec = executable

    task_str = " ".join(map(str, tasks))
    return f'run-parallel --parallel {jobs} --exec "{parallel_exec}" --tasks {task_str}'


def schedule(
    script: str,
    opts: Optional[SingleNodeOptions | MultiNodeOptions] = None,
    script_name: str = "auto_slurm.sh",
    cleanup: bool = True,
    skip_validation: bool = False,
) -> None:
    with open(script_name, "w") as f:
        f.write(script)

    cmdArgs = ""
    if opts is not None:
        cmdArgs = to_cmdline_flags(opts, skip_validation=skip_validation)

    os.system(f"sbatch {cmdArgs} {script_name}")

    if cleanup:
        os.remove(script_name)
