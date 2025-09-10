import sys
import os

sys.path.append(os.getcwd() + "/src")

import random
import argparse
import subprocess
from functools import partial
from multiprocessing.pool import Pool

from utils.results import gather_missing_indices
import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--entry", type=str, default="src/main.py")
parser.add_argument("--results", type=str, default="./")
parser.add_argument("--gpu", action="store_true", default=False)


def count(pre, it):
    print(pre, 0, end="\r")
    for i, x in enumerate(it):
        print(pre, i + 1, end="\r")
        yield x

    print()


if __name__ == "__main__":
    cmdline = parser.parse_args()

    pool = Pool()

    env = os.environ.copy()

    cmds = []
    e_to_missing = gather_missing_indices(
        cmdline.e, cmdline.runs, loader=Experiment.load
    )
    for path in cmdline.e:
        exp = Experiment.load(path)

        indices = list(count(path, e_to_missing[path]))
        if len(indices) and cmdline.gpu:
            idxs = " ".join([str(idx) for idx in indices])
            exe = f"python {cmdline.entry} --gpu -e {path} -i {idxs}"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
            cmds.append(exe)
        else:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(0.95 / len(indices))
            env["JAX_PLATFORM_NAME"] = "cpu"
            for idx in indices:
                exe = f"python {cmdline.entry} --silent -e {path} -i {idx}"
                cmds.append(exe)

    print(len(cmds))
    random.shuffle(cmds)
    res = pool.imap_unordered(
        partial(
            subprocess.run, shell=True, stdout=subprocess.PIPE, env=env
        ),
        cmds,
        chunksize=1,
    )
    for i, _ in enumerate(res):
        sys.stderr.write(f"\r{i + 1}/{len(cmds)}")
    sys.stderr.write("\n")

    pool.close()
