from typing import Callable, Dict, Iterable, List

from PyExpUtils.models.ExperimentDescription import (
    ExperimentDescription,
    loadExperiment,
)
from utils.PyExpUtils.results.sqlite import detectMissingIndices


def gather_missing_indices(
    experiment_paths: Iterable[str],
    runs: int,
    loader: Callable[[str], ExperimentDescription] = loadExperiment,
    base: str = "./",
):
    path_to_indices: Dict[str, List[int]] = {}

    for path in experiment_paths:
        exp = loader(path)

        indices = detectMissingIndices(exp, runs, base=base)
        indices = sorted(indices)
        path_to_indices[path] = indices

        size = exp.numPermutations() * runs
        print(path, f"{len(indices)} / {size}")

    return path_to_indices
