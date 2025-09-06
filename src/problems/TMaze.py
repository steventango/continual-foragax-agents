# Implement T-maze from https://papers.nips.cc/paper/2001/hash/a38b16173474ba8b1a95bcbc30d3b8a5-Abstract.html
from ml_instrumentation.Collector import Collector
from environments.TMaze import TMaze as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class TMaze(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(corridor_length=self.env_params.get('corridor_length', 10), 
                       seed=self.seed)
        self.actions = 4
        self.observations = (2, 2, 3)
        self.gamma = 0.95
