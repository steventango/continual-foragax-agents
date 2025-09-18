from ml_instrumentation.Collector import Collector
from environments.Debug import Debug as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class Debug(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        env = Env(self.seed, **self.env_params)
        self.env = env
        self.actions = 1

        self.observations = 1
        self.gamma = 0.99
