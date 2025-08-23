from ml_instrumentation.Collector import Collector
from environments.Foragax import Foragax as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class Foragax(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        env = Env(self.seed, **self.env_params)
        self.env = env
        self.actions = env.env.action_space(env.env.default_params).n

        self.observations = env.env.observation_space(env.env.default_params).shape
        self.gamma = 0.99
