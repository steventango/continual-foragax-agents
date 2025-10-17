from ml_instrumentation.Collector import Collector

from environments.Foragax import Foragax as Env
from environments.MCTSEnvWrapper import MCTSEnvWrapper
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class Foragax(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        env = Env(self.seed, **self.env_params)
        if self.exp.agent == "MCTS":
            env = MCTSEnvWrapper(env)

        self.env = env
        self.actions = env.env.action_space(env.env.default_params).n

        self.observations = env.env.observation_space(env.env.default_params).shape
        self.gamma = 0.99

    def getAgent(self):
        agent = super().getAgent()
        if self.exp.agent == "MCTS":
            agent.env = self.env  # type: ignore
        return agent
