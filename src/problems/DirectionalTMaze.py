from ml_instrumentation.Collector import Collector
from environments.DirectionalTMaze import DirectionalTMaze as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class DirectionalTMaze(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(corridor_length=self.env_params.get('corridor_length', 10), 
                       seed=self.seed)
        self.actions = 3
        self.observations = (2, 2, 3)
        self.gamma = 0.95
