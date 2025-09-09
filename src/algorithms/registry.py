from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.RandomAgent import RandomAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC
from algorithms.nn.W0_DQN import W0_DQN
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC


def getAgent(name) -> Type[BaseAgent]:
    if name == "W0-DQN":
        return W0_DQN

    if name.startswith("DQN"):
        return DQN

    if name == "EQRC":
        return EQRC

    if name == "ESARSA":
        return ESARSA

    if name == "SoftmaxAC":
        return SoftmaxAC

    if name == "Random":
        return RandomAgent

    raise Exception("Unknown algorithm")
