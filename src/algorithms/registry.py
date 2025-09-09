from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN_L2_Init import DQN_L2_Init
from algorithms.nn.EQRC import EQRC
from algorithms.RandomAgent import RandomAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC


def getAgent(name) -> Type[BaseAgent]:
    if name == "DQN":
        return DQN

    if name == "DQN_L2_Init":
        return DQN_L2_Init

    if name == "EQRC":
        return EQRC

    if name == "ESARSA":
        return ESARSA

    if name == "SoftmaxAC":
        return SoftmaxAC

    if name == "Random":
        return RandomAgent

    raise Exception("Unknown algorithm")
