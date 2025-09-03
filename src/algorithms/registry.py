from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN_Hare_and_Tortoise import DQN_Hare_and_Tortoise
from algorithms.nn.DQN_L2_Init import DQN_L2_Init
from algorithms.nn.DQN_Reset import DQN_Reset
from algorithms.nn.DQN_Shrink_and_Perturb import DQN_Shrink_and_Perturb
from algorithms.nn.DRQN import DRQN
from algorithms.nn.EQRC import EQRC
from algorithms.RandomAgent import RandomAgent
from algorithms.SearchAgent import SearchAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC


def getAgent(name) -> Type[BaseAgent]:
    if name == "DQN_L2_Init":
        return DQN_L2_Init
    if name == "DQN":
        return DQN
    
    if name == "DRQN":
        return DRQN

    if name.startswith("DQN_Reset"):
        return DQN_Reset

    if name.startswith("DQN_Shrink_and_Perturb"):
        return DQN_Shrink_and_Perturb

    if name.startswith("DQN_Hare_and_Tortoise"):
        return DQN_Hare_and_Tortoise

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

    if name.startswith("Search"):
        return SearchAgent


    raise Exception("Unknown algorithm")
