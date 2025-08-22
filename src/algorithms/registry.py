from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC
from algorithms.nn_old.DQN import DQN as DQNO
from algorithms.nn_old.EQRC import EQRC as EQRCO

from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC

def getAgent(name) -> Type[BaseAgent]:
    if name == 'DQN':
        return DQN

    if name == 'EQRC':
        return EQRC

    if name == 'DQNO':
        return DQNO

    if name == 'EQRCO':
        return EQRCO

    if name == 'ESARSA':
        return ESARSA

    if name == 'SoftmaxAC':
        return SoftmaxAC

    raise Exception('Unknown algorithm')
