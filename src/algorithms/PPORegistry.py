from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.ACConv import ActorCriticConv
from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.ESMAC import ESMAC


def getAgent(name):
    if name.startswith("RealTimeActorCriticConvPooling"):
        return RealTimeActorCriticConvPooling

    if name.startswith("RealTimeActorCriticConv"):
        return RealTimeActorCriticConv

    if name.startswith("RealTimeActorCriticMLPMulti"):
        return RealTimeActorCriticMLPMulti
    
    if name.startswith("ActorCriticConv"):
        return ActorCriticConv

    if name.startswith("RealTimeActorCriticMLP"):
        return RealTimeActorCriticMLP

    if name.startswith("ActorCriticMLP"):
        return ActorCriticMLP

    if name.startswith("ESMAC"):
        return ESMAC

    raise Exception("Unknown algorithm")
