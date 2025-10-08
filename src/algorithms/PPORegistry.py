from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACConvEmb import RealTimeActorCriticConvEmb
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.ACMLP import ActorCriticMLP

def getAgent(name):
    # if name.startswith("RealTimeActorCriticConv"):
    #     return RealTimeActorCriticConv
    
    # if name.startswith("RealTimeActorCriticConvPooling"):
    #     return RealTimeActorCriticConvPooling
    
    # if name.startswith("RealTimeActorCriticConvEmb"):
    #     return RealTimeActorCriticConvEmb
    
    if name.startswith("RealTimeActorCriticMLPMulti"):
        return RealTimeActorCriticMLPMulti

    if name.startswith("RealTimeActorCriticMLP"):
        return RealTimeActorCriticMLP
    
    if name.startswith("ActorCriticMLP"):
        return ActorCriticMLP

    raise Exception("Unknown algorithm")
