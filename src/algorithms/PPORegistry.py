from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACConvEmb import RealTimeActorCriticConvEmb
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP

def getAgent(name):
    if name.startswith("RealTimeActorCriticConv"):
        return RealTimeActorCriticConv
    
    if name.startswith("RealTimeActorCriticConvPooling"):
        return RealTimeActorCriticConvPooling
    
    if name.startswith("RealTimeActorCriticConvEmb"):
        return RealTimeActorCriticConvEmb
    
    if name.startswith("RealTimeActorCriticMLP"):
        return RealTimeActorCriticMLP

    raise Exception("Unknown algorithm")
