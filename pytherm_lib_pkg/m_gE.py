"""
NRTL with parameters from
https://onlinelibrary.wiley.com/doi/book/10.1002/9781118477304
https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118477304.app2
and simple tests
plot gamma, plot GE, plot bubble vle P vs x given T, flash VLE, flash LLE
"""
import numpy as np

def gammaNRTL(T,c_x,q_alpha, q_tau):
    #note that we used many lines for didatics
    #we can do it in few lines:
    #note that some expression occur more than once below
    #so it may be useful define it as a intermediary recurrent term here
    #and calculate it once to use it then several times
    #q_tau     = q_A/T
    q_G       = np.exp(-(q_alpha*q_tau))
    l_D       = ((1/((q_G.T) @ c_x)).T)
    q_E       = (q_tau*q_G) * l_D 
    gamma     = np.exp(((q_E+(q_E.T))-(((q_G * l_D) * (c_x.T)) @ (q_E.T))) @ c_x)
    return gamma
