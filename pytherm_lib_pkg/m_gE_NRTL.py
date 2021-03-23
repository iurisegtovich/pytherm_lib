"""
NRTL
uso:
alpha=[[0.,.4],[.4,0.]]
A=[[0.,1000.],[2000.,0.]]
ge=c_NRTL(alpha,A)
T=300
x=[.5,.5]
ge.gamma(T,x)
"""
import numpy as np

class c_NRTL():
    def __init__(self,q_alpha,q_A):
        self.q_alpha = np.asarray(q_alpha )
        self.q_A = np.asarray(q_A )
        
        
    def gamma(self,T,x):
    
        if type(isinstance(x,list)):
            n=len(x)
            c_x = np.asarray(x).reshape([n,1])
        elif isinstance(x, np.ndarray):
            if x.ndim==1:
                n=x.shape[0]
                c_x = x.reshape([n,1])
    
        return self.gamma_(T,c_x,self.q_alpha,self.q_A).reshape([n]) #outputs 1d

    def gamma_(self,T,c_x,q_alpha, q_A):
        '''T :  temperatura float escalar
        c_x : composição fração molar normalizada, float array1d matriz coluna
        q_alpha, q_tau : parametros, arra2d matriz quadrada'''
        

        
        q_tau     = q_A/T
        q_G       = np.exp(-(q_alpha*q_tau))
        l_D       = ((1/((q_G.T) @ c_x)).T)
        q_E       = (q_tau*q_G) * l_D 
        gamma     = np.exp(((q_E+(q_E.T))-(((q_G * l_D) * (c_x.T)) @ (q_E.T))) @ c_x)
        return gamma
