#!/usr/bin/env python
# coding: utf-8

import numpy as np

# # Flash-$TP$

# ## Initialize the EoS modeling for liquid and vapor using given pure substance properties.

# # provide initial guess for $\underline K$ and $\beta$

# In[7]:

def psat_wilson(T,P,Tc,Pc,acentric):
    #scalar or array
    #kwilson=psat/p
    return Pc*np.exp(+5.373*(1.+acentric)*(1.-Tc/T)) #Wilson 

# initial guesses
#K_iguess = np.exp(np.log(Pc/P)+5.373*(1.+acentric)*(1.-Tc/T)) #Wilson 
#K_iguess

#BETA_iguess = .5

# # Eq. 1 - The Rachford-Rice residue function

# In[9]:


def RES_RR(z,K,BETA,ncomp):
    RES = 0.
    for i in range(ncomp):
        RES += z[i]*( (K[i]-1.) / (1.+BETA*(K[i]-1.)) )            
    return RES


# In[10]:


def Newton(z,K,BETA,ncomp):
    RES=1
    TOL=1e-9
    MAXi=100
    i=0
    while (np.abs(RES)>TOL and i < MAXi):
        RES=RES_RR(z,K,BETA,ncomp)
        step=1e-5
        JAC=(RES_RR(z,K,BETA+step,ncomp)-RES_RR(z,K,BETA-step,ncomp))/(2*step)
        BETA-=RES/JAC
        i+=1
#         print(i)
    return BETA, i

def update_x(z,K,BETA,ncomp):
    x=np.zeros(ncomp)
    y=np.zeros(ncomp)
    for i in range(ncomp):
        x[i] = z[i]*( (1.) / (1+BETA*(K[i]-1.)) )            
        y[i] = K[i]*x[i]
    return x/np.sum(x), y/np.sum(y)


# In[14]:


def flash_TP(T,P,z,K,BETA,ncomp,eos):
    RES_flash=1
    TOL=1e-9

    j=0
    ii=0
    MAXj=100
    while (np.abs(RES_flash)>TOL and j<MAXj):
        K_ol=1.*K #copy
        #given K,calc BETA
        BETA,i=Newton(z,K,BETA)        
        ii+=i #acumulator for all NR iterations
        #given K,BETA calc x
        x,y=update_x(z,K,BETA)
#         print("x=",x)
#         print("y=",y)
        #given x calc fugacity_coeffs
        VL,_=eos.Volume(T=T,P=P,x=x) #expecting 2 sorted physically meaningful roots , discards the 2nd
        _,VV=eos.Volume(T=T,P=P,x=y) #expecting 2 sorted physically meaningful roots , discards the 1st
#         print(VL,VV)
        phiL=eos.fugacity_coeff(T=T,V=VL,x=x)
        phiV=eos.fugacity_coeff(T=T,V=VV,x=y)
#         print(phiL,phiV)
        #update K
        K=phiL/phiV #update K
#         print(K_ol)
#         print(K)
        RES_flash=np.linalg.norm(K_ol-K)
#         print(RES_flash)
        j+=1
    return x,y,BETA, ii, j

#ans = flash_TP(T=283,P=40e5,z=z,K=K_iguess,BETA=BETA_iguess)

#print("x=",ans[0])
#print("y=",ans[1])
#print("BETA=",ans[2])

#print("nit_NR (total)",ans[3], '(ave)', ans[3]/ans[4]//1)
#print("nit_xy",ans[4])


#sanity check
#equilibrium and mass balance conditions

#In[15]:


#x=ans[0]
#y=ans[1]
#BETA=ans[2]
#print("Beta=",BETA)

#sanity check
#VL=Volume(T=283, P=40e5, x=x)[0]
#print('VL=',VL)
#VV=Volume(T=283, P=40e5, x=y)[1]
#print('VV=',VV)
#phiL=fugacity_coeff(T=283,V=VL,x=x)
#print('phiL=',phiL)
#phiV=fugacity_coeff(T=283,V=VV,x=y)
#print('phiV=',phiV)

#print("x",x)
#print("x?N",np.sum(x))
#print("fL",phiL*x, "=?")
#print("fV",phiV*y)
#print("y",y)
#print("y?N",np.sum(y))

#print("x(1-B)+yB",y*(BETA)+x*(1-BETA))
#print("=z       ",z)


#In[ ]:




