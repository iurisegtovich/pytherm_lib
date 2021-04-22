#Px->Ty requires ~psat list of callables as psat(t)~
#Tx->py
#Py->Tx   reuires ~psat list of callables as psat(t)~
#Ty->px
import numpy as np
#easyest and more relevant:
#Tx -> Py





# ----


#def RES_RR_BETA(z,K,BETA):
    #RES = 0.
    #for i in range(Ncomp):
        #RES += z[i]*( (K[i]-1.) / (1.+BETA*(K[i]-1.)) )            
    #return RES

def RES_RR_T(z,BETA,P,x,y,T,Ncomp,eos):
    RES = 0.
    K = update_K(T,P,x,y,Ncomp,eos)
    for i in range(Ncomp):
        RES += z[i]*( (K[i]-1.) / (1.+BETA*(K[i]-1.)) )            
    return RES



#def Newton_BETA(z,K,BETA):
    #RES=1
    #TOL=1e-9
    #MAXi=100
    #i=0
    #while (np.abs(RES)>TOL and i < MAXi):
        #RES=RES_RR(z,K,BETA)
        #step=1e-5
        #JAC=(RES_RR(z,K,BETA+step)-RES_RR(z,K,BETA-step))/(2*step)
        #BETA-=RES/JAC
        #i+=1
        ##print(i)
    #return BETA

def Newton_T(z,K,BETA,P,x,y,T,Ncomp,eos):
    RES=1
    TOL=1e-9
    MAXi=100
    i=0
    while (np.abs(RES)>TOL and i < MAXi):
        RES =RES_RR_T(z,BETA,P,x,y,T,Ncomp,eos)
        step=1e-5
        JAC=(RES_RR_T(z,BETA,P,x,y,T+step,Ncomp,eos)-RES_RR_T(z,BETA,P,x,y,T-step,Ncomp,eos))/(2*step)
        T-=RES/JAC
        i+=1
#        print(i)
    return T,i

def update_x(z,K,BETA,Ncomp):
    x=np.zeros(Ncomp)
    y=np.zeros(Ncomp)
    for i in range(Ncomp):
        x[i] = z[i]*( (1.) / (1+BETA*(K[i]-1.)) )            
        y[i] = K[i]*x[i]
    return x/np.sum(x), y/np.sum(y)

def update_K(T,P,x,y,Ncomp,eos):
    #print(T,P,x)
    VL,_=eos.Volume(T=T,P=P,x=x)
    _,VV=eos.Volume(T=T,P=P,x=y)

    phiL=eos.fugacity_coeff(T=T,V=VL,x=x)
    phiV=eos.fugacity_coeff(T=T,V=VV,x=y)

    K=phiL/phiV
    return K



def flash_PBeta(T,P,z,K,BETA,Ncomp,eos):
    """BETA=1 is dew
    BETA=0 is bub"""
    RES_flash=1
    TOL=1e-9
    T = T*1. #guess
    j=0
    MAXj=1000
    while (np.abs(RES_flash)>TOL and j<MAXj):
        K_ol=1.*K #copy
        x,y=update_x(z,K,BETA,Ncomp)
        T,i = Newton_T(z,K,BETA,P,x,y,T,Ncomp,eos)
        K = update_K(T,P,x,y,Ncomp,eos)
        RES_flash=np.linalg.norm(K_ol-K)
        j+=1
    return T,x,y,K,i,j

#ans = flash_PBeta(T=160,P=10e5,z=z,K=K_iguess,BETA=0.5)

#print("x=",ans[0])
#print("y=",ans[1])
#print("T=",ans[2])


#--

def BubbleTy(P,x,eos,guessT,guessy):
    """temp scalar
       x array
       eos callable as eos.fug and eos.vol
       ge callable as ge.gamma
       psat list of psat at t"""
       
    #estimativa inicial de Tbolha
    T = guessT*1.
    #print('guess',P/1e5)
    y = guessy*1
    #print('guess',y)
    x=np.asarray(x)
    
    ncomp=len(x)
    #print(T,P,x,y,)

    ans = flash_PBeta(T=T,P=P,z=x,K=y/x,BETA=0.,Ncomp=ncomp,eos=eos)
    
    T, _x, y, _K, i, j = ans
    return  T, y, i, j

