#Px->Ty requires ~psat list of callables as psat(t)~
#Tx->py
#Py->Tx   reuires ~psat list of callables as psat(t)~
#Ty->px
import numpy as np

#easyest and more relevant:
#Tx -> Py
def BubblePy(T,x,eos,ge,psat):
    """temp scalar
       x array
       eos callable as eos.fug and eos.vol
       ge callable as ge.gamma
       psat list of psat at t
       ref Sandler 4th"""
       
    psat=np.asarray(psat)
    x=np.asarray(x)
    #estimativa inicial de Pbolha
    ncomp=len(x)
    fugl=np.zeros(ncomp)
    
    for i in range(ncomp):
        if psat[i] is not None:
            fugl[i]=psat[i]
        else:
            hiP=eos.Pc[i]*1. #sempre tem líq de Pc para cima, vamos usar fugL em Pc e esquecer a correção de poynt seja para + ou - pressão
            xpure=np.zeros(ncomp)
            xpure[i]=1.
            Vm_L,Vm_V = eos.Volume(T,hiP,xpure)
            phi_L=eos.fugacity_coeff(T,Vm_L,xpure)
            #print('phiv',Vm_V,phi_V)
            fugl[i]=phi_L[i]*x[i]*hiP #vetorial     
            
    P = np.sum(fugl*x) #vetorial
    #print('guess',P/1e5)
    y = x*fugl/P #vetorial
    #print('guess',y)
    
    j=0
    res_loopP=1
    soma_y=np.sum(y)
    
    y=y/soma_y #guess
    soma_y=1.
    
    tol_loopP=1e-6
    jmax=100
    while (res_loopP>tol_loopP and j<jmax):
        
        #fugacidade do liquido
        
        Vm_L,Vm_V = eos.Volume(T,P,x)
        phi_L=eos.fugacity_coeff(T,Vm_L,x)
        #print('phiv',Vm_V,phi_V)
        f_L=phi_L*x*P #vetorial        
        
        gamma = ge.gamma(T,x)
        f_L=gamma*x*fugl
        #print('gamma',gamma)
        
        res_loopY=1 #qq coisa maior q 1e-6
        tol_loopy=1e-6
        i=0
        imax=100
        while (res_loopY>tol_loopy and i<imax):
            #fugacidade do vapor
            Vm_L,Vm_V = eos.Volume(T,P,y/soma_y)
            phi_V=eos.fugacity_coeff(T,Vm_V,y/soma_y)
            #print('phiv',Vm_V,phi_V)
            f_V=phi_V*y*P #vetorial

            #calculando yi'
            y_novo=y*f_L/f_V

            res=y_novo-y #vetorial

            res_loopY=np.linalg.norm(res)

            soma_y=np.sum(y_novo) #normalizacao dos y p/ entrar na eq de estado como uma fracao

            y=y_novo*1 #cópia do array essencial para o método numérico #salvando o novo como o 'y velho' da iteracao anterior

            i=i+1
            #print(y)

        res_loopP=abs(soma_y-1)
        P=P*soma_y #atualiza P
        #print("Pbar=",P/1e5)
        j=j+1
    return  P, y, i, j
