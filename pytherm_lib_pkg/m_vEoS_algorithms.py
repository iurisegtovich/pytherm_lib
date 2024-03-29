import numpy as np
from scipy import optimize as opt

def calc_P_sat(vEoS_obj,T,iguess_P):#,index):
    #variáveis de método numérico
    RES=1 #resíduo inicial 1 -> objetivo ~zero
    TOL=1e-7 #tolerancia -> objetivo res < tol
    MAX=100 #numero máximo de iterações
    #estimativa inicial
    P=iguess_P
    #contagem de iterações
    i=0
    while(RES>TOL and i<MAX): #Kernel > Interrupt (console > Ctrl+C)
        x=np.array([1.]) #substância pura
        #volumes da cúbica
        V_L,V_V=vEoS_obj.Volume(T=T, P=P, x=x)
        #conferir se veio canditato L e V ou se veio trivial
        if np.abs(V_L-V_V)<1e-9: #caso so tenha sido encontrada uma raiz, não será possível calcular P_sat para essa temperatura por esse método
            print('solução trivial')
            return np.nan, -1 #return sem variável pois não há solução, o código q chama essa função precisa saber lidar com esse resultado excepcional!
        #coeficientes de fugacidade
        phiL=vEoS_obj.fugacity_coeff(T=T, V=V_L, x=x)
        phiV=vEoS_obj.fugacity_coeff(T=T, V=V_V, x=x)
        #atualiza pressão (método sandler adaptado para puro)
        P=P*(phiL/phiV) #no equilibrio, quando phiL=phiV, phiL/phiV=1 e a pressao nova == pressao velha
        RES=np.abs(phiL/phiV-1.) #residuo (módulo de phil/phiv - 1) -> quero que phil==phiv ... phil/phiv == 1, phil/phiv-1 ==> 0
        i=i+1 #contador
    #print(T,i)
    return P, i
    
def calc_Psat_curve(vEoS_obj,gridT,guess_P=100.):
    #grid feito usando a função linspace, gera pontos igualmente espaçados em escala linear. No caso: 100 pontos, entre 100 e Tc, inclusive.
    n=len(gridT)
    #vetor de pressoes igual a zero, para ser preenchida aos poucos
    gridP=np.zeros(n)
    #efetua primeiro calculo
    gridP[0],_=calc_P_sat(vEoS_obj,gridT[0],guess_P) #primeiro ponto
    #efetua demais cálculos
    for i in range(1,n): #demais pontos
        #print(i)
        gridP[i],_=calc_P_sat(vEoS_obj,gridT[i],gridP[i-1])
        #print(grid_P[i]) #habilitar essa linha - removendo o símbolo de comentário -- # -- - faz com que os resultados de cada iteração sejam exibidos na seção de impressão da célula
    return gridT, gridP

#alternativa no psat
#x=zeros(ncomp), x[index]=1.
