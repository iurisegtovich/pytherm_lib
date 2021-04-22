import numpy as np

def roundA(vT,prec,prec2=None):
    #prec should be xEy, i.e. q sig alg, any og
    vT=np.array(vT,dtype=np.float64)
    n=vT.shape[0]
    if prec==0:
        pass
    elif prec == 1.: #prec = minimum distance between succeccive elements of return array
        vT = [round(vT[i],0) for i in range(n)]#+- 0.5
    elif prec == 0.1:
        f=10 #1/prec
        vT =  [round(f*vT[i],0)/f for i in range(n)] #+- 0.5
    else: # prec == xEy:
        f=1/prec
        vT =  [round(f*vT[i],0)/f for i in range(n)] #+- 0.5
        if prec2 is not None:
            vT = np.array( [round(vT[i],prec2) for i in range(n)] )
    
    levels=sorted([*set(vT)])
    n=len(levels)
    return vT, levels, n

#roundPs, *_ = np.power(10,roundA(np.log10(vPs),1.,)) #each 10*j pressure level

#roundTs, *_ = roundA(vTs,1.,) #each integer-ish temperature level
