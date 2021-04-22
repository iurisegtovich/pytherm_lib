#requirements
#numdifftools #pip install numdifftools

import numdifftools as nd
import numpy as np



def minimize_step(objF,x0): #hard_quasi_newton
    Lmyhess = nd.Hessian(objF) #hessian lambda function 
    myhess = Lmyhess(x0) #hessian call at x0
    myhessinv = np.linalg.inv(myhess) #inverse
    
    #confer
    #test_identidade = myhess @ myhessinv
    #test_cond = np.linalg.cond(myhess)
    #li, U = np.linalg.eig(myhess)
    #test_pd = li>0 #assert true ture...true
     
    Lmygrad=nd.Gradient(objF) #grad lambda function
    mygrad=Lmygrad(x0) #grad call at x0
    
    #confer
    #np.isclose(mygrad,mygrad*0.)
    
    #print('myhessinv',myhessinv)
    #print('mygrad',mygrad)
    
    step = -myhessinv @ mygrad #broadcasting rules make it like mxn @ nx1 and returns 1d array
    
    #print('step',step)
    #input('step')
    return step, mygrad, myhessinv

def minimize_linesearch(objF,x0,step):
    #line search causes many additional objF calls, but runs without new hessians and prevents pingponging with hessians@grads overshhots
    #golden
    
    #%% find abc bounds for alpha
    a=0
    b=1

    
    #fa=objF(a*step+x0)
    #fb=objF(b*step+x0)
    
    
    #if fb<fa: #a ok, falta acertar c
        #c=2
        #fc=objF(c*step+x0)

        #i=0
        #imax=10
        #while i<imax and not fc>fb: 
            #c=c*2 #towards inf
            #fc=objF(c*step+x0)        
            #i+=1
        
    #else: #c ok, falta acertar b
        #c=b*1
        #fc=fb*1
        
        #i=0
        #imax=10
        #while i<imax and not fb<fa:
            #b=b/2 #towards zero
            #fb=objF(b*step+x0)
            #i+=1
    
    #alpha = gss(f=lambda alpha:objF(alpha*step+x0),
                #a=b,
                #b=c)
    
    #print(alpha)
    #print(objF(alpha*step+x0))
    #input('alpha')
    #return alpha
    
    from scipy import optimize as opt
    #alpha = gss(f=lambda alpha:objF(x0 + alpha*step),
                #a=a, 
                #b=c)
    #assert(fa<fb<fc)
    
    a,b,c,fa,fb,fc,_=opt.bracket(func=lambda alpha:objF(x0 + alpha*step),xa=a,xb=b)
    
    abc=np.array([a,b,c])
    fabc=np.array([fa,fb,fc])
    idx = np.argsort(abc)
    a,b,c=abc[idx]
    fa,fb,fc=fabc[idx]
    
    #print(a,b,c,fa,fb,fc)
    #input('paused')
    alpha=opt.golden(func=lambda alpha:objF(x0 + alpha*step), brack=(a, b, c))
    
    #print(alpha)
    #print(objF(alpha*step+x0))
    #input('alpha')
    return alpha    

#def gss(f,a,b):
    #invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    #invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2
    #tol=1e-3 #loose criteria for gss
    #h = b-a

    #Required steps to achieve tolerance
    #n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    #c = a + invphi2 * h
    #d = a + invphi * h
    #yc = f(c)
    #yd = f(d)

    #for k in range(n-1):
        #if yc < yd:
            #b = d
            #d = c
            #yd = yc
            #h = invphi * h
            #c = a + invphi2 * h
            #yc = f(c)
        #else:
            #a = c
            #c = d
            #yc = yd
            #h = invphi * h
            #d = a + invphi * h
            #yd = f(d)

    #if yc < yd:
        #alpha= (a+ d)/2
    #else:
        #alpha= (c+ b)/2

    #return alpha

def minimize(objF,x0):
    n=len(x0)
    
    res=1.
    tol=1e-4
    
    i=0
    imax=100
    
    f0=objF(x0)
    
    while res>tol and i<imax:
        xn=x0*1 #copy
        fn=f0*1 #copy
        
        step,mygrad,myhessinv = minimize_step(objF,xn)
        alpha = minimize_linesearch(objF,xn,step)
        
        x0 = xn + step*alpha
        
        f0=objF(x0)
        
        res = np.linalg.norm(x0-xn)/n #res x
        #res = np.abs(f0-fn) #res f
        
        i+=1
    
    #grad and hess after linesearch
    Lmyhess = nd.Hessian(objF) #hessian lambda function 
    myhess = Lmyhess(x0) #hessian call at x0
    myhessinv = np.linalg.inv(myhess) #inverse
    Lmygrad=nd.Gradient(objF) #grad lambda function
    mygrad=Lmygrad(x0) #grad call at x0
    #print('i',i)
    return x0, f0, mygrad, myhessinv,i
        
def test1():
    
    y = lambda x: np.exp(np.sum( .819273*x**2+.91823*x-.9236 )) #array in, scalar out
    
    xx, xf, xg, xh, nit = minimize(y,np.array([1.,2.]))
    
    print('xopt, fopt',xx)
    
    from scipy import optimize as opt
    ans=opt.minimize(y,np.array([1.,2.]))
    print(ans)
    
if __name__ == '__main__':
    test1()

#atolx
#atolf
#rtolx
#rtolf
#args
#linesearch
#prints
#asserts

def GaussNewton_STEP(model,par0,XEXP,YEXP,VAREXP,YC,NEXP,NVENT,NVSAI,NPAR):  #NOT TESTED
    
    for k in range(NEXP):
        YC[k,:]=model( XEXP[k,:], par0)
    
    DYO=YC-YEXP
    
    DFP = np.zeros([NEXP,NVSAI, NPAR])  #versao versatil para evitar vetores muito grandes, pode ser 100,1 ou 1,100 ou 10,10 e o resultado dá igual, só muda a gestão de mem
    
    
    for k in range(NEXP):
        Lmodel = lambda par: model( XEXP[k,:], par, )
        Lmygrad=nd.Gradient( Lmodel ) #grad lambda function
        mygrad=Lmygrad(par0) #grad call at x0
        print(mygrad)
        #input('mygrad')
        DFP[k,:,:] = mygrad #*1
    
    U = np.zeros([NPAR])
    for k in range(NEXP): #POSSO USAR UM FLATTEN EM NEXP*NVSAI
        U += DFP[k,:,:].T @ ( np.diag(1/VAREXP[k,:]) @ DYO[k,:])
    # = 1/2 * grad
    
    T=np.zeros([NPAR,NPAR]) #ok
    
    for k in range(NEXP):
        T += DFP[k,:,:].T @ ( np.diag(1/VAREXP[k,:]) @ DFP[k,:,:]) #diag(1/v), se fizer 1/diag(v) vai ter termos infinitos
    
    cova = np.linalg.inv(T) # = 2 * hessinv
    
    delP = - cova @ U.T #=2hesinv*1/2grad=hesinv*grad    
    
    return delP

#wlsq objf for GNs
def wlsqobjF(model,par0,XEXP,YEXP,VAREXP,YC,NEXP,NVENT,NVSAI,NPAR): #NOT TESTED
    #give YC memory
    f=0
    for k in range(NEXP):
        YC[k,:]=model( XEXP[k,:], par0)
        DYk=YC[k,:]-YEXP[k,:]
        f += DYk.T @ np.diag(1/VAREXP[k,:]) @ DYk
    return f

def GaussNewton_loop(model,x0,XEXP,YEXP,VAREXP,NEXP,NVENT,NVSAI,NPAR): #NOT TESTED
    
    YC=np.zeros_like(YEXP)
    
    n=len(x0)
    
    res=1.
    tol=1e-4
    
    i=0
    imax=100
    
    def objF(par0):
        return wlsqobjF(model,par0,XEXP,YEXP,VAREXP,YC,NEXP,NVENT,NVSAI,NPAR)    
    f0=objF(x0)
    
    while res>tol and i<imax:
        xn=x0*1 #copy
        fn=f0*1 #copy
        
        step = GaussNewton_STEP(model,xn,XEXP,YEXP,VAREXP,YC,NEXP,NVENT,NVSAI,NPAR)
        
        alpha = minimize_linesearch(objF,xn,step)
        
        x0 = xn + step*alpha
        
        f0=objF(x0)
        
        res = np.linalg.norm(x0-xn)/n #res x
        #res = np.abs(f0-fn) #res f
        
        i+=1
        return x0

def test2():
    NEXP=100
    NVENT=2
    NVSAI=3
    NPAR=3
    XEXP=np.random.random((100,2))
    YEXP=np.random.random((100,3))
    VAREXP=np.random.random((100,3))
    par = np.array([ .8193,.1238,.1245 ])
    
    def model(X, par):
        YC=np.zeros(3)
        YC[0]=np.exp(X[0]*par[0]) + (X[1]*par[1])**2
        YC[1]=X[0]*X[1]*par[0]*par[2]
        YC[2]=X[0]*par[1]/(X[1]*par[2])
        return YC
    
    for k in range(100):
        YEXP[k,:]=model(XEXP[k,:],par) + np.random.random(3)/100
        
    print(YEXP)
    input('paused')
    ans=GaussNewton_loop(model,par,XEXP,YEXP,VAREXP,NEXP,NVENT,NVSAI,NPAR) #NOT TESTED
    print(ans)
    
if __name__ == '__main__':
    test2()
