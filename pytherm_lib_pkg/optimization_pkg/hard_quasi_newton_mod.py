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

    
    fa=objF(a*step+x0)
    fb=objF(b*step+x0)
    
    
    if fb<fa: #b ok, falta acertar x
        c=2
        fc=objF(c*step+x0)

        i=0
        imax=10
        while i<imax and not fc>fb: 
            c=c*2 #towards inf
            fc=objF(c*step+x0)        
            i+=1
        
    else: #c ok, falta acertar b
        c=b*1
        fc=fb*1
        
        i=0
        imax=10
        while i<imax and not fb<fa:
            b=b/2 #towards zero
            fb=objF(b*step+x0)
            i+=1
    
    alpha = gss(f=lambda alpha:objF(alpha*step+x0),
                a=a,
                b=c)
    
    print(alpha)
    input('alpha')
    return alpha

def gss(f,a,b):
    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2
    tol=1e-3 #loose criteria for gss
    h = b-a

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        alpha= (a+ d)/2
    else:
        alpha= (c+ b)/2

    return alpha

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
