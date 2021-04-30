'''to do: regra de mistura com ge'''

'''equação de estado cúbica (peng robinson)
com regra de combinação clássica
e regras de mistura b:linear e a:quadrática
uso:

ncomp=2
cnames=["co2",     "benzene"]
Tc = [304.1, 562., ] #K floats
Pc = [73.8e5,    48.9e5, ] #Pa floats
acentric = [0.239,    0.212, ] #dimensionless floats
k = [[0., 0.,],
              [0.,0.,],] #dimensionless floats

eos=m_vEoS.c_vEoS(ncomp,Tc,Pc,acentric,k)

T=283.    #K
P=40e5 #Pa
x=[0.93, 0.07,] #fração normaliada; dtype=np.float64

VL,VV=eos.Volume(T,P,x)
PL=eos.Pressure(T,VL,x)
PV=eos.Pressure(T,VV,x)
phiL=eos.fugacity_coeff(T,VL,x)
phiV=eos.fugacity_coeff(T,VV,x)
HrL=eos.f_H_res(T,VL,x)
HrV=eos.f_H_res(T,VV,x)
SrL=eos.f_S_res(T,VL,x)
SrV=eos.f_S_res(T,VV,x)    
print('VL,VV,PL,PV,phiL,phiV,fL,fV,HrL,HrV,SrL,SrV')
print(VL,VV,PL,PV,fL,fV,HrL,PL*x*phiL,PV*x*phiV,HrV,SrL,SrV)

'''

#THE-LIBRARY#
import numpy as np
from scipy.constants import R as _R
#objetivo - código simples, sem polimorfismo, implementação de uma única EoS, toma Peng e Robinson como base, inclui modificações, fazer consistency checks nessa rotina (tipos e valores)
class c_vEoS(): #Peng Robinson
    def __init__(self,ncomp,Tc,Pc,acentric,k,iA=None,iApar=None,q_alpha=None,q_A=None): #roda uma vez para carregar as propriedades por componentes
               
        #Array dimensioning info
        self.ncomp = ncomp
        
        if iA is None: #termo atrativo
            self.iA=np.array( self.ncomp * ["PR76"] )
        else:
            self.iA=iA
        
        if iApar is None: #termo atrativo
            self.iApar = np.array([ [ acentric[i] ] for i in range(self.ncomp) ])
            #w_i=iApar[i,0]
        else:
            self.iApar=iApar
            #w_i = iApar[i,0] #pr76
            #mj,nj,Gj = iApar[j,0:3] #prat
        
        
        #q_alpha=np.zeros([2,2])
        #q_A=np.zeros([2,2])
        #q_alpha[0,1]=q_alpha[1,0]=.4
        #q_A[0,1]= 99.422
        #q_A[1,0]= 372.298
        self.q_A=np.asarray(q_A)
        self.q_alpha=np.asarray(q_alpha)
        
        
        
        #vEoS specific parameters 
        self.sigma = 1.0 + np.sqrt(2.)
        self.epsilon = 1.0 - np.sqrt(2.)
        self.ac = np.zeros(self.ncomp)
        self.bc = np.zeros(self.ncomp)
        self.k = np.zeros([self.ncomp,self.ncomp])
        
        #Extracted pure component properties
        self.Tc = np.zeros(self.ncomp) #needed at every alpha updating
        self.Pc = np.zeros(self.ncomp) #not really needed after initialization
        self.acentric = np.zeros(self.ncomp) #needed at every alpha updating
        
        self.kPR = np.zeros(self.ncomp) #needed at every alpha updating
        
        for i in range(self.ncomp):
            self.Tc[i]                     = Tc[i]
            self.Pc[i]                     = Pc[i]
            self.acentric[i]         = acentric[i]
        
        for i in range(self.ncomp):
            self.ac[i]                     = 0.45724*(_R**2)*((self.Tc[i])**2)/(self.Pc[i])
            self.bc[i]                     = 0.07780*_R*(self.Tc[i])/(Pc[i])

            for j in range(self.ncomp):
                self.k[i,j]                = k[i][j] #accepts array or list, copies all parameters
                
            if self.iA[i]=="PR76":
                self.kPR[i]                    = 0.37464 + 1.54226*acentric[i]-0.26992*(acentric[i])**2
            elif self.iA[i]=="PR78":
                self.kPR[i]                    = 0.379642 + 1.48503*acentric[i]-0.164423*(acentric[i])**2 +1.016666*(acentric[i])**3
            else:
                ...
                
        return #NoneTypeObj

    def Pressure(self,T,V,x):
        x=np.asarray(x,dtype=np.float64)
        bm=self._f_bmix(x)
        Aalpham,Aalpha=self._f_Aalphamix(T,x)
        P = (_R*T)/(V-bm) - Aalpham/((V+self.sigma*bm)*(V+self.epsilon*bm)) 
        return P

    def _f_Aalpha(self,T):
        alpha=np.zeros(self.ncomp)
        Aalpha=np.zeros(self.ncomp)
        for i in range(self.ncomp):
            
            if self.iA[i] in ["PR76","PR78"]:
                alpha[i] = (1. +self.kPR[i]*(1.-np.sqrt(T/self.Tc[i])))**2 #kPR comes from wacentric
            elif self.iA[i]=="PRAT":
                m_i=self.iApar[i][0]
                n_i=self.iApar[i][1]
                t_i=self.iApar[i][2]
                tr_i=T/self.Tc[i]
                alpha[i] = np.exp(m_i*(1-tr_i)*(np.abs(1-tr_i)**(t_i-1))+n_i*(1/tr_i-1))
            else:
                ...
                
            Aalpha[i] = self.ac[i]*alpha[i]
        return Aalpha

    def _f_bmix(self,x):
        bm = 0.
        for i in range(self.ncomp):
            bm += x[i]*self.bc[i]
        return bm
        
    #def _f_Aalphamix(self,T,x):
        #Aalpha=self._f_Aalpha(T)
        #Aalpham = 0.
        #for i in range(self.ncomp):
            #for j in range(self.ncomp):
                #Aalpham += x[i]*x[j]*np.sqrt(Aalpha[i]*Aalpha[j])*(1.-self.k[i,j])
        #return Aalpham, Aalpha
     
    def _f_dbdn(self,x):
        bm=self._f_bmix(x)
        dbdn=np.zeros(self.ncomp)
        for i in range(self.ncomp):
            dbdn[i]=self.bc[i]
        return dbdn, bm
        
    def _f_dAalphadn(self,T,x):
        Aalpham, Aalpha = self._f_Aalphamix(T,x)
        dAalphadn = np.zeros(self.ncomp)
        sum1 = 0.
        for i in range(self.ncomp):
            sum1 = 0.
            for j in range(self.ncomp):
                sum1 += x[j]*np.sqrt(Aalpha[j])*(1.-self.k[i,j])
            dAalphadn[i]=np.sqrt(Aalpha[i])*sum1
        return dAalphadn, Aalpham

    def Volume(self,T,P,x):
        x=np.asarray(x,dtype=np.float64)
    # T em unidade K
    # P em unidade Pa
    # x array normalizado

        bm=self._f_bmix(x)
        Aalpham,_=self._f_Aalphamix(T,x)
     
        c0 = -(bm**3)*self.sigma*self.epsilon + (-_R*T*self.sigma*self.epsilon*(bm**2)-bm*Aalpham)/P
        c1 = (bm**2)*(self.sigma*self.epsilon-self.epsilon-self.sigma) + ((_R*T)*(-self.sigma*bm-self.epsilon*bm) + Aalpham)/P
        c2 = self.epsilon*bm+self.sigma*bm-bm-_R*T/P
        c3 = 1.
        #print("cs",c3,c2,c1,c0)
        
        Vs=np.roots([c3,c2,c1,c0])
        Vs[np.logical_not(np.isreal(Vs))]=0.
        Vs=np.real(Vs)
        return np.array([np.nanmin(Vs[Vs>bm]),np.nanmax(Vs[Vs>bm])])

    #phase equilibrium common
    #def fugacity_coeff(self,T,V,x): #for a vdw1f mixrule cubic eos with sigma!=epsilon
        #x=np.asarray(x,dtype=np.float64)
        #P=self.Pressure(T,V,x)
        #dbdn,bm = self._f_dbdn(x)
        #dAalphadn, Aalpham = self._f_dAalphadn(T,x)
        #qsi = (1./(bm*(self.epsilon-self.sigma)))*np.log((V+self.epsilon*bm)/(V+self.sigma*bm))
        #lnPhi = np.zeros(self.ncomp)
        #for i in range(self.ncomp):
            #lnPhi[i] = ( #multiline
                #(dbdn[i]/bm)*((P*V)/(_R*T)-1.) #&
                #-np.log(P*(V-bm)/(_R*T)) #&
                #-(Aalpham/(_R*T))*qsi*((2.*dAalphadn[i]/Aalpham) #&
                #-(dbdn[i]/bm))
                                 #)#done
        #phi = np.exp(lnPhi)
        #return phi









#####
    def f_gamma_NRTL(self,T,c_x,q_alpha, q_A):
        '''T :  temperatura float escalar
        c_x : composição fração molar normalizada, float array1d matriz coluna
        q_alpha, q_tau : parametros, arra2d matriz quadrada'''
        
        q_tau     = q_A/T
        q_G       = np.exp(-(q_alpha*q_tau))
        l_D       = ((1/((q_G.T) @ c_x)).T)
        q_E       = (q_tau*q_G) * l_D 
        gamma     = np.exp(((q_E+(q_E.T))-(((q_G * l_D) * (c_x.T)) @ (q_E.T))) @ c_x)
        
        ge = 0
        ncomp=q_A.shape[0]
        for i in range(ncomp):
            ge += _R*T*np.log(gamma[i])*c_x[i,0]
        
        return gamma, ge

        
    def _f_Aalphamix(self, T, x):
        
        Aalpha = self._f_Aalpha(T)
        
        bm = self._f_bmix(x)
        

        
        ncomp=2
        c_x = np.reshape(x,[ncomp,1])
        
        am = 0.
        _,ge0 = self.f_gamma_NRTL(T,c_x,self.q_alpha, self.q_A) #!!!
        #print('ge0=',ge0)
        A1 = -0.64663 #posso estimar isso ?

        bm = self._f_bmix(x=x) 
        am = 0.

        soma1 = 0
        soma2 = 0
        for i in range(self.ncomp):
            soma1 += x[i]*Aalpha[i]/(self.bc[i])  #ok
            soma2 += x[i]*np.log(bm/self.bc[i]) #ok/
            
        am = bm*(ge0/A1 + soma1 + _R*T/A1 * soma2) 
        #print('am=',am)
        return am, Aalpha
     
    def _f_dqdn(self, x,T):
        
        Aalpham, Aalpha = self._f_Aalphamix(T,x)
        
        
        
        A1 = -0.64663
        bm = self._f_bmix(x=x)
        
        q = Aalpha/(self.bc*_R*T)
        
        
        ncomp=2
        c_x = np.reshape(x,[ncomp,1])
        
        d_q = np.zeros(self.ncomp)
        gamma,_ = self.f_gamma_NRTL(T,c_x,self.q_alpha, self.q_A) #!!!
        gamma=gamma[:,0]
        
        for i in range(self.ncomp):
            d_q[i] = (1/A1) * (np.log(gamma[i]) + np.log(bm/self.bc[i])+ self.bc[i]/bm-1)  + q[i]  #conferir derivada - resultados ok
        #print('d_q 1',d_q)
        
        #alt
        #for i in range(self.ncomp):
            #d_q[i] = (1/A1) * (np.log(gamma[i]) + np.log(bm/self.bc[i]) )  + q[i]   #conferir derivada - resultados estranhos
        #print('d_q 2',d_q)
        return d_q, Aalpham
        
    
    def fugacity_coeff(self,T,V,x): #for a vdw1f mixrule cubic eos with sigma!=epsilon
        x=np.asarray(x,dtype=np.float64)
        P=self.Pressure(T,V,x)
        dbdn,bm = self._f_dbdn(x)
        lnPhi = np.zeros(self.ncomp)

        P = self.Pressure(x=x,T=T,V=V)

        d_q,am = self._f_dqdn(x=x,T=T)
        
        lnPhi = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            
            lnPhi[i] = ( 
                          (self.bc[i]/bm)*(P*V/(_R*T) - 1)  #svnas
                          - np.log(P*(V - bm)/(_R*T)) 
                          - d_q[i] * 1/(self.epsilon-self.sigma) * np.log((V+self.epsilon*bm)/(V+self.sigma*bm))
                       )
            #não tem um bi/bm aqui ...?
            
        phi = np.exp(lnPhi)
        
        return phi
    
###

def test():
    import numpy as np
    ncomp=5
    cnames=np.array(["co2",     "benzene", "ethane", "ethanol", "methane"])
    Tc = np.array([304.1, 562., 305.3, 513.9, 190.555]) #K
    Pc = np.array([73.8e5,    48.9e5, 48.714e5, 61.4e5, 45.95e5]) #Pa
    acentric = np.array([0.239,    0.212, 0.099, 0.644, 0.008]) #dimensionless
    k = np.array([[0,0.,0.,0.,0.],
                  [0.,0,0.,0.,0.],
                  [0.,0.,0,0.,0.],
                  [0.,0.,0.,0,0.],
                  [0.,0.,0.,0.,0],]) #dimensionless
    print("@ input/system")
    print("ncomp :",ncomp)
    print("cnames :",cnames)
    print("Tc :",Tc)
    print("Pc :",Pc)
    print("acentric :",acentric)
    print("k :",k)
    T=283.    #K
    P=40e5 #Pa
    x=np.array([0.93, 0.01, 0.03, 0.02, 0.01])
    print("@ input/condition")
    print("T :",T)
    print("P :",P)
    print("x :",x)
    vEoS_obj=c_vEoS(ncomp,Tc,Pc,acentric,k)
    #output
    print("@ output")
    VL,VV=vEoS_obj.Volume(T,P,x)
    print("VL :",VL)
    print("VV :",VV)
    PL=vEoS_obj.Pressure(T,VL,x)
    PV=vEoS_obj.Pressure(T,VV,x)
    print("PL = PV :",PL,"=",PV)
    phiL=vEoS_obj.fugacity_coeff(T,VL,x)
    print("phiL :",phiL)
    phiV=vEoS_obj.fugacity_coeff(T,VV,x)
    print("phiV :",phiV)
    
    print("fV :",phiV*x*PV)
    print("fL :",phiL*x*PL)
    
    HrL=vEoS_obj.f_H_res(T,VL,x)
    print("HrL :",HrL)
    HrV=vEoS_obj.f_H_res(T,VV,x)
    print("HrV :",HrV)
    SrL=vEoS_obj.f_S_res(T,VL,x)
    print("SrL :",SrL)
    SrV=vEoS_obj.f_S_res(T,VV,x)
    print("SrV :",SrV)

    #optimization issues
    #i will recalc bmix and amix when calling phi and when calling v
    #optimizing this would require that either bmix were public and required prior to calc V
    # or that there were both a public calcv(t,p,x) and a private calv(t,p,x,am,bm) and either class variable bm and am wih status checking at every call or combo calls calcv(tpx) calcphi(tpx) calcv_and_phi(tpx) combinatorially.
    
