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
print('VL,VV,PL,PV,fL,fV,HrL,HrV,SrL,SrV')
print(VL,VV,PL,PV,fL,fV,HrL,HrV,SrL,SrV)

'''

#THE-LIBRARY#
import numpy as np
from scipy.constants import R as _R
#objetivo - código simples, sem polimorfismo, implementação de uma única EoS, toma Peng e Robinson como base, inclui modificações, fazer consistency checks nessa rotina (tipos e valores)
class c_vEoS(): #Peng Robinson
    def __init__(self,ncomp,Tc,Pc,acentric,k,iA=None,iApar=None): #roda uma vez para carregar as propriedades por componentes
               
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
            
        
        #vEoS specific parameters 
        self.sigma = 1.0 + np.sqrt(2.) #PR
        self.epsilon = 1.0 - np.sqrt(2.) #PR
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
            self.ac[i]                     = 0.45724*(_R**2)*((self.Tc[i])**2)/(self.Pc[i]) #PR
            self.bc[i]                     = 0.07780*_R*(self.Tc[i])/(Pc[i]) #PR

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
        
    def _f_Aalphamix(self,T,x):
        #regra de combinação classica com kij
        #regra de mistura quadrática
        Aalpha=self._f_Aalpha(T)
        Aalpham = 0.
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                                      #>>>Aalpha_comb[i,j]<<<<<<<<<<<<<<<<<<<<<<<<<#
                Aalpham += x[i]*x[j] * np.sqrt(Aalpha[i]*Aalpha[j])*(1.-self.k[i,j])
        
        #print('Aalpham=',Aalpham)
        return Aalpham, Aalpha
     
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

    def _f_dAalphadn_svnas(self,T,x):
        Aalpham, Aalpha = self._f_Aalphamix(T,x)
        dAalphadn = np.zeros(self.ncomp)
        sum1 = 0.
        for i in range(self.ncomp):
            sum1 = 0.
            for j in range(self.ncomp):
                sum1 += x[j]*np.sqrt(Aalpha[i]*Aalpha[j])*(1.-self.k[i,j])
            dAalphadn[i]=2*sum1 - Aalpham
        
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
    def fugacity_coeff(self,T,V,x): #for a vdw1f mixrule cubic eos with sigma!=epsilon
        x=np.asarray(x,dtype=np.float64)
        P=self.Pressure(T,V,x)
        dbdn,bm = self._f_dbdn(x)
        dAalphadn, Aalpham = self._f_dAalphadn(T,x)
        qsi = (1./(bm*(self.epsilon-self.sigma)))*np.log((V+self.epsilon*bm)/(V+self.sigma*bm))
        lnPhi = np.zeros(self.ncomp)
        lnPhi_svna = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            lnPhi[i] = ( #multiline
                (dbdn[i]/bm)*((P*V)/(_R*T)-1.) #&
                -np.log(P*(V-bm)/(_R*T)) #&
                -(Aalpham/(_R*T))*qsi*
                ((2.*dAalphadn[i]/Aalpham) -(dbdn[i]/bm))
                                 )#done
        #print('dq=',(Aalpham/(_R*T))*(1./(bm))*((2.*dAalphadn[:]/Aalpham) -(dbdn[:]/bm)) )
        
        #svna
        Isvna = (1./(self.epsilon-self.sigma)) * np.log((V+self.epsilon*bm)/(V+self.sigma*bm))
        q=Aalpham/(bm*_R*T)
        dAalphadn_svnas, Aalpham = self._f_dAalphadn_svnas(T,x)
        dqdn=q*(1+dAalphadn_svnas[:]/Aalpham-dbdn[:]/bm)
        for i in range(self.ncomp):
            lnPhi_svna[i] = ( #multiline
                (dbdn[i]/bm)*((P*V)/(_R*T)-1.) #&
                -np.log(P*(V-bm)/(_R*T)) #&
                -dqdn[i]*Isvna
                                 )#done
        
        #print('...',lnPhi_svna)
        #print('>>>',lnPhi)
        phi = np.exp(lnPhi_svna)
        return phi

    def _f_dAalphadT(self,T,x):
        dalphadT=np.zeros(self.ncomp)
        dAalphadT=np.zeros(self.ncomp)
        for i in range(self.ncomp):
            #alpha[i] = (1. + self.kPR[i]*(1.-np.sqrt(T/self.Tc[i])))**2
            dalphadT[i] = ( #multiline
                2.*(1. + self.kPR[i]*(1.-np.sqrt(T/self.Tc[i])))*
                self.kPR[i]*(-1.)*(
                1./(2.*(np.sqrt(T/self.Tc[i])))
                )*(1./self.Tc[i])
            )
            dAalphadT[i] = self.ac[i]*dalphadT[i]
        return dAalphadT

    def _f_dAalphamdT(self,T,x):
        _, Aalpha=self._f_Aalphamix(T,x)
        dAalphadT=self._f_dAalphadT(T,x)

        dAalphamdT = 0.
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                dAalphamdT += ( #multiline
                    x[i]*x[j]*
                    (1./(2.*np.sqrt(Aalpha[i]*Aalpha[j])))*
                    (Aalpha[i]*dAalphadT[j]+dAalphadT[i]*Aalpha[j])*
                    (1.-self.k[i,j])
                )
        #print(dAalphamdT)
        #numerical
        #AalphammaisT,_=self._f_Aalphamix(T+1e-3,x)
        #AalphammenosT,_=self._f_Aalphamix(T-1e-3,x)
        #dAalphamdT=(AalphammaisT-AalphammenosT)/(2.*1e-3)
        #print(dAalphamdT)
        return dAalphamdT

    #other spec flashes
    def f_H_res(self,T,V,x):
        x=np.asarray(x,dtype=np.float64)
        P=self.Pressure(T,V,x)
        bm=self._f_bmix(x)
        dAalphamdT = self._f_dAalphamdT(T,x)
        Aalpham, _ = self._f_Aalphamix(T,x)
        qsi = (1./(bm*(self.epsilon-self.sigma)))*np.log((V+self.epsilon*bm)/(V+self.sigma*bm)) #again
        H_res = P*V - _R*T + (T*dAalphamdT-Aalpham)*qsi
#        print(Aalpham)
        return H_res

    def f_S_res(self,T,V,x):
        x=np.asarray(x,dtype=np.float64)
        P=self.Pressure(T,V,x)
        bm=self._f_bmix(x)
        dAalphamdT = self._f_dAalphamdT(T,x)
        Aalpham, Aalpha = self._f_Aalphamix(T,x)
        qsi = (1./(bm*(self.epsilon-self.sigma)))*np.log((V+self.epsilon*bm)/(V+self.sigma*bm)) #again
        S_res = _R*np.log((P*(V-bm))/(_R*T)) + dAalphamdT*qsi
        return S_res

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
    fL=vEoS_obj.fugacity_coeff(T,VL,x)
    print("fL :",fL)
    fV=vEoS_obj.fugacity_coeff(T,VV,x)
    print("fV :",fV)
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
    
