# python -m numpy.f2py -c mPitzer_w_NaCl.f90 -m Pitzer

f = open('mPitzer_w_NaCl.f90',mode='r')
from numpy import f2py

source="".join( f.readlines() )

f2py.compile(source=source,modulename='Pitzer',extension='.f90')

import Pitzer

a = Pitzer.mpitzer_w_nacl(mnacl=1,t_k=300)

print("a = ", a)

