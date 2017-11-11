import numpy as np
from   numpy import exp, log, sqrt, maximum, minimum
import scipy.integrate as integrate
from solver.const import q, kBT, h_, h, pi

def LIFETIME():
   pass
def MOBILITY():
   pass

class TUNNELING:
   def __init__(self,t,mdiel,meff,l=1):
      self.__B = -4 * sqrt(2*mdiel) / (3* h_ * q) * t
      self.JC = 4*pi*meff* q /h**3 * q*kBT
      self.len = l

   def setEc(self,Ecl,Ecr,EBl,EBr):
      assert len(EBl) == len(EBr) == self.len,\
                        (len(EBl),len(EBr),self.len)
      assert len(Ecl) == len(Ecr) == self.len
      self.EBmax = maximum(EBl,EBr)
      self.EBmin = minimum(EBl,EBr)
      self.Ecmax = maximum(Ecl,Ecr)

   def WKB(self,E,i):
      A = 0
      if E > self.EBmax[i]:
         raise ValueError(("electron energy too high ({} > {}) WKB is used for tunneling only!!").format(E,Emax))
      elif E > self.EBmin[i]:
         ### FN tunneling
         A = (q*(self.EBmax[i]-E))**1.5
      else:
         ### Direct tunneling
         A = (q*(self.EBmax[i]-E))**1.5-(q*(self.EBmin[i]-E))**1.5

      TC = exp(self.__B/(self.EBmax[i]-self.EBmin[i])*A)
      return TC

   ### parameters should be arrays extracted from junctions
   def TSUESAKI(self,Efl,Efr):
      assert len(Efl)==len(Efr)==self.len,"Ef lengths not matched!"
      integrand = lambda Ex,i: log((1+exp(-(Ex-Efr[i])/kBT))/
                                   (1+exp(-(Ex-Efl[i])/kBT)))\
                              * self.WKB(Ex,i)
      J = [q * self.JC * integrate.quad(integrand,
                         self.Ecmax[i],self.EBmax[i],args=(i))[0]
                    for i in range(self.len)]
      return np.array(J)
