import numpy as np

from mysolver._solver import solver1D, solver2D
from mysolver.util    import solve_diag2
from mysolver.const   import *
mn = 1350000
mp = 450000

class dd_solver(solver1D) :

   def __init__ (self, dx, N):
      solver1D.__init__(self,dx,N)
      self._n_ = self.n[1:N+1]
      self._p_ = self.p[1:N+1]
      self._Efn_ = self.Efn[1:N+1]
      self._Efp_ = self.Efp[1:N+1]

   def set_Ecv (self,Ec,Ev) :
      assert Ec.shape[0] == self.N+2
      assert Ev.shape[0] == self.N+2
      self.Ec = Ec
      self.Ev = Ev
      self.__Ecmid = (Ec[0:self.N+1]+Ec[1:self.N+2])/2
      self.__Evmid = (Ev[0:self.N+1]+Ev[1:self.N+2])/2

   def set_BV4n (self, n0, nN) :
      assert n0 > 0 and nN > 0
      self.n[0] = n0
      self.n[-1] = nN

   def set_BV4p (self, p0, pN) :
      assert p0 > 0 and pN > 0
      self.p[0] = p0
      self.p[-1] = pN

   def solve_slotboom (self):
      self.__solve_slotboomn()
      self.__solve_slotboomp()

   def __solve_slotboomn (self):
      A = np.zeros([self.N,3])
      B = np.zeros(self.N)
      sl0 = self.n[0] / Nc * np.exp(self.Ec[0]/kBT)
      slN = self.n[-1] / Nc * np.exp(self.Ec[-1]/kBT)
      self.Efn[0]  = np.log(sl0) *kBT
      self.Efn[-1] = np.log(slN) *kBT

      #Jn[i+0.5] = mn*Nc*exp(-Ec[i+0.5]/kBT) * kBT*q *
      #            * (slotboom[i+1]-slotboom[i]) /dx
      assert self.__Ecmid.shape[0] == self.N + 1
      C       = mn*Nc*np.exp(-self.__Ecmid/kBT) * kBT * q /self.dx
      A[:,0] += C[0:self.N]
      A[:,1] -= C[0:self.N] + C[1:]
      A[:,2] += C[1:]
      A[0,0] = A[-1,2] = 0
      B[0]  -= C[0] * sl0
      B[-1] -= C[-1] * slN
      slotboom = solve_diag2(A,B)
      self.Efn[1:self.N+1] = np.log(slotboom) * kBT
      #self.n[1:self.N+1]  = Nc * slotboom * np.exp(self.Ec[]/kBT)

   def __solve_slotboomp (self):
      A = np.zeros([self.N,3])
      B = np.zeros(self.N)
      sl0 = self.p[0] / Nv * np.exp(-self.Ev[0]/kBT)
      slN = self.p[-1] / Nv * np.exp(-self.Ev[-1]/kBT)
      self.Efp[0]  = -np.log(sl0) *kBT
      self.Efp[-1] = -np.log(slN) *kBT
      #Jp[i+0.5] = -mp*Nv*exp(Ev[i+0.5]/kBT)*kBT*q \
      #            * (slotboom[i+1]-slotboom[i]) /dx
      
      assert self.__Evmid.shape[0] == self.N + 1
      C       = -mp*Nv*np.exp(self.__Evmid/kBT) * kBT * q /self.dx
      A[:,0] += C[0:self.N]
      A[:,1] -= C[0:self.N] + C[1:]
      A[:,2] += C[1:]
      A[0,0] = A[-1,2] = 0
      B[0]  -= C[0] * sl0
      B[-1] -= C[-1] * slN
      slotboom = solve_diag2(A,B)
      self.Efp[1:self.N+1] = - np.log(slotboom) * kBT

class current_solver(solver2D):
   def __init__(self,dx,dy):
      super(current_solver,self).__init__(dx,dy)
   def __continuity(self):
      pass
   def solve_np(self):
      pass
   def __WKB(self):
      pass
   def __SRH(self):
      pass

if __name__ == "__main__" :
   ctest = current_solver(1e-9,1e-9)
   ctest.solve_np()
