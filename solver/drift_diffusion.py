import numpy as np

from solver._solver import solver1D, solver2D
from solver.util    import solve_diag2
from solver.const   import q, kBT
mn = 1350000
mp = 450000

class current_solver(solver1D) :

   def __init__ (self, dx, N):
      solver1D.__init__(self,dx,N)
      self._n_ = self.n[1:N+1]
      self._p_ = self.p[1:N+1]
      self._Efn_ = self.Efn[1:N+1]
      self._Efp_ = self.Efp[1:N+1]

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
      ### Using Scharfetter-Gummel expression
      dnxAn = np.zeros(self.neighbor['x'].shape[1])
      dnxBn = np.zeros(self.neighbor['x'].shape[1])
      dnxAp = np.zeros(self.neighbor['x'].shape[1])
      dnxBp = np.zeros(self.neighbor['x'].shape[1])
      
      dnyAn = np.zeros(self.neighbor['y'].shape[1])
      dnyBn = np.zeros(self.neighbor['y'].shape[1])
      dnyAp = np.zeros(self.neighbor['y'].shape[1])
      dnyBp = np.zeros(self.neighbor['y'].shape[1])
      ix = 0
      iy = 0
      for m in self.meshes:
         jx = ix + m.Ny*(m.Nx-1)
         jy = iy + m.Nx*(m.Ny-1)
         if m.material is 'semiconductor':
            ##### for x-direction #####
            tx = np.diff(m.Ec,0).reshape(m.Ny*(m.Nx-1)) / kBT
            dnxAn[ix:jx]= q*m.Dn/(self.dx**2) * tx/(np.exp(-tx)-1)
            dnxBn[ix:jx]= q*m.Dn/(self.dx**2) * tx/(np.exp(tx)-1)

            ##### for y-direction #####
            ty = np.diff(m.Ec,1).reshape(m.Nx*(m.Ny-1)) / kBT
            dnyAn[iy:jy]= q*m.Dn/(self.dy**2) * ty/(np.exp(-ty)-1)
            dnyBn[iy:jy]= q*m.Dn/(self.dy**2) * ty/(np.exp(ty)-1)
         else:
            #### dummies for preventing singular matrix
            dnxAn[ix:jx] = 1
            dnxBn[ix:jx] = 1
            dnyAn[iy:jy] = 1
            dnyAn[iy:jy] = 1
         ix = jx
         iy = jy

      for j in self.junc:
         if not j.isins():
            tn = (self.Ec[j.idx[1]] - self.Ec[j.idx[0]]) / kBT
            dAn= q*j.m[0].Dn/(j.d**2) * tn/(np.exp(-tn)-1)
            dBn= q*j.m[0].Dn/(j.d**2) * tn/(np.exp(tn)-1)
         else:
            dAn = dBn = [0] * len(j)

         djAn.append(dAn)
         djBn.append(dBn)
      djAn = np.concatenate(djAn)
      djBn = np.concatenate(djBn)

      for c in self.contact:
         if not c.isins():
            tn = c.pflag*(c.Ec - self.Ec[c.idx]) / kBT
            dAn= q*c.m.Dn/(c.d**2)* tn/(np.exp(-tn)-1)
            dBn= q*c.m.Dn/(c.d**2)* tn/(np.exp(tn)-1)
         else:
            dAn = dBn = [0] * len(c)

         dcAn.append(dAn)
         dcBn.append(dBn)

   def solve_np(self):
      self.__continuity()
      pass
   def __WKB(self):
      pass
   def __SRH(self):
      pass

if __name__ == "__main__" :
   ctest = current_solver(1e-9,1e-9)
   ctest.solve_np()
