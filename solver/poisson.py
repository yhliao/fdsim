from __future__ import absolute_import, print_function, division

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

from solver._solver import solver1D, solver2D
from solver.const   import q, kBT
from solver.util    import myDamper

class p_solver1D(solver1D):

   def __init__(self,dx) :
      super(p_solver1D,self).__init__(dx)

   def construct_profile(self) : 
      super(p_solver1D,self).construct_profile()
      self.NB     = np.zeros(self.c_size)
      self.__EcBV  = np.zeros(self.c_size)
      self.__Ecoff = np.zeros(self.c_size)
      self.load_mesh(['NB'])
      self.NB *= q

      for j in self.junc:
         offset = -j.m[0].phiS + j.m[1].phiS
         if offset != 0:
            o = j.m[0].epr * offset / self.dx**2
            self.__Ecoff[j.idx[0]] += o
            self.__Ecoff[j.idx[1]] -= o

      dnx = np.concatenate([ [m.material.epr/self.dx**2] * (m.N-1)
                             for m in self.meshes ])
      djx = np.zeros(0)
      if len(self.junc):
         djx = np.array([j.m[0].epr/self.dx**2 for j in self.junc])
      dcx = np.zeros(0)
      if len(self.contact):
         dcx = [-j.m.epr/self.dx**2 for j in self.contact]

      d  = np.concatenate(( djx, djx, dnx, dnx,
                           -djx,-djx,-dnx,-dnx,dcx))
      
      L  = sp.coo_matrix((d,(self.op_row,self.op_col)))
      self.__L =  L.tocsr()
      print ("done, __L.shape=",self.__L.shape )
      # __L * Ec = (NB+n-p)*q/epr - __EcBV
      del L,d
      
   def reset_EcBV(self) : 
      self.__EcBV[:] = self.__Ecoff
      for c in self.contact :
         self.__EcBV[c.idx] += c.m.epr * c.Ec / self.dx**2
      for c in self.contact1 :
         self.__EcBV[c.idx] += c.Q / self.dx

   def solve_lpoisson(self) :
      charge = q * (self.p - self.n)
      Ec_new = spsolve(self.__L, self.NB + charge - self.__EcBV)
      self.Ev += Ec_new - self.Ec
      self.Ec[:] = Ec_new

   def solve_nlpoisson(self,tol=1e-5) :
      D = myDamper(1)
      dEc = np.zeros(self.c_size)
      err = 1
      time = 0
      while abs(err) >tol :
         self.calc_np()
         self.calc_it()
         charge = q * (self.p - self.n + self.Qit)
         LL = self.__L + sp.diags(charge*(q/kBT) - self.Dit*q,0
                                  ,format='csr')
         Laplacian = self.__L * self.Ec + self.__EcBV 
         dEc[:] = spsolve( LL , self.NB + charge - Laplacian)

         err  = dEc[np.argmax(abs(dEc))]
         dEc *= D(err)/err
         self.Ec += dEc
         self.Ev += dEc

         print ("1D poisson solver: {}th iteration, err={:.6}"
                  .format(time,err),end= "   \r")
         time += 1
      print ("\n1D poisson solver: converge!")
      self.write_mesh(['Ec','Ev'])
 
class p_solver2D(solver2D):

   def __init__ (self,dx,dy) :
      super(p_solver2D,self).__init__(dx,dy)

   def construct_profile(self) :
      super(p_solver2D,self).construct_profile()
      self.NB      = np.zeros(self.c_size)
      self.__EcBV  = np.zeros(self.c_size)
      self.__Ecoff = np.zeros(self.c_size)
      self.load_mesh(['NB'])
      self.NB *= q

      ## Handling Ec boundary offset
      for j in self.junc:
         offset = -j.m[0].phiS + j.m[1].phiS
         if offset != 0:
            o = j.m[0].epr * offset / j.d**2
            self.__Ecoff[j.idx[0]] += o
            self.__Ecoff[j.idx[1]] -= o

      ## use coordinate form to efficiently generate the matrix
      ## then convert to csr form
      print ("Constructing Laplacian sparse matrix ...",end=' ')
      
      dnx = np.concatenate([[m.material.epr/self.dx**2] *
                            (m.Ny*(m.Nx-1)) for m in self.meshes])
      dny = np.concatenate([ [m.material.epr/self.dy**2] *
                            (m.Nx*(m.Ny-1)) for m in self.meshes])
      dj = np.zeros(0)
      if len(self.junc):
         dj = np.concatenate([ [j.m[0].epr/j.d**2] * len(j)
                                        for j in self.junc])
      dc = np.zeros(0)
      if len(self.contact):
         dc = np.hstack([[-j.m.epr/j.d**2] * len(j.idx)
                                        for j in self.contact])

      d  = np.concatenate(( dj, dj, 
                            dnx, dnx, dny, dny,
                           -dj,-dj,
                           -dnx,-dnx,-dny,-dny, dc))

      L  = sp.coo_matrix((d,(self.op_row,self.op_col)))
      self.__L =  L.tocsr()
      print ("done, __L.shape=",self.__L.shape )
      # __L * Ec = (NB+n-p)*q/epr - __EcBV
      del L, d

   def reset_EcBV(self) :
      self.__EcBV[:] = self.__Ecoff
      for c in self.contact :
         self.__EcBV[c.idx] += c.m.epr * c.Ec / c.d**2

   def solve_lpoisson(self) :
      charge = q * (self.p - self.n)
      Ec_new = spsolve(self.__L, self.NB + charge - self.__EcBV)
      self.Ev += Ec_new - self.Ec
      self.Ec[:] = Ec_new

   def solve_nlpoisson(self,tol=1e-5) :
      D = myDamper(1)
      dEc = np.zeros(self.c_size)
      err = 1
      time = 0
      while abs(err) >tol :
         self.calc_np()
         charge = q * (self.p - self.n)
         LL = self.__L + sp.diags(charge,0,format='csr') *(q/kBT)
         Laplacian = self.__L * self.Ec + self.__EcBV 
         dEc[:] = spsolve( LL , self.NB + charge - Laplacian)

         err  = dEc[np.argmax(abs(dEc))]
         dEc *= D(err)/err
         self.Ec += dEc
         self.Ev += dEc

         print ("2D poisson solver: {}th iteration, err={:.6}"
                  .format(time,err),end= "   \r")
         time += 1
      print ("\n2D poisson solver: converge!")
      self.write_mesh(['Ec','Ev'])
