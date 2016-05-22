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
         offset = -j.m1.phiS + j.m2.phiS
         if offset != 0:
            self.__Ecoff[j.idx1] += j.m1.epr* offset / (self.dx**2)
            self.__Ecoff[j.idx2] -= j.m1.epr* offset / (self.dx**2)

      nx = self.neighbor
      jx = np.zeros([4,0])
      if len(self.junc):
         jx = np.hstack([[[j.idx1],
                          [j.idx2],
                          [j.m1.epr]] for j in self.junc])
      cx = np.zeros([2,0])
      if len(self.contact):
         cx = np.hstack([[[j.idx],
                          [j.material.epr]] for j in self.contact])
      
      row = np.concatenate((jx[0],jx[1],nx[0],nx[1],
                            jx[1],jx[0],nx[0],nx[1],cx[0]))
      col = np.concatenate((jx[1],jx[0],nx[1],nx[0],
                            jx[1],jx[0],nx[0],nx[1],cx[0]))
      d = [None] * 9
      d[0] = jx[2] / self.dx**2
      d[1] = jx[2] / self.dx**2
      d[2] = d[3] = nx[2] / self.dx**2
      d[4] = -jx[2] / self.dx**2
      d[5] = -jx[2] / self.dx**2
      d[6] = d[7] = -nx[2] / self.dx**2
      d[8] = -cx[1] / self.dx**2
      d  = np.concatenate(d)
      
      L  = sp.coo_matrix((d,(row,col)))
      self.__L =  L.tocsr()
      print ("done, __L.shape=",self.__L.shape )
      # __L * Ec = (NB+n-p)*q/epr - __EcBV
      del L

   def reset_EcBV(self) : 
      self.__EcBV[:] = self.__Ecoff
      for c in self.contact :
         self.__EcBV[c.idx] += c.material.epr * c.Ec / self.dx**2
      for c in self.contact1 :
         self.__EcBV[c.idx] += c.Q / self.dx

   def solve_lpoisson(self) :
      charge = q * (self.p - self.n)
      Ec_new = spsolve(self.__L, self.NB + charge - self.__EcBV)
      self.Ev += Ec_new - self.Ec
      self.Ec[:] = Ec_new

   def set_Dit(self,pos,Et,Dit) :
      pass
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

         print ("1D poisson solver: {}th iteration, err={:.6}"
                  .format(time,err),end= "   \r")
         time += 1
      print ("\n1D poisson solver: converge!")
 
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
      for j in self.junc['x']:
         offset = -j.m1.phiS + j.m2.phiS
         if offset != 0:
            self.__Ecoff[j.idx1] += j.m1.epr * offset / (self.dx**2)
            self.__Ecoff[j.idx2] -= j.m1.epr * offset / (self.dx**2)
      for j in self.junc['y']:
         offset = -j.m1.phiS + j.m2.phiS
         if offset != 0:
            self.__Ecoff[j.idx1] += j.m1.epr * offset / (self.dy**2)
            self.__Ecoff[j.idx2] -= j.m1.epr * offset / (self.dy**2)

      ## use coordinate form to efficiently generate the matrix
      ## then convert to csr form
      print ("Constructing Laplacian sparse matrix ...",end=' ')
      
      nx= self.neighbor['x']
      ny= self.neighbor['y']
      jx = jy = np.zeros([4,0])
      if len(self.junc['x']):
         jx = np.hstack([[ j.idx1, j.idx2,
                          [j.m1.epr] * len(j)]
                          for j in self.junc['x'] ])
      if len(self.junc['y']):
         jy = np.hstack([[ j.idx1, j.idx2,
                          [j.m1.epr] * len(j)]
                          for j in self.junc['y'] ])

      cx = cy = np.zeros([2,0])
      if len(self.contact['x']):
         cx= np.hstack([[j.idx, [j.material.epr] * len(j.idx)]
                         for j in self.contact['x']])
      if len(self.contact['y']):
         cy= np.hstack([[j.idx, [j.material.epr] * len(j.idx)]
                         for j in self.contact['y']])
      row = np.concatenate((jx[0],jx[1],jy[0],jy[1],
                            nx[0],nx[1],ny[0],ny[1],
                            jx[1],jx[0],jy[1],jy[0],
                            nx[0],nx[1],ny[0],ny[1],cx[0],cy[0]))

      col = np.concatenate((jx[1],jx[0],jy[1],jy[0],
                            nx[1],nx[0],ny[1],ny[0],
                            jx[1],jx[0],jy[1],jy[0],
                            nx[0],nx[1],ny[0],ny[1],cx[0],cy[0]))
      d = [None]*18
      d[0] = jx[2] / self.dx**2
      d[1] = jx[2] / self.dx**2
      d[2] = jy[2] / self.dy**2
      d[3] = jy[2] / self.dy**2

      d[4] = d[5] = nx[2] / self.dx**2
      d[6] = d[7] = ny[2] / self.dy**2

      d[8] = -jx[2] / self.dx**2
      d[9] = -jx[2] / self.dx**2
      d[10]= -jy[2] / self.dy**2
      d[11]= -jy[2] / self.dy**2
      
      d[12] = d[13] = -nx[2] / self.dx**2
      d[14] = d[15] = -ny[2] / self.dy**2
      d[16] = -cx[1] / self.dx**2
      d[17] = -cy[1] / self.dy**2

      d  = np.concatenate(d)

      L  = sp.coo_matrix((d,(row,col)))
      self.__L =  L.tocsr()
      del L
      print ("done, __L.shape=",self.__L.shape )

      # __L * Ec = (NB+n-p)*q/epr - __EcBV

   def reset_EcBV(self) :
      self.__EcBV[:] = self.__Ecoff
      for c in self.contact['x'] :
         self.__EcBV[c.idx] += c.material.epr * c.Ec / self.dx**2
      for c in self.contact['y'] :
         self.__EcBV[c.idx] += c.material.epr * c.Ec / self.dy**2

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
