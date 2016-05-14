from __future__ import absolute_import, print_function, division

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

from mysolver._solver import solver1D, solver2D
from mysolver.const   import q, kBT
from mysolver.util    import myDamper

class p_solver1D(solver1D):

   def __init__(self,dx) :
      super(p_solver1D,self).__init__(dx)

   def connect_meshes(self) : 
      super(p_solver1D,self).connect_meshes()
      self.NB_      = np.zeros(self.c_size)
      self.__EcBV  = np.zeros(self.c_size)
      self.__Ecoff = np.zeros(self.c_size)
      i = 0
      for m in self.meshes:
         j = i + m.size
         self.NB_[i:j] = m.NB*(q/m.material.epr)
         i = j
      assert i == self.c_size

      L  = sp.eye(self.c_size,format='lil')
      L *= -2./(self.dx**2)  
      for m in self.meshes :
         # neither left nor right edge
         for i in range(1,m.N-1) :
            L[i,i-1] = L[i,i+1] = 1./(self.dx**2)

         #### Handling left junctions ####
         L[m.l_idx,m.l_idx+1] = 1./(self.dx**2)
         edge = m.l_edge
         if edge >= 0 :
            L[m.l_idx,m.l_edge] = 1./(self.dx**2)
            self.__Ecoff[m.l_idx] += m.l_off / self.dx**2
         else:
            assert edge == -1
         #### Handling right junctions ####
         L[m.r_idx,m.r_idx-1] = 1./(self.dx**2)
         edge = m.r_edge
         if edge >= 0 :
            L[m.r_idx,m.r_edge] = 1./(self.dx**2)
            self.__Ecoff[m.r_idx] += m.r_off / self.dx**2
         else:
            assert edge == -1

      self.__L = L.tocsr()
      # __L * Ec = (NB+n-p)*q/epr - __EcBV
      del L

   def reset_EcBV(self) : 
      self.__EcBV[:] = self.__Ecoff
      for m in self.meshes :
         if m.l_edge == -1:
            self.__EcBV[m.l_idx] += m.l_Ec / self.dx**2 
         if m.r_edge ==-1:
            self.__EcBV[m.r_idx] += m.r_Ec / self.dx**2

   def solve_lpoisson(self) :
      _charge = self.p - self.n
      i = 0
      for m in self.meshes:
         j = i + m.size
         _charge[i:j] *= q/m.material.epr
         i = j
      Ec_new = spsolve(self.__L, self.NB_ + _charge - self.__EcBV)
      self.Ev += Ec_new - self.Ec
      self.Ec[:] = Ec_new

   def solve_nlpoisson(self,tol=1e-5) :
      D = myDamper(1)
      dEc = np.zeros(self.c_size)
      err = 1
      time = 0
      while abs(err) >tol :
         self.calc_np()
         _charge = self.p - self.n
         i = 0
         for m in self.meshes:
            j = i + m.size
            _charge[i:j] *= q/m.material.epr
            i = j
         LL = self.__L + sp.diags(_charge,0,format='csr') *(q/kBT)
         Laplacian = self.__L * self.Ec + self.__EcBV 
         dEc[:] = spsolve( LL , self.NB_ + _charge - Laplacian)

         err  = dEc[np.argmax(abs(dEc))]
         dEc *= D(err)/err
         self.Ec += dEc
         self.Ev += dEc

         print ("2D poisson solver: {}th iteration, err={:.6}"\
                  .format(time,err),end= "   \r")
         time += 1
      print ("\n2D poisson solver: converge!")
 
class p_solver2D(solver2D):

   def __init__ (self,dx,dy) :
      super(p_solver2D,self).__init__(dx,dy)

   def connect_meshes(self) :
      super(p_solver2D,self).connect_meshes()
      self.NB_     = np.zeros(self.c_size)
      self.__EcBV  = np.zeros(self.c_size)
      self.__Ecoff = np.zeros(self.c_size)
      i = 0
      for m in self.meshes:
         j = i + m.size
         self.NB_[i:j] = m.NB.reshape(m.size) * q /m.material.epr
         i = j
      assert i == self.c_size

      # Create laplacian operator on the vectorized E
      L  = sp.eye(self.c_size,format='lil')
      L *= -(2./(self.dx**2) + 2./(self.dy**2))  
      for m in self.meshes :
         # neither left nor right edge
         for n in range(m.Nx) :
            for i in range(m.l_idx[n]+1,m.r_idx[n]):
               L[i,i-1] =L[i,i+1]= 1./(self.dy**2)
         # neither top nor bottom edge
         for j in range(m.t_idx[-1]+1,m.b_idx[0]):
            L[j,j-m.Ny] = L[j,j+m.Ny] = 1./(self.dx**2)

         #### Handling left junctions ####
         for n,idx in enumerate(m.l_idx): 
            L[idx,idx+1] = 1./(self.dy**2)
            edge = m.l_edge[n]
            if edge >= 0 :
               L[idx,edge] = 1./(self.dy**2)
               self.__Ecoff[idx] += m.l_off[n] / self.dy**2
            else:
               assert edge == -1
         #### Handling right junctions ####
         for n,idx in enumerate(m.r_idx): 
            L[idx,idx-1] = 1./(self.dy**2)
            edge = m.r_edge[n]
            if edge >= 0 :
               L[idx,edge] = 1./(self.dy**2)
               self.__Ecoff[idx] += m.r_off[n] / self.dy**2
            else:
               assert edge == -1
         #### Handling top junctions ####
         for n,idx in enumerate(m.t_idx): 
            L[idx,idx+m.Ny] = 1./(self.dx**2)
            edge = m.t_edge[n]
            if edge >= 0 :
               L[idx,edge] = 1./(self.dx**2)
               self.__Ecoff[idx] += m.t_off[n] / self.dx**2
            else:
               assert edge == -1

         #### Handling bottom junctions ####
         for n,idx in enumerate(m.b_idx):
            L[idx,idx-m.Ny] = 1./(self.dx**2)
            edge = m.b_edge[n]
            if edge >= 0 :
               L[idx,edge] = 1./(self.dx**2)
               self.__Ecoff[idx] += m.b_off[n] / self.dx**2
            else:
               assert edge == -1

      self.__L = L.tocsr()
      # __L * Ec = (NB+n-p)*q/epr - __EcBV
      del L

   def reset_EcBV(self) :
      self.__EcBV[:] = self.__Ecoff
      for m in self.meshes :
         if m.l_len:
            self.__EcBV[m.l_boundary] += m.l_Ec / self.dy**2 
         if m.r_len:
            self.__EcBV[m.r_boundary] += m.r_Ec / self.dy**2
         if m.t_len:
            self.__EcBV[m.t_boundary] += m.t_Ec / self.dx**2
         if m.b_len:
            self.__EcBV[m.b_boundary] += m.b_Ec / self.dx**2

   def solve_lpoisson(self) :
      _charge = self.p - self.n
      i = 0
      for m in self.meshes:
         j = i + m.size
         _charge[i:j] *= q/m.material.epr
         i = j
      Ec_new = spsolve(self.__L, self.NB_ + _charge - self.__EcBV)
      self.Ev += Ec_new - self.Ec
      self.Ec[:] = Ec_new

   def solve_nlpoisson(self,tol=1e-5) :
      D = myDamper(1)
      dEc = np.zeros(self.c_size)
      err = 1
      time = 0
      while abs(err) >tol :
         self.calc_np()
         _charge = self.p - self.n
         i = 0
         for m in self.meshes:
            j = i + m.size
            _charge[i:j] *= q/m.material.epr
            i = j
         LL = self.__L + sp.diags(_charge,0,format='csr') *(q/kBT)
         Laplacian = self.__L * self.Ec + self.__EcBV 
         dEc[:] = spsolve( LL , self.NB_ + _charge - Laplacian)

         err  = dEc[np.argmax(abs(dEc))]
         dEc *= D(err)/err
         self.Ec += dEc
         self.Ev += dEc

         print ("2D poisson solver: {}th iteration, err={:.6}"\
                  .format(time,err),end= "   \r")
         time += 1
      print ("\n2D poisson solver: converge!")
