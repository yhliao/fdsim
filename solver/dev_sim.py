from __future__ import absolute_import, division, print_function

import numpy as np
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#from solver.const import kBT
from solver.poisson         import p_solver1D, p_solver2D
from solver.drift_diffusion import J_solver1D, J_solver2D

class dev_solver1D(p_solver1D, J_solver1D):
   def __init__(self,dx):
      super(dev_solver1D,self).__init__(dx)

   def construct_profile(self):
      super(dev_solver1D,self).construct_profile()
      self.Eclog = np.array(self.Ec)

   def solve(self,tol=1e-3):
      self.reset_EcBV()
      time = 0
      errE = 1
      while errE > tol:
         self.solve_nlpoisson(tol)
         self.solve_np()
         errE = max(abs(self.Ec-self.Eclog))
         time += 1
         print ("1D device solver: {}th iteration, err={:.6}"
                  .format(time,errE))
         self.Eclog[:] = self.Ec
         #self.visualize(['Ec','Ev','Efn','Efp'])
      print ("\n1D device solver: converge!")

class dev_solver2D(p_solver2D, J_solver2D):
   def __init__(self,dx,dy):
      super(dev_solver2D,self).__init__(dx,dy)

   def construct_profile(self):
      super(dev_solver2D,self).construct_profile()
      self.Eclog = np.array(self.Ec)

   def solve(self,tol=1e-5):
      self.reset_EcBV()
      time = 0
      errE = 1
      while errE > tol:
         self.solve_nlpoisson(tol)
         self.solve_np()
         errE = max(abs(self.Ec-self.Eclog))
         time += 1
         print ("2D device solver: {}th iteration, err={:.6}"
                  .format(time,errE))
         self.Eclog[:] = self.Ec
         #self.visualize(['Ec','Ev','Efn','Efp'])
      print ("\n2D device solver: converge!")
