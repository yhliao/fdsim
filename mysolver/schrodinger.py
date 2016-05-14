from poisson import p_solver
import numpy as np
from numpy      import linalg as la
from matplotlib import pyplot as plt
from const import *
from util  import myDamper

class schrodinger_solver:

   def __init__(self,dx,N) :
      self.dx = dx
      self.N  = N

      ## A: p^2 operator for electrons
      self.__P2  = np.eye(N,k=1) - 2* np.eye(N) + np.eye(N,k=-1)
      self.__P2 *= - h_**2 / (2*me) / dx**2
      # V: potential
      #self.Ec  = np.zeros(N)

      # E & phi : statioinary states solutions, only keep lv levels
      self.lv  = N
      self.__E   = np.zeros(N)
      self.__phi = np.zeros([N,N])

   def set_potential(self,v) :
      assert self.N == v.shape[0]
      self.Ec       = np.array(v)

   def solve_phi(self) :
      tosolve = np.array(self.__P2)
      ## create the hamiltonian matrix
      for i in range(self.N) :
         tosolve[i,i] += self.Ec[i] * q
      E , phi = la.eigh(tosolve)
      self.__E   = E  [  0:self.lv] / q
      self.__phi = phi[:,0:self.lv]

   def phi2n(self,Efn) :
      assert self.N  == Efn.shape[0]
      for i in range(self.lv) :
         n2d  = me*kBT*q/(pi*(h_**2))\
               *np.log(1+np.exp((Efn-self.__E[i])/kBT))
         self.n[1:-1] += n2d * self.__phi[:,i]**2 / self.dx

   def _show(self,f,t) :
      plt.figure()
      plt.plot(self.n)
      plt.show()
      time = 0
      for i in range(f,t) :
         time += 1
         if (time%5) == 1:
            plt.figure()
         plt.subplot(5,1, time % 5 )
         plt.plot(self.__phi[:,i] + self.__E[i])
         plt.plot(self.Ec)
         plt.xlabel("N")
         print "E = " + str(self.__E[i]) + " eV"
         #print "n2d~" +str(np.log(1+np.exp(-self.__E[i]/kBT/q)))
         if (time%5) == 0 or i==t-1:
            plt.show()

# Use the methods in poisson_solver to configure this solver
class ps_solver(p_solver,schrodinger_solver):
 
   def __init__(self,dx,N,lv) :
      p_solver          .__init__(self,dx,N)
      schrodinger_solver.__init__(self,dx,N)
      self.Ec_log   = np.zeros(N)

   def solve(self,Efn,tol=1e-5):
      d = myDamper(0.2)
      self.solve_nlpoisson(tol)
      self.show()
      self.Ec_log = np.array(self.Ec)
      errm = 1
      time = 0
      while abs(errm) > tol :
         time += 1
         self.solve_phi()
         self.phi2n(Efn)
         self.solve_lpoisson()
         dEc = self.Ec- self.Ec_log
         #errm = max(abs(dEc))
         errm = dEc[np.argmax(abs(dEc))]
         print "ps_solver (%dth iteration): errm = %f" %(time,errm)

         dEc *= d.damp(errm)/errm
         self.Ec     = self.Ec_log + dEc
         #self.show()
         self.Ec_log = np.array(self.Ec)
      print "converge!" 
