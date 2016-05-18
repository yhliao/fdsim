#******* Some utility function for the nonlinear solver  *****
# including: 1. time- and space-efficient solver for
#               a 2nd-order difference matrix
#            2. damping function which can detect oscillation
#               and update the damping factor accordingly
#*************************************************************
import numpy as np

def overlap(p1,d1,p2,d2):
   return p1+d1>p2 and p2+d2>p1

def calc_offset(p1,d1,p2,d2):
   off1 = p1 - p2
   (i,j) = (None,off1) if off1>=0 else (-off1,None)

   off2 = p1 + d1 - p2 -d2
   (k,l) = (-off2,None) if off2>0 else (None,off2)
   if l == 0:
      l = None

   return i,j,k,l

def solve_diag2(A,B):
   N = A.shape[0]
   assert A.shape[0] == B.shape[0],"Shapes of the two matrice do not agree!"
   assert A.shape[1] == 3, A
   assert A[0][0] == 0 and A[N-1][2] == 0

   X = np.array(A[:,1])
   O = np.array(B)
   for i in range(1,N) :
      s = A[i][0] / X[i-1]
      X[i] -= s * A[i-1][2]
      O[i] -= s * O[i-1]
   for i in range(N-1,0,-1) :
      s = A[i-1][2] / X[i]
      O[i-1] -= O[i] * s
   return O / X

class myDamper:
   p_flag = False
   n_flag = False
   osc    = 0
   update = 0
   def __init__ (self, init) :
      self.damp_init = init
      self.damper    = init
      self.latest    = init+1

   def reset(self) :
      #print "reset the damper from %f to %f" %(self.damper,self.damp_init)
      self.damper = self.damp_init
      self.p_flag = False
      self.n_flag = False
      self.latest = self.damp_init+1

   def __call__(self,dx) :
      if dx < -self.damper :
         if self.p_flag is True :
            self.osc_update()
            self.p_flag = False
         else:
            self.n_flag = True
         #print "damp %f to %f" %(dx,-self.damper)
         return -self.damper

      elif dx > self.damper:
         if self.n_flag is True :
            self.osc_update()
            self.n_flag = False
         else:
            self.p_flag = True
         #print "damp %f to %f" %(dx,self.damper)
         return self.damper

      elif abs(dx/self.latest + 1) < 1e-3:
         #print "endless oscillation detectd!"
         self.osc_update()
         self.latest = dx
         return self(dx) 
      else :
         #print "undamped! (%f)" %(dx)
         self.latest = dx
         return dx

   def osc_update(self) :
      self.osc += 1
      if self.osc > 0 :
         #:self.update:
         #print "damper updated"
         self.damper *= 0.5
         self.update += 1
         self.osc = 0

if __name__ == "__main__":
   a = myDamper(1)
   a.damp(4)
   a.damp(-3)
   a.reset()
   a.damp(5)
