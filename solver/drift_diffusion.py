from __future__ import absolute_import, print_function, division

import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from solver._solver import solver1D, solver2D
from solver.util    import overlap, calc_offset ,myDamper
from solver.const   import q , kBT
from solver.model   import TUNNELING

### TODO: mobility and lifetime variation changed with doping

###### Using Scharfetter-Gummel expression #######
# functions for n, p continuity equations        #
# t: -(Ei+1 - Ei)/kBT (both Ec, Ev)              #
# for hole current, the An, Bn should be swithed #
##################################################
def SGn(t,Dn,d):
   idx_c = abs(t) > 1e-7
   An = -Dn/(d**2) * np.ones(len(t))
   Bn =  Dn/(d**2) * np.ones(len(t))
   An[idx_c] *= -t[idx_c]/(np.exp(-t[idx_c])-1)
   Bn[idx_c] *=  t[idx_c]/(np.exp( t[idx_c])-1)
   return An, Bn

class J_solver1D(solver1D) :

   def __init__ (self, dx):
      super(J_solver1D,self).__init__(dx)

   def construct_profile(self):
      super(J_solver1D,self).construct_profile()
      self.__JnBV = np.zeros(self.c_size)
      self.__JpBV = np.zeros(self.c_size)
      
      nsize = self.neighbor.shape[1]
      self.dnn = [np.ones(nsize),np.ones(nsize)]
      self.dnp = [np.ones(nsize),np.ones(nsize)]
      jsize = len(self.junc)
      self.djn = [np.zeros(jsize),np.zeros(jsize)]
      self.djp = [np.zeros(jsize),np.zeros(jsize)]
      csize = len(self.contact)
      self.dcn = np.zeros(csize)
      self.dcp = np.zeros(csize)
      for j in self.junc:
         if j.isins():
            print("An insulating junction")

   def _continuity(self):
      ## free the spaces before creating new profile
      if hasattr(self,'__DJn'):
         del self.__DJn, self.__DJp
      #### for better code readability ####
      (dnn,dnp) = (self.dnn, self.dnp)
      (djn,djp) = (self.djn, self.djp)
      (dcn,dcp) = (self.dcn, self.dcp)

      i = 0
      for m in self.meshes:
         j = i + m.N - 1
         if m.material.type is 'semiconductor':
            t = -np.diff(m.Ec) / kBT
            dnn[0][i:j],dnn[1][i:j] = SGn(t,m.material.Dn,self.dx)
            dnp[1][i:j],dnp[0][i:j] = SGn(t,m.material.Dp,self.dx)
         i = j
      assert i == len(dnn[0])

      for m,j in enumerate(self.junc):
         if not j.isins():
            tn = -(self.Ec[j.idx[1]] - self.Ec[j.idx[0]]) / kBT
            djn[0][m],djn[1][m] = SGn(tn,j.m[0].Dn,j.d)
            tp = -(self.Ev[j.idx[1]] - self.Ev[j.idx[0]]) / kBT
            djp[1][m],djp[0][m] = SGn(tp,j.m[0].Dp,j.d)

      for m,c in enumerate(self.contact):
         if not c.isins():
            t = -(c.Ec - self.Ec[c.idx]) / kBT
            dcn[m] ,Yn = SGn(t, c.m.Dn, c.d)
            Yp, dcp[m] = SGn(t, c.m.Dp, c.d)
            self.__JnBV[c.idx] = -c.n * Yn
            self.__JpBV[c.idx] = -c.p * Yp

      dn = np.concatenate(( djn[1],-djn[0],-djn[1], djn[0], 
                            dnn[1],-dnn[0], dnn[0],-dnn[1],dcn))
      dp = np.concatenate(( djp[1],-djp[0],-djp[1], djp[0], 
                            dnp[1],-dnp[0], dnp[0],-dnp[1],dcp))
      DJn = sp.coo_matrix((dn,(self.op_row,self.op_col)))
      DJp = sp.coo_matrix((dp,(self.op_row,self.op_col)))
      self.__DJn = DJn.tocsr()
      self.__DJp = DJp.tocsr()

      ## for preventing singular matrix
      for m in self.meshes:
         if m.material.type is 'insulator':
            i = m.l_idx
            self.__DJn[i,i] += 1
            self.__DJp[i,i] += 1
      del DJn, DJp

   def solve_np(self,tol=1e-3):
      if not hasattr(self,'Efnlog'):
         self.Efnlog = np.array(self.Efn)
         self.Efplog = np.array(self.Efp)

      (errn,errp) = (1,1)
      time = 0
      while abs(errn) > tol or abs(errp) > tol:
         time += 1
         self._SRH()
         contn = self.__DJn - sp.diags(
                 self.p * self.__SRH[0,:],0,format='csr')
         contp = self.__DJp + sp.diags(
                 self.n * self.__SRH[0,:],0,format='csr')
         self.n[:] = spsolve(contn, self.__JnBV + self.__SRH[1,:])
         self.p[:] = spsolve(contp, self.__JpBV - self.__SRH[1,:])

         ### calculate the errors, and log the results
         self.calc_Ef()
         dEfn = self.Efn - self.Efnlog
         dEfp = self.Efp - self.Efplog
         errn = dEfn[np.argmax(abs(dEfn))]
         errp = dEfp[np.argmax(abs(dEfp))]
         print("1D current solver: {}th iteration,err={:.6},{:.6}"
                 .format(time,errn,errp),end= "......\r")
         self.Efnlog[:] = self.Efn
         self.Efplog[:] = self.Efp
      print ("\n1D current solver: converge!")

      self.write_mesh(['Efn','Efp','n','p'])
   def solve_current(self,tol=1e-3):
      self._continuity(tol)
      self.solve_np()
      pass

   def _SRH(self):
      if not hasattr(self,'__SRH'):
         self.__SRH = np.zeros([2,self.c_size])
         self.__SRH[0,:] = 1
      x = 0
      for m in self.meshes:
         mat = m.material
         y = x + m.size
         if mat.type is 'semiconductor':
            self.__SRH[0,x:y]=1/(mat.taun*(self.n[x:y]+mat.ni)+\
                                 mat.taup*(self.p[x:y]+mat.ni))
            self.__SRH[1,x:y]= - mat.ni**2 * self.__SRH[0,x:y]
         x = y
      assert x == self.c_size

class Tpath(object):

   def __init__(self,m,axis,j0,j1):
      assert m.material.type is 'insulator'
      self.ax    = axis
      self.mesh  = m

      self.mdiel = m.material.meff
      self.meff  = j0[0].m[1].meff
      self.t     = m.d[axis] * m.N[axis]
      i,j,k,l = calc_offset(j1[1],j1[2],j0[1],j0[2])
      self.cidx0 = j0[0].idx[0][j:l]
      self.Bidx0 = j0[0].idx[1][j:l]
      self.cidx1 = j1[0].idx[1][i:k]
      self.Bidx1 = j1[0].idx[0][i:k]
      assert len(self.cidx0) == len(self.cidx1)
      self.len = len(self.cidx0)
      self.sim = TUNNELING( self.t, self.mdiel,
                                    self.meff,  self.len)
      print ("*** Tpath in J_solver2D:"
             " A tunneling path has been logged\n" 
             "\t(direction: axis{}, thickness: {}, length: {})"
              .format(axis,self.t,self.len))
   def set(self,Ec):
      self.sim.setEc(Ec[self.cidx0],Ec[self.cidx1],
                     Ec[self.Bidx0],Ec[self.Bidx1])
   def calc_Jt(self,Efn,JnBV):
      J  = self.sim.TSUESAKI(Efn[self.cidx0],Efn[self.cidx1])

      lat = 1 if self.ax ==0 else 0
      print ("Tpath: tunneling current =", sum(J)*self.mesh.d[lat])
      bv = J / q / self.mesh.d[self.ax]
      #* self.mesh.d[lat] / q / self.mesh.d[self.ax]
      JnBV[self.cidx0] = -bv
      JnBV[self.cidx1] = bv

class J_solver2D(solver2D):

   def __init__(self,dx,dy):
      super(J_solver2D,self).__init__(dx,dy)
      self.paths = []

   def construct_profile(self):
      super(J_solver2D,self).construct_profile()
      self.__JnBV = np.zeros(self.c_size)
      self.__JpBV = np.zeros(self.c_size)
      nxsize = self.neighbor['x'].shape[1]
      nysize = self.neighbor['y'].shape[1]
      self.dnn = [[np.ones(nxsize), np.ones(nxsize)],
                  [np.ones(nysize), np.ones(nysize)]]
      self.dnp = [[np.ones(nxsize), np.ones(nxsize)],
                  [np.ones(nysize), np.ones(nysize)]]
      jsize = sum([len(j) for j in self.junc]) 
      self.djn = [np.zeros(jsize),np.zeros(jsize)]
      self.djp = [np.zeros(jsize),np.zeros(jsize)]
      csize = sum([len(c) for c in self.contact])
      self.dcn = np.zeros(csize)
      self.dcp = np.zeros(csize)
      for m in self.meshes:
         if m.material.type is 'insulator':
            mdiel = m.material.meff
            for ax in [0,1]:
               a = 0 if ax == 1 else 1
               for j1 in m.junc[ax][1]:
                  for j0 in m.junc[ax][0]:
                     if j1[0].m[0] == j0[0].m[1] and\
                        overlap(j1[1],j1[2],j0[1],j0[2]):
                        print ("A tunneling path is detected")
                        newp = Tpath(m,a,j0,j1)
                        self.paths.append(newp)

   def _continuity(self):
      ## free the spaces before creating new profile
      if hasattr(self,'__DJn'):
         del self.__DJn, self.__DJp
      #### for better code readability ####
      (dnn,dnp) = (self.dnn, self.dnp)
      (djn,djp) = (self.djn, self.djp)
      (dcn,dcp) = (self.dcn, self.dcp)

      ix = iy = 0
      for m in self.meshes:
         jx = ix + m.Ny*(m.Nx-1)
         jy = iy + m.Nx*(m.Ny-1)
         if m.material.type is 'semiconductor':
            ##### for x-direction #####
            tx= -np.diff(m.Ec,axis=0).reshape(m.Ny*(m.Nx-1))/kBT
            dnn[0][0][ix:jx],dnn[0][1][ix:jx]=\
                        SGn(tx,m.material.Dn,self.dx)
            dnp[0][1][ix:jx],dnp[0][0][ix:jx]=\
                        SGn(tx,m.material.Dp,self.dx)
            ##### for y-direction #####
            ty= -np.diff(m.Ec,axis=1).T.reshape(m.Nx*(m.Ny-1))/kBT
            dnn[1][0][iy:jy],dnn[1][1][iy:jy]=\
                        SGn(ty,m.material.Dn,self.dy)
            dnp[1][1][iy:jy],dnp[1][0][iy:jy]=\
                        SGn(ty,m.material.Dp,self.dy)
         ix = jx
         iy = jy
      assert ix == len(dnn[0][0])
      assert iy == len(dnn[1][0])

      m = 0
      for j in self.junc:
         n = m + len(j)
         if not j.isins():
            tn = -(self.Ec[j.idx[1]] - self.Ec[j.idx[0]]) / kBT
            djn[0][m:n],djn[1][m:n] = SGn(tn,j.m[0].Dn,j.d)
            tp = -(self.Ev[j.idx[1]] - self.Ev[j.idx[0]]) / kBT
            djp[1][m:n],djp[0][m:n] = SGn(tp,j.m[0].Dp,j.d)
         m = n
      assert m == len(djn[0])

      m = 0
      for c in self.contact:
         n = m + len(c)
         if not c.isins():
            t = -(c.Ec - self.Ec[c.idx]) / kBT
            dcn[m:n] ,Yn = SGn(t, c.m.Dn, c.d)
            Yp, dcp[m:n] = SGn(t, c.m.Dp, c.d)
            self.__JnBV[c.idx] = -c.n * Yn
            self.__JpBV[c.idx] = -c.p * Yp
         m = n
      assert m == len(dcn)

      dn = np.concatenate(( djn[1], -djn[0], -djn[1], djn[0], 
                   dnn[0][1],-dnn[0][0],dnn[1][1],-dnn[1][0],
                   dnn[0][0],-dnn[0][1],dnn[1][0],-dnn[1][1], dcn))
      dp = np.concatenate(( djp[1], -djp[0], -djp[1], djp[0], 
                   dnp[0][1],-dnp[0][0],dnp[1][1],-dnp[1][0],
                   dnp[0][0],-dnp[0][1],dnp[1][0],-dnp[1][1], dcp))
      DJn = sp.coo_matrix((dn,(self.op_row,self.op_col)))
      DJp = sp.coo_matrix((dp,(self.op_row,self.op_col)))
      self.__DJn = DJn.tocsr()
      self.__DJp = DJp.tocsr()
      ## for preventing singular matrix
      for m in self.meshes:
         if m.material.type is 'insulator':
            i = m.l_idx
            self.__DJn[i,i] += 1
            self.__DJp[i,i] += 1
      del DJn, DJp

   def solve_np(self,tol=1e-5,SRH=True):
      if not hasattr(self,'Efnlog'):
         self.Efnlog = np.array(self.Efn)
         self.Efplog = np.array(self.Efp)

      (errn,errp) = (1,1)
      time = 0
      while errn > tol or errp > tol:
         time += 1
         (contn,contp) = (self.__DJn, self.__DJp)
         (bvn,  bvp  ) = (self.__JnBV,self.__JpBV)
         if SRH:
            self._SRH()
            contn = contn -\
               sp.diags(self.p*self.__SRH[0,:],0,format='csr')
            contp = contp +\
               sp.diags(self.n*self.__SRH[0,:],0,format='csr')
            bvn += self.__SRH[1,:]
            bvp -= self.__SRH[1,:]

         self.n[:] = spsolve(contn, bvn)
         self.p[:] = spsolve(contp, bvp)

         ### calculate the errors, and log the results
         self.calc_Ef()
         errn = max(abs(self.Efn-self.Efnlog))
         errp = max(abs(self.Efp-self.Efplog))
         print("2D current solver: {}th iteration,err={:.6},{:.6}"
                 .format(time,errn,errp),end= "......\r")
         self.Efnlog[:] = self.Efn
         self.Efplog[:] = self.Efp
      print ("\n2D current solver: converge!")

   def solve_current(self,tol=1e-3,SRH=True,tunneling=False):
      self._continuity()
      if not hasattr(self,'Efnold'):
         self.Efnold = np.array(self.Efn)
         self.Efpold = np.array(self.Efp)

      if tunneling:
         ## setup for tunneling calculation
         for p in self.paths:
            p.set(self.Ec)
         errn = errp = 1
         while errn > tol or errp > tol:

            self.solve_np(tol,SRH)
            for p in self.paths:
               p.calc_Jt(self.Efn,self.__JnBV)
            ## calculate error and log Ef
            errn = max(abs(self.Efn-self.Efnold))
            errp = max(abs(self.Efp-self.Efpold))
            self.Efnold[:] = self.Efn
            self.Efpold[:] = self.Efp
         print ("**** J_solver 2D.solve_current"
                " (with tunneling simulation): converge!****")
      else:
         self.solve_np(tol)
      self.write_mesh(['Efn','Efp','n','p'])

   def _SRH(self):
      if not hasattr(self,'__SRH'):
         self.__SRH = np.zeros([2,self.c_size])
         self.__SRH[0,:] = 1
      x = 0
      for m in self.meshes:
         mat = m.material
         y = x + m.size
         if mat.type is 'semiconductor':
            self.__SRH[0,x:y]=1/(mat.taun*(self.n[x:y]+mat.ni)+\
                                 mat.taup*(self.p[x:y]+mat.ni))
            self.__SRH[1,x:y]= - mat.ni**2 * self.__SRH[0,x:y]
         x = y
      assert x == self.c_size

