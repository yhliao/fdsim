from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from solver.const import kBT, mdb
from solver.util import overlap, calc_offset

V0   =  4.5
pcol = { 'Ec' :'blue','Ev' :'green' ,
         'n'  :'red' ,'p'  :'yellow',
         'Efn':'red' ,'Efp':'yellow' }

#******* Classes for handling junctions & contacts  *******
#* **They contain the properties of a junction or contact *
#* needed for handling and solving the problems           *
#**********************************************************
class junction(object):
   def __init__(self,idx1,idx2,m1,m2,d):
      if type(idx1) is int and type(idx2) is int:
         self.idx = [[idx1],[idx2]]
      else:
         assert len(idx1) == len(idx2),\
            ("Error!!, index length doesn't match")
         self.idx = [idx1, idx2]
      self.m   = [m1,m2]
      self.d   = d
      ### Dit handling ###
      self.Dit = [None    ,None]
      self.Et  = [m1.Eg/2 ,m2.Eg/2]
   def isins(self):
      return not all([m.type is "semiconductor" for m in self.m])
   def __len__(self):
      return len(self.idx[0])
   def set_it(self,i,dist,Et):
      assert i == 0 or i ==1
      self.Dit[i] = dist
      self.Et [i] = Et

class contact(object):
   __slots__ = ['idx','m','d','pflag','Ec','n','p']
   def __init__(self,idx,m,d,pflag):
      assert pflag == 1 or pflag == -1
      self.idx   = idx
      self.m     = m
      self.d     = d
      self.pflag = pflag
      self.Ec = V0 - m.phiS
      if m.type is "semiconductor":
         self.n  = m.ni
         self.p  = m.ni
   @property
   def V(self):
      return V0 - (self.Ec + self.m.phiS)
   @V.setter
   def V(self,v):
      self.Ec = V0 - self.m.phiS - v
   def __len__(self):
      if type(self.idx) is int:
         return 1
      else:
         return len(self.idx)
   def isins(self):
      return (self.m.type is "insulator")

class contact1(object):
   ## NOTE: only supported by 1-D solver
   __slots__ = ['idx','m','pflag','Q','d']
   def __init__(self,idx,m,d,flag):
      self.idx   = idx
      self.m     = m
      self.pflag = flag
      self.Q     = 0
      self.d     = d
#************* Base class for all solvers  ****************
#* ** It defines the common routines which will be used   *
#* to construct and manage the 1-D representative vectors *
#**********************************************************
class __solver(object):

   meshes  = []
   junc    = []
   contact = []
   c_size = 0
   def __init__():
      pass
   ############# Mesh adding & Edges handling #############
   ## ** This function should be called before starting  ##
   ##  solving. It is implemented differently for        ##
   ##  different-dimensional solver                      ##
   ########################################################
   def add_mesh(self,N,pos,material):
      pass
   ######## Initializing the representative vector ########
   ## ** This function should be called only once after  ##
   ##  finishing adding all meshes                       ##
   ########################################################
   def construct_profile(self):
      print ("========== solver.construct_profile() ==========")
      print ("Generating the complete profile...")
      self.Ec   = np.zeros(self.c_size) 
      self.Ev   = np.zeros(self.c_size) 
      self.Efn  = np.zeros(self.c_size) 
      self.Efp  = np.zeros(self.c_size) 
      self.n    = np.zeros(self.c_size) 
      self.p    = np.zeros(self.c_size) 
      self.Qit  = np.zeros(self.c_size) 
      self.Dit  = np.zeros(self.c_size) 
      x = 0
      for i,m in enumerate(self.meshes):
         y = x + m.size
         self.Ec[x:y] = V0 - m.material.phiS
         self.Ev[x:y] = V0 - m.material.phiS - m.material.Eg
         if m.material.type is 'semiconductor' :
            self.n[x:y]    = m.material.ni
            self.p[x:y]    = m.material.ni
            self.Efn[x:y]  = V0 - m.material.phiS - m.material.Eg/2
            self.Efp[x:y]  = V0 - m.material.phiS - m.material.Eg/2
         elif not m.material.type is 'insulator':
            raise ValueError, \
             "Error, material type (%s) unknown!"\
              %(material.type)
         x = y
      assert x == self.c_size

   #############################################################
   ##   Interfaces for loading/ writing the content of meshes ##
   ## ** Avoid calling them frequently, manipulate the        ##
   ## representative 1-D vector directly!!!                   ##
   #############################################################
   def load_mesh(self,nlist) :
      assert type(nlist) is list
      i = 0
      for num,m in enumerate(self.meshes) :
         s = m.size
         j = i + s
         for name in nlist:
            try:
               vector = self.__dict__[name] 
               d = m.__dict__[name]
               vector[i:j] = d.reshape(s)
            except KeyError:
               print ("no {} attribute for mesh #{}"
                     .format(name,num))
         i = j
      assert i == self.c_size
   def write_mesh(self,nlist) :
      assert type(nlist) is list
      i = 0
      for num,m in enumerate(self.meshes) :
         s = m.size
         j = i + s
         for name in nlist:
            try:
               p = m.__dict__[name]
               p[:] = self.__dict__[name][i:j].reshape(m.N)
            except KeyError: 
               print ("no {} attribute for mesh #{}"
                     .format(name,num))
         i = j
      assert i == self.c_size

   ############ Carrier-Related Calculation routines ###########
   # ** same for all solvers                                  ##
   #############################################################
   def calc_np(self):
      i = 0
      for m in self.meshes :
         j = i + m.size
         if m.material.type is "semiconductor":
            self.n[i:j] = m.material.Nc * \
                         np.exp(( self.Efn[i:j]-self.Ec[i:j])/kBT)
            self.p[i:j] = m.material.Nv * \
                         np.exp((-self.Efp[i:j]+self.Ev[i:j])/kBT)
         elif not m.material.type is "insulator":
            raise ValueError,\
             "Error!! Material %s unkown!!" % m.material.type
         i = j
      assert i == self.c_size
   def calc_Ef(self):
      i = 0
      for m in self.meshes :
         j = i + m.size
         if m.material.type is "semiconductor":
            self.Efn[i:j] = self.Ec[i:j] + \
                        kBT * np.log(self.n[i:j]/m.material.Nc)
            self.Efp[i:j] = self.Ev[i:j] - \
                        kBT * np.log(self.p[i:j]/m.material.Nv)
         elif not m.material.type is "insulator":
            raise ValueError,\
             "error!! Material %s unkown!!" % m.material.type
         i = j
      assert i == self.c_size

   ### Visualization, different for different dimensions ###
   def visualize(self,vlist):
      pass

#********** Base class for handling 1-D problem ***********
#* ** When handling heterojunctions, meshes with diffrent *
#* materials should be added, their properties will be    *
#* handled by the solver                                  *
#* ** It can be the base class of practical solvers, e.g. *
#* poisson solver and drift-diffusion solver              *
#**********************************************************
class solver1D(__solver):

   class mesh1D(object):
      def __init__ (self,dx,N,pos,material):
         self.pos = pos
         self.N   = N
         self.dx  = dx
         self.material = material
         self.NB = np.zeros(N)
         self.Jn = np.zeros(N)
         self.Jp = np.zeros(N)
         self.Ec = (V0 - material.phiS) * np.ones(N)
         self.Ev = self.Ec - material.Eg
         if material.type == 'semiconductor' :
            self.n    = material.ni * np.ones(N)
            self.p    = material.ni * np.ones(N)
            self.Efn  = (self.Ec + self.Ev) /2
            self.Efp  = (self.Ec + self.Ev) /2
            self.GRR  = np.zeros(N)
         elif material.type == 'insulator' :
            pass
         else:
            raise ValueError, \
             "mesh2D() Error, material type (%s) unknown!"\
              %(material.type)
         self.vx = dx * (np.arange(N)+pos)
      def idxlog(self,startidx):
         self.l_idx = startidx
         self.r_idx = startidx + self.N-1
      @property
      def size(self):
         return self.N

   contact1 = []
   neighbor = []
   def __init__ (self, dx) :
      self.dx = dx

   def add_mesh(self,N,pos,material) :
      if any([overlap(pos,N,m.pos,m.N) for m in self.meshes]) :
         raise RuntimeError,\
          "solver1D.add_mesh ERROR, mesh overlap detected!!"
      new = self.mesh1D(self.dx,N,pos,mdb[material])
      new.idxlog(self.c_size)
      print ("solver1D.add_mesh: mesh #{} added, idx={}~{}"\
       .format(len(self.meshes),new.l_idx, new.r_idx))
      for m in self.meshes:
         ##### left junction #####
         if pos == m.pos + m.N:
            print ("\t*Junction at left of the new mesh detected:"
                   " logged as junc[{}]".format(len(self.junc)))
            self.junc.append(junction(m.r_idx,new.l_idx,
                              m.material, new.material,self.dx))
         if pos + N == m.pos:
            print ("\t*Junction at right of the new mesh detected:"
                  " logged as junc[{}]".format(len(self.junc)))
            self.junc.append(junction(new.r_idx,m.l_idx,
                              new.material, m.material,self.dx))
      self.meshes.append(new)
      self.c_size += N
      return new

   def add_contact(self,x,type=0):
      ci = [len(self.contact), len(self.contact1)]
      if any([m.pos<=x<=m.pos+m.N-1 for m in self.meshes]):
         raise ValueError,\
         ("error! contact should not overlap with meshes")
      for n,m in enumerate(self.meshes):
         if x == m.pos - 1:
            print("solver1D.add_contact: type {} contact #{} added"
                  "\n\t*At left of mesh #{}"
                  .format(type,ci[type],n))
            if type is 0:
               new = contact(m.l_idx,m.material,self.dx,-1)
               self.contact.append(new)
               return new
            elif type is 1:
               new = contact1(m.l_idx,m.material,self.dx,-1)
               self.contact1.append(new)
               return new

         if x == m.pos + m.N:
            print("solver1D.add_contact: type {} contact #{} added"
                  "\n\t*At right of mesh #{}"
                  .format(type,ci[type],n))
            if type is 0:
               new = contact(m.r_idx,m.material,self.dx,1)
               self.contact.append(new)
               return new
            elif type is 1:
               new = contact1(m.r_idx,m.material,self.dx,1)
               self.contact1.append(new)
               return new
      else:
         print("No connection with mesh detected, ignored")

   def calc_it(self):
      for j in self.junc:
         for i in [0,1]:
            if not j.Dit[i] is None:
               self.Qit[j.idx[i]] = j.Dit[i] * \
                                    (self.Ec[j.idx[i]] -j.Et[i] - self.Efn[j.idx[i]]) / self.dx
               self.Dit[j.idx[i]] = j.Dit[i] / self.dx

   def construct_profile(self):
      super(solver1D,self).construct_profile()
      print ("logging indice of neighboring pairs",end='...')
      nn = []
      for m in self.meshes:
         n = np.empty([2,m.N-1])
         n[0,:] = np.arange(m.l_idx,m.r_idx,dtype=int)
         n[1,:] = n[0,:]+1
         nn.append(n)
      self.neighbor = nx = np.hstack([n for n in nn])
      print ("done")

      print ("logging indice of jundction pairs",end='...')
      j = np.zeros([2,0])
      if len(self.junc):
         j = np.hstack([ j.idx for j in self.junc ])
      print ("done")

      print ("logging indice of all contacts",end='...')
      c = np.zeros(0)
      if len(self.contact):
         c= [c.idx for c in self.contact]
      print ("done")

      self.op_row = np.concatenate((j[0],j[1],j[1],j[0],
                                    nx[0],nx[1],nx[0],nx[1],c))
      self.op_col = np.concatenate((j[1],j[0],j[1],j[0],
                                    nx[1],nx[0],nx[0],nx[1],c))

   def visualize(self,vlist):
      plt.figure()
      """for m in self.meshes:
         for v in vlist:
            if hasattr(m,v):
               plt.plot(m.vx,m.__dict__[v],color=pcol[v])"""
      for v in vlist:
         plt.plot(self.__dict__[v],color=pcol[v])
      plt.xlim([0,self.c_size])
      plt.show()

#********** Base class for handling 2-D problem ***********
#* ** It defines how to relate the meshes to a 1-D        *
#* representative vector, and how to manage the boundary  *
#* of the meshes.                                         *
#* ** It can be the base class of practical solvers, e.g. *
#* poisson solver and drift-diffusion solver.             *
#**********************************************************
class solver2D(__solver):

   class mesh2D(object):
      
      def __init__ (self, dx, dy, N, pos, material) :
         assert len(N) == 2
         self.pos = pos
         self.N   = N
         self.Nx  = N[0]
         self.Ny  = N[1]
         self.dx  = dx
         self.dy  = dy
         self.material = material

         self.NB  = np.zeros(N)
         self.Jn = np.zeros(N)
         self.Jp = np.zeros(N)
         self.Ec  = (V0 - material.phiS) * np.ones(N)
         self.Ev  = self.Ec - material.Eg
         if material.type == 'semiconductor' :
            self.n    = material.ni * np.ones(N)
            self.p    = material.ni * np.ones(N)
            self.Efn  = (self.Ec + self.Ev) /2
            self.Efp  = (self.Ec + self.Ev) /2
            self.GRR  = np.zeros(N)
         elif material.type == 'insulator' :
            pass
         else:
            raise ValueError, \
             "mesh2D() Error, material type (%s) unknown!"\
              %(material.type)

         x = dx * (np.arange(N[0]) + pos[0])
         y = dy * (np.arange(N[1]) + pos[1])
         self.vx, self.vy = np.meshgrid(x,y)
         self.vx = self.vx.T
         self.vy = self.vy.T

      def idxlog(self,startidx):
         self.l_idx = np.arange(self.Nx)*self.Ny + startidx
         self.t_idx = np.arange(self.Ny) + startidx
         self.r_idx = self.l_idx + self.Ny -1
         self.b_idx = self.t_idx + self.Ny * (self.Nx -1)
         # moving junction management function to the solvers 

      @property
      def size(self):
         return self.Nx * self.Ny

   # containing the indice of neighboring pair
   neighbor = {'x':[], 'y':[]}
   def __init__ (self, dx, dy) :
      self.dx = dx
      self.dy = dy

   def add_mesh(self, N, pos=[0,0], material='Si') :
      assert len(N)==2, "N must be a list containing the 2-D sizes"
      ##### Should not exit overlaps between meshes #####
      if any([overlap(pos[0],N[0],m.pos[0],m.N[0]) 
          and overlap(pos[1],N[1],m.pos[1],m.N[1]) 
          for m in self.meshes]):
         raise RuntimeError,\
          "solver2D.add_mesh ERROR, mesh overlap detected!!"

      new = self.mesh2D(self.dx,self.dy, N, pos, mdb[material])
      new.idxlog(self.c_size)
      print ("solver2D.add_mesh: mesh #{} added, idx={}~{}".format
       (len(self.meshes),self.c_size,self.c_size+N[0]*N[1]-1))

      ### detect junctions between new and old meshes ###
      for m in self.meshes:

         ###### junction at top or bottom of the new mesh ########
         if overlap(pos[1],N[1],pos[1],m.N[1]):
            i, j, k, l = calc_offset(pos[1],N[1],m.pos[1],m.N[1])
            ##### top junction #####
            if pos[0] == m.pos[0]+m.N[0]: 
               assert len(new.t_idx[i:k]) == len(m.b_idx[j:l])
               print ("\t*Junction at top of the new mesh: "
                "overlap length=", len(new.t_idx[i:k]))
               print ("\t --> logged as junc[{}]"
                      .format(len(self.junc)))
               self.junc.append(junction(m.  b_idx[j:l],
                                         new.t_idx[i:k],
                               m.material, new.material,self.dx))

            ###### bottom junction #####
            elif pos[0] + N[0] == m.pos[0]:
               assert len(new.b_idx[i:k]) == len(m.t_idx[j:l])
               print ("\t*Junction at bottom of the new mesh: "
                "overlap lenth=", len(new.b_idx[i:k]))
               print ("\t --> logged as junc[{}]"
                      .format(len(self.junc)))
               self.junc.append(junction(new.b_idx[i:k],
                                         m.  t_idx[j:l],
                                new.material, m.material, self.dx))

         ###### junction at left or right of the new mesh ######
         if overlap(pos[0],N[0],m.pos[0],m.N[0]):
            i, j, k, l = calc_offset(pos[0],N[0],m.pos[0],m.N[0])
            ##### right junction #####
            if pos[1] + N[1] == m.pos[1]:
               assert len(new.r_idx[i:k]) == len(m.l_idx[j:l])
               print ("\t*Junction at right of the new mesh:",
                "overlap lenth=", len(new.r_idx[i:k]))
               print ("\t --> logged as junc[{}]"
                      .format(len(self.junc)))
               self.junc.append(junction(new.r_idx[i:k],
                                         m.  l_idx[j:l],
                                new.material, m.material, self.dy))
            ###### left junction #######
            elif pos[1] == m.pos[1] + m.N[1]:
               assert len(new.l_idx[i:k]) == len(m.r_idx[j:l])
               print ("\t*Junction at left of the new mesh:",
                "overlap lenth=", len(new.l_idx[i:k]))
               print ("\t --> logged as junc[{}]"
                      .format(len(self.junc)))
               self.junc.append(junction(m.  r_idx[j:l],
                                         new.l_idx[i:k],
                                  m.material, new.material,self.dy))

      self.meshes.append(new)
      self.c_size += N[0] * N[1]
      return new

   def add_contact(self,x,y) :
      ci = len(self.contact)

      #### y contanct ####
      if type(x) is list and len(x)==2 and type(y) is int:
         Nx = x[1] - x[0] + 1
         assert Nx > 0, "require x[1] > x[0]!!"
         if any([overlap(x[0],Nx,m.pos[0],m.N[0]) and\
                 m.pos[1]<= y <= m.pos[1] + m.N[1] -1
                 for m in self.meshes]):
            raise ValueError,\
            ("error! contact should not overlap with meshes")

         for n,m in enumerate(self.meshes):
            if overlap(x[0],Nx,m.pos[0],m.N[0]):
               _,i,_,k = calc_offset(x[0],Nx,m.pos[0],m.N[0])
               if y == m.pos[1] - 1:
                  print("solver2D.add_contact: contact #{} added"
                        "\n\t*At left of mesh #{}, length={}"
                        .format(ci,n,len(m.l_idx[i:k])))
                  new = contact(m.l_idx[i:k],m.material,self.dy,-1)
                  self.contact.append(new)
                  return new
               elif y == m.pos[1] + m.N[1]:
                  print("solver2D.add_contact: contact #{} added"
                        "\n\t*At right of mesh #{}, length={}"
                        .format(ci,n,len(m.r_idx[i:k])))
                  new = contact(m.r_idx[i:k],m.material,self.dy,1)
                  self.contact.append(new)
                  return new
         else:
            print("No connection with mesh detected, ignored")
      #### x contact ####
      elif type(y) is list and len(y)==2 and type(x) is int:
         Ny = y[1] - y[0] + 1
         assert Ny > 0, "require y[1] > y[0]!!"
         if any([overlap(y[0],Ny,m.pos[1],m.N[1]) and\
                 m.pos[0]<= x <= m.pos[0] + m.N[0] -1
                 for m in self.meshes]):
            raise ValueError,\
            ("error! contact should not overlap with meshes")

         for n,m in enumerate(self.meshes):
            if overlap(y[0],Ny,m.pos[1],m.N[1]):
               _,i,_,k = calc_offset(y[0],Ny,m.pos[1],m.N[1])
               if x == m.pos[0] - 1:
                  print("solver2D.add_contact: contact #{} added"
                        "\n\t*At top of mesh #{}, length={}"
                        .format(ci,n,len(m.t_idx[i:k])))
                  new = contact(m.t_idx[i:k],m.material,self.dx,-1)
                  self.contact.append(new)
                  return new
               elif x == m.pos[0] + m.N[0]:
                  print("solver2D.add_contact: contact #{} added"
                        "\n\t*At bottom of mesh #{}, length={}"
                        .format(ci,n,len(m.b_idx[i:k])))
                  new = contact(m.b_idx[i:k],m.material,self.dx,1)
                  self.contact.append(new)
                  return new
         else:
            print("No connection with mesh detected, ignored")
      else:
         raise ValueError,"Wrong Format!\n<usage>:"\
              "add_contact([x1,x2],y) or add_contact(x,[y1,y2])"

   def construct_profile(self):
      super(solver2D,self).construct_profile()
      print ("logging indice of neighboring pairs inside",end='...')
      nn = {'x':[],'y':[]}
      for m in self.meshes:
         xx = np.zeros([2,m.Ny*(m.Nx-1)])
         yy = np.zeros([2,m.Nx*(m.Ny-1)])
         for i in range(m.Nx-1) :
            xx[0,i*m.Ny:(i+1)*m.Ny] = m.t_idx + i*m.Ny
         xx[1,:] = xx[0,:] + m.Ny
         for i in range(m.Ny-1) :
            yy[0,i*m.Nx:(i+1)*m.Nx] = m.l_idx + i
         yy[1,:] = yy[0,:] + 1
         nn['x'].append(xx)
         nn['y'].append(yy)
      self.neighbor['x'] = nx = np.hstack([n for n in nn['x']])
      self.neighbor['y'] = ny = np.hstack([n for n in nn['y']])
      print ("done")

      print ("logging indice of jundction pairs",end='...')
      j = np.zeros([2,0])
      if len(self.junc):
         j = np.hstack([ j.idx for j in self.junc ])
      print ("done")

      print ("logging indice of all contacts",end='...')
      c = np.zeros(0)
      if len(self.contact):
         c= np.concatenate([c.idx for c in self.contact])
      print ("done")

      ### for creating operator matrice in the future
      self.op_row = np.concatenate((j[0],j[1],j[1],j[0],
                                    nx[0],nx[1],ny[0],ny[1],
                                    nx[0],nx[1],ny[0],ny[1],c))

      self.op_col = np.concatenate((j[1],j[0],j[1],j[0],
                                    nx[1],nx[0],ny[1],ny[0],
                                    nx[0],nx[1],ny[0],ny[1],c))

   #############################################################
   ##        Visulization of the semiconductor meshes         ##
   ## ** Use write_mesh to update the mesh before calling it. ##
   #############################################################
   def visualize(self,vlist):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for m in self.meshes:
         for v in vlist:
            if hasattr(m,v):
               ax.plot_surface(m.vx,m.vy,m.__dict__[v],\
                      rstride=2,cstride=2,color=pcol[v])

      plt.show()
