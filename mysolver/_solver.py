from __future__ import absolute_import, print_function

import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from mysolver.const import kBT, mdb
from mysolver.util import overlap, calc_offset

V0   =  4.5
pcol = {'Ec':'b','Ev':'g','n':'r','p':'y'}
#************* Base class for all solvers  ****************
#* ** It defines the common routines which will be used   *
#* to construct and manage the 1-D representative vectors *
#**********************************************************
class __solver(object):

   meshes = []
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
      print ("========== solver.connect_meshes() ==========")
      print ("Generating the complete profile...")
      self.Ec   = np.zeros(self.c_size) 
      self.Ev   = np.zeros(self.c_size) 
      self.Efn  = np.zeros(self.c_size) 
      self.Efp  = np.zeros(self.c_size) 
      self.n    = np.zeros(self.c_size) 
      self.p    = np.zeros(self.c_size) 
      x = 0
      for i,m in enumerate(self.meshes):
         y = x + m.size
         #NOTE: boundary handling will be substituted by contact object

         self.Ec[x:y] = V0 - m.material.phiS
         self.Ev[x:y] = V0 - m.material.phiS - m.material.Eg
         if m.material.type == 'semiconductor' :
            self.n[x:y]    = m.material.ni
            self.p[x:y]    = m.material.ni
            self.Efn[x:y]  = V0 - m.material.phiS - m.material.Eg/2
            self.Efp[x:y]  = V0 - m.material.phiS - m.material.Eg/2
         elif m.material.type != 'insulator':
            raise ValueError, \
             "mesh2D() Error, material type (%s) unknown!"\
              %(material.type)
         x = y
      assert x == self.c_size

   #############################################################
   ##   Interfaces for loading/ writing the content of meshes ##
   ## ** Avoid calling them frequently, manipulate the        ##
   ## representative 1-D vector directly!!!                   ##
   #############################################################
   def load_mesh(self,name) :
      vector = self.__dict__[name] 
      i = 0
      for m in self.meshes :
         s = m.size
         j = i + s
         d = m.__dict__[name]
         vector[i:j] = d.reshape(s) 
         i = j
      assert i == self.c_size
   def write_mesh(self,name) :
      i = 0
      for m in self.meshes :
         s = m.size
         j = i + s
         p = m.__dict__[name]
         p[:] = self.__dict__[name][i:j].reshape(m.N)
         i = j
      assert i == self.c_size

   ############ Carrier-Related Calculation routines ###########
   # ** same for all solvers                                  ##
   #############################################################
   def calc_np(self):
      i = 0
      for m in self.meshes :
         j = i + m.size
         if m.material.type == "semiconductor":
            self.n[i:j] = m.material.Nc * \
                         np.exp(( self.Efn[i:j]-self.Ec[i:j])/kBT)
            self.p[i:j] = m.material.Nv * \
                         np.exp((-self.Efp[i:j]+self.Ev[i:j])/kBT)
         elif m.material.type != "insulator":
            raise ValueError,\
             "Error!! Material %s unkown!!" % m.material.type
         i = j
      assert i == self.c_size
   def calc_Ef(self):
      i = 0
      for m in self.meshes :
         j = i + m.size
         if m.material.type == "semiconductor":
            self.Efn[i:j] = self.Ec[i:j] + \
                        kBT * np.log(self.n[i:j]/m.material.Nc)
            self.Efp[i:j] = self.Ev[i:j] - \
                        kBT * np.log(self.p[i:j]/m.material.Nv)
         elif m.material.type != "insulator":
            raise ValueError,\
             "error!! Material %s unkown!!" % m.material.type
         i = j
      assert i == self.c_size

   ### Visualization, different for different dimensions ###
   def visualize(self,vlist):
      pass

class contact(object):
   __slots__ = ['idx','material','pflag','Ec','n','p']
   def __init__(self,idx,m,flag):
      self.idx      = idx
      self.material = m
      self.pflag    = flag

      self.Ec = V0 - m.phiS
      self.n  = m.ni
      self.p  = m.ni
   @property
   def V(self):
      return V0 - (self.Ec + self.material.phiS)
   @V.setter
   def V(self,v):
      self.Ec = V0 - self.material.phiS - v

#********** Base class for handling 1-D problem ***********
#* ** When handling heterojunctions, meshes with diffrent *
#* materials should be added, their properties will be    *
#* handled by the solver                                  *
#* ** It can be the base class of practical solvers, e.g. *
#* poisson solver and drift-diffusion solver              *
#**********************************************************
class solver1D(__solver):

   class mesh1D(object):
   ## this mesh doesn't contain calculable data (Ec,Ev,n,p,etc.)##
      def __init__ (self,dx,N,pos,material):
         self.pos = pos
         self.N   = N
         self.dx  = dx
         self.material = material
         self.NB  = np.zeros(N)
      def idxlog(self,startidx):
         self.l_idx = startidx
         self.r_idx = startidx + self.N-1
         """self.l_edge = self.r_edge = -1
         self.l_off  = self.r_off  = 0
      def create_boundary(self):
         if self.l_edge == -1:
            print ("\t*mesh1D: left boundary detected")
            self.l_Ec = 0
            self.l_n  = 0
            self.l_p  = 0
         if self.r_edge == -1:
            print ("\t*mesh1D: right boundary detected")
            self.r_Ec = 0
            self.r_n  = 0
            self.r_p  = 0"""#NOTE:to be removed

      @property
      def size(self):
         return self.N

   junct = []
   neighbor = []
   def __init__ (self, dx) :
      print ("========= solver1D initialization ==========")
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
            print ("\t*Junction at left of the new mesh detected")
            self.junct.append([m.r_idx,    new.l_idx,
                               m.material, new.material])
            """new.l_edge = m.r_idx
            new.l_off  = -new.material.phiS + m.material.phiS
            m  .r_edge = m.l_idx
            m  .r_off  =  new.material.phiS - m.material.phiS"""
         if pos + N == m.pos:
            print ("\t*Junction at right of the new mesh detected")
            self.junct.append([new.r_idx,    m.l_idx,
                               new.material, m.material.phiS])
            """new.r_edge = m.l_idx
            new.r_off  = -new.material.phiS + m.material.phiS
            m  .l_edge = m.r_idx
            m  .l_off  =  new.material.phiS - m.material.phiS"""
      self.meshes.append(new)
      self.c_size += N

   def construct_profile(self):
      super(solver1D,self).construct_profile()
      #TODO : neighbor handling

   def visualize(self,vlist):
      fig = plt.figure()
      for v in vlist:
         plt.plot(self.__dict__[v],label=v,color=pcol[v])
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

   # will be used in constructing operator matrice
   # each entry of the list contain indice and material type
   junc     = {'x':[], 'y':[]}
   contact  = {'x':[], 'y':[]}
   # containing the indice of neighboring pair
   neighbor = {'x':[], 'y':[]}
   def __init__ (self, dx, dy) :
      print ("========= solver2D initialization ==========")
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
               self.junc['x'].append([m.  b_idx[j:l],
                                      new.t_idx[i:k],
                                      m.material, new.material])
            ###### bottom junction #####
            elif pos[0] + N[0] == m.pos[0]:
               assert len(new.b_idx[i:k]) == len(m.t_idx[j:l])
               print ("\t*Junction at bottom of the new mesh: "
                "overlap lenth=", len(new.b_idx[i:k]))
               self.junc['x'].append([new.b_idx[i:k],
                                      m.  t_idx[j:l],
                                      new.material, m.material])

         ###### junction at left or right of the new mesh ######
         if overlap(pos[0],N[0],m.pos[0],m.N[0]):
            i, j, k, l = calc_offset(pos[0],N[0],m.pos[0],m.N[0])
            ##### right junction #####
            if pos[1] + N[1] == m.pos[1]:
               assert len(new.r_idx[i:k]) == len(m.l_idx[j:l])
               print ("\t*Junction at right of the new mesh:",
                "overlap lenth=", len(new.r_idx[i:k]))
               self.junc['y'].append([new.r_idx[i:k],
                                      m.  l_idx[j:l],
                                      new.material, m.material])
            ###### left junction #######
            elif pos[1] == m.pos[1] + m.N[1]:
               assert len(new.l_idx[i:k]) == len(m.r_idx[j:l])
               print ("\t*Junction at left of the new mesh:",
                "overlap lenth=", len(new.l_idx[i:k]))
               self.junc['y'].append([m.  r_idx[j:l],
                                      new.l_idx[i:k],
                                      m.material, new.material])

      self.meshes.append(new)
      self.c_size += N[0] * N[1]

   def add_contact(self,x,y) :

      cix = len(self.contact['x'])
      ciy = len(self.contact['y'])

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
                  print("y contact #{} set at left of mesh #{}"\
                        .format(ciy,n))
                  new = contact(m.l_idx[i:k],m.material,0)
                  self.contact['y'].append(new)
                  break
               elif y == m.pos[1] + m.N[1]:
                  print("y contact #{} set at right of mesh #{}"\
                        .format(ciy,n))
                  new = contact(m.r_idx[i:k],m.material,1)
                  self.contact['y'].append(new)
                  break
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
                  print("x contact #{} set at top of mesh #{}"\
                        .format(cix,n)) 
                  new = contact(m.t_idx[i:k],m.material,0)
                  self.contact['x'].append(new)
                  break
               elif x == m.pos[0] + m.N[0]:
                  print("x contact #{} set at bottom of mesh #{}"\
                        .format(cix,n))
                  new = contact(m.b_idx[i:k],m.material,1)
                  self.contact['x'].append(new)
                  break
         else:
            print("No connection with mesh detected, ignored")
      else:
         raise ValueError,"Wrong Format!\n<usage>:"\
              "add_contact([x1,x2],y) or add_contact(x,[y1,y2])"

   def construct_profile(self):
      super(solver2D,self).construct_profile()
      print ("logging indice neighboring pairs",end='...')
      nn = {'x':[],'y':[]}
      for m in self.meshes:
         xx = np.zeros([2,m.Ny*(m.Nx-1)],dtype=int)
         yy = np.zeros([2,m.Nx*(m.Ny-1)],dtype=int)
         for i in range(m.Nx-1) :
            xx[0,i*m.Ny:(i+1)*m.Ny] = m.t_idx + i*m.Ny
         xx[1,:] = xx[0,:] + m.Ny
         for i in range(m.Ny-1) :
            yy[0,i*m.Nx:(i+1)*m.Nx] = m.l_idx + i
         yy[1,:] = yy[0,:] + 1
         nn['x'].append(xx)
         nn['y'].append(yy)
      self.neighbor['x'] = np.hstack([n for n in nn['x']])
      self.neighbor['y'] = np.hstack([n for n in nn['y']])
      print ("done")

   #############################################################
   ##        Visulization of the semiconductor meshes         ##
   ## ** Use write_mesh to update the mesh before calling it. ##
   #############################################################
   def visualize(self,vlist):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for m in self.meshes:
         for v in vlist:
            if m.material.type == 'semiconductor':
               ax.plot_surface(m.vx,m.vy,m.__dict__[v],\
                      rstride=2,cstride=2,color=pcol[v])

      plt.show()