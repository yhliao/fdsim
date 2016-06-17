from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, Normalize

from solver.const import kBT, mdb ,q
from solver.util import overlap, calc_offset

V0   =  4.5
pcol = { 'Ec' :'blue','Ev' :'green' ,
         'n'  :'red' ,'p'  :'yellow',
         'Efn':'red' ,'Efp':'yellow' }

scol = { 'Ec' :'winter','Ev' :'summer' ,
         'n'  :'autumn' ,'p'  :'spring',
         'Efn':'autumn' ,'Efp':'spring' }

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
   #__slots__ = ['idx','m','d','Ec','n','p']
   def __init__(self,idx,m,d):
      if type(idx) is int:
         self.idx = [idx]
      else:
         self.idx   = idx
      self.m     = m
      self.d     = d
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

class __solver(object):

   """
   ************* Base class for all solvers  ****************
   * ** It defines the common routines which will be used   *
   * to construct and manage the 1-D representative vectors *
   **********************************************************
   """

   def __init__(self):
      self.meshes  = []
      self.junc    = []
      self.contact = []
      self.c_size = 0

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
      #self.ni   = np.zeros(self.c_size) 
      #self.taun = np.zeros(self.c_size)
      #self.taup = np.zeros(self.c_size)
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
            #self.ni[x:y]   = m.material.ni
            self.Efn[x:y]  = V0 - m.material.phiS - m.material.Eg/2
            self.Efp[x:y]  = V0 - m.material.phiS - m.material.Eg/2
         elif not m.material.type is 'insulator':
            raise ValueError, \
             "Error, material type (%s) unknown!" %(material.type)
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
               pass
               #print ("no {} attribute for mesh #{}"
               #      .format(name,num))
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
               pass
               #print ("no {} attribute for mesh #{}"
               #      .format(name,num))
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

class solver1D(__solver):

   """
   ********** Base class for handling 1-D problem ***********
   * ** When handling heterojunctions, meshes with diffrent *
   * materials should be added, their properties will be    *
   * handled by the solver                                  *
   * ** It can be the base class of practical solvers, e.g. *
   * poisson solver and drift-diffusion solver              *
   **********************************************************
   """

   def __init__ (self, dx) :
      super(solver1D,self).__init__()
      self.contact1 = []
      self.neighbor = []
      self.dx = dx

   def add_mesh(self,N,pos,material) :
      if any([overlap(pos,N,m.pos,m.N) for m in self.meshes]) :
         raise RuntimeError,\
          "solver1D.add_mesh ERROR, mesh overlap detected!!"
      new = mesh1D(self.dx,N,pos,mdb[material])
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
               new = contact(m.l_idx,m.material,self.dx)
               self.contact.append(new)
               return new
            elif type is 1:
               new = contact1(m.l_idx,m.material,self.dx)
               self.contact1.append(new)
               return new

         if x == m.pos + m.N:
            print("solver1D.add_contact: type {} contact #{} added"
                  "\n\t*At right of mesh #{}"
                  .format(type,ci[type],n))
            if type is 0:
               new = contact(m.r_idx,m.material,self.dx)
               self.contact.append(new)
               return new
            elif type is 1:
               new = contact1(m.r_idx,m.material,self.dx)
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
         c= [c.idx[0] for c in self.contact]
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
         plt.plot(self.__dict__[v],color=pcol[v],linewidth=2)
      plt.xlim([0,self.c_size])
      plt.show()

class mesh2D(object):

   def __init__ (self, dx, dy, N, pos, material) :
      assert len(N) == 2
      self.junc    = [ [[],[]] , [[],[]] ]
      #self.contact = [ [[],[]] , [[],[]] ]
      self.pos = pos
      self.N   = N
      self.Nx  = N[0]
      self.Ny  = N[1]
      self.dx  = dx
      self.dy  = dy
      self.d   = [dx,dy]
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
         #self.GRR  = np.zeros(N)
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

      ### position of the mesh
      self.extent = (dx* pos[0],        ## left
                     dx* (pos[0]+N[0]), ## right
                     dy* (pos[1]+N[1]), ## bottom
                     dy* pos[1]       ) ## top

   def idxlog(self,startidx):
      self.l_idx = np.arange(self.Nx)*self.Ny + startidx
      self.t_idx = np.arange(self.Ny) + startidx
      self.r_idx = self.l_idx + self.Ny -1
      self.b_idx = self.t_idx + self.Ny * (self.Nx -1)
      # moving junction management function to the solvers 

   @property
   def size(self):
      return self.Nx * self.Ny

   def cshow(self,name):
      if hasattr(self,name):
         data = self.__dict__[name]
         if name=='n' or name=='p':
            N = LogNorm(vmin=data.min(),vmax=data.max())
            title = "Carrier concentration "+name+'$\ (m^{-3})$'
         elif name=='Ec'or name=='Ev'or name=='Efn'or name=='Efp':
            N = Normalize(vmin=data.min(),vmax=data.max())
            title = "Energy " + name + ' (eV)'
         else: ## Jn, Jp, NB
            print ("Warning! Showing function for " + name +
                   " has not been implemented, ignored..")
            return
         plt.matshow(data.T,extent=self.extent,
                     norm=N,cmap=scol[name])
         plt.title(title)
         plt.colorbar()
         plt.show()
      else:
         print (("mesh.show: Warning, this mesh (type:{})"
                 "doesn't has attribute {}, ignored")
                .format(self.material.type,name))

class solver2D(__solver):

   """
   ********** Base class for handling 2-D problem ***********
   * ** It defines how to relate the meshes to a 1-D        *
   * representative vector, and how to manage the boundary  *
   * of the meshes.                                         *
   * ** It can be the base class of practical solvers, e.g. *
   * poisson solver and drift-diffusion solver.             *
   **********************************************************
   """

   # containing the indice of neighboring pair
   def __init__ (self, dx, dy) :
      super(solver2D,self).__init__()
      self.neighbor = {'x':[], 'y':[]}
      self.dx = dx
      self.dy = dy

   def add_mesh(self, N, pos=[0,0], material='Si') :

      def log_junc(old,new,axis,edge):
         i, j, k, l = calc_offset(new.pos[axis],new.N[axis],
                                  m.  pos[axis],m.  N[axis])
         if edge=='top':
            assert axis == 1
            (idx0,mat0) = (old.b_idx[j:l],old.material)
            (idx1,mat1) = (new.t_idx[i:k],new.material)
            d = self.dx
         elif edge == 'bottom' :
            assert axis == 1
            (idx0,mat0) = (new.b_idx[i:k],new.material)
            (idx1,mat1) = (old.t_idx[j:l],old.material)
            d = self.dx
         elif edge == 'left' :
            assert axis == 0
            (idx0,mat0) = (old.r_idx[j:l],old.material)
            (idx1,mat1) = (new.l_idx[i:k],new.material)
            d = self.dy
         elif edge == 'right' :
            assert axis == 0
            (idx0,mat0) = (new.r_idx[i:k],new.material)
            (idx1,mat1) = (old.l_idx[j:l],old.material)
            d = self.dy
         else:
            raise ValueError,("edge should be one of "
            "'top' or 'bottom or 'left' or 'right'")
         assert len(idx0) == len(idx1)
         print ("\t*Junction at",edge,"of the new mesh: "
                "overlap length=", len(idx0))
         print ("\t --> logged as junc[{}]".format(len(self.junc)))
         junc = junction(idx0,idx1,mat0,mat1,d)
         self.junc.append(junc)

         if mat0.type == mat1.type =='insulator':
            print ("\tReminder: The tunneling current for an"+
                   " insulator stack won't be calculated")
         elif old.material.type is "insulator":
            pflag = 0 if (edge=='bottom'or edge =='right') else 1
            old.junc[axis][pflag].append((junc,j,len(idx0)))
         elif new.material.type is "insulator":
            pflag = 0 if (edge=='top'or edge =='left') else 1
            new.junc[axis][pflag].append((junc,i,len(idx1)))

      assert len(N)==2, "N must be a list containing the 2-D sizes"
      ##### Should not exit overlaps between meshes #####
      if any([overlap(pos[0],N[0],m.pos[0],m.N[0]) 
          and overlap(pos[1],N[1],m.pos[1],m.N[1]) 
          for m in self.meshes]):
         raise RuntimeError,\
          "solver2D.add_mesh ERROR, mesh overlap detected!!"

      new = mesh2D(self.dx,self.dy, N, pos, mdb[material])
      new.idxlog(self.c_size)
      print ("solver2D.add_mesh: mesh #{} added, idx={}~{}".format
       (len(self.meshes),self.c_size,self.c_size+N[0]*N[1]-1))

      ### detect junctions between new and old meshes ###
      for m in self.meshes:

         ###### junction at top or bottom of the new mesh ########
         if overlap(pos[1],N[1],m.pos[1],m.N[1]):
            ##### top junction #####
            if pos[0] == m.pos[0]+m.N[0]:
               log_junc(m,new,1,'top')
            ###### bottom junction #####
            elif pos[0] + N[0] == m.pos[0]:
               log_junc(m,new,1,'bottom')

         ###### junction at left or right of the new mesh ######
         if overlap(pos[0],N[0],m.pos[0],m.N[0]):
            ###### left junction #######
            if pos[1] == m.pos[1] + m.N[1]:
               log_junc(m,new,0,'left')
            ##### right junction #####
            elif pos[1] + N[1] == m.pos[1]:
               log_junc(m,new,0,'right')

      self.meshes.append(new)
      self.c_size += N[0] * N[1]
      return new

   def add_contact(self,x,y) :

      def log_contact(m,pos,N,axis,edge,n):
         _,i,_,k = calc_offset(pos,N,m.pos[axis],m.N[axis])
         if edge=='top':
            assert axis == 1
            idx   = m.t_idx[i:k]
         elif edge == 'bottom' :
            assert axis == 1
            idx   = m.b_idx[i:k]
         elif edge == 'left' :
            assert axis == 0
            idx   = m.l_idx[i:k]
         elif edge == 'right' :
            assert axis == 0 
            idx   = m.r_idx[i:k]
         else:
            raise ValueError,("edge should be one of "
            "'top' or 'bottom or 'left' or 'right'")

         print("solver2D.add_contact: contact #{} added"
               "\n\t*At {} of mesh #{}, length={}"
               .format(len(self.contact),edge,n,len(idx)))
         pflag = 1 if (edge=='bottom' or edge=='right') else 0
         d     = self.dx if axis==1 else self.dy
         new   = contact(idx,m.material,d)
         self.contact.append(new)
         #if m.material.type is 'insulator':
         #   m.contact[axis][pflag].append((new,i,len(idx)))
         return new

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
               #_,i,_,k = calc_offset(x[0],Nx,m.pos[0],m.N[0])
               if y == m.pos[1] - 1:
                  return log_contact(m,x[0],Nx,0,'left',n)
               elif y == m.pos[1] + m.N[1]:
                  return log_contact(m,x[0],Nx,0,'right',n)
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
               #_,i,_,k = calc_offset(y[0],Ny,m.pos[1],m.N[1])
               if x == m.pos[0] - 1:
                  return log_contact(m,y[0],Ny,1,'top',n)
               elif x == m.pos[0] + m.N[0]:
                  return log_contact(m,y[0],Ny,1,'bottom',n)
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
            if m.material.type is 'semiconductor':
               ax.plot_surface(m.vx,m.vy,m.__dict__[v],\
                      rstride=2,cstride=2,cmap=scol[v])
      plt.show()

   def summarize(self):
      from solver.drift_diffusion import SGn
      for c in self.contact:
         lat = self.dx if c.d==self.dy else self.dy
         dE = self.Ec[c.idx] - c.Ec
         An,Bn = SGn(-dE/kBT, c.m.Dn, c.d) 
         Bp,Ap = SGn(-dE/kBT, c.m.Dp, c.d) 
         Jn = q* (An * c.n + Bn * self.n[c.idx]) * c.d
         Jp = q* (Ap * c.p + Bp * self.p[c.idx]) * c.d
         D = c.m.epr * dE / c.d
         c.Jn = sum(Jn) * lat
         c.Jp = sum(Jp) * lat
         c.Q  = sum(D)  * lat
