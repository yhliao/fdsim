#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver2D
from solver.const import mdb, kBT
import solver.model as model
import numpy as np
import csv
import pickle
import pylab

NSD =1e20*1e6
Lg  = 25e-9
tinsf = 1.2e-9
tinsb = tinsf
Tch   = 8e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.2e-9
dy = 1e-9
Vref=0.0
PHIG=4.5
NBODY=1e21
Vds = 0.5
#############################################
s = dev_solver2D(dx,dy)
#m0 = s.add_mesh(N=[10,100] , pos=[-10,0], material='Si') #gate
m1 = s.add_mesh(N=[int(np.round((tinsf/dx))),int(np.round((Lg/dy)))] , pos=[0,0] ,material='SiO2')
m2 = s.add_mesh(N=[int(np.round((Tch/dx))),int(np.round(((Lg+Ls+Ld)/dy)))], pos=[int(np.round((tinsf/dx))),int(-np.round((Ls/dy)))],material='Si')
m3 = s.add_mesh(N=[int(np.round((tinsb/dx))),int(np.round((Lg/dy)))] , pos=[int(np.round(((tinsf+Tch)/dx))),0] ,material='SiO2')
#m0.NB[:] = 1E19 * 1E6
m2.NB[:] = -NBODY
m2.NB[:, 0:int(np.round(Ls/dy))] = NSD
m2.NB[:, int(np.round(((Lg+Ld)/dy))):] = NSD

cg = s.add_contact(-1,[0,int(np.round(Lg/dy))]) #(x=-11), y=0 to 100
cs = s.add_contact([int(np.round(tinsf/dx)),int(np.round(((Tch+tinsf)/dx)))],int(-np.round(Ls/dy+1)))
cd = s.add_contact([int(np.round(tinsf/dx)),int(np.round(((Tch+tinsf)/dx)))],int(np.round(((Lg+Ld)/dy))))
cgb = s.add_contact(int(np.round((Tch+tinsf+tinsb)/dx)),[0, int(np.round((Lg/dy)))])


ni = mdb['Si'].ni

cs.n = NSD
cs.p = ni**2/NSD
cd.n = NSD
cd.p = ni**2/NSD

cg.n = 1e19 * 1e6
cg.p = 1e1 * 1e6
cgb.n = 1e19 * 1e6
cgb.p = 1e1 * 1e6

#s.visualize(['Ec','Ev'])
s.construct_profile()

### The B.C can be either vector or number
cs.V = Vref
cd.V = Vref+Vds

f = open("SOIinfo-T.csv",'wb')
writer = csv.writer(f)
step = 20
Vg = np.linspace(0,1,step)
Vgeff = Vg+Vref-(PHIG-(mdb['Si'].phiS+mdb['Si'].Eg/2+ kBT*np.log(NBODY/ni)))-kBT*np.log(NBODY*NSD/ni**2)

IDn   = np.empty(step)
IDp   = np.empty(step)
IDn_t = np.empty(step)
IDp_t = np.empty(step)

Ign   = np.empty(step)
Igp   = np.empty(step)
Ign_t = np.empty(step)
Igp_t = np.empty(step)


Qg_array  = []

model.HighFieldDep = True
for n,V in enumerate(Vgeff):
   cg.V= V
   cgb.V= V
   #filename1 = "FDSOI_Vgeff{}.dat".format(V)
   #output    = open(filename1,'wb')
   s.solve(1e-3,False,False)
   (IDn[n],IDp[n]) = (cd.In, cd.Ip)
   (Ign[n],Igp[n]) = (cg.In, cg.Ip)
   print ("**** Vgeff={}, IDn={}, Ig={} ***".format(V,-cd.In,cg.In))
   Qg_array.append(cg.D)
   #s.visualize(['Ec','Ev','Efn','Efp'])
   #m2.cshow('n')
   #pickle.dump(s,output)
pylab.plot(Vg, IDn)

pylab.plot(Vg, IDn)
pylab.yscale('log')

s.visualize(['Ec','Ev','Efn','Efp'])

### Simple access for the displacements at the contact
print (cs.D)
prefix='FiNFET_LG25EOT1TCH8_VD05'
np.savetxt(prefix+'Vgeff_data.txt', Vg, delimiter=',')
np.savetxt(prefix+'IDn_data.txt', IDn, delimiter=',')
np.savetxt(prefix+'Qg_data_new.txt', np.transpose(Qg_array), delimiter=',')
pylab.show()
