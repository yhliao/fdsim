#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver2D
import numpy as np
import csv
import pickle


s = dev_solver2D(1e-9,1E-9)
m0 = s.add_mesh(N=[10,100] , pos=[-10,0], material='Si')
m1 = s.add_mesh(N=[2,100] , pos=[0,0] ,material='SiO2')
m2 = s.add_mesh(N=[10,180], pos=[2,-40],material='Si')
m0.NB[:] = 1E19 * 1E6
m2.NB[:] = -1e16 * 1e6
m2.NB[:, 0:40] = 1E18 * 1E6
m2.NB[:, 140:] = 1E18 * 1E6
cg = s.add_contact(-11,[0,100])
cs = s.add_contact([2,12],-41)
cd = s.add_contact([2,12],140)

cs.n = 1e18 * 1e6
cs.p = 1e2 * 1e6
cd.n = 1e18 * 1e6
cd.p = 1e2 * 1e6

cg.n = 1e19 * 1e6
cg.p = 1e1 * 1e6

s.visualize(['Ec','Ev'])
s.construct_profile()

### The B.C can be either vector or number
cs.V = 0.87 * np.ones(10)
cd.V = 1.87

f = open("SOIinfo-T.csv",'wb')
writer = csv.writer(f)
step = 30
Vg = np.linspace(0,2,step)

IDn   = np.empty(step)
IDp   = np.empty(step)
IDn_t = np.empty(step)
IDp_t = np.empty(step)

Ign   = np.empty(step)
Igp   = np.empty(step)
Ign_t = np.empty(step)
Igp_t = np.empty(step)


for n,V in enumerate(Vg):
   cg.V= V
   #filename1 = "FDSOI_Vg{}.dat".format(V)
   #output    = open(filename1,'wb')
   s.solve(1e-3,True,False)
   (IDn[n],IDp[n]) = (cd.In, cd.Ip)
   (Ign[n],Igp[n]) = (cg.In, cg.Ip)
   print ("**** VG={}, IDn={}, Ig={} ***".format(V,-cd.In,cg.In))

   #s.visualize(['Ec','Ev','Efn','Efp'])
   #m2.cshow('n')
   #pickle.dump(s,output)

writer.writerow(Vg)
writer.writerow(IDn)
writer.writerow(IDp)
writer.writerow(Ign)
writer.writerow(Igp)
### Simple access for the displacements at the contact
print cs.D
"""writer.writerow(IDn_t)
writer.writerow(IDp_t)
writer.writerow(Ign_t)
writer.writerow(Igp_t)"""
