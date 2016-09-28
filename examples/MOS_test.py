#!/usr/bin/ipython
#from solver.drift_diffusion import J_solver1D
from solver.dev_sim import dev_solver2D
import numpy as np
import csv
import pickle


s = dev_solver2D(2e-9,2E-9)
m0 = s.add_mesh(N=[40,50] , pos=[-40,0], material='Si')
m1 = s.add_mesh(N=[4,50] , pos=[0,0] ,material='SiO2')
m2 = s.add_mesh(N=[100,90], pos=[4,-20],material='Si')
m0.NB[:] = 1E19 * 1E6
m2.NB[:] = -1e16 * 1e6
m2.NB[0:20, 0:20] = 1E19 * 1E6
m2.NB[0:20,70:90] = 1E19 * 1E6
cg = s.add_contact(-41,[0,50])
cs = s.add_contact([4,10],-21)
cd = s.add_contact([4,10],70)
cb = s.add_contact(104,[0,50])
s.construct_profile()

cs.n = 1e19 * 1e6
cs.p = 1e1 * 1e6
cd.n = 1e19 * 1e6
cd.p = 1e1 * 1e6

cg.n = 1e19 * 1e6
cg.p = 1e1 * 1e6
cb.n = 1e4  * 1e6
cb.p = 1e16 * 1e6

s.visualize(['Ec','Ev','Efn','Efp'])

cs.V = 0.87
cd.V = 1.87
cb.V = 0

step = 20
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
   filename1 = "MOS_Vg{:.2f}.dat".format(V)
   output    = open(filename1,'wb')
   s.solve(1e-3,True,False)
   (IDn[n],IDp[n]) = ( cd.Jn, cd.Jp)
   (Ign[n],Igp[n]) = ( cg.Jn, cg.Jp)
   print ("**** VG={}, IDn={}, Ig={} ***".format(V,cd.Jn,cg.Jn))

   s.visualize(['Ec','Ev','Efn','Efp'])
   m2.cshow('n')

   pickle.dump(s,output)

f = open("info1.csv",'wb')
writer = csv.writer(f)
writer.writerow(Vg)
writer.writerow(IDn)
writer.writerow(IDp)
writer.writerow(Ign)
writer.writerow(Igp)
