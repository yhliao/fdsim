#!/usr/bin/ipython
from solver.dev_sim import dev_solver2D
import numpy as np
import csv
import pickle

s = pickle.load(open('MOS_Vg0.00.dat','rb'))
cg = s.contact[0]
cs = s.contact[1]
cd = s.contact[2]
cb = s.contact[3]
m2 = s.meshes[2]

f = open("info2.csv",'wb')
writer = csv.writer(f)
step = 10
Vg = np.linspace(0,-1,step)

IDn   = np.empty(step)
IDp   = np.empty(step)

Ign   = np.empty(step)
Igp   = np.empty(step)

for n,V in enumerate(Vg):
   cg.V= V
   #filename1 = "MOS_Vg{:.3F}.dat".format(V)
   #output    = open(filename1,'wb')
   s.solve(1e-3,True,False)
   (IDn[n],IDp[n]) = (cd.In, cd.Ip)
   (Ign[n],Igp[n]) = (cg.In, cg.Ip)
   print ("**** VG={}, IDn={}, Ig={} ***".format(V,cd.In,cg.In))

   s.visualize(['Ec','Ev'])
   m2.cshow('n')

   #pickle.dump(s,output)

writer.writerow(Vg)
writer.writerow(IDn)
writer.writerow(IDp)
writer.writerow(Ign)
writer.writerow(Igp)
