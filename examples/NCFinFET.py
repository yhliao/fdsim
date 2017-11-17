#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver2D
from solver.const import mdb
import numpy as np
import csv
import pickle
import pylab

################################################################################

NSD =1e20*1e6
Lg  = 25e-9
tinsf = 2e-9
tinsb = 2e-9
Tch   = 10e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.5e-9
dy = 1e-9
alpha = -3e9 #;%m/F alpha=-1.61379679e-2
beta = 6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
tfe = 3e-9#;%m
Vref = 1.0
################################################################################

s = dev_solver2D(dx,dy)
#m0 = s.add_mesh(N=[10,100] , pos=[-10,0], material='Si') #gate
m1 = s.add_mesh(N=[int(np.round((tinsf/dx))),int(np.round((Lg/dy)))] , pos=[0,0] ,material='SiO2')
m2 = s.add_mesh(N=[int(np.round((Tch/dx))),int(np.round(((Lg+Ls+Ld)/dy)))], pos=[int(np.round((tinsf/dx))),int(-np.round((Ls/dy)))],material='Si')
m3 = s.add_mesh(N=[int(np.round((tinsb/dx))),int(np.round((Lg/dy)))] , pos=[int(np.round(((tinsf+Tch)/dx))),0] ,material='SiO2')
#m0.NB[:] = 1E19 * 1E6
m2.NB[:] = -1e16 * 1e6
m2.NB[:, 0:int(np.round(Ls/dy))] = NSD
m2.NB[:, int(np.round(((Lg+Ld)/dy))):] = NSD

cg = s.add_contact(-1,[0,int(np.round(Lg/dy))]) #(x=-11), y=0 to 100
cs = s.add_contact([int(np.round(tinsf/dx)),int(np.round(((Tch+tinsf)/dx)))],int(-np.round(Ls/dy+1)))
cd = s.add_contact([int(np.round(tinsf/dx)),int(np.round(((Tch+tinsf)/dx)))],int(np.round(((Lg+Ld)/dy))))
cgb = s.add_contact(int(np.round((Tch+tinsf+tinsb)/dx)),[0, int(np.round((Lg/dy)))])


def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx

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
cd.V = Vref+0.05

f = open("SOIinfo-T.csv",'wb')
writer = csv.writer(f)
step = 20
Vg = np.linspace(0,1,step)+Vref

IDn   = np.empty(step)
IDp   = np.empty(step)
IDn_t = np.empty(step)
IDp_t = np.empty(step)

Ign   = np.empty(step)
Igp   = np.empty(step)
Ign_t = np.empty(step)
Igp_t = np.empty(step)

prefix='NC1_0005V_25nm_'
Vg_file = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn_file = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg_file = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')


prefix='0005V_25nm_'
Vg_file2 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn_file2 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg_file2 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')


Qg_array  = []

for n,V in enumerate(Vg):
   valueV,indexvg = find_nearest(Vg,V)
   cg.V= V*np.ones(int(np.round((Lg/dy))))-VFE(Qg_file[:,indexvg],alpha,beta,tfe)
   cgb.V= V*np.ones(int(np.round((Lg/dy))))-VFE(Qg_file[:,indexvg],alpha,beta,tfe)

   #filename1 = "FDSOI_Vg{}.dat".format(V)
   #output    = open(filename1,'wb')
   s.solve(1e-3,False,False)
   (IDn[n],IDp[n]) = (cd.In, cd.Ip)
   (Ign[n],Igp[n]) = (cg.In, cg.Ip)
   print ("**** VG={}, IDn={}, Ig={} ***".format(V,-cd.In,cg.In))
   Qg_array.append(cg.D)
   #s.visualize(['Ec','Ev','Efn','Efp'])
   #m2.cshow('n')
   #pickle.dump(s,output)

pylab.figure(1)
pylab.plot(Vg, IDn,'o')
pylab.plot(Vg, IDn_file,'s')
pylab.plot(Vg, IDn_file2,'>')


pylab.figure(2)
pylab.plot(Vg, IDn,'o')
pylab.plot(Vg, IDn_file,'s')
pylab.plot(Vg, IDn_file2,'>')
pylab.yscale('log')

s.visualize(['Ec','Ev','Efn','Efp'])

### Simple access for the displacements at the contact
print (cs.D)
prefix='NC2_0005V_25nm_'
np.savetxt(prefix+'Vg_data.txt', Vg, delimiter=',')
np.savetxt(prefix+'IDn_data.txt', IDn, delimiter=',')
np.savetxt(prefix+'Qg_data_new.txt', np.transpose(Qg_array), delimiter=',')
pylab.show()
