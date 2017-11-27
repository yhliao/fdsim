#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver2D
from solver.const import mdb, kBT
import numpy as np
import csv
import pickle
import pylab

################################################################################
tins_array = {0.8e-9,1.2e-9,1.6e-9}
for tinsf in tins_array:
    device='ncfet'
    NSD =1e20*1e6
    Lg  = 25e-9
    #tinsf = 1.2e-9
    tinsb = tinsf
    Tch   = 8e-9
    Ls = 20e-9
    Ld = 20e-9
    dx = 0.2e-9
    dy = 1e-9
    Vref=0.0
    PHIG=4.5
    NBODY=1e21
    Vds = 0.05
    alpha = -3e9 #;%m/F alpha=-1.61379679e-2
    beta = 6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
    tfe = 3e-9#;%m
    Vref = 1.0
    vgpoints=20
    iterations_fe=3

    ################################################################################
    def VFE(qfe,alpha,beta,tfe):
       return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx],idx

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
    Vg = np.linspace(0,1,vgpoints)
    Vgeff = Vg+Vref-(PHIG-(mdb['Si'].phiS+mdb['Si'].Eg/2+ kBT*np.log(NBODY/ni)))-kBT*np.log(NBODY*NSD/ni**2)


    IDn   = np.empty(vgpoints)
    IDp   = np.empty(vgpoints)
    IDn_t = np.empty(vgpoints)
    IDp_t = np.empty(vgpoints)

    Ign   = np.empty(vgpoints)
    Igp   = np.empty(vgpoints)
    Ign_t = np.empty(vgpoints)
    Igp_t = np.empty(vgpoints)

    Qg_array  = []
    Vfe_array = []

    vfe_voltage = np.zeros(int(np.round((Lg/dy))))
    for n,V in enumerate(Vgeff):
       for i in range(iterations_fe):
           cg.V = V*np.ones(int(np.round((Lg/dy))))-vfe_voltage
           cgb.V = cg.V
           s.solve(1e-3,False,False)
           vfe_voltage_old = vfe_voltage
           vfe_voltage = VFE(cg.D,alpha,beta,tfe)
           print ('VFE sum diff:')
           print (np.sum(abs(vfe_voltage_old-vfe_voltage)))

       (IDn[n],IDp[n]) = (cd.In, cd.Ip)
       (Ign[n],Igp[n]) = (cg.In, cg.Ip)
       print ("**** VG={}, IDn={}, Ig={} ***".format(V,-cd.In,cg.In))
       Qg_array.append(cg.D)
       Vfe_array.append(vfe_voltage)
       #s.visualize(['Ec','Ev','Efn','Efp'])
       #m2.cshow('n')
       #pickle.dump(s,output)

    pylab.figure(1)
    pylab.plot(Vg, IDn,'o')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'o')
    pylab.yscale('log')

    #s.visualize(['Ec','Ev','Efn','Efp'])

    ### Simple access for the displacements at the contact
    #save vfe, potential front, potentia
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_VDS'+str(Vds)
    np.savetxt(prefix+'_Vg_data.txt', Vg, delimiter=',')
    np.savetxt(prefix+'_IDn_data.txt', IDn, delimiter=',')
    np.savetxt(prefix+'_Qg_data_new.txt', np.transpose(Qg_array), delimiter=',')
    np.savetxt(prefix+'_Vfe_data_new.txt', np.transpose(Vfe_array), delimiter=',')
pylab.show()
