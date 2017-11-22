import numpy as np
import csv
import pickle
import pylab

device='ncfet'
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
alpha = -3e9 #;%m/F alpha=-1.61379679e-2
beta = 6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
tfe = 0#;%m
Vref = 1.0
vgpoints=20

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx

tfe_array = {0,1e-9,2e-9,3e-9}#;%m

for tfe in tfe_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

tfe = 0#;%m
Lg_array = {20e-9,25e-9,30e-9}#;%m
for Lg in Lg_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(3)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(4)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

tfe = 3e-9#;%m
Lg_array = {20e-9,25e-9,30e-9}#;%m
for Lg in Lg_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(3)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(4)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

tfe = 0#;%m
Lg=25e-9
Tch_array = {6e-9,8e-9,10e-9}#;%m
for Tch in Tch_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(5)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(6)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

tfe = 3e-9#;%m
Lg=25e-9
Tch_array = {6e-9,8e-9,10e-9}#;%m
for Tch in Tch_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(5)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(6)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')


tfe = 0#;%m
Lg=25e-9
Tch=8e-9
tins_array = {0.8e-9,1.2e-9,1.6e-9}#;%m
for tinsf in tins_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(7)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(8)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

tfe = 3e-9#;%m
Lg=25e-9
Tch=8e-9
tins_array = {0.8e-9,1.2e-9,1.6e-9}#;%m
for tinsf in tins_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(7)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(8)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')


tfe = 0#;%m
Lg=25e-9
Tch=8e-9
tinsf=1.2e-9
NBODY_array = {1e21}
for NBODY in NBODY_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(9)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(10)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')


NBODY_array = {1e24,5e24}#;%m+'_NBODY'+str(NBODY)
for NBODY in NBODY_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(9)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(10)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

tfe = 3e-9#;%m
NBODY_array = {1e21}
for NBODY in NBODY_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(9)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(10)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')


NBODY_array = {1e24,5e24}#;%m+'_NBODY'+str(NBODY)
for NBODY in NBODY_array:
    prefix='../results_juan/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(9)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(10)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

'''
SSfinfet = 1e3*np.diff(Vg0)/ np.diff(np.log10(IDn0))
SSncfinfet = 1e3*np.diff(Vg0)/ np.diff(np.log10(IDn1))

pylab.figure(3)
pylab.plot(Vg0[:-1], SSfinfet,'o-')
pylab.plot(Vg0[:-1], SSncfinfet,'s-')

SSfinfet = 1e3*np.diff(Vg0)/ np.diff(np.log10(IDn0))
SSncfinfet = 1e3*np.diff(Vg0)/ np.diff(np.log10(IDn1))

pylab.figure(4)
pylab.plot(Vg0[:-1], SSfinfet,'o-')
pylab.plot(Vg0[:-1], SSncfinfet,'s-')
'''

pylab.show()
