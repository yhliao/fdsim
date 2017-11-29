import numpy as np
import csv
import pickle
import pylab

device='ncfet'
NSD =1e20*1e6
Lg  = 25e-9
tinsf = 0.8e-9
tinsb = tinsf
Tch   = 11e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.2e-9
dy = 1e-9
Vref=0.0
PHIG=4.5
NBODY=1e21
Vds = 0.05
Ec = 2.5e7 #0.162742753246 *1e6/1e-2 #MV/cm to V/m
Pr = 22.3298216479 *1e-6/1e-4  #uC/cm^2 to C/m^2
alpha = -((3*np.sqrt(3))/2.0)*Ec/Pr
beta = ((3*np.sqrt(3))/2.0)*Ec/Pr**3

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx

tfe_array = {0,1e-9,2e-9,3e-9,4e-9,5e-9}

vth_sat =[]
for tfe in tfe_array:
    prefix='../results_juan2/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

    SS = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
    pylab.figure(3)
    pylab.plot(Vg[:-1], SS,'-')
    axes = pylab.gca()
    axes.set_ylim([50,100])



    value,indexvg = find_nearest(IDn,100)
    vth_sat.append(Vg[indexvg])

pylab.figure(4)
pylab.plot(np.linspace(0.1, 1, num=len(Vfe[:,1])), Vfe,'-')

pylab.figure(5)
pylab.plot( Vfe[1,:],Qg[1,:],'s')

pylab.figure(6)
pylab.plot( Vfe[-1,:],Qg[-1,:],'o')

tfe=5e-9
qfe = np.linspace(-0.2,0.2,40)
pylab.figure(5)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')


pylab.figure(6)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

Vds = 1.05
tfe_array = [0,1e-9,2e-9,3e-9]#;%m
vth_lin =[]
for tfe in tfe_array:
    prefix='../results_juan2/'+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

    value,indexvg = find_nearest(IDn,100)
    vth_lin.append(Vg[indexvg])

pylab.figure(7)
print(vth_sat)
print(vth_lin)
dibl = [(x - y)/(0.5-0.05) for x, y in zip(vth_lin,vth_sat)]#np.map(np.operator.sub, vth_sat, vth_lin)# np.array(vth_sat)-np.array(vth_lin)
print (tfe_array)
print (dibl)
pylab.plot(tfe_array, dibl,'-')

'''
Vds = 0.5
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
