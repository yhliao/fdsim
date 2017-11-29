import numpy as np
import csv
import pickle
import pylab

device='ncfet'
NSD = 2e+26
Lg  = 25e-9
tinsf = 0.5e-9
tinsb = tinsf
Tch   = 11e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.1e-9
dy = 2.5e-9
Vref=0.0
PHIG=4.388
NBODY=5e24
Vds = 0.05
Ec = 2.5e7 #0.162742753246 *1e6/1e-2 #MV/cm to V/m
Pr = 22.3298216479 *1e-6/1e-4  #uC/cm^2 to C/m^2
alpha = -((3*np.sqrt(3))/2.0)*Ec/Pr
beta = ((3*np.sqrt(3))/2.0)*Ec/Pr**3
print (alpha,beta)
#alpha = -3e9 #;%m/F alpha=-1.61379679e-2
#beta = 6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
#tfe = 3e-9#;%m
Vref = 0.0
vgpoints=20
iterations_fe=3
rootfolderdata ='../results_juan4/'

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx

tfe_array = {3e-9,5e-9}

vth_sat =[]
vth_lin =[]
############################################## LINEAR NC-FET ##################
for tfe in tfe_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
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
    axes.set_xlim([0,0.2])

    Ith = 100
    value,indexvg = find_nearest(IDn,Ith)
    invt = np.log(IDn[indexvg+1]/IDn[indexvg-1])/(Vg[indexvg+1]-Vg[indexvg-1])
    I0 = IDn[indexvg+1]/(np.exp(Vg[indexvg+1]*invt))
    vth = np.log(Ith/I0)/invt
    vth_lin.append(vth)

    pylab.figure(5)
    pylab.plot( Vfe[1,:],Qg[1,:],'s')

    pylab.figure(6)
    pylab.plot( Vfe[-1,:],Qg[-1,:],'o')
    qfe = np.linspace(-0.2,0.2,40)
    pylab.figure(5)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')
    pylab.figure(6)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

pylab.figure(4)
pylab.plot(np.linspace(0.1, 1, num=len(Vfe[:,1])), Vfe,'-')



tfe=5e-9
qfe = np.linspace(-0.2,0.2,40)
pylab.figure(5)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

pylab.figure(6)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

############################################## SAT NC-FET ##################
Vds = 1.05
tfe_array = [3e-9,5e-9]#;%m

for tfe in tfe_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

    value,indexvg = find_nearest(IDn,Ith)
    invt = np.log(IDn[indexvg+1]/IDn[indexvg-1])/(Vg[indexvg+1]-Vg[indexvg-1])
    I0 = IDn[indexvg+1]/(np.exp(Vg[indexvg+1]*invt))
    vth = np.log(Ith/I0)/invt
    vth_sat.append(vth)
'''
pylab.figure(7)
print(vth_sat)
print(vth_lin)
dibl = [(x - y)/1 for x, y in zip(vth_lin,vth_sat)]#np.map(np.operator.sub, vth_sat, vth_lin)# np.array(vth_sat)-np.array(vth_lin)
print (tfe_array)
print (dibl)
pylab.plot(tfe_array, dibl,'o')
'''
pylab.figure(8)
pylab.plot(np.linspace(0.1, 1, num=len(Vfe[:,1])), Vfe,'-')

############################################## SAT FET ##################
PHIG=4.388
tinsf=0.8e-9
tfe_array = {0}

for tfe in tfe_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

    SS = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
    pylab.figure(3)
    pylab.plot(Vg[:-1], SS,'-')
    axes = pylab.gca()
    axes.set_ylim([50,100])
    axes.set_xlim([0,0.4])

    value,indexvg = find_nearest(IDn,Ith)
    invt = np.log(IDn[indexvg+1]/IDn[indexvg-1])/(Vg[indexvg+1]-Vg[indexvg-1])
    I0 = IDn[indexvg+1]/(np.exp(Vg[indexvg+1]*invt))
    vth = np.log(Ith/I0)/invt
    vth_sat.append(vth)

############################################## LINEAR FET ##################
Vds=0.05
tfe_array = {0}

for tfe in tfe_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

    value,indexvg = find_nearest(IDn,Ith)
    invt = np.log(IDn[indexvg+1]/IDn[indexvg-1])/(Vg[indexvg+1]-Vg[indexvg-1])
    I0 = IDn[indexvg+1]/(np.exp(Vg[indexvg+1]*invt))
    vth = np.log(Ith/I0)/invt
    vth_lin.append(vth)

tfe_array = [3e-9,5e-9,0]
print ('dibl final calculation')
pylab.figure(7)
print(vth_sat)
print(vth_lin)
dibl = [(x - y)*1000 for x, y in zip(vth_lin,vth_sat)]#np.map(np.operator.sub, vth_sat, vth_lin)# np.array(vth_sat)-np.array(vth_lin)
print (tfe_array)
print (dibl)
pylab.plot(tfe_array, dibl,'o')

x_label = ['VGS (V)','VGS (V)','VGS (V)','x/Lg (nm/nm)','VFE (V)','VFE (V)', 'TFE (nm)','x/Lg (nm/nm)']

y_label = ['IDS (A/m)','IDS (A/m)','SS (mV/dec)','VFE (V)','Q (C/m^2)','Q (C/m^2)', 'DIBL (mV/V)','VFE (V)']
###################
for i in range(1,9):
    print (i)
    pylab.figure(i)
    axes = pylab.gca()

    def axis_setting(axes,linewidth):
        for axis in ['top','bottom','left','right']:
          axes.spines[axis].set_linewidth(linewidth)

    axis_setting(axes,2)

    axes.set_xlabel(x_label[i-1], fontsize=20)
    axes.set_ylabel(y_label[i-1], fontsize=20)
    axes.tick_params(axis='both', which='major', labelsize=20)
    axes.tick_params(axis='both', which='minor', labelsize=20)
    pylab.savefig(str(i)+'.png', bbox_inches='tight')

pylab.show()
