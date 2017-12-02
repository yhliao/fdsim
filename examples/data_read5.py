import numpy as np
import csv
import pickle
import pylab
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

device='ncfet'
NSD = 2e+26
#Lg  = 20e-9
tinsf = 0.5e-9
tinsb = tinsf
Tch   = 6e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.25e-9
dy = 1.0e-9
Vref=0.0
PHIG=4.388
NBODY=5e24
Vds = 0.05
Ec = 1.4*1e8#2.5e7 #0.162742753246 *1e6/1e-2 #MV/cm to V/m
Pr = 15*1e-6*1e4#22.3298216479 *1e-6/1e-4  #uC/cm^2 to C/m^2
alpha = -((3*np.sqrt(3))/2.0)*Ec/Pr
beta = ((3*np.sqrt(3))/2.0)*Ec/Pr**3
print (alpha,beta)
#alpha = -3e9 #;%m/F alpha=-1.61379679e-2
#beta = 6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
tfe = 3e-9#;%m
Vref = 0.0
vgpoints=20
iterations_fe=3
rootfolderdata ='../results_juan5/'

Ith=1000

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx



vth_sat =[]
vth_lin =[]
ss_finfet = []
ss_ncfinfet = []
ioff_finfet = []
ioff_ncfinfet = []
############################################## LINEAR NC-FET ##################
tfe = 0#;%m
Vds = 0.05
Lg_array =  [16e-9,14e-9,12e-9,10e-9]
for Lg in Lg_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    '''pylab.figure(1)
    pylab.plot(Vg, IDn,'-')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')'''

    vgii = np.linspace(0, 1, 751)
    ius = InterpolatedUnivariateSpline(Vg, IDn)
    idi = ius(vgii)
    vgmodel = InterpolatedUnivariateSpline(idi, vgii)
    vth_lin.append(vgmodel(Ith))

    '''pylab.figure(10)
    pylab.plot(Vg, IDn,'o')
    pylab.yscale('log')

    pylab.figure(10)
    pylab.plot(vgii, idi,'-')
    pylab.yscale('log')'''


    '''pylab.figure(5)
    pylab.plot( Vfe[1,:],Qg[1,:],'s')

    pylab.figure(6)
    pylab.plot( Vfe[-1,:],Qg[-1,:],'o')
    qfe = np.linspace(-0.2,0.2,40)
    pylab.figure(5)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')
    pylab.figure(6)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')'''
############################################## SAT NC-FET ##################
Vds = 1.05
Lg_array =  [16e-9,14e-9,12e-9,10e-9]
for Lg in Lg_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    #pylab.figure(1)
    #pylab.plot(Vg, IDn,'-')

    pylab.figure(1)
    pylab.plot(Vg, IDn,'-')
    pylab.yscale('log')

    '''value,indexvg = find_nearest(IDn,Ith)
    invt = np.log(IDn[indexvg+1]/IDn[indexvg-1])/(Vg[indexvg+1]-Vg[indexvg-1])
    I0 = IDn[indexvg+1]/(np.exp(Vg[indexvg+1]*invt))
    vth = np.log(Ith/I0)/invt'''
    vgii = np.linspace(0, 1, 751)
    ius = InterpolatedUnivariateSpline(Vg, IDn)
    idi = ius(vgii)
    vgmodel = InterpolatedUnivariateSpline(idi, vgii)
    vth_sat.append(vgmodel(Ith))

    SS = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
    pylab.figure(3)
    #pylab.plot(Lg, SS[1],'o',markersize=10)
    ss_finfet.append(SS[1])
    axes = pylab.gca()
    #axes.set_ylim([50,150])
    axes.set_xlim([9e-9,17e-9])

    ioff_finfet.append(IDn[0])
############################################### base line define without high-k
Vds = 0.05
tfe = 3e-9#;%m
############################################## LINEAR NC-FET ##################
Vds = 0.05
for Lg in Lg_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    '''pylab.figure(1)
    pylab.plot(Vg, IDn,'--')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')'''



    #Ith = 100
    vgii = np.linspace(0, 1, 751)
    ius = InterpolatedUnivariateSpline(Vg, IDn)
    idi = ius(vgii)
    vgmodel = InterpolatedUnivariateSpline(idi, vgii)
    vth_lin.append(vgmodel(Ith))

pylab.figure(4)
pylab.plot(np.linspace(0.1, 1, num=len(Vfe[:,1])), Vfe,'-')

qfe = np.linspace(-0.2,0.2,40)
pylab.figure(5)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

pylab.figure(6)
pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')

############################################## SAT NC-FET ##################
Vds = 1.05
Lg_array =  [16e-9,14e-9,12e-9,10e-9]
for Lg in Lg_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    #pylab.figure(1)
    #pylab.plot(Vg, IDn,'--')

    pylab.figure(2)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

    vgii = np.linspace(0, 1, 751)
    ius = InterpolatedUnivariateSpline(Vg, IDn)
    idi = ius(vgii)
    vgmodel = InterpolatedUnivariateSpline(idi, vgii)
    vth_sat.append(vgmodel(Ith))


    '''pylab.figure(5)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')
    pylab.figure(6)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')'''

    SS = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
    #pylab.figure(3)
    #pylab.plot(Lg, SS[1],'s',markersize=10)
    ss_ncfinfet.append(SS[1])
    axes = pylab.gca()
    #axes.set_ylim([50,100])

    ioff_ncfinfet.append(IDn[0])

pylab.figure(9)
pylab.plot(Lg_array, ioff_finfet,'o-',markersize=10)
pylab.plot(Lg_array, ioff_ncfinfet,'s-',markersize=10)
pylab.yscale('log')
axes = pylab.gca()
axes.set_xlim([9e-9,17e-9])

pylab.figure(3)
pylab.plot(Lg_array, ss_finfet,'o-',markersize=10)
pylab.plot(Lg_array, ss_ncfinfet,'s-',markersize=10)
axes = pylab.gca()
axes.set_xlim([9e-9,17e-9])

pylab.figure(5)
pylab.plot( Vfe[1,:],Qg[1,:],'s')

pylab.figure(6)
pylab.plot( Vfe[-1,:],Qg[-1,:],'o')
qfe = np.linspace(-0.2,0.2,40)

pylab.figure(8)
pylab.plot(np.linspace(0.1, 1, num=len(Vfe[:,1])), Vfe,'-')

tfe_array = [16e-9,14e-9,12e-9,10e-9,16e-9,14e-9,12e-9,10e-9]
print ('dibl final calculation')
pylab.figure(7)
print(vth_sat)
print(vth_lin)
dibl = [(x - y)*1000 for x, y in zip(vth_lin,vth_sat)]#np.map(np.operator.sub, vth_sat, vth_lin)# np.array(vth_sat)-np.array(vth_lin)
print (tfe_array)
print (dibl)
pylab.plot(tfe_array[:4], dibl[:4],'o-',markersize=10)
pylab.plot(tfe_array[4:], dibl[4:],'s-',markersize=10)

axes = pylab.gca()
axes.set_ylim([20,200])

pylab.figure(10)
pylab.plot(tfe_array[:4], vth_sat[:4],'o-',markersize=10)
pylab.plot(tfe_array[4:], vth_sat[4:],'s-',markersize=10)
################################
'../results_juan6/'
Vds = 1.05
Lg =  16e-9
Tch_array = [8e-9, 10e-9]
for Tch in Tch_array:
    prefix=rootfolderdata+device+'_Lg'+str(Lg)+'_tins'+str(tinsf)+'_Tch'+str(Tch)+'_tfe'+str(tfe)+'_NBODY'+str(NBODY)+'_Ec'+str(Ec)+'_Pr'+str(Pr)+'_PHIG'+str(PHIG)+'_VDS'+str(Vds)
    Vg = np.loadtxt(prefix+'_Vg_data.txt', delimiter=',')
    IDn = np.loadtxt(prefix+'_IDn_data.txt', delimiter=',')
    Qg = np.loadtxt(prefix+'_Qg_data_new.txt', delimiter=',')
    Vfe = np.loadtxt(prefix+'_Vfe_data_new.txt', delimiter=',')

    #pylab.figure(1)
    #pylab.plot(Vg, IDn,'--')

    pylab.figure(11)
    pylab.plot(Vg, IDn,'--')
    pylab.yscale('log')

    vgii = np.linspace(0, 1, 751)
    ius = InterpolatedUnivariateSpline(Vg, IDn)
    idi = ius(vgii)
    vgmodel = InterpolatedUnivariateSpline(idi, vgii)
    vth_sat.append(vgmodel(Ith))


    '''pylab.figure(5)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')
    pylab.figure(6)
    pylab.plot( VFE(qfe,alpha,beta,tfe), qfe,'-')'''

    SS = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
    #pylab.figure(3)
    #pylab.plot(Lg, SS[1],'s',markersize=10)
    ss_ncfinfet.append(SS[1])
    axes = pylab.gca()
    #axes.set_ylim([50,100])

    ioff_ncfinfet.append(IDn[0])

#################################
x_label = ['VGS (V)','VGS (V)','Lg (nm)','x/Lg (nm/nm)','VFE (V)','VFE (V)', 'LG (nm)','x/Lg (nm/nm)', 'LG (nm)', 'LG (nm)']

y_label = ['IDS (A/m)','IDS (A/m)','SS (mV/dec)','VFE (V)','Q (C/m^2)','Q (C/m^2)', 'DIBL (mV/V)','VFE (V)', 'IOFF (A/m)', 'Vth,sat (V)' ]
###################
for i in range(1,11):
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

    #pylab.savefig('ncfinfet'+str(i)+'.png', bbox_inches='tight')

pylab.show()
