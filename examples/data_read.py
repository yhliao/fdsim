import numpy as np
import csv
import pickle
import pylab

NSD =1e20*1e6
Lg  = 25e-9
tinsf = 1e-9
tinsb = 1e-9
Tch   = 10e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.5e-9
dy = 1e-9

alpha = -3e9 #;%m/F alpha=-1.61379679e-2
beta =6.5e11 #;%6e11;%C^2m^5/F 6.52372352e-5*1e12
tfe=3e-9#;%m

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],idx

prefix='0005V_25nm_'
Vg0 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn0 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg0 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')

prefix='05V_25nm_'
Vg = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')

prefix='NC_05V_25nm_'
Vg2 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn2 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg2 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')


prefix='NC2_05V_25nm_'
Vg3 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn3 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg3 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')


prefix='NC3_05V_25nm_'
Vg4 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn4 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg4 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')


prefix='NC1_0005V_25nm_'
Vg5 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn5 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg5 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')

prefix='NC2_0005V_25nm_'
Vg6 = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn6 = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg6 = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')

pylab.figure(1)
pylab.plot(Vg, IDn0,'-')
pylab.plot(Vg, IDn,'o')
pylab.plot(Vg, IDn2, 's')
pylab.plot(Vg, IDn3, '<')
pylab.plot(Vg, IDn4, '*')
pylab.plot(Vg, IDn5, '-')
pylab.plot(Vg, IDn6, '-')
pylab.figure(2)
pylab.plot(Vg, IDn0,'-')
pylab.plot(Vg, IDn,'o')
pylab.plot(Vg, IDn2, 's')
pylab.plot(Vg, IDn3, '<')
pylab.plot(Vg, IDn4, '*')
pylab.plot(Vg, IDn5, '-')
pylab.plot(Vg, IDn6, '-')
pylab.yscale('log')


SSfinfet = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn))
SSncfinfet = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn4))

pylab.figure(3)
pylab.plot(Vg[:-1], SSfinfet,'o-')
pylab.plot(Vg[:-1], SSncfinfet,'s-')

SSfinfet = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn0))
SSncfinfet = 1e3*np.diff(Vg)/ np.diff(np.log10(IDn6))

pylab.figure(4)
pylab.plot(Vg[:-1], SSfinfet,'o-')
pylab.plot(Vg[:-1], SSncfinfet,'s-')

'''

pylab.figure(3)
pylab.plot(np.linspace(0,1,int(np.round((Lg/dy)))), Qg)
pylab.figure(4)
pylab.plot(np.linspace(0,1,int(np.round((Lg/dy)))), VFE(Qg,alpha,beta,tfe))


test = np.loadtxt('test_data.txt',  delimiter=',')

print (test[:,0])

arr = []

arr.append(Vg)
arr.append(IDn)
np.savetxt('test_data.txt',  np.transpose(arr), delimiter=',')
'''
value,indexvg = find_nearest(Vg,0.3)
print(indexvg)
print(Qg[:,6])



pylab.show()
