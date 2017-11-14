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

'''
prefix='05V_25nm_'
np.savetxt(prefix+'Vg_data.txt', Vg, delimiter=',')
np.savetxt(prefix+'IDn_data.txt', IDn, delimiter=',')
np.savetxt(prefix+'Qg_data.txt', cg.D, delimiter=',')
'''
prefix='05V_25nm_'
Vg = np.loadtxt(prefix+'Vg_data.txt', delimiter=',')
IDn = np.loadtxt(prefix+'IDn_data.txt', delimiter=',')
Qg = np.loadtxt(prefix+'Qg_data_new.txt', delimiter=',')

pylab.figure(1)
pylab.plot(Vg, IDn)
pylab.figure(2)
pylab.plot(Vg, IDn)
pylab.yscale('log')
pylab.figure(3)
pylab.plot(np.linspace(0,1,int(np.round((Lg/dy)))), Qg)
pylab.figure(4)
pylab.plot(np.linspace(0,1,int(np.round((Lg/dy)))), VFE(Qg,alpha,beta,tfe))

'''
test = np.loadtxt('test_data.txt',  delimiter=',')

print (test[:,0])

arr = []

arr.append(Vg)
arr.append(IDn)
np.savetxt('test_data.txt',  np.transpose(arr), delimiter=',')
'''
pylab.show()
