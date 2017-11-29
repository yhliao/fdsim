import numpy as np
import pylab

alpha = -0.03787026/2.0
beta = 0.0001519/4.0
tfe=1
Pr = np.sqrt(-alpha/beta) #uC/cm^2
Ec = -2/(3*np.sqrt(3))*alpha*Pr #MV/cm



print (Ec,Pr)

Ec = 0.162742753246
Pr = 22.3298216479

def VFE(qfe,alpha,beta,tfe):
   return 2*alpha*tfe*qfe+4*beta*tfe*qfe**3

def EFE(qfe,alpha,beta):
   return 2*alpha*qfe+4*beta*qfe**3

qfe = np.linspace(-40,40,40)
E_fe = EFE(qfe,alpha,beta)
pylab.figure(1)
pylab.plot(E_fe,qfe,'o')


axes = pylab.gca()
axes.set_xlim([-2,2])
axes.set_ylim([-40,40])


#for fe+tcad code
#Ec = 0.162742753246 *1e6/1e-2 #MV/cm to V/m
Ec = 3.8e7
Pr = 22.3298216479 *1e-6/1e-4  #uC/cm^2 to C/m^2
alpha = -((3*np.sqrt(3))/2.0)*Ec/Pr
beta = ((3*np.sqrt(3))/2.0)*Ec/Pr**3
print (Ec,Pr)
print ('alpha: %E' % alpha)
print ('beta: %E' % beta)
pylab.show()
