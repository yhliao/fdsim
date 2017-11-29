Results here are for GF NC-FET (tinsf=0.5nm) and Reference device (tinsf=0.5+1.5nm/5=0.8nm):

device='ncfet'
NSD = 2e+26
Lg  = 25e-9
tinsf = 0.8e-9
tinsb = tinsf
Tch   = 11e-9
Ls = 20e-9
Ld = 20e-9
dx = 0.1e-9
dy = 2.5e-9
Vref=0.0
PHIG=4.188
NBODY=5e24
Vds = 1.05
Ec = 2.5e7 #0.162742753246 *1e6/1e-2 #MV/cm to V/m
Pr = 22.3298216479 *1e-6/1e-4  #uC/cm^2 to C/m^2
alpha = -((3*np.sqrt(3))/2.0)*Ec/Pr
beta = ((3*np.sqrt(3))/2.0)*Ec/Pr**3
print (alpha,beta)
