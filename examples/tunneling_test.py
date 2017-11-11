#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver2D
import matplotlib.pyplot as plt
import numpy as np
s = dev_solver2D(1e-10,1e-10)
m1 = s.add_mesh([15,100],[0,0], 'Si')
m2 = s.add_mesh([15,10],[0,100],'SiO2')
m3 = s.add_mesh([15,100],[0,110],'Si')
c1 = s.add_contact([0,15],-1)
c2 = s.add_contact([0,15],210)

m1.NB[:] = 1e19 * 1e6
m3.NB[:] = 1e19 * 1e6


#c2.V = 0
c1.n = 1e19 * 1e6
c2.n = 1e19 * 1e6

c1.p = 1e1 * 1e6
c2.p = 1e1 * 1e6
s.construct_profile()
s.visualize(['Ec','Ev','Efn','Efp'])

step = 10
Vg = np.linspace(0,1,10)
It0 = []
It1 = []
for V in Vg:
   c1.V = V
   s.solve(1e-5,SRH=True,tunneling=True)
   It0.append(-c2.In)
   #s.solve(1e-5,SRH=True,tunneling=True)
   #s.reset_EcBV()
   #s.solve_nlpoisson()
   #s.write_mesh(['Ec','Ev','n','p'])
   #Jt1.append(-c2.Jn)
   #s.visualize(['Ec','Ev','Efn','Efp'])
   #s.visualize(['n','p'])

s.visualize(['Ec','Ev','Efn','Efp'])
plt.plot(Vg,It0)#,Vg,Jt1)
plt.show()
