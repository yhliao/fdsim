#!/usr/bin/ipython
from solver.poisson import p_solver1D

s = p_solver1D(1e-10)
m1 = s.add_mesh(N=10, pos=0 ,material='SiO2')
m2 = s.add_mesh(N=40, pos=10,material='Si')
m1.NB[:] = 1e17 * 1e6
m2.NB[:] = -1e18 * 1e6
c1 = s.add_contact(-1)
#c2 = s.add_contact(30)
c1.V = 0.5
#c2.V = -0.5
s.construct_profile()
s.visualize(['Ec','Ev'])

s.reset_EcBV()
s.solve_nlpoisson()


#s.write_mesh(['Ec','Ev','En','Ep'])
s.visualize(['Ec','Ev','Efn'])
s.visualize(['n'])

