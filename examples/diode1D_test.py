#!/usr/bin/ipython
import sys
sys.path.append("../")
from solver.dev_sim import dev_solver1D

s = dev_solver1D(200e-9)
m1 = s.add_mesh(N=100, pos=0 ,material='Si')
m2 = s.add_mesh(N=100, pos=100,material='Si')
m1.NB[:] =  1e14 * 1e6
m2.NB[:] = -1e14 * 1e6
c1 = s.add_contact(-1)
c2 = s.add_contact(200)
c1.V = 0.4
c2.V = 0.
c1.n = 1e14 * 1e6
c1.p = 1e6  * 1e6
c2.n = 1e6  * 1e6
c2.p = 1e14 * 1e6

s.construct_profile()
s.solve(1e-5,SRH=True)

s.visualize(['Ec','Ev','Efn','Efp'])
s.visualize(['n','p'])
#s.solve_nlpoisson()
