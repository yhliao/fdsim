#!/usr/bin/ipython
#from solver.drift_diffusion import J_solver2D
from solver.dev_sim import dev_solver2D
import pickle

s = dev_solver2D(1e-9,1e-9)
m1 = s.add_mesh(N=[15,25], pos=[10,15],material='Ge')
m2 = s.add_mesh(N=[15,25], pos=[25,15],material='Si')
m3 = s.add_mesh(N=[30,15], pos=[10,0] ,material='Si')
m4 = s.add_mesh(N=[10,40], pos=[0,0]  ,material='Si')
m5 = s.add_mesh(N=[40, 5], pos=[0,40] ,material='Si')
c1 = s.add_contact(-1,[0,10])
c2 = s.add_contact(40,[0,10])
s.construct_profile()
s.visualize(['Ec','Ev','Efn','Efp'])
#s.meshes[0].NB[:] = 1e19 * 1e6
#m2.NB[:] = -1e19 * 1e6
#m3.NB[:] = -1e19 * 1e6
#m4.NB[:] = -1e19 * 1e6
#m5.NB[:] = -1e19 * 1e6


c1.p = 1e18 * 1e6
c2.p = 2e18 * 1e6
c1.V = 0.5
c2.V = -0.5




#s.reset_EcBV()
#s.solve_nlpoisson(1e-3)

s.solve()
#s.write_mesh(['Ec','Ev','n','p'])
s.visualize(['Ec','Ev'])
s.visualize(['n',])
