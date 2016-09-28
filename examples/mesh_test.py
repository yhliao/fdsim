#!/usr/bin/ipython 
from solver._solver import solver2D

s = solver2D(1e-9,1e-9)
s.add_mesh([15,25], [10,15],'Ge')
s.add_mesh([15,25], [25,15],'Si')
s.add_mesh([30,15], [10,0] ,'Si')
s.add_mesh([10,40], [0,0]  ,'Si')
s.add_mesh([40, 5], [0,40] ,'Si')
s.meshes[0].NB[:] = 1e18 * 1e6
s.construct_profile()
s.visualize(['Ec','Ev'])

