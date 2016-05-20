from __future__ import division
import numpy as np
from solver.const    import q, kBT
from solver.poisson import p_solver1D
from solver.util    import myDamper
#### constants for FE #######
alpha = -4.89e7
phiB = 0.7 # buit-in barrier
#############################
N   = 100
ti  = 5
tSi = 300
dx = 1e-9
s = p_solver1D(dx)
m1 = s.add_mesh(ti ,0  ,'PZT')
m2 = s.add_mesh(N  ,ti  ,'PZT')
m3 = s.add_mesh(ti ,ti+N,'PZT')
m4 = s.add_mesh(tSi,2*ti+N,'Si')

Ec_log = np.empty(N+2*ti+tSi)
Ev_log = np.empty(N+2*ti+tSi)
cl = s.add_contact(-1)
cr = s.add_contact(N+2*ti+tSi)
s.construct_profile()
doping = 1e18 * 1e6
s.NB[2*ti+N:] = doping * q
########################
cl.V = -0
cr.V = 0
s.Efn[0:ti]  = cl.Ec  - phiB
s.Efn[ti+N:] = cr.Ec  - 1.12/2 + kBT * np.log(doping/m4.material.ni)
s.Efp[ti+N:] = cr.Ec  - 1.12/2 + kBT * np.log(doping/m4.material.ni)
s.reset_EcBV()
s.solve_nlpoisson()
Ec_log[:] = s.Ec
Ev_log[:] = s.Ev
s.visualize(['Ec','Ev','Efn'])
s.visualize(['p'])
err = 1
D = myDamper(1)
while err > 1e-3:
   VFE = s.Ec[ti+N-1] - s.Ec[ti]
   P   = VFE / (2*(N*dx)*alpha)
   #P =0
   #Qitl = Ditl*(ps.Ec[LN+ti]  - ETL - Ef[LN+ti] )
   #Qitr = Ditr*(ps.Ec[N-1-ti]- ETR - Ef[N-1-ti] )
   s.NB[ti]     = -P / dx
   s.NB[ti+N-1] = P / dx

   s.solve_nlpoisson()
   err = max(abs(Ec_log - s.Ec))
   dE = (s.Ec-Ec_log) * D(err) / err
   s.Ec = Ec_log + dE
   s.Ev = Ev_log + dE
   print "err=%.10f" %err
   #ps.show()
   Ec_log[:] = s.Ec
   Ev_log[:] = s.Ev
   s.visualize(['Ec','Ev','Efn','Efp'])

s.visualize(['Ec','Ev','Efn'])
s.visualize(['p'])

