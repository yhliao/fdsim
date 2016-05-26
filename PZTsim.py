#!/usr/bin/python
from __future__ import division, print_function

import argparse
from sys import argv
parser = argparse.ArgumentParser()
parser.add_argument('--Dit' ,default=0,type=float)
parser.add_argument('--tPZT',required=True,type=int)
parser.add_argument('--ND'  ,required=True,type=float)
config = parser.parse_args(argv[1:])

import numpy as np
from xlutils.copy import copy
import xlwt, xlrd

from solver.const    import q, kBT
from solver.poisson import p_solver1D
from solver.util    import myDamper

N      = config.tPZT
doping = config.ND  * 1e6
Dit    = config.Dit * 1E4

base = 4
ti  = 5
tSi = 50
dx = 1e-9
total = base + N + 2*ti + tSi
filename = "PZT_stack_QV({}_{}nm).xls".format(config.ND,N)
try:
   rb = xlrd.open_workbook(filename)
   wb = copy(rb)
   print ("using existing ", filename)
except IOError:
   wb = xlwt.Workbook()
   print (filename, "not found, creating a new one")

ws = wb.add_sheet('Dit{}'.format(config.Dit))
#### constants for FE #######
phiB = 0.7 # buit-in barrier
#############################

##### initialize the simulation profile ###### 
s = p_solver1D(dx)
if base > 0:
   m0 = s.add_mesh(base,-base,'base')
m1 = s.add_mesh(ti ,0  ,'PZTi')
m2 = s.add_mesh(N  ,ti  ,'PZT')
m3 = s.add_mesh(ti ,ti+N,'PZTi')
m4 = s.add_mesh(tSi,2*ti+N,'Si')
m4.NB[:] = doping
Ec_log = np.empty(total)
Ev_log = np.empty(total)
cl = s.add_contact(-base-1)
cr = s.add_contact(N + 2*ti+tSi)
s.junc[3].Dit[1] = Dit
s.construct_profile()

#######################
cr.V = 0
Efr = cr.Ec - 1.12/2 + kBT * np.log(doping/m4.material.ni)
s.Efn[base+ti+N:] = Efr
s.Efp[base+ti+N:] = Efr

ws.write(0,0,'Q(C/m^2)')
ws.write(0,1,'Vfe')
ws.write(0,2,'Vpoly')
ws.write(0,3,'Vtotal')
row = 0
for V in np.linspace(0.2,-4,60):
   row = row+1
   cl.V = V
   s.reset_EcBV()
   s.solve_nlpoisson()
   s.Efn[0:ti+base]  = s.Ec[base]  - phiB
   s.Efp[0:ti+base]  = s.Ec[base]  - phiB
   Q     = (s.Ec[0] - cl.Ec)/dx * m0.material.epr
   Vfe   = -s.Ec[base] + s.Ec[-tSi-1]
   Vpoly = cr.Ec - s.Ec[-tSi]
   print ("Q= {} C/m^2".format(Q))
   Vtot  = Vfe + Vpoly
   print ("V= {} V".format(Vtot))
   print ("Vapp={}".format(V))
   ws.write(row,0,Q)
   ws.write(row,1,Vfe)
   ws.write(row,2,Vpoly)
   ws.write(row,3,Vtot)
   print (s.Qit[-tSi])
   Ec_log[:] = s.Ec
   Ev_log[:] = s.Ev
   #s.write_mesh(['Ec','Ev','Efn'])
   #s.visualize(['Ec','Ev','Efn'])
wb.save(filename)
