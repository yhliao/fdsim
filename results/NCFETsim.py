#!/usr/bin/python
import argparse
from sys import argv
parser = argparse.ArgumentParser()
parser.add_argument('--Dit' ,default=0,type=float)
parser.add_argument('--tPZT',required=True,type=int)
parser.add_argument('--ND'  ,required=True,type=float)
config = parser.parse_args(argv[1:])

import xlrd, xlwt
import numpy as np
from matplotlib import pyplot as plt

N      = config.tPZT
doping = config.ND  * 1e6
Dit    = config.Dit * 1e4
filename  = "PZT_stack_QV({}_{}nm).xls".format(config.ND,N)
sheetname = "Dit{}".format(config.Dit)
picname   = "FIG/poly{}_FE{}nm_Dit{:.1E}.png"\
            .format(config.ND,N,config.Dit)

def FEplusMOS(Qstack,Vstack,QMOS,VMOS,ID) :
   UNDEF = 100
   Vstack = np.interp(QMOS,Qstack,Vstack,left=UNDEF, right=UNDEF)
   valid_idx = (Vstack!=UNDEF)
   QMOS   = QMOS[valid_idx]
   VNCFET = VMOS[valid_idx] + Vstack[valid_idx]
   ID     = ID[valid_idx]
   return QMOS, VNCFET, ID

def show():
   fig = plt.figure(figsize=(20,10))
   ### plot the Q-V relations
   plt.subplot(1,3,1)
   plt.plot(V_NCFET,Q_MOS, label='NCFET')
   plt.plot(VMOS,   QMOS,  label='MOS')
   plt.legend(loc='upper right')
   plt.title('Q-V relations')
   plt.xlabel('VG')
   plt.ylabel('QG')

   ### Plot the I-V relations
   plt.subplot(1,3,2)
   plt.semilogy(V_NCFET,I_D, label='NCFET')
   plt.semilogy(VMOS,   ID,  label='MOS')
   plt.legend(loc='upper right')
   plt.title('I-V relations')
   plt.xlabel('VG')
   plt.ylabel('ID')

   plt.subplot(1,3,3)
   plt.plot(V_NCFET,I_D, label='NCFET')
   plt.plot(VMOS,   ID,  label='MOS')
   plt.legend(loc='upper right')
   plt.title('I-V relations')
   plt.xlabel('VG')
   plt.ylabel('ID')
   #plt.show()
   fig.savefig(picname)

def calcSS(VG,ID):
   i1 = np.min(np.where(ID>1E-5))
   i2 = np.max(np.where(ID<1E-1))
   deltaV = VG[i2] - VG[i1]
   multI  = ID[i2] / ID[i1]
   SS = deltaV / np.log10(multI) * 1000
   print SS

try:
   wb = xlrd.open_workbook(filename)
   sFE  = wb.sheet_by_name(sheetname)
except:
   print "No existing {}.{} will simulate it first..."\
            .format(filename,sheetname)
   import os
   os.system('../PZTsim.py --ND {} --tPZT {} --Dit {}'
               .format(config.ND,config.tPZT,config.Dit))
   wb = xlrd.open_workbook(filename)
   sFE  = wb.sheet_by_name(sheetname)

wb = xlrd.open_workbook('MOS_IQV.xls')
sMOS = wb.sheet_by_name('mos')

Qstack = -np.array(sFE.col_values(0,start_rowx=1))
Vstack = -np.array(sFE.col_values(3,start_rowx=1))
assert len(Qstack) == len(Vstack)
QMOS   = np.array(sMOS.col_values(1,start_rowx=1))
VMOS   = np.array(sMOS.col_values(0,start_rowx=1))
ID     = np.array(sMOS.col_values(2,start_rowx=1))
assert len(QMOS)== len(VMOS) == len(ID)
Q_MOS, V_NCFET, I_D = FEplusMOS(Qstack,Vstack,QMOS,VMOS,ID)
calcSS(VMOS,ID)
calcSS(V_NCFET,I_D)
show()
