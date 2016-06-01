#!/usr/bin/python
import xlrd, xlwt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.rcParams['text.usetex']        = True
#rc('font',family='serif')

UNDEF = 1000
def calcSS(VG,ID):
   i1 = np.min(np.where(ID>1E-5))
   i2 = np.max(np.where(ID<1E-1))
   deltaV = VG[i2] - VG[i1]
   multI  = ID[i2] / ID[i1]
   SS = deltaV / np.log10(multI) * 1000
   return SS

def calc_C(V,Q,factor=1):
   CC = np.diff(Q)/np.diff(V) / factor
   QQ = (Q[1:] + Q[:-1]) / 2 /factor
   return QQ, CC

class NCFET_calc:
   def __init__(self):
      ######## read the MOSFET data #########
      wb = xlrd.open_workbook('DATA/MOS_IQV.xls')
      sMOS = wb.sheet_by_name('mos')
      self.QMOS   = np.array(sMOS.col_values(1,start_rowx=1)) 
      self.VMOS   = np.array(sMOS.col_values(0,start_rowx=1))
      self.IDMOS  = np.array(sMOS.col_values(2,start_rowx=1))
      assert len(self.QMOS)== len(self.VMOS) == len(self.IDMOS)

      ### plot the Q-C and ID-VG relations for MOS first ###
      self.fig = plt.figure(figsize=(12,6),tight_layout=True)
      self.ax1 = self.fig.add_subplot(1,2,1)
      self.ax2 = self.fig.add_subplot(1,2,2)

      Q1, C1 = calc_C(self.VMOS,self.QMOS)
      self.ax1.plot(Q1,C1,label=' C_MOS')
      self.ax2.semilogy(self.VMOS,self.IDMOS,
                        label='MOSFET, SS= {0:.2f}mV/dec'
                        .format(calcSS(self.VMOS,self.IDMOS)))

      ############ plot & subplot configurations ###########
      xt = [0,0.01,0.02,0.03,0.04]
      self.ax1.set_xticks(xt)
      self.ax1.set_ylim([0,0.1])
      self.ax1.set_xlabel('Q_{stack} (C/m^2)',fontsize=15)
      self.ax1.set_ylabel('C (F/m^2)',fontsize=15)
      self.ax1.set_xlim([0,0.04])
      self.ax1.grid()

      self.ax2.set_xlabel('V_G (V)',fontsize=15)
      self.ax2.set_ylabel('I_D (uA/um)',fontsize=15)
      self.ax2.set_ylim([1e-6,1e3])
      self.ax2.grid()

      self.i = 0

   def calc4stack(self,N,doping,Dit,factor=1):
      filename  = "DATA/PZT_stack_QV({}_{}nm).xls".format(doping,N)
      sheetname = "Dit{}".format(Dit)

      wb   = xlrd.open_workbook(filename)
      sFE  = wb.sheet_by_name(sheetname)
      Qstack = -np.array(sFE.col_values(0,start_rowx=1)) 
      Vstack = -np.array(sFE.col_values(3,start_rowx=1))
      assert len(Qstack) == len(Vstack)
      self.i += 1

      ### plot Q-C relation ###
      Q,C = calc_C(Vstack,Qstack)
      self.ax1.plot(Q,-C,'--',\
                    label="-C_stack{} ".format(self.i))
      ### ID-VG simulation ###
      QNCFET,VNCFET,INCFET = self.FEplusMOS(Qstack,Vstack,factor)
      SSNCFET = calcSS(VNCFET,INCFET)
      self.ax2.semilogy(VNCFET,INCFET,\
         label="NCFET"+str(self.i)+", SS= {0:.2f}mV/dec".format(SSNCFET))

   def FEplusMOS(self,Qstack,Vstack,factor) :
      Vstack = np.interp(self.QMOS*factor,\
                         Qstack,Vstack,left=UNDEF,right=UNDEF)
      valid_idx = (Vstack!=UNDEF)
      QMOS   = self.QMOS [valid_idx]
      VNCFET = self.VMOS [valid_idx] + Vstack[valid_idx]
      ID     = self.IDMOS[valid_idx]
      return QMOS, VNCFET, ID

   def show(self):
      self.ax1.legend(loc='upper left')
      self.ax2.legend(loc='lower right')
      plt.show()

   def save(self,figname):
      self.ax1.legend(loc='upper left')
      self.ax2.legend(loc='lower right')
      self.fig.savefig(figname)

######### plot ###############################
if __name__ == '__main__':
   N      = 700
   doping = 4e19 
   Dit    = 0e0
   cc = NCFET_calc()

   for i, factor in enumerate([1]):
      cc.calc4stack(N,doping,Dit,factor)

   cc.show()
   cc.save('test')
