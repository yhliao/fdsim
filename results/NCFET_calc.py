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

mk = ['d','o','s','^','v']
class NCFET_calc:
   def __init__(self):
      ######## read the MOSFET data #########
      wb = xlrd.open_workbook('DATA/MOS_IQV.xls')
      sMOS = wb.sheet_by_name('mos')
      self.QMOS   = np.array(sMOS.col_values(1,start_rowx=1)) 
      self.VMOS   = np.array(sMOS.col_values(0,start_rowx=1))
      self.IDMOS  = np.array(sMOS.col_values(2,start_rowx=1))
      assert len(self.QMOS)== len(self.VMOS) == len(self.IDMOS)

      self.i = 0
      ### plot the Q-C and ID-VG relations for MOS first ###
      self.fig = plt.figure(figsize=(12,6),tight_layout=True)
      self.ax1 = self.fig.add_subplot(1,2,1)
      self.ax2 = self.fig.add_subplot(1,2,2)

      self.ax1.set_title('(a)')
      self.ax2.set_title('(b)')

      Q1, C1 = calc_C(self.VMOS,self.QMOS)
      self.ax1.plot(Q1,C1,label='$C_{MOS}$',
                          linewidth=2)
      self.ax2.semilogy(self.VMOS,self.IDMOS,
                        linewidth=2,
                        label='$MOSFET,\ SS= {0:.2f}mV/dec$'
                        .format(calcSS(self.VMOS,self.IDMOS)))


      ############ plot & subplot configurations ###########
      xt = [0,0.01,0.02,0.03,0.04]
      self.ax1.set_xticks(xt)
      self.ax1.set_ylim([0,0.1])
      self.ax1.set_xlabel(r'$Charge,\ Q(C/m^2)$',fontsize=14)
      self.ax1.set_ylabel('$Capacitance,\ C(F/m^2)$'
                          ,fontsize=14)
      self.ax1.set_xlim([0,0.037])
      self.ax1.grid()

      self.ax2.set_xlabel('$Gate\ Voltage,\ V_G (V)$',fontsize=14)
      self.ax2.set_ylabel('$Drain\ Current,\ I_D(\mu A/\mu m)'\
                          '\ (log\ scale)$',
                          fontsize=14)
      self.ax2.set_xlim([-0.,1.6])
      self.ax2.set_ylim([1e-6,1e3])
      self.ax2.grid()

   def calc4stack(self,N,doping,Dit,factor=1,label="",Qf=0):
      filename  = "DATA/PZT_stack_QV({}_{}nm).xls".format(doping,N)
      sheetname = "Dit{}".format(Dit)

      nameC = "$-C_{stack" + str(self.i+1) + '}$'
      nameC += label
      nameI = "$NCFET_{stack" + str(self.i+1) + '},$'

      wb   = xlrd.open_workbook(filename)
      sFE  = wb.sheet_by_name(sheetname)
      Qstack = -np.array(sFE.col_values(0,start_rowx=1)) + Qf
      Vstack = -np.array(sFE.col_values(3,start_rowx=1))
      assert len(Qstack) == len(Vstack)

      ### plot Q-C relation ###
      Q,C = calc_C(Vstack,Qstack)
      self.ax1.plot(Q,-C,
                    marker=mk[self.i],
                    linewidth=1.5,
                    label=nameC)
      ### ID-VG simulation ###
      QNCFET,VNCFET,INCFET = self.FEplusMOS(Qstack,Vstack,factor)
      SSNCFET = calcSS(VNCFET,INCFET)
      nameI += "$\ SS={:.2f}mV/dec$".format(SSNCFET)
      self.ax2.semilogy(VNCFET,INCFET,
                        marker=mk[self.i],
                        linewidth=1.5,
                        label= nameI)

      self.i += 1
      return SSNCFET

   def FEplusMOS(self,Qstack,Vstack,factor) :
      Vstack = np.interp(self.QMOS*factor,\
                         Qstack,Vstack,left=UNDEF,right=UNDEF)
      valid_idx = (Vstack!=UNDEF)
      QMOS   = self.QMOS [valid_idx]
      VNCFET = self.VMOS [valid_idx] + Vstack[valid_idx]
      ID     = self.IDMOS[valid_idx]
      return QMOS, VNCFET, ID

   def show(self,IDscale='log'):
      self.ax2.set_yscale(IDscale)
      self.ax1.legend(loc='upper left',fontsize=10)
      self.ax2.legend(loc='upper right',fontsize=11)
      plt.show()

   def save(self,figname,IDscale='log'):
      self.ax2.set_yscale(IDscale)
      self.ax1.legend(loc='upper left',fontsize=10)
      self.ax2.legend(loc='upper right',fontsize=11)
      self.fig.savefig(figname)

class NCFET_calc2:
   def __init__(self):
      ######## read the MOSFET data #########
      wb = xlrd.open_workbook('DATA/MOS_IQV.xls')
      sMOS = wb.sheet_by_name('mos')
      self.QMOS   = np.array(sMOS.col_values(1,start_rowx=1)) 
      self.VMOS   = np.array(sMOS.col_values(0,start_rowx=1))
      self.IDMOS  = np.array(sMOS.col_values(2,start_rowx=1))
      assert len(self.QMOS)== len(self.VMOS) == len(self.IDMOS)

      self.i = 0
      ### plot the Q-C and ID-VG relations for MOS first ###
      self.fig = plt.figure(figsize=(15,5),tight_layout=True)
      self.ax1 = self.fig.add_subplot(1,3,1)
      self.ax2 = self.fig.add_subplot(1,3,2)
      self.ax3 = self.fig.add_subplot(1,3,3)

      self.ax1.set_title('(a)')
      self.ax2.set_title('(b)')
      self.ax3.set_title('(c)')

      Q1, C1 = calc_C(self.VMOS,self.QMOS)
      self.ax1.plot(Q1,C1,label='$C_{MOS}$',
                          linewidth=2)
      self.ax2.semilogy(self.VMOS,self.IDMOS,
                        linewidth=2,
                        label='$MOSFET,\ SS= {0:.2f}mV/dec$'
                        .format(calcSS(self.VMOS,self.IDMOS)))
      self.ax3.plot(self.VMOS,self.IDMOS,
                    linewidth=2,
                    label='$MOSFET$')
      ############ plot & subplot configurations ###########

      xt = [0,0.01,0.02,0.03,0.04]
      self.ax1.set_xticks(xt)
      self.ax1.set_ylim([0,0.1])
      self.ax1.set_xlabel(r'$Charge,\ Q(C/m^2)$',fontsize=14)
      self.ax1.set_ylabel('$Capacitance,\ C(F/m^2)$'
                          ,fontsize=14)
      self.ax1.set_xlim([-0.001,0.035])
      self.ax1.grid()

      self.ax2.set_xlabel('$Gate\ Voltage,\ V_G (V)$',fontsize=14)
      self.ax2.set_ylabel('$Drain\ Current,\ I_D(\mu A/\mu m)'\
                          '\ (log\ scale)$',
                          fontsize=14)
      self.ax2.set_xlim([-0.1,1.5])
      self.ax2.set_ylim([1e-6,1e3])
      self.ax2.grid()

      self.ax3.set_xlabel('$Gate\ Voltage,\ V_G(V)$',fontsize=14)
      self.ax3.set_ylabel('$Drain\ Current'\
                          '\ I_D(\mu A/\mu m)\ (linear)$'
                         ,fontsize=14)
      self.ax3.set_xlim([-0.1,1.5])
      self.ax3.set_ylim([0,1e3])
      self.ax3.grid()
   def calc4stack(self,N,doping,Dit,factor=1,label=""):
      filename  = "DATA/PZT_stack_QV({}_{}nm).xls".format(doping,N)
      sheetname = "Dit{}".format(Dit)

      nameC = "$-C_{stack" + str(self.i+1) + '}$'
      nameC += label
      nameI = "$NCFET_{stack" + str(self.i+1) + '}$'


      wb   = xlrd.open_workbook(filename)
      sFE  = wb.sheet_by_name(sheetname)
      Qstack = -np.array(sFE.col_values(0,start_rowx=1)) 
      Vstack = -np.array(sFE.col_values(3,start_rowx=1))
      assert len(Qstack) == len(Vstack)

      ### plot Q-C relation ###
      Q,C = calc_C(Vstack,Qstack)
      self.ax1.plot(Q,-C,
                    marker=mk[self.i],
                    linewidth=1.5,
                    label=nameC)
      ### ID-VG simulation ###
      QNCFET,VNCFET,INCFET = self.FEplusMOS(Qstack,Vstack,factor)
      SSNCFET = calcSS(VNCFET,INCFET)
      SSmsg = "$,\ SS={:.2f}mV/dec$".format(SSNCFET)
      self.ax2.semilogy(VNCFET,INCFET,
                        marker=mk[self.i],
                        linewidth=1.5,
                        label= nameI+SSmsg)

      self.ax3.plot(VNCFET,INCFET,
                    marker=mk[self.i],
                    linewidth=1.5,
                    label= nameI)
      self.i += 1
      return SSNCFET
   def FEplusMOS(self,Qstack,Vstack,factor) :
      Vstack = np.interp(self.QMOS*factor,\
                         Qstack,Vstack,left=UNDEF,right=UNDEF)
      valid_idx = (Vstack!=UNDEF)
      QMOS   = self.QMOS [valid_idx]
      VNCFET = self.VMOS [valid_idx] + Vstack[valid_idx]
      ID     = self.IDMOS[valid_idx]
      return QMOS, VNCFET, ID

   def show(self):
      self.ax1.legend(loc='upper left',fontsize=8.5)
      self.ax2.legend(loc='lower right',fontsize=8.5)
      self.ax3.legend(loc='upper right',fontsize=10)
      plt.show()

   def save(self,figname,IDscale='log'):
      self.ax1.legend(loc='upper left',fontsize=8.5)
      self.ax2.legend(loc='lower right',fontsize=8.5)
      self.ax3.legend(loc='upper right',fontsize=10)
      self.fig.savefig(figname)
######### plot ###############################
if __name__ == '__main__':
   N      = 1800
   doping = 1e19 
   Dit    = 0e0
   factor = 1
   cc = NCFET_calc()

   for (N,doping) in [(1300,1e19),(920,2e19),
                      (780,3e19),(700,4e19)]:
      cc.calc4stack(N,doping,Dit,factor)

   cc.show()
   cc.save('optimized')
