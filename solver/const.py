from __future__ import division
import numpy as np

### Useful constants #####
q   = 1.6e-19 # C
kBT = 8.6173324e-5 * 300 # eV
ep0 = 8.852e-12 # F/m

### Nc calculation ###
me  = 9.10938E-31 #kg
h_  = 1.05457E-34 #J*s
h   = 6.62607E-34 #J*s
pi  = 3.14159

class Si:
   #__slots__=['type','epr','Nc','Nv','Dn','Dp',\
   #           'taun','taup','meff','Eg','phiS']
   type = 'semiconductor'
   epr  = 11.7*ep0
   Nc   = 3.2e19*1e6
   Nv   = 1.8e19*1e6
   Dn   = .1430 * kBT
   Dp   = .0470 * kBT
   taun = 1e-8
   taup = 1e-8
   meff = me * 0.98
   Eg   = 1.12 # eV
   ni   = np.sqrt(Nc*Nv*np.exp(-Eg/kBT))
   phiS = 4.05
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      raise AttributeError("Edit const.py to change the parameters")

class Ge:
   #__slots__=['type','epr','Nc','Nv','Dn','Dp',\
   #           'taun','taup','meff','Eg','phiS']
   type = 'semiconductor'
   epr  = 16*ep0
   Nc   = 3.2e19*1e6
   Nv   = 1.8e19*1e6
   Dn   = .3900 * kBT
   Dp   = .1900 * kBT
   taun = 1e-5
   taup = 1e-5
   meff = me * 1.64
   Eg   = 0.66 # eV
   ni   = np.sqrt(Nc*Nv*np.exp(-Eg/kBT))
   phiS = 4.15
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      raise AttributeError( "Edit const.py to change the parameters")

class SiO2:
   #__slots__=['type','epr','Eg','meff','phiS']
   type = 'insulator'
   epr  = 3.9*ep0
   Eg   = 9 # eV
   meff = me * 0.3
   phiS = 0.95
   Dn   = 10e-8*.3900 * kBT #TODO: this is just a trick 10e-8*
   Dp   = 10e-8*.1900 * kBT #TODO: this is just a trick 10e-8*
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      print ("Edit const.py to change the parameters")
      raise AttributeError("?")

class PZT:
   #__slots__=['type','epr','Eg','meff','phiS']
   type = 'insulator'
   alpha = -4.89e7
   epr  = 1/(2*alpha)
   Eg   = 3# eV
   meff = me * 2
   phiS = 3.35
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      print ("Edit const.py to change the parameters")
      raise AttributeError ("?")

class PZTi:
   #__slots__=['type','epr','Nc','Nv','Dn','Dp','Eg','phiS']
   type = 'semiconductor'
   epr  = 100*ep0
   Nc   = 3.2e19*1e6
   Nv   = 1.8e19*1e6
   Dn   = .1430 * kBT
   Dp   = .470 * kBT
   Eg   = 3 # eV
   ni   = np.sqrt(Nc*Nv*np.exp(-Eg/kBT))
   phiS = 3.35
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      raise AttributeError("Edit const.py to change the parameters")

class VirtualOxide:
   #__slots__=['type','epr','Eg','phiS']
   type = 'insulator'
   epr  = 4*ep0
   Eg   = 4 # eV
   phiS = 3
   def __init__ (self):
      pass
   def __setattr__ (self,name,value):
      print ("Error!! all parameters of the material are constant")
      print ("Edit const.py to change the parameters")
      raise AttributeError

mdb = {"Si":Si(), "Ge":Ge(), "SiO2":SiO2(),\
      'PZT':PZT(), 'PZTi':PZTi(), 'base': VirtualOxide()}
