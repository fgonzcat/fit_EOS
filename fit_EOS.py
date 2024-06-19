#!/usr/bin/env python
"""
-----------------------------------------------------------------------------------------------|
 FITTING EQUATIONS OF STATE USING VINET / BIRCH-MURNHAGAN 3RD-ORDER                            |
The Birch-Murnhagan EOS of 3rd order is given by                                               |
                                                                                               |
$$    P (V)= \frac{3}{2} K_0 \left(\left(\frac{V_0}{V}\right)^{7/3} -                          |
                 \left(\frac{V_0}{V}\right)^{5/3}\right)                                       |
         \left(1 + \frac{3}{4}\left(K_0'-4\right)(\left(\frac{V_0}{V}\right)^{2/3}-1)\right)$$ |
                                                                                               |
and the Birch-Murnhagan EOS of 4th order is given by                                           |
                                                                                               |
$$    P (V)= \frac{3}{2} K_0 \left[\left(\frac{V_0}{V}\right)^{7/3}                            |
       -  \left(\frac{V_0}{V}\right)^{5/3}\right]                                              |
          \left[1 + \frac{3}{4}\left(K_0'-4\right)                                             |
          \left(\left(\frac{V_0}{V}\right)^{2/3}-1\right)                                      |
          + \frac{1}{24}\left(9{K_0'}^2-63K_0'+9K_0K_0''+143\right)                            |
          \left(\left(\frac{V_0}{V}\right)^{2/3}-1\right)^2\right],$$                          |
                                                                                               |
where $K_0\equiv K(V_0)$, $K_0'\equiv K'(P=0)$, and $K_0'' \equiv K''(P=0)$, with              |
$K'(P)\equiv \left(\frac{\partial K}{\partial P}\right)$ and                                   |
$K''(P)\equiv \left(\frac{\partial^2 K}{\partial P^2}\right)$.                                 |
Note that the derivatives are taken with respect to pressure, not volume.                      |
In fact, $K'(P)= -K'(V)V/K(V)$.                                                                |
                                                                                               |
The Vinet EOS is given by                                                                      |
                                                                                               |
$$P(V)= 3K_0 \left(\frac{1.0-\eta}{\eta^2}\right)                                              |
         x e^{ \frac{3}{2}(K_0'-1)(1-\eta) };\qquad \eta=(V/V_0)^{1/3}$$                       |
                                                                                               |
I provide a new log-log polynomial EOS fit:                                                    |
         lnV = a + b*lnP + c*lnP^2 + d*lnP^3  <==>  V(P) = exp(a)*P^{b+c*lnP+d*(lnP)^2}        |
                                                                                               |
                                                                                               |
Reported Errors:                                                                               |
   residuals = P_fit(V)-P                                                                      |
   sigma     = std(residuals)                                                                  |
   RMSE      = sqrt(mean(residuals)^2)                                                         |
   R2        =  1 - sum(residuals^2)                                                           |
   chi_squared = sum(residuals**2 / dP**2)                                                     |
       chi^2 < 1 ---> Line passes more than 2/3 of 1 sigma / sigma too big / Overfit           |
       chi^2 > 1 ---> Line misses more than 1/3 of 1 sigma / sigma too small / inadeq. model   |
       chi^2 = 1 ---> Line passes through 2/3 of 1 sigma & 95% of 2 sigma                      |
                      Model with chi^2 closest to 1, wins.                                     |
                                                                                               |
Felipe Gonzalez                                                          Berkeley, 05/19/2023  |
-----------------------------------------------------------------------------------------------|
Last modified on:                                                                  05/18/2024
"""
from pylab import *
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import sys,os, subprocess
import random



filename = "/home/fgonzalez/EOS_Fe_sol_6000K.dat"  # Example, must be provided as argument
colV = 5  # Default column number for V[A^3]
colP = 11  # Default column number for P[GPa]
colPE = 12  # Default column number for P_error
BM_deg = 3   # Birch-Murnaghan of 3rd order by default
PTarget=-1   # Ptarget to guess the volume
P1 =    -1   # To calculate integral V(P)dP from P1 to PTarget
print_table = False
deleting_points_test = False
show_plots = True 
V0_as_param = False
fbv_path = os.path.expanduser("~/scripts/fbv")
try:
 subprocess.run([fbv_path],check=True)
 fbv_exists = True
except:
 fbv_exists = False




'''
WELCOME MESSAGE
'''
print("\n**** FITTING EOS ****")
print("by Felipe Gonzalez [Berkeley 05-19-2023]\n")
if len(sys.argv) == 1:
 message = """
 
 This code fits an isotherm using Birch-Murnhagan, Vinet, and log-log [Berkeley 05-19-23]
 
 Usage: {0}  EOS_Fe_sol_6000K.dat
 Usage: {0}  EOS_Fe_sol_6000K.dat   V[A^3]-col P[GPa]-col P_error-col
 Usage: {0}  EOS_Fe_sol_6000K.dat       6         12          13
 Usage: {0}  EOS_H2O_liq_7000K.dat      6         12          13  --BM4 --V0-as-param
 Usage: {0}   ... -p              --> print V(P) 
 Usage: {0}   ... --test          --> deleting-points performance test 
 Usage: {0}   ... --noplots       --> don't plot 
 Usage: {0}   ... --BM2           --> Birch-Murnaghan 2th order
 Usage: {0}   ... --BM3           --> Birch-Murnaghan 3th order (default)
 Usage: {0}   ... --BM4           --> Birch-Murnaghan 4th order
 Usage: {0}   ... --V0-as-param   --> Treat V0 as another fitting parameter [ do not force the default P(V0) = P0, where P0=min(P), V0=max(P) ]

 No arguments assumes V[A^3]-col= 6, P[GPa]-col= 12,  P_error-col= 13

 Reported Errors:
   residuals = P_fit(V)-P
   sigma     = std(residuals) 
   RMSE      = sqrt(mean(residuals)^2)
   R2        =  1 - sum(residuals^2)
   chi_squared = sum(residuals**2 / dP**2)
       chi^2 < 1 ---> Line passes more than 2/3 of 1 sigma / sigma too big / Overfit
       chi^2 > 1 ---> Line misses more than 1/3 of 1 sigma / sigma too small / inadeq. model
       chi^2 = 1 ---> Line passes through 2/3 of 1 sigma & 95% of 2 sigma 
                      Model with chi^2 closest to 1, wins. 
 
 """.format(sys.argv[0])
 
 print(message)
 exit()
else:
 filename = sys.argv[1] 
 if not os.path.isfile(filename):
  print("Error: File '{}' does not exist.".format(filename))
  exit()

 if '-p' in sys.argv:
  print_table = True
  idx = sys.argv.index('-p')
  sys.argv.pop(idx)
 if '--test' in sys.argv:
  idx = sys.argv.index('--test')
  sys.argv.pop(idx)
  deleting_points_test = True
 if '--noplots' in sys.argv:
  idx = sys.argv.index('--noplots')
  sys.argv.pop(idx)
  show_plots = False 
 if '--V0-as-param' in sys.argv:
  idx = sys.argv.index('--V0-as-param')
  sys.argv.pop(idx)
  V0_as_param = True
 if '--BM2' in sys.argv:
  idx = sys.argv.index('--BM2')
  sys.argv.pop(idx)
  BM_deg = 2
 if '--BM3' in sys.argv:
  idx = sys.argv.index('--BM3')
  sys.argv.pop(idx)
  BM_deg = 3
 if '--BM4' in sys.argv:
  idx = sys.argv.index('--BM4')
  sys.argv.pop(idx)
  BM_deg = 4
 if '--P1' in sys.argv:
  idx = sys.argv.index('--P1')
  sys.argv.pop(idx)
  P1 = float(sys.argv.pop(idx))


 try:
  PTarget=float(sys.argv[5])
 except:
  pass



 if len(sys.argv)==2: pass
 elif len(sys.argv) >= 4 and all(arg.isdigit() for arg in sys.argv[2:5]):
  colV,colP,colPE = [ int(c)-1 for c in sys.argv[2:5] ]
 else:
  print ("Your input: " , sys.argv)
  print("You need to specify the column numbers: V[A^3]-col P[GPa]-col P_error-col")
  exit()




#------------------------#
# SETTING FIGURE PARAMS  #
#------------------------#
GPaA3_to_eV=0.0062415091
Ha_to_eV = 27.211386
Ha_to_meV = Ha_to_eV * 1000
fig_size = [700/72.27 ,520/72.27]
#fig_size = [350/72.27 ,250/72.27]
params = {'axes.labelsize': 22, 'legend.fontsize': 16,
          'xtick.labelsize': 22, 'ytick.labelsize': 22,
          'xtick.major.size': 14,'ytick.major.size': 14,
          'xtick.minor.size': 7,'ytick.minor.size': 7,
          'xtick.direction': 'in', 'ytick.direction': 'in',
          'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
          'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
          'text.usetex': False, 'figure.figsize': fig_size, 'axes.linewidth': 2,
          'xtick.major.pad': 5,
          'ytick.major.pad': 10,
          'figure.subplot.bottom': 0.110,'figure.subplot.top': 0.975,'figure.subplot.left': 0.160,'figure.subplot.right': 0.960}


kB = 0.000086173303372 # eV/K
rcParams.update(params)



#-----------------------------------#
# FUNCTIONAL FORMS OF FIT FUNCTIONS #
#-----------------------------------#
def P_V_BM2(V, V0,K0):
    # 2nd order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    P = 1.5*K0 * (f**7 - f**5) 
    return P

def P_V_BM3(V, V0,K0,K0p):
    # 3rd order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    P = 1.5*K0 * (f**7 - f**5) * (1 + 0.75*(K0p-4)*(f*f - 1))
    return P

def P_V_BM4(V, V0,K0,K0p,K0pp):
    # 4th order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    P = 1.5*K0 * (f**7 - f**5) * (1 + 0.75*(K0p-4)*(f*f - 1)  + (1.0/24)*(9*K0p*K0p - 63*K0p + 9*K0*K0pp + 143) *(f*f - 1)*(f*f - 1) )
    return P


def E_V_BM3(V, V0, K0, K0p, E0):
    f = V0/V
    E = E0 + (9*V0*K0/16)*0.006241509125883 * ( (f**(2.0/3.0)-1)**3 * K0p + (f**(2.0/3.0)-1)**2 * (6-4*(f**(2.0/3.0))))
    return E

def VinetPressure(V, V0,K0,K0p):
  #if V<0: return 0
  x  = (V/V0)**(1.0/3.0)
  xi = 1.5*(K0p-1.0);
  P  = 3.0*K0 * (1.0-x)/x/x * np.exp( xi*(1.0-x) );
  return P

if   BM_deg==2: P_V_BM = lambda V, V0,K0:          P_V_BM2(V, V0,K0)
elif BM_deg==3: P_V_BM = lambda V, V0,K0,K0p:      P_V_BM3(V, V0,K0,K0p)
elif BM_deg==4: P_V_BM = lambda V, V0,K0,K0p,K0pp: P_V_BM4(V, V0,K0,K0p,K0pp)



#filename = "EOS_MgSiO3FeO_"+str(T0)+"K.dat"
#print ("These are the cols",colV,colP,colPE)
#if colV*colP*colPE == 0:
# colP = 12 -1
# colPE= 13 -1
# colV = 6  -1

try:
 data = np.loadtxt(filename, usecols=(3,colV,7,9,colP,colPE,14,15), dtype=float, comments='#')  # N, V, rho, T, P, PE, E, EE
 data  = data[data[:,1].argsort()][::-1]  # Sort by increasing volumes and then invert
 N = data[0,0]
 V   = data[:,1]
 rho = data[:,2]
 T   = data[0][3]
 P   = data[:,4]
 dP  = data[:,5]
 E   = data[:,6] * Ha_to_eV
 dE  = data[:,7] * Ha_to_eV
except:
 #V,P,dP  = np.loadtxt(filename, usecols=(colV,colP,colPE), dtype=float, comments='#', unpack=True)  # x, y, yerr 
 data    = np.loadtxt(filename, usecols=(colV,colP,colPE), dtype=float, comments='#',unpack=True)  # x, y, yerr 
 sorted_indices = data[0].argsort()[::-1]    # Sort by increasing volumes and then invert
 data = data[:, sorted_indices]
 positive = data[1] >= 0.0  # (P>=0)
 V,P,dP  = [ arr[positive] for arr in data] 
 #dP = 3*dP


try:
 T0=int(T)
except:
 T0=0

# PRINTING THE ORIGINAL DATA
print ("\n#EOS Data: " + filename)
for j in range(len(V)):
 print("i= %2i  V[A^3]= %9.4f  P[GPa]= %9.4f %6.4f" % (j, V[j], P[j], dP[j]) ) 


#------------------------#
#       LOG-LOG          #
#------------------------#
# FITTING A POLYNOMIAL IN LOG-LOG SPACE
P_residual = 10.0 + abs(min(P))        # Shift P by 10 upwards to prevent P=0.0 generating problems
V_residual = 10.0 + abs(min(V))        # Shift V in case V does not represent volumes and may be negative 
log_P = np.log(P + P_residual)
log_V = np.log(V + V_residual)
loglog_fit = lambda x: np.poly1d(np.polyfit(log_P, log_V,3))(x)  # lnV = a + b*lnP + c*lnP^2 + d*lnP^3  <==>  V(P) = exp(a)*P^{b+c*lnP+d*(lnP)^2}
pp = np.linspace(min(P),max(P))
#V_loglogfit = lambda p: np.exp(loglog_fit(np.log(p+P_residual))) - V_residual          #V(P)
V_loglogfit = InterpolatedUnivariateSpline(pp, np.exp(loglog_fit(np.log(pp+P_residual))) - V_residual )         #V(P)
try:
 P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp[::-1]), pp[::-1]) #P(V)
except:
 P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp), pp) #P(V)

#------------------------#
#       SPLINE           #
#------------------------#
#spl_V = InterpolatedUnivariateSpline(P[sorted_indices],V[sorted_indices])  # V(P)
sorted_indices = P.argsort()
try:
 P_spline = InterpolatedUnivariateSpline(V[::-1],P[::-1])  # P(V)
except: 
 P_spline = InterpolatedUnivariateSpline(V,P)  # P(V)
V_spline = InterpolatedUnivariateSpline(P[sorted_indices],V[sorted_indices])  # V(P)
#dP_spline = InterpolatedUnivariateSpline(P[sorted_indices], dP[sorted_indices])



#------------------------#
#    BIRCH MURNAGHAN     #
#------------------------#
'''
Forcing P(V0) = P0 = min(P). V0 is not a parameter
'''
k0   = -V[0]*(P[-1]-P[0])/(V[-1]-V[0])  #max(P)/10
k0p  = 4
k0pp = -(9*k0p*k0p -63*k0p + 143)/(9*k0)
## Only K0,K0p as parameters. Forcing P(V0)=P0 = min(P) with V0 = max(V)
if   BM_deg==2:
 initial_guess = (k0)  # Initial guess of parameters K0, K0p
 p_BM = lambda v,K0: min(P) + P_V_BM2(v, max(V),K0)
elif BM_deg==3:
 initial_guess = (k0, k0p)  # Initial guess of parameters K0, K0p
 p_BM = lambda v,K0,K0p: min(P) + P_V_BM3(v, max(V),K0,K0p)
elif BM_deg==4:
 initial_guess = (k0, k0p, k0pp)  # Initial guess of parameters K0, K0p, K0pp
 p_BM = lambda v,K0,K0p,K0pp: min(P) + P_V_BM4(v, max(V),K0,K0p,K0pp)


#popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess) #, maxfev=10000)
popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess , sigma=dP, absolute_sigma=True) #, maxfev=10000)
Perr_BM = np.sqrt(np.diag(pcov_BM))
print ("Birch-Murnaghan of degree",BM_deg,"\n")
if   BM_deg==2:
 print ( "BM fit:       V0[A^3]= %9.4f            K0[GPa]= %9.4f %7.4f  %s"  % ( max(V), popt_BM[0], Perr_BM[0], "                            # Forcing P(V0)=P0 = min(P)" ) )
elif BM_deg==3:
 print ( "BM fit:       V0[A^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( max(V), popt_BM[0], Perr_BM[0], popt_BM[1], Perr_BM[1], "  # Forcing P(V0)=P0 = min(P)" ) )
elif BM_deg==4:
 print ( "BM fit:       V0[A^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f  K0pp[1/GPa]= %7.4f %4.4f%s"  % ( max(V), popt_BM[0], Perr_BM[0], popt_BM[1], Perr_BM[1],popt_BM[2], Perr_BM[2], "  # Forcing P(V0)=P0 = min(P)" ) )
#print("COVARIANT BM:",pcov_BM)


'''
## V0,K0,K0p as parameters.
'''
k0   = -V[0]*(P[-1]-P[0])/(V[-1]-V[0])  #max(P)/10
k0p  = 4
k0pp = -(9*k0p*k0p -63*k0p + 143)/(9*k0)
if   BM_deg==2: initial_guess = (max(V),k0)            # Initial guess of parameters V0, K0
elif BM_deg==3: initial_guess = (max(V),k0, k0p)       # Initial guess of parameters V0, K0, K0p
elif BM_deg==4: initial_guess = (max(V),k0, k0p,k0pp)  # Initial guess of parameters V0, K0, K0p, K0pp

BM_bounds = [2*max(V),2*k0]
lower_BM_bounds = [0,0]
if   BM_deg==3:
 BM_bounds += [3*k0p]
 lower_BM_bounds = [0,0,0]
elif BM_deg==4:
 BM_bounds += [3*k0p,20]
 lower_BM_bounds = [0,0,0,-20]
#print("Lower Bounds", lower_BM_bounds)
#print("Upper Bounds", BM_bounds)
try:
 npopt_BM, npcov_BM= curve_fit(P_V_BM, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True, bounds=(lower_BM_bounds,BM_bounds) , maxfev=10000)
except:
 npopt_BM, npcov_BM= curve_fit(P_V_BM, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True,                                      maxfev=10000)
Perr_BM = np.sqrt(np.diag(npcov_BM))
if   BM_deg==2:
 print ( "BM fit:       V0[A^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  %s"  % ( npopt_BM[0], Perr_BM[0], npopt_BM[1], Perr_BM[1], "                            # V0 as param" ) )
elif BM_deg==3:
 print ( "BM fit:       V0[A^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( npopt_BM[0], Perr_BM[0], npopt_BM[1], Perr_BM[1], npopt_BM[2], Perr_BM[2], "  # V0 as param" ) )
elif BM_deg==4:
 print ( "BM fit:       V0[A^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f  K0pp[1/GPa]= %7.4f %4.4f%s"  % ( npopt_BM[0], Perr_BM[0], npopt_BM[1], Perr_BM[1], npopt_BM[2], Perr_BM[2], npopt_BM[3], Perr_BM[3], "  # V0 as param" ) )


#------------------------#
#         VINET          #
#------------------------#
initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, max(V),K0,K0p)
popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
Perr_Vinet = np.sqrt(np.diag(pcov_Vinet))
print ( "Vinet fit:    V0[A^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( max(V), popt_Vinet[0], Perr_Vinet[0], popt_Vinet[1], Perr_Vinet[1], "  # Forcing P(V0)=P0 = min(P)" ) )

initial_guess = (max(V),k0, k0p)  # Initial guess of parameters K0, K0p
try:
 npopt_Vinet, npcov_Vinet = curve_fit(VinetPressure, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True, bounds=(0,[3*max(V),2*k0,100]) ) #, maxfev=1000000 )
except:
 npopt_Vinet, npcov_Vinet = curve_fit(VinetPressure, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True,                                     maxfev=1000000 )
Perr_Vinet = np.sqrt(np.diag(npcov_Vinet))
print ( "Vinet fit:    V0[A^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( npopt_Vinet[0], Perr_Vinet[0], npopt_Vinet[1], Perr_Vinet[1], npopt_Vinet[2], Perr_Vinet[2], "  # V0 as param" ) )


# PLOTTING THE FIT WHERE P( max(V)=V0 ) = min(P) = P0
if V0_as_param:
 P_BM = lambda v:  P_V_BM(v, *npopt_BM)
 P_Vinet = lambda v: VinetPressure(v, *npopt_Vinet)
else:
 P_BM = lambda v: min(P) + P_V_BM(v, max(V), *popt_BM)
 P_Vinet = lambda v: p_Vinet(v, *popt_Vinet)


#------------------------#
#       PLOTTING         #
#------------------------#
fig = figure(1)   
ax = subplot(111)
#ax.errorbar(data[:,1], data[:,4], data[:,5], ls='', color='b', marker='s', ms=15, capsize=10,mfc='None', mec='b', mew=2, label=r'$P(V)$' + filename) ## all data
#ax.errorbar(V, P, dP, marker='o', ls='',c='r',ms=10, capsize=10, mfc='pink', mec='r', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0)+' (filtered data)')  ## filtered data
ax.errorbar(V, P, dP, marker='o', ls='',c='b',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0))
#ax.errorbar(V, P, dP,  marker='o', ls='-',c='b',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0))  ## filtered data
#ax.errorbar(V, E, dE,  marker='o', ls='-',c='b',ms=10, capsize=10, mfc='lightblue', mec='blue', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0))  ## filtered data


#vs = linspace(0.7*min(V), 1.1*max(V), 500)
dv = (max(V)-min(V))/10
vs = linspace(min(V)-2*dv, max(V)+dv, 500)
ps = P_BM(vs) #[P_BM(v) for v in vs]
ax.plot(vs, ps,'r-', lw=4,label='$P(V)$ BM fit')
ax.plot(vs, P_Vinet(vs),'--', c='limegreen',lw=3,label='$P(V)$ Vinet fit')
ax.plot(vs, P_spline(vs),'--',dashes=[10,3], c='darkblue', lw=2, label='$P(V)$ spline fit')
ax.plot(vs, P_loglogfit(vs),'k',dashes=[5,1,1,1],lw=2,label=r'Log-Log polyfit ($\ln V=a + b*\ln P + c*\ln^2 P + d*\ln^3 P$)')
if fbv_exists:
 p_list =linspace(1.1*min(P),0.9*max(P),100)
 vols_fbv = [  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True ))  for p in p_list ]
 ax.plot(vols_fbv, p_list , '-', dashes=[5,2,2,2], c='purple', lw=2, label='fbv')
ax.set_xlabel("Volume ($\AA^3$)")
ax.set_ylabel("Pressure (GPa)")
ax.set_xlim(min(V)-2*dv,max(V)+2*dv)
ax.set_ylim(0.9*min(P),1.5*max(P))

legend()
#ax.set_xscale('log')
#ax.set_yscale('log')
#savefig('PV.png')




#----------------------------------------------#
# Difference between the fit and the points    #
#----------------------------------------------#
fig2 = figure(2)
ax2 = subplot(111)
ps=np.array(ps)
plot(vs,ps-ps,'k--' ,label='$P$ Data')

ax2.errorbar(V, (P_BM(V)-P), 0*dP, color='red', marker='s', ms=12, capsize=10, mfc='pink', mec='red', mew=2, label=r'$P_{\rm BM}-P$')
ax2.errorbar(V, (P_Vinet(V)-P), 0*dP, color='limegreen', marker='v', ms=10, capsize=10, mfc='None', mec='limegreen', mew=2,label=r'$P_{\rm Vinet}-P$')
ax2.errorbar(V, (P_loglogfit(V)-P), 0*dP, color='k', lw=1, marker='d', ms=12, capsize=6, mfc='yellow', mec='k', mew=2, label=r'$P_{\rm log-log}-P$')
ax2.errorbar(V, P-P, dP, marker='o', ls='', c='blue', ms=14, capsize=10, mfc='lightblue', mec='blue', mew=2, lw=2, zorder=-1, label=r'$P(V)$ ' + filename)

ax2.set_ylabel(r"$P_{\rm fit}-P_{\rm data}$ (GPa)")

# COMMENT THE BLOCK ABOVE AN UNCOMMENT THE ONE BELOW TO PLOT IN TERMS OF PV
#ax2.errorbar(V, (fPV(V)-P)*V*GPaA3_to_eV, dP*V*GPaA3_to_eV, color='b', marker='s', c='b',ms=10, capsize=10, mfc='None', mec='b', mew=2,label=r'$P-P_{\rm BM}$')
#ax2.errorbar(V, (P_Vinet(V)-P)*V*GPaA3_to_eV, dP*V*GPaA3_to_eV, color='green', marker='d', c='green',ms=10, capsize=10, mfc='None', mec='green', mew=2,label=r'$P-P_{\rm Vinet}$')
#ax2.plot(V,(P-fPV(V))*V*GPaA3_to_eV,'ro', mfc='w',ms=10,mew=2,label='$P(V)$ at T='+str(T0)+'K')
#ax2.set_xlim(600,1050)
#ax2.set_ylabel("$PV$ (eV)")

### REPORT ###
predictors = ['BM',  'Vinet',  'loglog' ]
P_fit = { 'BM': P_BM, 'Vinet': P_Vinet, 'loglog': P_loglogfit}
Nparams = { 'BM': len(popt_BM),  'Vinet': len(popt_Vinet), 'loglog': 4 }
residuals = { predictor : P_fit[predictor](V)-P for predictor in predictors }
sigma     = { predictor : std(residuals[predictor]) for predictor in predictors }
RMSE      = { predictor : sqrt(mean(residuals[predictor]**2)) for predictor in predictors }
R2        = { predictor : 1 - sum(residuals[predictor]**2) / sum((P-mean(P))**2) for predictor in predictors }
chi_squared = {predictor:  sum( residuals[predictor]**2 / dP**2)/(len(dP)-Nparams[predictor])  for predictor in predictors}
#for predictor in predictors:
# print ("RESIDUALS",predictor,":", residuals[predictor])
# print ("RESIDUALS^2",predictor,":", residuals[predictor]**2)
# print ("SIGMA      ",predictor,":", sigma[predictor])
# print ("error bars dP :", dP   )
# print ("error bars dP2:", dP**2)
# print ("My X2=", sum(      residuals[predictor]**2 / dP**2)/(len(dP)-Nparams[predictor]))
# ## P-values using the chi-squared distribution
# #degrees_of_freedom = len(predictors) - 1
# #p_values = {predictor: chi2.sf(chi_squared[predictor], degrees_of_freedom) for predictor in predictors}
sorted_predictors = sorted(predictors, key=lambda p: R2[p])

print("\nRoot Mean Square Error of each fit:")
for p in sorted_predictors:
 print("FIT: %-9s  RMSE_P[GPa]= %9.6f  std(residuals)= %9.6f  R2=  %10.8f  chi^2=  %10.8f" % (p, RMSE[p], sigma[p], R2[p], chi_squared[p]) )







#----------------------------------------------#
#  PRINT INTERPOLATED TABLE AND                #
#  plot the fbv, spline BM & Vinet curves      #
#----------------------------------------------#
if print_table:
 print("\n# PRINTING TABLE OF INTERPOLATED VOLUMES")
 spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs[::-1]), vs[::-1])  # V(P)
 spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs[::-1]), vs[::-1])  # V(P)
 #p_list = arange(10,500,10)
 #p_list = arange(1e-10,201,1)
 p_list = linspace(1e-10,max(ps),40)
 print("# Plotting the predicted volumes for 20 pressures equally spread on the interval:")
 ax.plot(spl_V_BM(p_list), p_list , 'o', mfc='None', mec='r', ms=8, mew=2, label=r'$V_{\rm BM}(P)$ inv')
 ax.plot(spl_V_Vinet(p_list), p_list , 's', mfc='None', mec='limegreen', ms=8, mew=2, label='Vinet $V(P)$ inv')
 # Weighted fit by Burkhard fbv
 try:
  vols_fbv = [  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True ))  for p in p_list ]
  ax.plot(vols_fbv, p_list , '*', mfc='w', mec='purple', ms=12, mew=2,label='fbv')
  for j,p in enumerate(p_list):
   print ("P[GPa]=  %9.2f  V_BM[A^3]=  %9.4f  V_Vinet[A^3]=  %9.4f  V_loglog[A^3]=  %9.4f  V_fbv[A^3]=  %9.4f  (V_Vinet-V_BM)/V_Vinet[%%]= %7.4f" % (p, spl_V_BM(p), spl_V_Vinet(p), V_loglogfit(p), vols_fbv[j], (1-spl_V_BM(p)/spl_V_Vinet(p))*100 )  )
 except:
  for j,p in enumerate(p_list):
   print ("P[GPa]=  %9.2f  V_BM[A^3]=  %9.4f  V_Vinet[A^3]=  %9.4f  V_loglog[A^3]=  %9.4f  (V_Vinet-V_BM)/V_Vinet[%%]= %7.4f" % (p, spl_V_BM(p), spl_V_Vinet(p), V_loglogfit(p), (1-spl_V_BM(p)/spl_V_Vinet(p))*100 )  )

ax.legend()


if PTarget>0:
 print("\nVolume at P_Target")
 p = PTarget
 V_BM = 0.0
 V_Vinet = 0.0
 try:
  each = 1 if P_BM(vs)[0]<P_BM(vs)[1] else -1
  spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs)[::each], vs[::each])  # V(P)
  V_BM = spl_V_BM(PTarget)
 except:
  pass 
 try:
  each = 1 if P_Vinet(vs)[0]<P_Vinet(vs)[1] else -1
  spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs)[::each], vs[::each])  # V(P)
  V_Vinet = spl_V_Vinet(PTarget)
 except:
  pass
 print ("P_Target[GPa]=  %9.2f  V_BM[A^3]=      %9.4f" % (PTarget, V_BM)    )
 print ("P_Target[GPa]=  %9.2f  V_Vinet[A^3]=   %9.4f" % (PTarget, V_Vinet) )
 print ("P_Target[GPa]=  %9.2f  V_loglog[A^3]=  %9.4f" % (PTarget, V_loglogfit(PTarget)) )
 print ("P_Target[GPa]=  %9.2f  V_spline[A^3]=  %9.4f" % (PTarget, V_spline(PTarget)) )
 if fbv_exists:
  v_fbv =  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True )) 
  print ("P_Target[GPa]=  %9.2f  V_fbv[A^3]=     %9.4f" % (PTarget, v_fbv) )


 #def integrand_with_error(P):  return (V_spline.derivative()(P) * dP_spline(P))**2
 #dGInt, _ = quad(V_spline, 148.878, PTarget)
 #dGIntE = np.sqrt(quad(integrand_with_error, 148.878, PTarget)[0])
 #print("dGInt=",dGInt*0.00022937123,dGIntE*0.00022937123)
 
 if V_BM != 0.0:
  PBest = min(P, key=lambda p: abs(p-PTarget))
  print("PBest[GPa]= %9.1f"% (PBest) )
  dGInt, _ = quad( spl_V_BM , P1, PTarget)
  print("Integral from P0[GPa]= %9.1f to P_Target[GPa]= %9.1f:  dGInt= %14.12f " % (P1,PTarget, dGInt*0.00022937123) )
 

 
 




#----------------------------------------------#
#  PRINTING RESULTS OF DELETING POINTS TEST    #
#  when the --test flag is active              #
#----------------------------------------------#
if deleting_points_test:   #  --test 
 V_org = array(V)
 P_org = array(P)
 dP_org = array(dP)

 print("\n# DELETING POINTS TEST: removing one by one")
 predictors = ['BM',  'Vinet',  'loglog']
 if fbv_exists:  predictors += [ 'fbv' ]
 
 counter_best  = {predictor: 0 for predictor in predictors}
 counter_worst = {predictor: 0 for predictor in predictors}
 for k in range(1,len(V_org)):
  #print ("Deleting point " + str(k))
  V0 = V_org[k]
  P0 = P_org[k]
  V = delete(V_org, k)
  P = delete(P_org, k)
  dP= delete(dP_org, k)
  #print ("Point removed:", V0,P0)
  #if k==3: ax.errorbar(V, P, dP, marker='D', ls='', ms=15, capsize=10, mfc='None', mew=2, zorder=5,label=r'$P(V)$ removing P0= '+str(P0))


  #------------------------#
  #    BIRCH MURNAGHAN     #
  #------------------------#
  ## Only K0,K0p as parameters. Forcing P(V0)=P0 = min(P)
  if   BM_deg==3:
   initial_guess = (k0, k0p)  # Initial guess of parameters K0, K0p
   p_BM = lambda v,K0,K0p: min(P) + P_V_BM(v, max(V),K0,K0p)
  elif BM_deg==4:
   initial_guess = (k0, k0p, k0pp)  # Initial guess of parameters K0, K0p
   p_BM = lambda v,K0,K0p,K0pp: min(P) + P_V_BM(v, max(V),K0,K0p,K0pp)
  try:
   popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=10000)
   P_BM = lambda v: min(P) + P_V_BM(v, max(V), *popt_BM)
   #print ("BM ,K0, K0p: ", max(V), popt_BM)
  except:
   pass

  #if k==1: ax.plot(vs,P_BM(vs),'-',alpha=0.5, label='$P(V)$ removing P0= '+str(P0))
  #if k==1: ax.errorbar(V, P, dP, marker='D', ls='', ms=15, capsize=10, mfc='None', mew=2, zorder=5,label=r'$P(V)$ removing P0= '+str(P0))
  #if k==1: ax2.errorbar(V_org, (P_BM(V_org)-P_org), 0*dP_org, color='red', marker='s', ms=12, capsize=10, mfc='w', mec='red', mew=2,alpha=0.5, label=r'$P_{\rm BM}-P$')

  #------------------------#
  #         VINET          #
  #------------------------#
  initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
  p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, max(V),K0,K0p)
  popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
  P_Vinet = lambda v: p_Vinet(v, *popt_Vinet)
  #print ("Vinet V0,K0, K0p: ", max(V), popt_Vinet)

  #------------------------#
  #       LOG-LOG          #
  #------------------------#
  # FITTING A POLYNOMIAL IN LOG-LOG SPACE
  P_residual = 10.0 + abs(min(P))        # Shift P by 10 upwards to prevent P=0.0 generating problems
  V_residual = 10.0 + abs(min(V))        # Shift V in case V does not represent volumes and may be negative 
  log_P = np.log(P + P_residual)
  log_V = np.log(V + V_residual)
  loglog_fit = lambda x: np.poly1d(np.polyfit(log_P, log_V,3))(x)  # lnV = a + b*lnP + c*lnP^2 + d*lnP^3  <==>  V(P) = exp(a)*P^{b+c*lnP+d*(lnP)^2}
  pp = np.linspace(min(P),max(P))
  #V_loglogfit = lambda p: np.exp(loglog_fit(np.log(p+P_residual))) - V_residual          #V(P)
  V_loglogfit = InterpolatedUnivariateSpline(pp, np.exp(loglog_fit(np.log(pp+P_residual))) - V_residual )         #V(P)
  each = 1 if V_loglogfit(pp)[0]<V_loglogfit(pp)[1] else -1
  P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp[::each]), pp[::each]) #P(V)



  command = "grep -v '^#' " +filename+ "| awk 'NR!= " +str(k)+ "+1{print }' > tmp.tmp"
  subprocess.check_output(command, shell=True)
  command = "~/scripts/fbv " + ' tmp.tmp ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(P0) + " | awk '/NewV/{print $NF}' "
  v_fbv = 0
  if fbv_exists: v_fbv = float(subprocess.check_output(command, shell=True))

  V_BM = 0.0
  V_Vinet = 0.0
  try:
   each = 1 if P_BM(vs)[0]<P_BM(vs)[1] else -1
   spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs[::each]), vs[::each])  # V(P)
   spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs[::each]), vs[::each])  # V(P)
   V_BM = spl_V_BM(P0)
   V_Vinet = spl_V_Vinet(P0)
  except:
   pass
  
  V_BM_err = 100*(V_BM/V0-1)
  V_Vinet_err = 100*(V_Vinet/V0-1)
  #print("El V_loglog de 100 es", V_loglogfit(100))
  V_loglogfit_err = 100*(V_loglogfit(P0)/V0-1)
  if v_fbv != 0:   V_fbv_err = 100*(v_fbv/V0-1)

  vlist =  { 'BM': V_BM_err, 'Vinet': V_Vinet_err, 'loglog': V_loglogfit_err}
  if fbv_exists:  vlist =  { 'BM': V_BM_err, 'Vinet': V_Vinet_err, 'loglog': V_loglogfit_err, 'fbv': V_fbv_err }
  ax.legend()

  sorted_vlist = sorted(vlist.items(), key=lambda item: abs(item[1]))
  #print ("Best:", sorted_vlist[0][0], "Worst:", sorted_vlist[-1][0])
  sorted_vlist = sorted(vlist.items(), key=lambda item: abs(item[1]))
  #print ("Best:", sorted_vlist[0][0], "Worst:", sorted_vlist[-1][0])
  counter_best[sorted_vlist[0][0]] += 1 
  counter_worst[sorted_vlist[-1][0]] += 1 

  if fbv_exists:
   print ("P0[GPa]=  %7.2f  V0[A^3]=  %7.2f  V_BM[A^3]=  %7.2f  V_Vinet[A^3]=  %7.2f  V_loglog[A^3]=  %7.2f  V_fbv[A^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f  V_fbv_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0), v_fbv,   V_BM_err, V_Vinet_err, V_loglogfit_err, V_fbv_err )  )
  else:
   print ("P0[GPa]=  %7.2f  V0[A^3]=  %7.2f  V_BM[A^3]=  %7.2f  V_Vinet[A^3]=  %7.2f  V_loglog[A^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0),   V_BM_err, V_Vinet_err, V_loglogfit_err)  )

 print ("BEST SCORES:  ", counter_best, "out of", len(V_org)-1)
 print ("WORST SCORES: ", counter_worst, "out of", len(V_org)-1)
 print ("BEST FOR EXTRAPOLATIONS:  ", sorted_vlist[0][0])
 overall_scores = {predictor: counter_best[predictor] - counter_worst[predictor] for predictor in predictors}
 best_predictor = max(overall_scores, key=overall_scores.get)
 print ("Overall scores:", overall_scores)
 print ("Best predictor:", best_predictor)


 print("\n# DELETING TWO RANDOM POINTS TEST: (P0,V0) is one of them")
 #predictors = ['BM',  'Vinet',  'loglog', 'fbv' ]
 predictors = ['BM',  'Vinet',  'loglog']
 counter_best  = {predictor: 0 for predictor in predictors}
 counter_worst = {predictor: 0 for predictor in predictors}
 trials = 20
 for _ in range(trials):
  kk = random.sample(range(1,len(V_org)-1), 2)
  V = delete(V_org, kk)
  P = delete(P_org, kk)
  dP= delete(dP_org, kk)
  #print ( "Removing points:",kk ,  [ ("V=",data[:,1][k],"P=",data[:,4][k]) for k in kk] )
  V0 = V_org[kk[0]]
  P0 = P_org[kk[0]]

  ### V0,K0,K0p as parameters
  #initial_guess = (2*max(V), max(P)/10, 4)  # Initial guess of parameters V0,K0, K0p
  #popt_Press, pcov_Press = curve_fit(P_V_BM, V, P, p0=initial_guess, maxfev=10000)
  #P_BM = lambda v:P_V_BM(v, *popt_Press)
  ##print ("BM V0,K0, K0p: ", popt_Press)
  #initial_guess = popt_BM  # Initial guess of parameters V0,K0, K0p
  #popt_Vinet, pcov_Vinet = curve_fit(VinetPressure, V, P, p0=initial_guess, maxfev=1000000)
  #P_Vinet = lambda v:VinetPressure(v, *popt_Vinet)
  ##print ("Vinet V0,K0, K0p: ", popt_Vinet)

  #------------------------#
  #    BIRCH MURNAGHAN     #
  #------------------------#
  ## Only K0,K0p as parameters. Forcing P(V0)=P0 = min(P)
  if   BM_deg==3:
   initial_guess = (k0, k0p)  # Initial guess of parameters K0, K0p
   p_BM = lambda v,K0,K0p: min(P) + P_V_BM(v, max(V),K0,K0p)
  elif BM_deg==4:
   initial_guess = (k0, k0p, k0pp)  # Initial guess of parameters K0, K0p
   p_BM = lambda v,K0,K0p,K0pp: min(P) + P_V_BM(v, max(V),K0,K0p,K0pp)
  popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=10000)
  P_BM = lambda v: min(P) + P_V_BM(v, max(V), *popt_BM)
  #print ("BM ,K0, K0p: ", max(V), popt_BM)


  #------------------------#
  #         VINET          #
  #------------------------#
  initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
  p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, max(V),K0,K0p)
  popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
  P_Vinet = lambda v: p_Vinet(v, *popt_Vinet)
  #print ("Vinet V0,K0, K0p: ", max(V), popt_Vinet)


  #------------------------#
  #       LOG-LOG          #
  #------------------------#
  # FITTING A POLYNOMIAL IN LOG-LOG SPACE
  #V_loglog= 0.0
  #try:
  P_residual = 10.0 + abs(min(P))        # Shift P by 10 upwards to prevent P=0.0 generating problems
  V_residual = 10.0 + abs(min(V))        # Shift V in case V does not represent volumes and may be negative 
  log_P = np.log(P + P_residual)
  log_V = np.log(V + V_residual)
  degp = 3 if len(log_P)>3 else 2
  loglog_fit = lambda x: np.poly1d(np.polyfit(log_P, log_V, degp))(x)  # lnV = a + b*lnP + c*lnP^2 + d*lnP^3  <==>  V(P) = exp(a)*P^{b+c*lnP+d*(lnP)^2}
  pp = np.linspace(min(P),max(P))
  #V_loglogfit = lambda p: np.exp(loglog_fit(np.log(p+P_residual))) - V_residual          #V(P)
  #print("775: pp=",pp)
  #print("776: logP=",  log(pp+P_residual)   )
  #print("777: V=", loglog_fit( log(pp+P_residual) )  )
  V_loglogfit = InterpolatedUnivariateSpline(pp, np.exp(loglog_fit(np.log(pp+P_residual))) - V_residual )         #V(P)
  each = 1 if V_loglogfit(pp)[0]<V_loglogfit(pp)[1] else -1
  P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp[::each]), pp[::each]) #P(V)
  #except:
  # pass


  command = "grep -v '^#' " +filename+ "| awk 'NR!= " +str(kk[0])+ "+1  && NR!= " +str(kk[1])+ "+1{print }' > tmp.tmp"
  subprocess.check_output(command, shell=True)
  command = "~/scripts/fbv " + ' tmp.tmp ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(P0) + " | awk '/NewV/{print $NF}' "
  v_fbv = 0
  if fbv_exists: v_fbv = float(subprocess.check_output(command, shell=True))


  V_BM = 0.0
  V_Vinet = 0.0
  try:
   each = 1 if P_BM(vs)[0]<P_BM(vs)[1] else -1
   spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs[::each]), vs[::each])  # V(P)
   spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs[::each]), vs[::each])  # V(P)
   V_BM = spl_V_BM(P0)
   V_Vinet = spl_V_Vinet(P0)
  except:
   pass

  
  V_BM_err = 100*(V_BM/V0-1)
  V_Vinet_err = 100*(V_Vinet/V0-1)
  V_loglogfit_err = 100*(V_loglogfit(P0)/V0-1)
  if v_fbv !=0: V_fbv_err = 100*(v_fbv/V0-1)

  #vlist =  { 'BM': V_BM_err, 'Vinet': V_Vinet_err, 'loglog': V_loglogfit_err, 'fbv': V_fbv_err }
  vlist =  { 'BM': V_BM_err, 'Vinet': V_Vinet_err, 'loglog': V_loglogfit_err}
  sorted_vlist = sorted(vlist.items(), key=lambda item: abs(item[1]))
  #print ("Best:", sorted_vlist[0][0], "Worst:", sorted_vlist[-1][0])
  counter_best[sorted_vlist[0][0]] += 1 
  counter_worst[sorted_vlist[-1][0]] += 1 

  #print ("P0[GPa]=  %7.2f  V0[A^3]=  %7.2f  V_BM[A^3]=  %7.2f  V_Vinet[A^3]=  %7.2f  V_loglog[A^3]=  %7.2f  V_fbv[A^3]=  %7.2f  V_BM_err[%%]= %6.3f  V_Vinet_err[%%]= %6.3f  V_loglog_err[%%]= %6.3f  V_fbv_err[%%]= %6.3f"  % (P0, V0, spl_V_BM(P0),  spl_V_Vinet(P0), V_loglogfit(P0), v_fbv,   V_BM_err, V_Vinet_err, V_loglogfit_err, V_fbv_err )  )
  print ("P0[GPa]=  %7.2f  V0[A^3]=  %7.2f  V_BM[A^3]=  %7.2f  V_Vinet[A^3]=  %7.2f  V_loglog[A^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0),   V_BM_err, V_Vinet_err, V_loglogfit_err )  )

 print ("BEST SCORES:  ", counter_best, "out of", trials)
 print ("WORST SCORES: ", counter_worst, "out of", trials)
 overall_scores = {predictor: counter_best[predictor] - counter_worst[predictor] for predictor in predictors}
 best_predictor = max(overall_scores, key=overall_scores.get)
 print ("Overall scores:", overall_scores)
 print ("Best predictor two-points test:", best_predictor)




ax2.set_xlabel("Volume ($\AA^3$)")
#ax2.set_ylim(2*min(P_Vinet(V)-P),2.1*max(P_Vinet(V)-P))
ymax = max(abs(P_Vinet(V)-P)) if max(abs(P_Vinet(V)-P)) > max(dP) else max(dP)
ax2.set_ylim(-3*ymax,3*ymax)
ax2.set_xlim(min(V) - dv,max(V) + dv)
ax2.legend(loc='best')

#geometry = plt.get_current_fig_manager().window.geometry()
## Extract xmin, xmax, ymin, ymax from the QRect object
#xmin = geometry.x()
#xmax = geometry.x() + geometry.width()
#ymin = geometry.y()
#ymax = geometry.y() + geometry.height()
#fig.canvas.manager.window.setGeometry(100, 100, xmax, ymax)  # Set position for fig
#fig2.canvas.manager.window.setGeometry(100+xmax+50, 100, xmax, ymax)  # Set position for fig2

#savefig('diff.png')

if show_plots: show()
