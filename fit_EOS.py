#!/usr/bin/env python 
"""
-----------------------------------------------------------------------------------------------|
 FITTING EQUATIONS OF STATE USING VINET / BIRCH-MURNHAGAN / F-f / LOG-LOG                      |
Repository @ https://github.com/fgonzcat/fit_EOS.git                                           |
                                                                                               |
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
   R2        =  1 - sum(residuals^2)/sum(Pi - mean(P))                                         |
   chi_squared = sum(residuals**2 / dP**2)                                                     |
   reduced_chi_squared = chi_squared/(N-p)                                                     |
       chi^2 ~ 1 ---> model is consistent with data within error bars.                         |
       chi^2 < 1 ---> possible overfitting or overestimated uncertainties                      |
       chi^2 > 1 ---> model may be inadequate or error bars underestimated                     |
                 Among multiple models, the one with reduced_chi_squared                       |
                 closest to 1 is generally preferred.                                          |
                                                                                               |
Felipe Gonzalez                                                          Berkeley, 05/19/2023  |
-----------------------------------------------------------------------------------------------|
Last modified on:                                                                  05/01/2025
"""
from pylab import *
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import sys,os, subprocess
import random
import matplotlib.gridspec as gridspec



filename = "/home/fgonzalez/EOS_Fe_sol_6000K.dat"  # Example, must be provided as argument
colV = 5  # Default column number for V[A^3]
colVE= -1 # Default column number for volume error dV[A^3]
colP = 11  # Default column number for P[GPa]
colPE = 12  # Default column number for P error dP[GPa]
BM_deg = 3   # Birch-Murnaghan of 3rd order by default
PTarget=-1   # Ptarget to guess the volume
P1 =    -1   # To calculate integral V(P)dP from P1 to PTarget
print_table = False
deleting_points_test = False
show_plots = True 
show_F_plot = False
V0_as_param = False
Merge_Figures=False
fbv_path = os.path.expanduser("~/scripts/fbv")
try:
 #subprocess.run([fbv_path],check=True)
 fbv_exists = os.path.isfile(fbv_path)
except:
 fbv_exists = False





'''
WELCOME MESSAGE
'''
print("\n**** FITTING EOS ****")
print("by Felipe Gonzalez [Berkeley 05-19-2023]\n")
if len(sys.argv) == 1:
 message = """
 
 This code fits an isotherm using Birch-Murnhagan, Vinet, F-f, and log-log [Berkeley 05-19-23]
 
 Usage: {0}  EOS_Fe_sol_6000K.dat
 Usage: {0}  EOS_Fe_sol_6000K.dat   V[Å^3]-col P[GPa]-col P_error-col
 Usage: {0}  EOS_Fe_sol_6000K.dat       6         12          13
 Usage: {0}  EOS_Fe_sol_6000K.dat   V[Å^3]-col V[Å^3]-col  P[GPa]-col P_error-col
 Usage: {0}  EOS_Fe_sol_6000K.dat       6         7           12          13

 Usage: {0}  EOS_H2O_liq_7000K.dat      6         12          13  --BM4 --V0-as-param
 Usage: {0}  $EOS                       6         12          13  --PTarget 150   --BM4  --test -p --show-F-plot --merge-plots
 Usage: {0}   ... -p              --> print V(P) 
 Usage: {0}   ... --test          --> deleting-points performance test 
 Usage: {0}   ... --noplots       --> don't plot 
 Usage: {0}   ... --BM2           --> Birch-Murnaghan 2th order
 Usage: {0}   ... --BM3           --> Birch-Murnaghan 3th order (default)
 Usage: {0}   ... --BM4           --> Birch-Murnaghan 4th order
 Usage: {0}   ... --V0-as-param   --> Treat V0 as another fitting parameter [ do not force the default P(V0) = P0, where P0=min(P), V0=max(P) ]
 Usage: {0}   ... --PTarget 150   --> prints the volume at P_Target from each model and the integral  ∆G = int_P1^P_Target V(P) dP = G(P_Target) - G(P1)
 Usage: {0}   ... --P1      110   --> Changes P1 for the integral above to 110 GPa. 
 Usage: {0}   ... --merge-plots   --> Just one P(V) figure with two plots instead of separate figures
 Usage: {0}   ... --show-F-plot   --> Show the F(f) plot to check how the fit performed in F-f space 

 No arguments assumes V[Å^3]-col= 6, P[GPa]-col= 12,  P_error-col= 13

 Reported Errors:
   residuals = P_fit(V)-P
   sigma     = std(residuals)
   RMSE      = sqrt(mean(residuals)^2)
   R2        =  1 - sum(residuals^2)/sum(Pi - mean(P))
   chi_squared = sum(residuals**2 / dP**2)
   reduced_chi_squared = chi_squared/(N-p)
       chi^2 ~ 1 ---> model is consistent with data within error bars.
       chi^2 < 1 ---> possible overfitting or overestimated uncertainties.
       chi^2 > 1 ---> model may be inadequate or error bars underestimated.
                 Among multiple models, the one with reduced_chi_squared.
                 closest to 1 is generally preferred.

 
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
 if '--merge-plots' in sys.argv:
  idx = sys.argv.index('--merge-plots')
  sys.argv.pop(idx)
  Merge_Figures = True
 if '--show-F-plot' in sys.argv:
  idx = sys.argv.index('--show-F-plot')
  sys.argv.pop(idx)
  show_F_plot   = True
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
 if '--PTarget' in sys.argv:
  idx = sys.argv.index('--PTarget')
  sys.argv.pop(idx)
  PTarget = float(sys.argv.pop(idx))



 if len(sys.argv)==2: pass
 elif len(sys.argv) >= 5:
  if len(sys.argv) >= 6 and all(arg.isdigit() for arg in sys.argv[2:6]):   # like  fit_EOS.py $EOS 5 7 2 4
   colV,colVE,colP,colPE = [ int(c)-1 for c in sys.argv[2:6] ]
  elif all(arg.isdigit() for arg in sys.argv[2:5]):                        # like  fit_EOS.py $EOS 12 14 15
   colV,colP,colPE = [ int(c)-1 for c in sys.argv[2:5] ]
 else:
  print ("Your input: " , sys.argv)
  print("You need to specify the column numbers: V[Å^3]-col P[GPa]-col P_error-col")
  exit()




#------------------------#
# SETTING FIGURE PARAMS  #
#------------------------#
GPaA3_to_eV=0.0062415091
Ha_to_eV = 27.211386
Ha_to_meV = Ha_to_eV * 1000
kB = 0.000086173303372 # eV/K
fig_size = [700/72.27 ,520/72.27]
#fig_size = [700/72.27 ,720/72.27]
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
          'figure.subplot.bottom': 0.120,'figure.subplot.top': 0.975,'figure.subplot.left': 0.160,'figure.subplot.right': 0.960}


rcParams.update(params)



#-----------------------------------#
# FUNCTIONAL FORMS OF FIT FUNCTIONS #
#-----------------------------------#
def P_V_BM2(V, V0,K0):
    # 2nd order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    P = 1.5*K0 * (f**7 - f**5) 
    return P
def dP_V_BM2(V, V0,K0, dV0,dK0):
    # Error in P: dP = sqrt( [ (∂P/∂K0)dK0 ] ^2 )
    f = (V0/V)**(1.0/3)
    dPdK0 = P_V_BM2(V, V0,K0)/K0
    dp = abs(dPdK0)*dK0
    return dp

def P_V_BM3(V, V0,K0,K0p):
    # 3rd order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    f5 = f*f*f*f*f
    P = 1.5*K0 * f5*(f**2 - 1.0) * (1 + 0.75*(K0p-4)*(f*f - 1))
    return P
def dP_V_BM3(V, V0,K0,K0p, dV0,dK0,dK0p,  cov_mat):
    # Error in P: dP = sqrt( [ (∂P/∂K0)dK0] ^2 + [ (∂P/∂K0p)dK0p] ^2  )
    f = (V0/V)**(1.0/3)
    f2 = f*f
    f5 = f*f*f*f*f
    f7 = f5*f2
    f2m1 = f2 - 1.0

    A = f5*f2m1
    B = (1 + 0.75*(K0p-4)*(f2  - 1))

    dPdK0 = P_V_BM3(V, V0,K0,K0p)/K0
    dPdK0p = 1.5*K0 * A *  0.75*f2m1 
    # ∂P/∂V0= 3/2 K0 *[ dAdV0 * B + A * dBdV0]
    dAdV0 = (7.0*f7  - 5.0*f5)/(3*V0)
    dBdV0 = 0.5 *(K0p - 4) *f2/V0
    dPdV0  = 1.5*K0 * (dAdV0*B + A*dBdV0)
    # If V0 is NOT a param, 
    # cov_mat =  [ Cov(K0,K0)   Cov(K0,K0p) 
    #              Cov(K0p,K0)  Cov(K0p,K0p)  ]
    # cov_mat =  [ Cov(V0,V0)   Cov(V0,K0)  Cov(V0, K0p)
    #              Cov(K0,V0)   Cov(K0,K0)  Cov(K0, K0p)
    #              Cov(K0p,V0)  Cov(K0p,K0) Cov(K0, K0p) ]
    cov_terms = 0.0
    if dV0==0:
     cov_terms += 2*dPdK0*dPdK0p*cov_mat[0][1] 
    else:
     cov_terms += 2*dPdV0*dPdK0*cov_mat[0][1] +  2*dPdV0*dPdK0p*cov_mat[0][2] +  2*dPdK0*dPdK0p*cov_mat[1][2]  
    dp = sqrt( dPdV0*dPdV0*dV0*dV0 + dPdK0*dPdK0*dK0*dK0  + dPdK0p*dPdK0p*dK0p*dK0p  + dPdV0*dPdV0*dV0*dV0  + 1.0*cov_terms)
    return dp

def P_V_BM4(V, V0,K0,K0p,K0pp):
    # 4th order Birch-Murnaghan
    f = (V0/V)**(1.0/3)
    P = 1.5*K0 * (f**7 - f**5) * (1 + 0.75*(K0p-4)*(f*f - 1)  + (1.0/24)*(9*K0p*K0p - 63*K0p + 9*K0*K0pp + 143) *(f*f - 1)*(f*f - 1) )
    return P
def dP_V_BM4(V, V0,K0,K0p,K0pp, dV0,dK0,dK0p,dK0pp, cov_mat):
    # Error in P: dP = sqrt( [ (∂P/∂K0)dK0] ^2 + [ (∂P/∂K0p)dK0p] ^2 +  [ (∂P/∂K0pp)dK0pp] ^2 )
    f = (V0/V)**(1.0/3)
    dPdK0 = P_V_BM4(V, V0,K0,K0p,K0pp)/K0
    dPdK0p = 1.5*K0 * (f**7 - f**5) * ( 0.75*(f*f - 1)  + (1.0/24)*(18*K0p - 63 ) *(f*f - 1)*(f*f - 1) )
    dPdK0pp= 1.5*K0 * (f**7 - f**5) * ( (1.0/24)*( 9*K0 ) *(f*f - 1)*(f*f - 1) )
    dPdV0 = 0.0 # for now...
    dp = sqrt( dPdV0*dPdV0*dV0*dV0 + dPdK0*dPdK0*dK0*dK0  + dPdK0p*dPdK0p*dK0p*dK0p + dPdK0pp*dPdK0pp*dK0pp*dK0pp  )
    return dp

def E_V_BM3(V, V0, K0, K0p, E0):
    f = V0/V
    E = E0 + (9*V0*K0/16)*GPaA3_to_eV * ( (f**(2.0/3.0)-1)**3 * K0p + (f**(2.0/3.0)-1)**2 * (6-4*(f**(2.0/3.0))))
    return E

def VinetPressure(V, V0,K0,K0p):
  #if V<0: return 0
  x  = (V/V0)**(1.0/3.0)
  xi = 1.5*(K0p-1.0)
  P  = 3.0*K0 * (1.0-x)/x/x * np.exp( xi*(1.0-x) )
  return P
def dP_VinetPressure(V, V0,K0,K0p, dV0,dK0,dK0p, cov_mat):
  # Error in P: dP = sqrt( [ (∂P/∂K0)dK0] ^2 + [ (∂P/∂K0p)dK0p] ^2  )
  x  = (V/V0)**(1.0/3.0)
  xi = 1.5*(K0p-1.0)
  dPdK0  = VinetPressure(V, V0,K0,K0p)/K0 
  dPdK0p = VinetPressure(V, V0,K0,K0p) *  1.5*(1.0-x) 
  # ∂P/∂V0= -(K0/V0) (1/eta - 2/eta^2) exp( xi*(1.0-eta) ) + (P(V)/V0) *eta (K0'-1)
  dPdV0 =  -(K0/V0) *(x -2)/(x*x)* np.exp( xi*(1.0-x) )  + VinetPressure(V, V0,K0,K0p) * x*(K0p-1)/(2*V0)
  cov_terms = 0.0
  if dV0==0:
   cov_terms += 2*dPdK0*dPdK0p*cov_mat[0][1] 
  else:
   cov_terms += 2*dPdV0*dPdK0*cov_mat[0][1] +  2*dPdV0*dPdK0p*cov_mat[0][2] +  2*dPdK0*dPdK0p*cov_mat[1][2]  
  dp = sqrt( dPdV0*dPdV0*dV0*dV0 +  dPdK0*dPdK0*dK0*dK0  + dPdK0p*dPdK0p*dK0p*dK0p  + cov_terms)
  return dp


def SplineError(x_data, y_data, yerr, y_spline_func, x0):
 '''
 S(x) = InterpolatedUnivariateSpline(x_data, y_data)
 sigma_S(x0) = sqrt( sum_i [∂S(x0)/∂yi]sigma_yi   )
 '''
 perturbation = min(yerr)
 partial_derivatives = []
 for i in range(len(y_data)):
     y_perturbed = y_data.copy()
     y_perturbed[i] += perturbation
     k = 3 if len(x_data)>3 else 2
     spline_perturbed = InterpolatedUnivariateSpline(x_data, y_perturbed, w=1/yerr**2, k=k)
     y_spline_perturbed = spline_perturbed(x0)
     partial_derivative = (y_spline_perturbed - y_spline_func(x0)) / perturbation
     partial_derivatives.append(partial_derivative)
     #xx = linspace(min(x_data),max(x_data))
     #ax3.plot(xx,spline_perturbed(xx),'--', lw=1)
 partial_derivatives = np.array(partial_derivatives)
 propagated_error_at_x0 = np.sqrt(np.sum((partial_derivatives * yerr)**2))
 return propagated_error_at_x0



#--------------------------#
#  READ INPUT DATA TABLE   #
#--------------------------#
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
 dV = 0.0
except:
 #V,P,dP  = np.loadtxt(filename, usecols=(colV,colP,colPE), dtype=float, comments='#', unpack=True)  # x, y, yerr 
 if colVE>0:  
  if colPE<0:
   data    = np.loadtxt(filename, usecols=(colV,colVE,colP), dtype=float, comments='#',unpack=True)  # x, y, yerr 
  else:
   data    = np.loadtxt(filename, usecols=(colV,colVE,colP,colPE), dtype=float, comments='#',unpack=True)  # x, y, yerr 
 else:
  if colPE<0:
   data    = np.loadtxt(filename, usecols=(colV,colP), dtype=float, comments='#',unpack=True)  # x, y, yerr 
  else:
   data    = np.loadtxt(filename, usecols=(colV,colP,colPE), dtype=float, comments='#',unpack=True)  # x, y, yerr 
 sorted_indices = data[0].argsort()[::-1]    # Sort by increasing volumes and then invert
 data = data[:, sorted_indices]
 positive = data[1] >= 0.0  # (P>=0)
 if colVE>0:  
  if colPE<0:
   V,dV,P     = [ arr[positive] for arr in data] 
  else:
   V,dV,P,dP  = [ arr[positive] for arr in data] 
 else:
  if colPE<0:
   V,P     = [ arr[positive] for arr in data] 
  else:
   V,P,dP  = [ arr[positive] for arr in data] 
  dV = 0.0

#dP = 10*dP  # Increase the error bars

minP=min(P)
maxV=max(V)
if colPE<0: dP=zeros(len(V))
dP = np.where(dP == 0, 1e-10, dP)

if len(V)<=4: BM_deg=3   # If the user requests BM4, but you have  4 data points, BM4 is not possible. Force BM3

try:
 T0=int(T)
except:
 T0=0

# PRINTING THE ORIGINAL DATA
print ("\n#EOS Data: " + filename)
for j in range(len(V)):
 print("i= %2i  V[Å^3]= %9.4f  P[GPa]= %9.4f %6.4f" % (j, V[j], P[j], dP[j]) ) 


#------------------------#
#       LOG-LOG          #
#------------------------#
# FITTING A POLYNOMIAL IN LOG-LOG SPACE
P_residual = 10.0 + abs(min(P))        # Shift P by 10 upwards to prevent P=0.0 generating problems
V_residual = 10.0 + abs(min(V))        # Shift V in case V does not represent volumes and may be negative 
log_P = np.log(P + P_residual)
log_V = np.log(V + V_residual)

delta_lnP = abs(dP/(P + P_residual))
k = 3 if len(V)>3  else 2 # There should be more than just 3 points to fit usually, but if only 3 are available, use degree 2 
try: 
 if len(V)==4:
  coeffs_loglogP                     = np.polyfit(log_V, log_P, k,  cov=False)
  errors_loglogP = [0,0,0,0]
 else:
  coeffs_loglogP, cov_matrix_loglogP = np.polyfit(log_V, log_P, k,  cov=True)
  errors_loglogP = sqrt(diag(cov_matrix_loglogP))
 A=coeffs_loglogP[-1]; dA=errors_loglogP[-1]
 B=coeffs_loglogP[-2]; dB=errors_loglogP[-2]
 C=coeffs_loglogP[-3]; dC=errors_loglogP[-3]
 D=0.0; dD=0.0         
 if k==3:  D=coeffs_loglogP[-4]; dD=errors_loglogP[-4]
 polynomial_loglog_fit = lambda lnV: np.poly1d(coeffs_loglogP)(lnV)  # lnP = a + b*lnV + c*lnV^2 + d*lnV^3  <==>  P(V) = exp(a)*V^{b+c*lnV+d*(lnV)^2}
 dlnP = lambda lnV:  sqrt( dA**2 + (lnV*dB)**2 + (lnV*lnV*dC)**2 + (lnV*lnV*lnV*dD)**2 ) 
 vv =  np.linspace(min(V),max(V))
 P_loglogfit = InterpolatedUnivariateSpline(vv,  np.exp(polynomial_loglog_fit( np.log(vv+V_residual) )) - P_residual    )  # P(V)
 P_loglog_err  = lambda v:  P_loglogfit(v) * dlnP( log(v) )
 try:
  sorted_indices = vv[::-1].argsort()
  P_sorted = P_loglogfit(vv[::-1][sorted_indices])
  V_sorted =             vv[::-1][sorted_indices] 
  # Remove duplicates 
  _, unique_indices = np.unique(P_sorted, return_index=True)
  P_unique = P_sorted[unique_indices]
  V_unique = V_sorted[unique_indices]
  #V_loglogfit = InterpolatedUnivariateSpline(P_loglogfit(vv[::-1]), vv[::-1]) #V(P)
  V_loglogfit = InterpolatedUnivariateSpline(P_unique, V_unique) #V(P)
 except:
  V_loglogfit = InterpolatedUnivariateSpline(P_loglogfit(vv), vv) #V(P)
 V_loglog_err  = lambda p: abs(V_loglogfit.derivative()(p)) * P_loglog_err( V_loglogfit(p) )  
except:
 print("\nCan't fit a log-log function with just ", len(V),"points. Skipping log-log fit...")





#------------------------#
#       SPLINE           #
#------------------------#
try:
 if any(dP==0): P_spline = InterpolatedUnivariateSpline(V[::-1],P[::-1], k=k)  # P(V)
 else:          P_spline = InterpolatedUnivariateSpline(V[::-1],P[::-1], w=1/dP[::-1]**2, k=k)  # P(V)
except: 
 if any(dP==0):  P_spline = InterpolatedUnivariateSpline(V,P, k=k)  # P(V)
 else:           P_spline = InterpolatedUnivariateSpline(V,P, w=1/dP**2, k=k)  # P(V)
sorted_indices = P.argsort()
P_sorted = P[sorted_indices]
V_sorted = V[sorted_indices]
# Remove duplicates 
_, unique_indices = np.unique(P_sorted, return_index=True)
P_unique = P_sorted[unique_indices]
V_unique = V_sorted[unique_indices]

V_spline = InterpolatedUnivariateSpline(P_unique,V_unique, k=k)  # V(P)
#V_spline = InterpolatedUnivariateSpline(P[sorted_indices],V[sorted_indices])  # V(P)
#dP_spline = InterpolatedUnivariateSpline(P[sorted_indices], dP[sorted_indices])



#------------------------#
#    BIRCH MURNAGHAN     #
#------------------------#
'''
Forcing P(V0) = P0 = min(P). V0 is not a parameter
'''
if   BM_deg==2: P_V_BM = lambda V, V0,K0:          P_V_BM2(V, V0,K0)
elif BM_deg==3: P_V_BM = lambda V, V0,K0,K0p:      P_V_BM3(V, V0,K0,K0p)
elif BM_deg==4: P_V_BM = lambda V, V0,K0,K0p,K0pp: P_V_BM4(V, V0,K0,K0p,K0pp)

if   BM_deg==2: dP_V_BM = lambda V, V0,K0,dV0,dK0, cov_mat:                     dP_V_BM2(V, V0,K0,dV0,dK0, cov_mat)
elif BM_deg==3: dP_V_BM = lambda V, V0,K0,K0p,dV0,dK0,dK0p, cov_mat:            dP_V_BM3(V, V0,K0,K0p,dV0,dK0,dK0p, cov_mat)
elif BM_deg==4: dP_V_BM = lambda V, V0,K0,K0p,K0pp,dV0,dK0,dK0p,dK0pp, cov_mat: dP_V_BM4(V, V0,K0,K0p,K0pp,dV0,dK0,dK0p,dK0pp, cov_mat)


k0   = -V[0]*(P[-1]-P[0])/(V[-1]-V[0])  #max(P)/10
k0p  = 4
k0pp = -(9*k0p*k0p -63*k0p + 143)/(9*k0)
## Only K0,K0p as parameters. Forcing P(V0)=P0 = min(P) with V0 = max(V)
if   BM_deg==2:
 initial_guess = (k0)  # Initial guess of parameters K0, K0p
 p_BM = lambda v,K0: minP + P_V_BM2(v, maxV,K0)
elif BM_deg==3:
 initial_guess = (k0, k0p)  # Initial guess of parameters K0, K0p
 p_BM = lambda v,K0,K0p: minP + P_V_BM3(v, maxV,K0,K0p)
elif BM_deg==4:
 initial_guess = (k0, k0p, k0pp)  # Initial guess of parameters K0, K0p, K0pp
 p_BM = lambda v,K0,K0p,K0pp: minP + P_V_BM4(v, maxV,K0,K0p,K0pp)


'''
## BM with NO V0-as-param (only K0 and K0p are parameters) 
'''
#popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess) #, maxfev=10000)
popt_BM, pcov_BM= curve_fit(p_BM, V, P, p0=initial_guess , sigma=dP, absolute_sigma=True, maxfev=10000)
Perr_BM = np.sqrt(np.diag(pcov_BM))
print ("Birch-Murnaghan of degree",BM_deg,"\n")
if   BM_deg==2:
 print ( "BM fit:       V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  %s"  % ( maxV, popt_BM[0], Perr_BM[0], "                       # Forcing P(V0)=P0 = min(P)" ) )
elif BM_deg==3:
 print ( "BM fit:       V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( maxV, popt_BM[0], Perr_BM[0], popt_BM[1], Perr_BM[1], "  # Forcing P(V0)=P0 = min(P)" ) )
elif BM_deg==4:
 print ( "BM fit:       V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f  K0pp[1/GPa]= %7.4f %4.4f%s"  % ( maxV, popt_BM[0], Perr_BM[0], popt_BM[1], Perr_BM[1],popt_BM[2], Perr_BM[2], "  # Forcing P(V0)=P0 = min(P)" ) )
#print("COVARIANT BM:",pcov_BM)


'''
## V0,K0,K0p as parameters.
'''
k0   = -V[0]*(P[-1]-P[0])/(V[-1]-V[0])  #max(P)/10
k0p  = 4
k0pp = -(9*k0p*k0p -63*k0p + 143)/(9*k0)
if   BM_deg==2: initial_guess = (maxV,k0)            # Initial guess of parameters V0, K0
elif BM_deg==3: initial_guess = (maxV,k0, k0p)       # Initial guess of parameters V0, K0, K0p
elif BM_deg==4: initial_guess = (maxV,k0, k0p,k0pp)  # Initial guess of parameters V0, K0, K0p, K0pp

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
nPerr_BM = np.sqrt(np.diag(npcov_BM))
if   BM_deg==2:
 print ( "BM fit:       V0[Å^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  %s"  % ( npopt_BM[0], nPerr_BM[0], npopt_BM[1], nPerr_BM[1], "                       # V0 as param" ) )
elif BM_deg==3:
 print ( "BM fit:       V0[Å^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( npopt_BM[0], nPerr_BM[0], npopt_BM[1], nPerr_BM[1], npopt_BM[2], nPerr_BM[2], "  # V0 as param" ) )
elif BM_deg==4:
 print ( "BM fit:       V0[Å^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f  K0pp[1/GPa]= %7.4f %4.4f%s"  % ( npopt_BM[0], nPerr_BM[0], npopt_BM[1], nPerr_BM[1], npopt_BM[2], nPerr_BM[2], npopt_BM[3], nPerr_BM[3], "  # V0 as param" ) )


#------------------------#
#         VINET          #
#------------------------#
initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, maxV,K0,K0p)
popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
Perr_Vinet = np.sqrt(np.diag(pcov_Vinet))
print ( "Vinet fit:    V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( maxV, popt_Vinet[0], Perr_Vinet[0], popt_Vinet[1], Perr_Vinet[1], "  # Forcing P(V0)=P0 = min(P)" ) )

initial_guess = (maxV,k0, k0p)  # Initial guess of parameters K0, K0p
try:
 npopt_Vinet, npcov_Vinet = curve_fit(VinetPressure, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True, bounds=(0,[3*max(V),2*k0,100]) ) #, maxfev=1000000 )
except:
 npopt_Vinet, npcov_Vinet = curve_fit(VinetPressure, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True,                                     maxfev=1000000 )
nPerr_Vinet = np.sqrt(np.diag(npcov_Vinet))
print ( "Vinet fit:    V0[Å^3]= %9.4f %9.4f  K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( npopt_Vinet[0], nPerr_Vinet[0], npopt_Vinet[1], nPerr_Vinet[1], npopt_Vinet[2], nPerr_Vinet[2], "  # V0 as param" ) )


#-----------------------------------------------#
#        BM & VINET P(V) & dP(V) functions      #
#-----------------------------------------------#
# Here I simplify the functions so they only depend on one variable: V
if V0_as_param:
 P_BM =    lambda v:       P_V_BM(v, *npopt_BM)
 dP_BM=    lambda v:       dP_V_BM(v, *npopt_BM, *nPerr_BM, npcov_BM)
 P_Vinet = lambda v:       VinetPressure(v, *npopt_Vinet)
 dP_Vinet= lambda v:       dP_VinetPressure(v, *npopt_Vinet, *nPerr_Vinet, npcov_Vinet)
else:
 P_BM =    lambda v:       minP + P_V_BM(v, maxV, *popt_BM)
 dP_BM=    lambda v:       dP_V_BM(v, maxV, *popt_BM, 0, *Perr_BM, pcov_BM)
 P_Vinet = lambda v:       p_Vinet(v, *popt_Vinet)
 dP_Vinet= lambda v:       dP_VinetPressure(v, maxV, *popt_Vinet, 0, *Perr_Vinet, pcov_Vinet)


#----------------------------------------------#
#            PLOTTING FIGURE 1                 #
#              P(V) and fits                   #
#----------------------------------------------#
if Merge_Figures:
 fig_size = [700/72.27 ,590/72.27]
 rcParams.update({ 'figure.figsize': fig_size, 'figure.subplot.bottom': 0.090 } )

 fig = figure('Pressure vs. Volume')   
 gs = gridspec.GridSpec(2, 1)
 ax = subplot(gs[0:1,0])
 ax2 = subplot(gs[1,0], sharex= ax)
 setp(ax.get_xticklabels(),visible=False)
 subplots_adjust(hspace=0.0)
else:
 fig = figure('Pressure vs. Volume')   
 ax = subplot(111)
 fig2 = figure('Residuals')
 ax2 = subplot(111)

#ax.errorbar(data[:,1], data[:,4], data[:,5], ls='', color='b', marker='s', ms=15, capsize=10,mfc='None', mec='b', mew=2, label=r'$P(V)$' + filename) ## all data
#ax.errorbar(V, P, dP, marker='o', ls='',c='r',ms=10, capsize=10, mfc='pink', mec='r', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0)+' (filtered data)')  ## filtered data
ax.errorbar(V, P, dP, marker='o', ls='',c='b',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$')
#ax.errorbar(V, P, dP,  marker='o', ls='-',c='b',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0))  ## filtered data
#ax.errorbar(V, E, dE,  marker='o', ls='-',c='b',ms=10, capsize=10, mfc='lightblue', mec='blue', mew=2, zorder=5,label=r'$P(V)$ at T='+str(T0))  ## filtered data


#vs = linspace(0.7*min(V), 1.1*max(V), 500)
dv = (max(V)-min(V))/10
vs = linspace(min(V)-2*dv, max(V)+dv, 500)
ps = P_BM(vs) #[P_BM(v) for v in vs]
ax.plot(vs, ps,'r-', lw=4,label='$P(V)$ BM fit')
ax.plot(vs, P_Vinet(vs),'--', c='limegreen',lw=3,label='$P(V)$ Vinet fit')
ax.plot(vs, P_spline(vs),'--',dashes=[8,2], c='m', lw=2, label='$P(V)$ spline fit')
if len(V)>3:  # FIXME: Can't fit a log-log with just 3 points, so P_loglogfit never happened above
 ax.plot(vs, P_loglogfit(vs),'k',dashes=[5,1,1,1],lw=2,label=r'Log-Log polyfit '+'\n'+r'($\ln P=a + b*\ln V + c*\ln^2 V + d*\ln^3 V$)')

if fbv_exists:
 p_list =linspace(1.1*min(P),0.9*max(P),100)
 vols_fbv = [  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True ))  for p in p_list ]
 ax.plot(vols_fbv, p_list , '-', dashes=[5,2,2,2], c='purple', lw=2, label='fbv')
ax.set_xlabel("Volume ($\AA^3$)")
ax.set_ylabel("Pressure (GPa)")
ax.set_xlim(min(V)-2*dv,max(V)+2*dv)
ax.set_ylim(0.9*min(P),1.5*max(P))

dPs = dP_BM(vs)
ax.fill_between(vs, ps-dPs, ps+dPs, color='red', alpha=0.1)
#pp = P_loglogfit(vs)
#dPs = P_loglog_err(vs)
#ax.fill_between(vs, pp-dPs, pp+dPs, color='blue', alpha=0.1)
ps = P_Vinet(vs)
dPs = dP_Vinet(vs)
ax.fill_between(vs, ps-dPs, ps+dPs, color='limegreen', alpha=0.1)
#ax.set_xscale('log')
#ax.set_yscale('log')
#savefig('PV.png')


#----------------------------------------------#
#            PLOTTING FIGURE 2                 #
# Difference between the fit and the points    #
#----------------------------------------------#
ps=np.array(ps)
ax2.plot(vs,ps-ps,'k--' ,label='$P$ Data')

dPs = dP_BM(V) #, *popt_BM, *Perr_BM)
ax2.errorbar(V, (P_BM(V)-P), 0*dPs, color='red', marker='s', ms=12, capsize=10, mfc='pink', mec='red', mew=2, label=r'$P_{\rm BM}-P$')
ax2.errorbar(V, (P_Vinet(V)-P), 0*dP, color='limegreen', marker='v', ms=10, capsize=10, mfc='w', mec='limegreen', mew=2,label=r'$P_{\rm Vinet}-P$')
if len(V)>3: ax2.errorbar(V, (P_loglogfit(V)-P), 0*dP, color='k', lw=1, marker='d', ms=12, capsize=6, mfc='yellow', mec='k', mew=2, label=r'$P_{\rm log-log}-P$')
ax2.errorbar(V, P-P, dP, marker='o', ls='', c='blue', ms=14, capsize=10, mfc='lightblue', mec='blue', mew=2, lw=2, zorder=-1, label=r'$P(V)$ ')


# COMMENT THE BLOCK ABOVE AN UNCOMMENT THE ONE BELOW TO PLOT IN TERMS OF PV
#ax2.errorbar(V, (fPV(V)-P)*V*GPaA3_to_eV, dP*V*GPaA3_to_eV, color='b', marker='s', c='b',ms=10, capsize=10, mfc='None', mec='b', mew=2,label=r'$P-P_{\rm BM}$')
#ax2.errorbar(V, (P_Vinet(V)-P)*V*GPaA3_to_eV, dP*V*GPaA3_to_eV, color='green', marker='d', c='green',ms=10, capsize=10, mfc='None', mec='green', mew=2,label=r'$P-P_{\rm Vinet}$')
#ax2.plot(V,(P-fPV(V))*V*GPaA3_to_eV,'ro', mfc='w',ms=10,mew=2,label='$P(V)$ at T='+str(T0)+'K')
#ax2.set_xlim(600,1050)
#ax2.set_ylabel("$PV$ (eV)")



#----------------------------------------------#
#            PLOTTING FIGURE 3                 #
# Raymond Jeanloz's F-vs-f                     #
# see http://doi.org/10.1029/GL008i012p01219   #
# see http://doi.org/10.1063/1.333139          #
#----------------------------------------------#
V_org = array(V)
P_org = array(P)
dV_org = 0.0
if not isinstance(dV,float): dV_org = array(dV)
P_org = array(P)
dP_org = array(dP)


V0 = 1.0
P_shift = 0.0
if min(V)>1:
 #**** THIS IS JUST A TRICK  *****#
 #   ---- Re-scale x-axis, shift y-axis ----
 # When volumes are not normalized (i.e., V is not equal to V/V0), then min(V)>1.
 # Therefore, I will normalize the volumes by V0 = max(V) for F-f fit purposes only.
 # The problem with this normalization is that when V=V0, then f(V0)=0 by definition and F(V0) becomes undefined (division by zero).
 #  - I solve this by ignoring F[0] and considering only the rest of the data, V=V[1:], P=P[1:] (ignoring the largest volume max(V)=V0 point)
 # In addition, since I usually fit T>0 isotherms, I have P(V0) = min(P) > 0. Thus, the expansion
 # P=3K0*f(1+2f)^(5/2)(1+2xi*f+4*zeta*f^2+...)  and F=K0*(1+2xi*f+4*zeta*f^2+...)
 # does not make much sense because P(f=0)= 0. But it does make sense if I treat P as thermal pressure, P-P[0]=Pth.
 #  - Therefore, I shift down all pressures P by P_shift=P[0], so Pth=P-P[0] looks like a zero-Kelvin isotherm
 #    that satisfies Pth(V0) = 0 by definition, so I force the fitted pressure to satisfy P(V0)= min(P) = P[0] 
 #    and the F-f fit is done over Pth[1:]
 V0 = 1.0*max(V_org) 
 #V0 = 59.14
 V=V_org[1:]
 if not isinstance(dV,float): dV = dV_org[1:]
 P_shift = P_org[0]
 P=P_org[1:] - P_shift    # ~Thermal P
 dP=dP_org[1:]
f = 0.5*( (V0/V)**(2.0/3) -1 )
F = P / ( 3*f * (1+2*f)**(5.0/2)  )
#dF = abs(F/P)*dP

dV0=0.00   # No error in the measurement of V0, but here for consistency
dfdV0=  (V0/V)**(2.0/3)/(3*V0)
dfdV = -(V0/V)**(2.0/3)/(3*V)
df2 = (dfdV*dV)**2 + (dfdV0*dV0)**2
dFdf = -( F/f + 5*F/(1+2*f) ) 
dFdP = F/P
dF = sqrt( (dFdP*dP)**2 + (dFdf)**2*df2  )


F_f_deg = 2 if BM_deg ==4 else 1
try:
 coeffs, cov_matrix = np.polyfit(f, F, F_f_deg, w=1/dF, cov=True)
except:
 coeffs             = np.polyfit(f, F, F_f_deg, w=1/dF, cov=False)
 cov_matrix = zeros(len(coeffs))
#coeffs, cov_matrix = np.polyfit(f, F, F_f_deg, cov=True)
#print("Coefficients F vs f",coeffs)
def linearF(x, a,b): return  a*x+b
popt_Ff, pcov_Ff= curve_fit(linearF, f, F,  sigma=dF) #, absolute_sigma=True) #, maxfev=10000)
#print("Curve fit Coefficients F vs f:",popt_Ff)

errors = sqrt(diag(cov_matrix)) 
if len(V)==2:
 errors = [0, 0]
 cov_matrix = [0, 0]
K0 = coeffs[-1]  # The last one in polyfit is the x**0 coefficient [F(x) = F[0] * f**deg + ... + F[deg]]
K0E = errors[-1]
#xi = (3/4)*(K0p-4)                              --> K0p = xi/(3/4) + 4 = [a1/(2K0)]/(3/4) + 4
K0p = coeffs[-2]/(2*K0*3/4.0 ) + 4               # Because a1 = +2 K0 xi = (+2 K0) (3/4)(K0'-4) in F=  a0 + a1 f, with a0= K0
K0pE= abs(K0p-4)*sqrt( (errors[-2]/coeffs[-2])**2 + (errors[-1]/coeffs[-1])**2 ) 
if BM_deg==4:
 #zeta = (3/8)*[K0 K0pp + K0p*(K0p-7) + 143/9]    --> K0pp= [ zeta/(3/8) - 143/9 - K0p*(K0p-7) ]/K0 = [ [a2/(4K0)]/(3/8) - 143/9 - K0p*(K0p-7) ]/K0
 K0pp = ( (coeffs[-3]/(4*K0))/(3/8) - 143/9 - K0p*(K0p-7) )/K0
 K0ppE = 0.0
F_vs_f_fit = lambda fi: np.poly1d(coeffs)(fi) 
if F_f_deg==1:
 dF_vs_f_fit = lambda fi: sqrt( (fi*errors[0])**2 + errors[1]**2 )
elif F_f_deg==2:
 dF_vs_f_fit = lambda fi: sqrt( (fi*fi*errors[0])**2 + (fi*errors[1])**2 + errors[2]**2 )
ff = linspace(0*min(f), max(f))


def P_Ff(v):
 f = 0.5*( (V0/v)**(2.0/3) -1 )
 pp= F_vs_f_fit(f) * ( 3*f * (1+2*f)**(5.0/2)  ) + P_shift
 return pp
def dP_Ff(v):
 f = 0.5*( (V0/v)**(2.0/3) -1 )
 return ( P_Ff(v)/F_vs_f_fit(f) ) * dF_vs_f_fit(f)


if show_F_plot:
 #rcParams.update(params)
 fig3 = figure(3)
 ax3 = subplot(211)
 ax3.set_xlabel(r'$f= \frac{1}{2}[ (V_0/V)^{2/3} -1 ]$')
 ax3.set_ylabel(r'$F= P/[ 3f \; (1+2f)^{5/2}  ] $')
 ax3.plot( ff, F_vs_f_fit(ff), 'r--' , label='Weighted fit ($w_i=1/\delta F_i$)')
 #ax3.plot( ff, linearF(ff, *popt_Ff), 'k--' , label='curve fit')
 FFerr = dF_vs_f_fit(ff)
 ax3.fill_between(ff, F_vs_f_fit(ff) - FFerr, F_vs_f_fit(ff) + FFerr, color='red', zorder=-1, alpha=0.1)
 ax3.errorbar(f, F, dF , marker='o', ls='-',c='b',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$F(f)$')
 ax3.legend(loc='best')
 ax3.set_xlim(0,1.1*max(f))

 ax4 = subplot(212)
 dfdV_org = -(V0/V_org)**(2.0/3)/(3*V_org)
 dfdV0_org=  (V0/V_org)**(2.0/3)/(3*V0)
 df2_org = (dfdV_org*dV_org)**2 + (dfdV0_org*dV0)**2
 #ax4.errorbar(1-V_org/V0, P_org, yerr=dP_org, xerr=abs(1/dfdV_org)*sqrt(df2_org)/V0, marker='o', ls='',c='b',ms= 8, capsize= 6, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$')
 ax4.errorbar(1-V_org/V0, P_org, yerr=dP_org, xerr=sqrt( dV_org**2+(V_org*dV0/V0)**2 )/V0, marker='o', ls='',c='b',ms= 8, capsize= 6, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$P(V)$')
 #ax4.plot(1-vs/V0, P_BM(vs), 'k-', label='$P(V)$ BM fit')
 ax4.plot(1-vs/V0, P_Ff(vs), 'm-', dashes=[5,1,1,1], lw=2, label='$P_{F-f}(V)$ fit')
 ax4.plot(1-vs/V0, 0*vs, 'k--')
 ax4.fill_between(1-vs/V0, P_Ff(vs) - dP_Ff(vs), P_Ff(vs) + dP_Ff(vs), color='red', zorder=-1, alpha=0.1)
 ax4.legend(loc='best')
 ax4.set_xlabel(r'$1-V/V_0$')
 ax4.set_ylabel(r'Pressure (GPa)')
 subplots_adjust(hspace=0.35)
 ##ax3.set_ylim(0,440)
 ##ax3.set_xlim(0,max(1-vs/V0))
 ##ax3.set_ylim(0,1.1*max(P))


#fig5 = figure('Z=PV/NkT')
#ax5 = subplot(111)
#GPaA3_to_eV = 0.0062415091
#N = 1
#T = 300
#Z = P_org*V_org*GPaA3_to_eV/(N*kB*T)
#ax5.plot(V_org, Z, label='$Z=PV/NkT$')
#ax5.plot(V_org, 0*Z+1,'--')

#F_spline = InterpolatedUnivariateSpline(f,F)
#F_wspline = InterpolatedUnivariateSpline(f,F, w=1/dF**2)
#dF_spline = [ SplineError(f,F,dF,F_spline, fi) for fi in ff  ]
#ax3.errorbar(ff, F_vs_f_fit(ff), dF_spline, marker='.', ls='-',c='r',ms=10, capsize=10, mfc='lightblue', mec='b', mew=2, zorder=5,label=r'$F(f)$')



V = V_org
P = P_org
dP = dP_org
ax.plot(vs, P_Ff(vs), '-', c='orange',dashes=[5,2,1,1], lw=2, label='$P(V)$ $F$-$f$ fit')
ff = 0.5*( (V0/vs)**(2.0/3) -1 )
FFerr = sqrt( (errors[0]*ff)**2 + errors[1]**2 ) 
dPs = FFerr * P_Ff(vs) /F_vs_f_fit(ff)
#ax.fill_between(vs,  P_Ff(vs)-dPs,  P_Ff(vs)+dPs, color='orange',zorder=-1, alpha=0.1 )

ax2.plot(V, (P_Ff(V) -P), ls='-', color='orange', marker='D', ms= 5, mfc='orange', mec='k', mew=1, zorder=10, label='$P_{F-f}(V)$')
if BM_deg==4:
 print ( "F-f fit:      V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f  K0pp= %7.4f %7.4f  %s"  % ( V0    , K0, K0E, K0p, K0pE, K0pp, K0ppE, "  # V0 as param" ) )
else:
 print ( "F-f fit:      V0[Å^3]= %9.4f            K0[GPa]= %9.4f %7.4f  K0p= %7.4f %7.4f %s"  % ( V0    , K0, K0E, K0p, K0pE, "  # V0 as param" ) )


ax2.set_xlabel("Volume ($\AA^3$)")
ax2.set_ylabel(r"$P_{\rm fit}-P_{\rm data}$ (GPa)")
#ax2.set_ylim(2*min(P_Vinet(V)-P),2.1*max(P_Vinet(V)-P))
ymax = max(abs(P_BM(V)-P)) if max(abs(P_BM(V)-P)) > max(dP) else max(dP)
ax2.set_ylim(-3*ymax,3*ymax)
ax2.set_xlim(min(V) - dv,max(V) + dv)
ax.legend()
ax2.legend(loc=1)




### REPORT ###
if len(V)>3:
 predictors = ['BM',  'Vinet',  'loglog' ]
 P_fit = { 'BM': P_BM, 'Vinet': P_Vinet, 'loglog': P_loglogfit}
 Nparams = { 'BM': len(popt_BM),  'Vinet': len(popt_Vinet), 'loglog': 4}
elif len(V)==3:
 predictors = ['BM',  'Vinet' ]
 P_fit = { 'BM': P_BM, 'Vinet': P_Vinet}
 Nparams = { 'BM': len(popt_BM),  'Vinet': len(popt_Vinet), 'loglog': 4}
else:
 print("\n Can't fit with just 2 points... bye.")
 exit()
if len(dP)==4: Nparams['loglog'] = 3 # avoid dividing by 0
predictors  += ['F-f']
P_fit['F-f'] = P_Ff
Nparams['F-f'] = 2
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
sorted_predictors = sorted(predictors, key=lambda p: chi_squared[p])

print("\nRoot Mean Square Error of each fit:")
for p in sorted_predictors:
 print("FIT: %-9s  RMSE_P[GPa]= %9.6f  std(residuals)= %9.6f  R2=  %10.8f  chi^2=  %10.8f" % (p, RMSE[p], sigma[p], R2[p], chi_squared[p]) )






#----------------------------------------------#
#  PRINT & PLOT INTERPOLATED TABLE AND         #
#  plot the fbv, spline BM & Vinet curves      #
#  when the -p flag is active                  #
#----------------------------------------------#
if print_table:
 print("\n# PRINTING TABLE OF INTERPOLATED VOLUMES")
 vs = linspace(min(V), max(V), 500)
 sorted_indices = P_BM(vs).argsort()
 spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs)[sorted_indices], vs[sorted_indices])  # V(P)
 spl_V_Ff    = InterpolatedUnivariateSpline(    P_Ff(vs)[sorted_indices], vs[sorted_indices])  # V(P)
 spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs)[sorted_indices], vs[sorted_indices])  # V(P)
 #p_list = arange(10,500,10)
 #p_list = arange(1e-10,201,1)
 Np = 40
 p_list = linspace(1e-10,max(ps), Np)
 print("# Plotting the predicted volumes for",Np,"pressures equally spread on the interval:")
 ax.plot(spl_V_BM(p_list), p_list , 'o', mfc='None', mec='r', ms=8, mew=2, label=r'$V_{\rm BM}(P)$ inv')
 ax.plot(spl_V_Vinet(p_list), p_list , 's', mfc='None', mec='limegreen', ms=8, mew=2, label=r'$V_{\rm Vinet}(P)$ inv')
 ax.legend()
 # Weighted fit by Burkhard fbv
 if fbv_exists:
  vols_fbv = [  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True ))  for p in p_list ]
  ax.plot(vols_fbv, p_list , '*', mfc='w', mec='purple', ms=12, mew=2,label='fbv')
  for j,p in enumerate(p_list):
   print ("P[GPa]=  %9.2f  V_BM[Å^3]=  %9.4f  V_Vinet[Å^3]=  %9.4f  V_loglog[Å^3]=  %9.4f  V_fbv[Å^3]=  %9.4f  (V_Vinet-V_BM)/V_Vinet[%%]= %7.4f" % (p, spl_V_BM(p), spl_V_Vinet(p), V_loglogfit(p), vols_fbv[j], (1-spl_V_BM(p)/spl_V_Vinet(p))*100 )  )
 else:
  for j,p in enumerate(p_list):
   print ("P[GPa]=  %9.2f  V_BM[Å^3]=  %9.4f  V_Ff[Å^3]=  %9.4f  V_Vinet[Å^3]=  %9.4f  V_loglog[Å^3]=  %9.4f  (V_Vinet-V_BM)/V_Vinet[%%]= %7.4f" % (p, spl_V_BM(p), spl_V_Ff(p), spl_V_Vinet(p), V_loglogfit(p), (1-spl_V_BM(p)/spl_V_Vinet(p))*100 )  )



#----------------------------------------------#
#  P_TARGET                                    #
#  Providing V(P_Target) for each fit          #
#  when the --PTarget flag is active           #
#----------------------------------------------#
if PTarget>0:
 print("\nVolume at P_Target")
 p = PTarget
 V_BM = 0.0
 V_Ff = 0.0
 V_Vinet = 0.0
 V_BMerr = 0.0
 V_Fferr = 0.0
 V_Vinet_err = 0.0
 V_spline_err  = abs(V_spline.derivative()(PTarget)) * SplineError(V[::-1],P[::-1],dP[::-1],P_spline, V_spline(PTarget))
 if len(V)>3: # Can't do a log-log fit with just 3 points
  V_loglog_err  = V_loglog_err(PTarget) 
 #V_loglog_err  = V_loglogfit(PTarget) * sqrt( da**2 + (lnP*db)**2 + (lnP*lnP*dc)**2 + (lnP*lnP*lnP*dd)**2 ) 
 '''
 A model of two parameters, say P(V)= a*V**b, will have error bars of da and db in the covariance matrix: 
   Perr = da,db = np.sqrt(np.diag(pcov))         <-- errors in each of the fitting parameters
 Then, for a given V0, the error in P(V0) is given by
   dP = sqrt( [(∂P/∂a)da]^2 + [(∂P/∂b)db]^2 )    <-- with derivatives evaluated at V0
 Since propagation of errors tells us that
   dP = |(∂P/∂V)| dV
 we can calculate the associated error in calculating V(P) using
   dV = dP/ |(∂P/∂V)| = |(∂V/∂P)| dP
 '''
 try:
  sorted_indices = P_BM(vs).argsort()
  spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs)[sorted_indices], vs[sorted_indices])  # V(P)
  sorted_indices = P_Ff(vs).argsort()
  spl_V_Ff    = InterpolatedUnivariateSpline(    P_Ff(vs)[sorted_indices], vs[sorted_indices])  # V(P)
  V_BM = spl_V_BM(PTarget)
  V_Ff = spl_V_Ff(PTarget)
  V_BMerr = abs(spl_V_BM.derivative()(PTarget)) * dP_BM(V_BM) 
  V_Fferr = abs(spl_V_Ff.derivative()(PTarget)) * dP_Ff(V_Ff) 
 except:
  pass 

 try:
  sorted_indices = P_Vinet(vs).argsort()
  spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs)[sorted_indices], vs[sorted_indices])  # V(P)
  V_Vinet = spl_V_Vinet(PTarget)
  V_Vinet_err = abs(spl_V_Vinet.derivative()(PTarget)) * dP_Vinet(V_Vinet) 
 except:
  pass
 print ("P_Target[GPa]=  %9.2f  V_BM[Å^3]=      %9.4f ± %6.4f" % (PTarget, V_BM, V_BMerr)    )
 print ("P_Target[GPa]=  %9.2f  V_F-f[Å^3]=     %9.4f ± %6.4f" % (PTarget, V_Ff, V_Fferr)    )
 print ("P_Target[GPa]=  %9.2f  V_Vinet[Å^3]=   %9.4f ± %6.4f" % (PTarget, V_Vinet, V_Vinet_err) )
 if len(V)>3:
  print ("P_Target[GPa]=  %9.2f  V_loglog[Å^3]=  %9.4f ± %6.4f" % (PTarget, V_loglogfit(PTarget), V_loglog_err) )
 print ("P_Target[GPa]=  %9.2f  V_spline[Å^3]=  %9.4f ± %6.4f" % (PTarget, V_spline(PTarget), V_spline_err) )
 if fbv_exists:
  v_fbv =  float(subprocess.check_output(fbv_path +' '+ filename + ' ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(p) + " | awk '/NewV/{print $NF}' ", shell=True )) 
  print ("P_Target[GPa]=  %9.2f  V_fbv[Å^3]=     %9.4f" % (PTarget, v_fbv) )

 xx = linspace(0, V_BM)
 ax.plot(xx, 0*xx + PTarget, '--', c='grey')
 yy = linspace(0, PTarget)
 ax.plot(0*yy+V_BM, yy, '--', c='grey')

 #def integrand_with_error(P):  return (V_spline.derivative()(P) * dP_spline(P))**2
 #dGInt, _ = quad(V_spline, 148.878, PTarget)
 #dGIntE = np.sqrt(quad(integrand_with_error, 148.878, PTarget)[0])
 #print("dGInt=",dGInt*0.00022937123,dGIntE*0.00022937123)
 
 if V_BM != 0.0:
  PBest = min(P, key=lambda p: abs(p-PTarget))
  print("PBest[GPa]= %9.1f"% (PBest) )
  dGInt, _ = quad( spl_V_BM , P1, PTarget)
  print("Integral from P1[GPa]= %6.2f to P_Target[GPa]= %6.2f:  ∆G[eV]= %14.12f   ∆G[Ha]= %14.12f  " % (P1,PTarget, dGInt*GPaA3_to_eV, dGInt*GPaA3_to_eV/Ha_to_eV) )
  if P1>0:
   yy = linspace(P1,PTarget)
   ax.fill_betweenx( yy, 0*yy , 0*yy + spl_V_BM(yy), color='r',alpha=0.3)
   ax.text( 0.45*(max(V)+min(V)), 0.5*(PTarget+P1), r'$\int V dP$')
 

 
 




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
  #*** BIRCH MURNAGHAN ****#
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
  #*******  VINET  ********#
  #------------------------#
  initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
  p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, max(V),K0,K0p)
  popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
  P_Vinet = lambda v: p_Vinet(v, *popt_Vinet)
  #print ("Vinet V0,K0, K0p: ", max(V), popt_Vinet)

  #------------------------#
  #*****  LOG-LOG  ********#
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
  sorted_indices = V_loglogfit(pp).argsort()
  P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp)[sorted_indices], pp[sorted_indices]) #P(V)



  command = "grep -v '^#' " +filename+ "| awk 'NR!= " +str(k)+ "+1{print }' > tmp.tmp"
  subprocess.check_output(command, shell=True)
  command = "~/scripts/fbv " + ' tmp.tmp ' + str(colV+1) + ' ' + str(colP+1) + ' ' + str(colPE+1) + ' ' + str(P0) + " | awk '/NewV/{print $NF}' "
  v_fbv = 0
  if fbv_exists: v_fbv = float(subprocess.check_output(command, shell=True))

  V_BM = 0.0
  V_Vinet = 0.0
  try:
   sorted_indices = P_BM(vs).argsort()
   spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs)[sorted_indices], vs[sorted_indices])  # V(P)
   spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs)[sorted_indices], vs[sorted_indices])  # V(P)
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

  sorted_vlist = sorted(vlist.items(), key=lambda item: abs(item[1]))
  #print ("Best:", sorted_vlist[0][0], "Worst:", sorted_vlist[-1][0])
  sorted_vlist = sorted(vlist.items(), key=lambda item: abs(item[1]))
  #print ("Best:", sorted_vlist[0][0], "Worst:", sorted_vlist[-1][0])
  counter_best[sorted_vlist[0][0]] += 1 
  counter_worst[sorted_vlist[-1][0]] += 1 

  if fbv_exists:
   print ("P0[GPa]=  %7.2f  V0[Å^3]=  %7.2f  V_BM[Å^3]=  %7.2f  V_Vinet[Å^3]=  %7.2f  V_loglog[Å^3]=  %7.2f  V_fbv[Å^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f  V_fbv_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0), v_fbv,   V_BM_err, V_Vinet_err, V_loglogfit_err, V_fbv_err )  )
  else:
   print ("P0[GPa]=  %7.2f  V0[Å^3]=  %7.2f  V_BM[Å^3]=  %7.2f  V_Vinet[Å^3]=  %7.2f  V_loglog[Å^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0),   V_BM_err, V_Vinet_err, V_loglogfit_err)  )

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
  #*** BIRCH MURNAGHAN ****#
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
  #*******  VINET  ********#
  #------------------------#
  initial_guess = (k0, k0p)  # Initial guess of Vinet parameters K0, K0p
  p_Vinet  = lambda v,K0,K0p: min(P) + VinetPressure(v, max(V),K0,K0p)
  popt_Vinet, pcov_Vinet = curve_fit(p_Vinet, V, P, p0=initial_guess, sigma=dP, absolute_sigma=True) #, maxfev=1000000)
  P_Vinet = lambda v: p_Vinet(v, *popt_Vinet)
  #print ("Vinet V0,K0, K0p: ", max(V), popt_Vinet)


  #------------------------#
  #*****  LOG-LOG  ********#
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
  sorted_indices = V_loglogfit(pp).argsort()
  P_loglogfit = InterpolatedUnivariateSpline(V_loglogfit(pp)[sorted_indices], pp[sorted_indices]) #P(V)
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
   sorted_indices = P_BM(vs).argsort()
   spl_V_BM    = InterpolatedUnivariateSpline(    P_BM(vs)[sorted_indices], vs[sorted_indices])  # V(P)
   spl_V_Vinet = InterpolatedUnivariateSpline( P_Vinet(vs)[sorted_indices], vs[sorted_indices])  # V(P)
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

  #print ("P0[GPa]=  %7.2f  V0[Å^3]=  %7.2f  V_BM[Å^3]=  %7.2f  V_Vinet[Å^3]=  %7.2f  V_loglog[Å^3]=  %7.2f  V_fbv[Å^3]=  %7.2f  V_BM_err[%%]= %6.3f  V_Vinet_err[%%]= %6.3f  V_loglog_err[%%]= %6.3f  V_fbv_err[%%]= %6.3f"  % (P0, V0, spl_V_BM(P0),  spl_V_Vinet(P0), V_loglogfit(P0), v_fbv,   V_BM_err, V_Vinet_err, V_loglogfit_err, V_fbv_err )  )
  print ("P0[GPa]=  %7.2f  V0[Å^3]=  %7.2f  V_BM[Å^3]=  %7.2f  V_Vinet[Å^3]=  %7.2f  V_loglog[Å^3]=  %7.2f  V_BM_err[%%]= %7.3f  V_Vinet_err[%%]= %7.3f  V_loglog_err[%%]= %7.3f"  % (P0, V0, V_BM,  V_Vinet, V_loglogfit(P0),   V_BM_err, V_Vinet_err, V_loglogfit_err )  )

 print ("BEST SCORES:  ", counter_best, "out of", trials)
 print ("WORST SCORES: ", counter_worst, "out of", trials)
 overall_scores = {predictor: counter_best[predictor] - counter_worst[predictor] for predictor in predictors}
 best_predictor = max(overall_scores, key=overall_scores.get)
 print ("Overall scores:", overall_scores)
 print ("Best predictor two-points test:", best_predictor)





#geometry = plt.get_current_fig_manager().window.geometry()
## Extract xmin, xmax, ymin, ymax from the QRect object
#xmin = geometry.x()
#xmax = geometry.x() + geometry.width()
#ymin = geometry.y()
#ymax = geometry.y() + geometry.height()
#fig.canvas.manager.window.setGeometry(100, 100, xmax, ymax)  # Set position for fig
#fig2.canvas.manager.window.setGeometry(100+xmax+50, 100, xmax, ymax)  # Set position for fig2

#savefig('Figure2.png')
figure('Pressure vs. Volume')  # Just bring it to the front
#savefig('Figure1.png')
if show_plots: show()
