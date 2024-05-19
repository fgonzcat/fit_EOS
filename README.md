# fit_EOS
Fitting Birch Murnaghan, Vinet, and log-log curves to P-V data

## Fitting Equations of State using Vinet / Birch-Murnahan / log-log to P-V data
The Birch-Murnhagan EOS of 3rd order is given by

$$    P (V)= \frac{3}{2} K_0 \left(\left(\frac{V_0}{V}\right)^{7/3} -         \left(\frac{V_0}{V}\right)^{5/3}\right)   \left(1 + \frac{3}{4}\left(K_0'-4\right)\left(\left(\frac{V_0}{V}\right)^{2/3}-1\right)\right)$$ 

and the Birch-Murnhagan EOS of 4th order is given by

$$    P (V)= \frac{3}{2} K_0 \left[\left(\frac{V_0}{V}\right)^{7/3} -  \left(\frac{V_0}{V}\right)^{5/3}\right]   \left[1 + \frac{3}{4}\left(K_0'-4\right)\left(\left(\frac{V_0}{V}\right)^{2/3}-1\right)  + \frac{1}{24}\left(9{K_0'}^2-63K_0'+9K_0K_0''+143\right)\left(\left(\frac{V_0}{V}\right)^{2/3}-1\right)^2\right],$$ 

where $K_0\equiv K(V_0)$, $K_0'\equiv K'(P=0)$, and $K_0'' \equiv K''(P=0)$, with  $K'(P)\equiv \left(\frac{\partial K}{\partial P}\right)$ and  $K''(P)\equiv \left(\frac{\partial^2 K}{\partial P^2}\right)$. Note that the derivatives are taken with respect to pressure, not volume. In fact, $K'(P)= -K'(V)V/K(V)$.

The Vinet EOS is given by

$$P(V)= 3K_0 \left(\frac{1.0-\eta}{\eta^2}\right)  x e^{ \frac{3}{2}(K_0'-1)(1-\eta) };\qquad \eta=(V/V_0)^{1/3}$$

I also provide a new log-log polynomial EOS fit:
$$\ln V = a + b*\ln P + c*(\ln P)^2 + d*(\ln P)^3  \Rightarrow V(P) = {\rm e}^aP^{b+c\ln P+d(\ln P)^2}$$


### Reported Errors:
-   residuals = $P_{\rm fit}(V)-P$                                                                      
-   sigma     = std(residuals)
-   RMSE      = sqrt(mean(residuals)^2)
-   R2        =  1 - sum(residuals^2)
-   $\chi^2$ = sum(${\rm residuals}^2$ / $dP^2$)
  - $\chi^2 < 1$ ---> Line passes more than 2/3 of 1 sigma / sigma too big / Overfit
  - $\chi^2 > 1$ ---> Line misses more than 1/3 of 1 sigma / sigma too small / inadeq. model
  - $\chi^2 = 1$ ---> Line passes through 2/3 of 1 sigma & 95% of 2 sigma.
  - Model with $\chi^2$ closest to 1, wins.                                     
                                                                                               

## Executing
With no arguments, executing the code provides help:
```bash
./fit_EOS.py
```
```python
**** FITTING EOS ****
by Felipe Gonzalez [Berkeley 05-19-2023]



 This code fits an isotherm using Birch-Murnhagan, Vinet, and log-log [Berkeley 05-19-23]

 Usage: ./fit_EOS.py  EOS_Fe_sol_6000K.dat
 Usage: ./fit_EOS.py  EOS_Fe_sol_6000K.dat   V[A^3]-col P[GPa]-col P_error-col
 Usage: ./fit_EOS.py  EOS_Fe_sol_6000K.dat       6         12          13
 Usage: ./fit_EOS.py  EOS_H2O_liq_7000K.dat      6         12          13  --BM4 --V0-as-param
 Usage: ./fit_EOS.py   ... -p          ... --> print V(P)
 Usage: ./fit_EOS.py   ... --test      ... --> deleting-points performance test
 Usage: ./fit_EOS.py   ... --noplots       --> don't plot
 Usage: ./fit_EOS.py   ... --BM2           --> Birch-Murnaghan 2th order
 Usage: ./fit_EOS.py   ... --BM3           --> Birch-Murnaghan 3th order (default)
 Usage: ./fit_EOS.py   ... --BM4           --> Birch-Murnaghan 4th order
 Usage: ./fit_EOS.py   ... --V0-as-param   --> Treat V0 as another fitting parameter [ do not force the default P(V0) = P0, where P0=min(P), V0=max(P) ]

 No arguments assumes V[A^3]-col= 6, P[GPa]-col= 12,  P_error-col= 13
```

## Example: Equation of state of water at 7000 K
```bash
cat EOS_H2O_liq_7000K.dat
```

```
H2O001	72O+144H	N=	216	V[A^3]=	615.399662	rho[g/cc]=	3.5	T[K]=	7000	P[GPa]=	248.553	0.448	E[Ha]=	-14.990529	0.064178	t=	0.5	0.4
H2O002	72O+144H	N=	216	V[A^3]=	582.134815	rho[g/cc]=	3.7	T[K]=	7000	P[GPa]=	288.881	0.593	E[Ha]=	-13.635456	0.069719	t=	0.5	0.4
H2O003	72O+144H	N=	216	V[A^3]=	552.281748	rho[g/cc]=	3.9	T[K]=	7000	P[GPa]=	331.785	0.617	E[Ha]=	-12.290564	0.056398	t=	0.5	0.4
H2O004	72O+144H	N=	216	V[A^3]=	525.341176	rho[g/cc]=	4.1	T[K]=	7000	P[GPa]=	377.525	0.582	E[Ha]=	-10.942243	0.049417	t=	0.5	0.4
H2O005	72O+144H	N=	216	V[A^3]=	500.906701	rho[g/cc]=	4.3	T[K]=	7000	P[GPa]=	432.452	0.78	E[Ha]=	-8.923608	0.086511	t=	0.5	0.4
H2O006	72O+144H	N=	216	V[A^3]=	478.644181	rho[g/cc]=	4.5	T[K]=	7000	P[GPa]=	484.784	0.778	E[Ha]=	-7.411852	0.093474	t=	0.5	0.4
H2O007	72O+144H	N=	216	V[A^3]=	458.276344	rho[g/cc]=	4.7	T[K]=	7000	P[GPa]=	544.4  	0.767	E[Ha]=	-5.482341	0.081054	t=	0.5	0.4
H2O008	72O+144H	N=	216	V[A^3]=	439.571187	rho[g/cc]=	4.9	T[K]=	7000	P[GPa]=	606.946	0.763	E[Ha]=	-3.557931	0.076483	t=	0.5	0.4
H2O009	72O+144H	N=	216	V[A^3]=	422.333102	rho[g/cc]=	5.1	T[K]=	7000	P[GPa]=	675.005	0.921	E[Ha]=	-1.334811	0.096554	t=	0.5	0.4
H2O010	72O+144H	N=	216	V[A^3]=	406.396003	rho[g/cc]=	5.3	T[K]=	7000	P[GPa]=	748.008	0.993	E[Ha]=	0.888587	0.129627	t=	0.5	0.4
H2O011	72O+144H	N=	216	V[A^3]=	391.617966	rho[g/cc]=	5.5	T[K]=	7000	P[GPa]=	823.765	0.685	E[Ha]=	2.938162	0.093997	t=	0.5	0.4
```

```bash
./fit_EOS.py EOS_H2O_liq_7000K.dat 6 12 13
```

```python
**** FITTING EOS ****
by Felipe Gonzalez [Berkeley 05-19-2023]

BM fit:       V0[A^3]=  615.3997            K0[GPa]=  630.4528  4.6157  K0p[GPa]=  3.2871  0.0274   # Forcing P(V0)=P0 = min(P)
BM fit:       V0[A^3]= 4753.5151 1692.8541  K0[GPa]=    1.6525  1.6265  K0p[GPa]=  4.0657  0.0181   # V0 as param
Vinet fit:    V0[A^3]=  615.3997            K0[GPa]=  636.8488  5.1443  K0p[GPa]=  3.2062  0.0446   # Forcing P(V0)=P0 = min(P)
Vinet fit:    V0[A^3]= 1846.1990  952.8444  K0[GPa]=   16.5966 37.5304  K0p[GPa]=  5.4149  1.6295   # V0 as param

#EOS Data: EOS_H2O_liq_7000K.dat
i=  0  V[A^3]=  615.3997  P[GPa]=  248.5530
i=  1  V[A^3]=  582.1348  P[GPa]=  288.8810
i=  2  V[A^3]=  552.2817  P[GPa]=  331.7850
i=  3  V[A^3]=  525.3412  P[GPa]=  377.5250
i=  4  V[A^3]=  500.9067  P[GPa]=  432.4520
i=  5  V[A^3]=  478.6442  P[GPa]=  484.7840
i=  6  V[A^3]=  458.2763  P[GPa]=  544.4000
i=  7  V[A^3]=  439.5712  P[GPa]=  606.9460
i=  8  V[A^3]=  422.3331  P[GPa]=  675.0050
i=  9  V[A^3]=  406.3960  P[GPa]=  748.0080
i= 10  V[A^3]=  391.6180  P[GPa]=  823.7650

Root Mean Square Error of each fit:
FIT: BM         RMSE_P[GPa]=  1.653522  std(residuals)=  1.614589  R2=  0.99991781  chi^2=  7.24491905
FIT: Vinet      RMSE_P[GPa]=  1.432710  std(residuals)=  1.411193  R2=  0.99993829  chi^2=  4.76497275
FIT: loglog     RMSE_P[GPa]=  0.912606  std(residuals)=  0.912606  R2=  0.99997496  chi^2=  2.62694304
```

## Plots generated
<img src="https://github.com/fgonzcat/fit_EOS/blob/main/PV_isotherm.png?raw=true" alt="Alt text" width="600">
<img src="https://github.com/fgonzcat/fit_EOS/blob/main/Pdiff_vs_V.png?raw=true" alt="Alt text" width="600">

