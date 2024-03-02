# fit_EOS
Fitting Birch Murnaghan, Vinet, and log-log curves to P-V data

## Fitting Equations of State using Vinet / Birch-Murnahan / log-log to P-V data
The Birch-Murnhagan EOS of 3rd order is given by

$$    P (V)= \frac{3}{2} K_0 \left(\left(\frac{V_0}{V}\right)^{7/3} -         \left(\frac{V_0}{V}\right)^{5/3}\right)   \left(1 + \frac{3}{4}\left(K_0'-4\right)(\left(\frac{V_0}{V}\right)^{2/3}-1)\right)$$ 

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

 Usage: ./fit_EOS.py  /home/fgonzalez/EOS_Fe_sol_6000K.dat
 Usage: ./fit_EOS.py  /home/fgonzalez/EOS_Fe_sol_6000K.dat   V[A^3]-col P[GPa]-col P_error-col
 Usage: ./fit_EOS.py  /home/fgonzalez/EOS_Fe_sol_6000K.dat       6         12          13
 Usage: ./fit_EOS.py   ... -p          ... --> print V(P)
 Usage: ./fit_EOS.py   ... --test      ... --> deleting-points performance test
 Usage: ./fit_EOS.py   ... --noplots

 No arguments assumes V[A^3]-col= 6, P[GPa]-col= 12,  P_error-col= 13
```

## Examples
<img src="https://github.com/fgonzcat/fit_EOS/blob/main/PV_isotherm.png?raw=true" alt="Alt text" width="600">
<img src="https://github.com/fgonzcat/fit_EOS/blob/main/Pdiff_vs_V.png?raw=true" alt="Alt text" width="600">

