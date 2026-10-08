import numpy as np
from skfuzzy import control as ctrl
import sympy as sp
from sympy import *
import time
np.random.seed(42)
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
ms = 500
mus = 40
ks = 10000
kt = 150000
cs = 1000


e_max = 0.025
de_max = 3.5
f_max = 900
ks_nl = 0.3 * ks


error = ctrl.Antecedent(np.linspace(-1, 1, 500), 'error')
delta_error = ctrl.Antecedent(np.linspace(-1, 1, 500), 'delta_error')

a1, b1, c1, m, c2, b2, a2, x1, z, h2, h3 = symbols('a1, b1, c1, m, c2, b2, a2, x1, z, h2, h3')



#a1<x<b1
b_1_1 = sp.integrate(z*((x1-a1)/(m-a1)-z*((x1-a1)/(m-a1)-h2 * (x1-b1)/(m-b1))),
                     (z,0,(x1*(m-b1)-a1*(m-b1))/(x1*((m-b1)+h2*(a1-m))-(a1*(m-b1)+b1*h2*(a1-m)))))
b_1_2 = sp.integrate(z,
                     (z,0,(x1*(m-b1)-a1*(m-b1))/(x1*((m-b1)+h2*(a1-m))-(a1*(m-b1)+b1*h2*(a1-m)))))

z_b_1_1_A2 = b_1_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_1_2_A2 = b_1_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})

z_b_1_1_A3 = b_1_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_1_2_A3 = b_1_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})

z_b_1_1_A4 = b_1_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_1_2_A4 = b_1_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})

z_b_1_1_A5 = b_1_1.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
z_b_1_2_A5 = b_1_2.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})

ZA2a1b1 = z_b_1_1_A2/(2*z_b_1_2_A2)
ZA2a1b1s = sp.simplify(ZA2a1b1)
ZA3a1b1 = z_b_1_1_A3/(2*z_b_1_2_A3)
ZA3a1b1s = sp.simplify(ZA3a1b1)
ZA4a1b1 = z_b_1_1_A4/(2*z_b_1_2_A4)
ZA4a1b1s = sp.simplify(ZA4a1b1)
ZA5a1b1 = z_b_1_1_A5/(2*z_b_1_2_A5)
ZA5a1b1s = sp.simplify(ZA5a1b1)

#b1<x<c1
b_2_1 = sp.integrate(z*((x1-a1)/(m-a1)-z*((x1-a1)/(m-a1)-h2*(x1-b1)/(m-b1))),
                     (z,0,1))
b_2_2_1 = sp.integrate((z),
                     (z,0,1))
z_b_2_1_A2 = b_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_2_2_1_A2 = b_2_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_2_1_A2 = z_b_2_1_A2/z_b_2_2_1_A2
zz_b_2_1_A2s = sp.simplify(zz_b_2_1_A2)

z_b_2_1_A3 = b_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_2_2_1_A3 = b_2_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_2_1_A3 = z_b_2_1_A3/z_b_2_2_1_A3
zz_b_2_1_A3s = sp.sympify(zz_b_2_1_A3)

z_b_2_1_A4 = b_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_2_2_1_A4 = b_2_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_2_1_A4 = z_b_2_1_A4/z_b_2_2_1_A4
zz_b_2_1_A4s = sp.sympify(zz_b_2_1_A4)

z_b_2_1_A5 = b_2_1.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
z_b_2_2_1_A5 = b_2_2_1.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
zz_b_2_1_A5 = z_b_2_1_A5/z_b_2_2_1_A5
zz_b_2_1_A5s = sp.sympify(zz_b_2_1_A5)

b_2_2 = sp.integrate(z*(h3*(x1-c1)/(m-c1)+z*(h2*(x1-b1)/(m-b1)-h3*(x1-c1)/(m-c1))),
                     (z,(x1*h3*(b1-m)-c1*h3*(b1-m))/(x1*(h3*(b1-m)+h2*(m-c1))-(c1*h3*(b1-m)+b1*h2*(m-c1))),1))
b_2_2_2 = sp.integrate((z),
                     (z,(x1*h3*(b1-m)-c1*h3*(b1-m))/(x1*(h3*(b1-m)+h2*(m-c1))-(c1*h3*(b1-m)+b1*h2*(m-c1))),1))

z_b_2_1_A22 = b_2_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_2_2_A22 = b_2_2_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_2_2_A22 = z_b_2_1_A22/z_b_2_2_A22
zz_b_2_2_A22s = sp.simplify(zz_b_2_2_A22)

z_b_2_1_A33 = b_2_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_2_2_A33 = b_2_2_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_2_2_A33 = z_b_2_1_A33/z_b_2_2_A33
zz_b_2_2_A33s = sp.simplify(zz_b_2_2_A33)

z_b_2_1_A44 = b_2_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_2_2_A44 = b_2_2_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_2_2_A44 = z_b_2_1_A44/z_b_2_2_A44
zz_b_2_2_A44s = sp.simplify(zz_b_2_2_A44)

z_b_2_1_A55 = b_2_2.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
z_b_2_2_A55 = b_2_2_2.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
zz_b_2_2_A55 = z_b_2_1_A55/z_b_2_2_A55
zz_b_2_2_A55s = sp.simplify(zz_b_2_2_A55)

ZA2b1c1s = 1/2 * (zz_b_2_1_A2s+zz_b_2_2_A22s)
ZA3b1c1s = 1/2 * (zz_b_2_1_A3s+zz_b_2_2_A33s)
ZA4b1c1s = 1/2 * (zz_b_2_1_A4s+zz_b_2_2_A44s)
ZA5b1c1s = 1/2 * (zz_b_2_1_A5s+zz_b_2_2_A55s)

#c1<x<m
b_3_1 = sp.integrate(z*((x1-a1)/(m-a1)-z*((x1-a1)/(m-a1)-h2*(x1-b1)/(m-b1))),
                     (z,0,1))
b_3_2_1 = sp.integrate((z),
                     (z,0,1))
z_b_3_1_A2 = b_3_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_3_2_1_A2 = b_3_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_3_1_A2 = z_b_3_1_A2/z_b_3_2_1_A2
zz_b_3_1_A2s = sp.simplify(zz_b_3_1_A2)

z_b_3_1_A3 = b_3_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_3_2_1_A3 = b_3_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_3_1_A3 = z_b_3_1_A3/z_b_3_2_1_A3
zz_b_3_1_A3s = sp.simplify(zz_b_3_1_A3)

z_b_3_1_A4 = b_3_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_3_2_1_A4 = b_3_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_3_1_A4 = z_b_3_1_A4/z_b_3_2_1_A4
zz_b_3_1_A4s = sp.simplify(zz_b_3_1_A4)

z_b_3_1_A5 = b_3_1.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
z_b_3_2_1_A5 = b_3_2_1.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
zz_b_3_1_A5 = z_b_3_1_A5/z_b_3_2_1_A5
zz_b_3_1_A5s = sp.simplify(zz_b_3_1_A5)

b_3_2 = sp.integrate(z*(h3*(x1-c1)/(m-c1)+z*(h2*(x1-b1)/(m-b1)-h3*(x1-c1)/(m-c1))),
                     (z,0,1))
b_3_2_2 = sp.integrate((z),
                     (z,0,1))

z_b_3_1_A22 = b_3_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_3_2_1_A22 = b_3_2_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_3_1_A22 = z_b_3_1_A22/z_b_3_2_1_A22
zz_b_3_1_A22s = sp.simplify(zz_b_3_1_A22)

z_b_3_1_A33 = b_3_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_3_2_1_A33 = b_3_2_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_3_1_A33 = z_b_3_1_A33/z_b_3_2_1_A33
zz_b_3_1_A33s = sp.simplify(zz_b_3_1_A33)

z_b_3_1_A44 = b_3_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_3_2_1_A44 = b_3_2_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_3_1_A44 = z_b_3_1_A44/z_b_3_2_1_A44
zz_b_3_1_A44s = sp.simplify(zz_b_3_1_A44)

z_b_3_1_A55 = b_3_2.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
z_b_3_2_1_A55 = b_3_2_2.subs({a1:0.25,b1:0.5,c1:0.75,m:1,c2:1.25,b2:1.5,a2:1.75,h2:0.6, h3:0.5})
zz_b_3_1_A55 = z_b_3_1_A55/z_b_3_2_1_A55
zz_b_3_1_A55s = sp.simplify(zz_b_3_1_A55)

ZA2c1ms = 1/2 * (zz_b_3_1_A2s + zz_b_3_1_A22s)
ZA3c1ms = 1/2 * (zz_b_3_1_A3s + zz_b_3_1_A33s)
ZA4c1ms = 1/2 * (zz_b_3_1_A4s + zz_b_3_1_A44s)
ZA5c1ms = 1/2 * (zz_b_3_1_A5s + zz_b_3_1_A55s)

#m<x<c2
b_4_1 = sp.integrate(z*((a2-x1)/(a2-m)-z*((a2-x1)/(a2-m)-h2*(b2-x1)/(b2-m))),
                     (z,0,1))
b_4_2_1 = sp.integrate((z),
                     (z,0,1))

z_b_4_1_A1 = b_4_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
z_b_4_2_1_A1 = b_4_2_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
zz_b_4_1_A1 = z_b_4_1_A1/z_b_4_2_1_A1
zz_b_4_1_A1s = sp.simplify(zz_b_4_1_A1)

z_b_4_1_A2 = b_4_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_4_2_1_A2 = b_4_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_4_1_A2 = z_b_4_1_A2/z_b_4_2_1_A2
zz_b_4_1_A2s = sp.simplify(zz_b_4_1_A2)

z_b_4_1_A3 = b_4_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_4_2_1_A3 = b_4_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_4_1_A3 = z_b_4_1_A3/z_b_4_2_1_A3
zz_b_4_1_A3s = sp.simplify(zz_b_4_1_A3)

z_b_4_1_A4 = b_4_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_4_2_1_A4 = b_4_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_4_1_A4 = z_b_4_1_A4/z_b_4_2_1_A4
zz_b_4_1_A4s = sp.simplify(zz_b_4_1_A4)

b_4_2 = sp.integrate(z*(h3*(c2-x1)/(c2-m)+z*(h2*(b2-x1)/(b2-m)-h3*(c2-x1)/(c2-m))),
                     (z,0,1))
b_4_2_2 = sp.integrate((z),
                     (z,0,1))

z_b_4_1_A11 = b_4_2.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
z_b_4_2_1_A11 = b_4_2_2.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
zz_b_4_1_A11 = z_b_4_1_A11/z_b_4_2_1_A11
zz_b_4_1_A11s = sp.simplify(zz_b_4_1_A11)

z_b_4_1_A22 = b_4_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_4_2_1_A22 = b_4_2_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_4_1_A22 = z_b_4_1_A22/z_b_4_2_1_A22
zz_b_4_1_A22s = sp.simplify(zz_b_4_1_A22)

z_b_4_1_A33 = b_4_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_4_2_1_A33 = b_4_2_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_4_1_A33 = z_b_4_1_A33/z_b_4_2_1_A33
zz_b_4_1_A33s = sp.simplify(zz_b_4_1_A33)

z_b_4_1_A44 = b_4_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_4_2_1_A44 = b_4_2_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_4_1_A44 = z_b_4_1_A44/z_b_4_2_1_A44
zz_b_4_1_A44s = sp.simplify(zz_b_4_1_A44)

ZA1mc2s = 1/2 * (zz_b_4_1_A1s + zz_b_4_1_A11s)
ZA2mc2s = 1/2 * (zz_b_4_1_A2s + zz_b_4_1_A22s)
ZA3mc2s = 1/2 * (zz_b_4_1_A3s + zz_b_4_1_A33s)
ZA4mc2s = 1/2 * (zz_b_4_1_A4s + zz_b_4_1_A44s)

#c2<x<b2
b_5_1 = sp.integrate(z*((a2-x1)/(a2-m)-z*((a2-x1)/(a2-m)-h2*(b2-x1)/(b2-m))),
                     (z,0,1))
b_5_2_1 = sp.integrate((z),
                     (z,0,1))
z_b_5_1_A1 = b_5_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
z_b_5_2_1_A1 = b_5_2_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
zz_b_5_1_A1 = z_b_5_1_A1/z_b_5_2_1_A1
zz_b_5_1_A1s = sp.simplify(zz_b_5_1_A1)

z_b_5_1_A2 = b_5_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_5_2_1_A2 = b_5_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_5_1_A2 = z_b_5_1_A2/z_b_5_2_1_A2
zz_b_5_1_A2s = sp.simplify(zz_b_5_1_A2)

z_b_5_1_A3 = b_5_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_5_2_1_A3 = b_5_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_5_1_A3 = z_b_5_1_A3/z_b_5_2_1_A3
zz_b_5_1_A3s = sp.simplify(zz_b_5_1_A3)

z_b_5_1_A4 = b_5_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_5_2_1_A4 = b_5_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_5_1_A4 = z_b_5_1_A4/z_b_5_2_1_A4
zz_b_5_1_A4s = sp.simplify(zz_b_5_1_A4)

b_5_2 = sp.integrate(z*(h3*(c2-x1)/(c2-m)+z*(h2*(b2-x1)/(b2-m)-h3*(c2-x1)/(c2-m))),
                     (z,(x1*h3*(b2-m)-c2*h3*(b2-m))/(x1*(h3*(b2-m)+h2*(m-c2))-(c2*h3*(b2-m)+b2*h2*(m-c2))),1))
b_5_2_2 = sp.integrate((z),
                     (z,(x1*h3*(b2-m)-c2*h3*(b2-m))/(x1*(h3*(b2-m)+h2*(m-c2))-(c2*h3*(b2-m)+b2*h2*(m-c2))),1))

z_b_5_1_A11 = b_5_2.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
z_b_5_2_A11 = b_5_2_2.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
zz_b_5_1_A11 = z_b_5_1_A11/z_b_5_2_A11
zz_b_5_1_A11s = sp.simplify(zz_b_5_1_A11)

z_b_5_1_A22 = b_5_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_5_2_A22 = b_5_2_2.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
zz_b_5_1_A22 = z_b_5_1_A22/z_b_5_2_A22
zz_b_5_1_A22s = sp.simplify(zz_b_5_1_A22)

z_b_5_1_A33 = b_5_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_5_2_A33 = b_5_2_2.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
zz_b_5_1_A33 = z_b_5_1_A33/z_b_5_2_A33
zz_b_5_1_A33s = sp.simplify(zz_b_5_1_A33)

z_b_5_1_A44 = b_5_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_5_2_A44 = b_5_2_2.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
zz_b_5_1_A44 = z_b_5_1_A44/z_b_5_2_A44
zz_b_5_1_A44s = sp.simplify(zz_b_5_1_A44)

ZA1c2b2s = 1/2 * (zz_b_5_1_A1s + zz_b_5_1_A11s)
ZA2c2b2s = 1/2 * (zz_b_5_1_A2s + zz_b_5_1_A22s)
ZA3c2b2s = 1/2 * (zz_b_5_1_A3s + zz_b_5_1_A33s)
ZA4c2b2s = 1/2 * (zz_b_5_1_A4s + zz_b_5_1_A44s)

#b2<x<a2
b_6_1 = sp.integrate(z*((a2-x1)/(a2-m)-z*((a2-x1)/(a2-m)-h2*(b2-x1)/(b2-m))),
                     (z,0,(x1*(m-b2)-a2*(m-b2))/(x1*((m-b2)+h2*(a2-m))-(a2*(m-b2)+b2*h2*(a2-m)))))
b_6_2_1 = sp.integrate((z),
                     (z,0,(x1*(m-b2)-a2*(m-b2))/(x1*((m-b2)+h2*(a2-m))-(a2*(m-b2)+b2*h2*(a2-m)))))

z_b_6_1_A1 = b_6_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
z_b_6_2_1_A1 = b_6_2_1.subs({a1:-1.75,b1:-1.5,c1:-1.25,m:-1,c2:-0.75,b2:-0.5,a2:-0.25,h2:0.6, h3:0.5})
ZA1b2a2 = 1/2 * (z_b_6_1_A1/z_b_6_2_1_A1)
ZA1b2a2s = sp.simplify(ZA1b2a2)

z_b_6_1_A2 = b_6_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
z_b_6_2_1_A2 = b_6_2_1.subs({a1:-1.25,b1:-1,c1:-0.75,m:-0.5,c2:-0.25,b2:0,a2:0.25,h2:0.6, h3:0.5})
ZA2b2a2 = 1/2 * (z_b_6_1_A2/z_b_6_2_1_A2)
ZA2b2a2s = sp.simplify(ZA2b2a2)

z_b_6_1_A3 = b_6_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
z_b_6_2_1_A3 = b_6_2_1.subs({a1:-0.75,b1:-0.5,c1:-0.25,m:0,c2:0.25,b2:0.5,a2:0.75,h2:0.6, h3:0.5})
ZA3b2a2 = 1/2 *(z_b_6_1_A3/z_b_6_2_1_A3)
ZA3b2a2s = sp.simplify(ZA3b2a2)

z_b_6_1_A4 = b_6_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
z_b_6_2_1_A4 = b_6_2_1.subs({a1:-0.25,b1:0,c1:0.25,m:0.5,c2:0.75,b2:1,a2:1.25,h2:0.6, h3:0.5})
ZA4b2a2 = 1/2 * (z_b_6_1_A4/z_b_6_2_1_A4)
ZA4b2a2s = sp.simplify(ZA4b2a2)

fA1mc2 = sp.lambdify(x1, ZA1mc2s, modules=['numpy'])
fA1c2b2 = sp.lambdify(x1, ZA1c2b2s, modules=['numpy'])
fA1b2a2 = sp.lambdify(x1, ZA1b2a2s, modules=['numpy'])

fA2a1b1 = sp.lambdify(x1, ZA2a1b1s, modules=['numpy'])
fA2b1c1 = sp.lambdify(x1, ZA2b1c1s, modules=['numpy'])
fA2c1m = sp.lambdify(x1, ZA2c1ms, modules=['numpy'])
fA2mc2 = sp.lambdify(x1, ZA2mc2s, modules=['numpy'])
fA2c2b2 = sp.lambdify(x1, ZA2c2b2s, modules=['numpy'])
fA2b2a2 = sp.lambdify(x1, ZA2b2a2s, modules=['numpy'])

fA3a1b1 = sp.lambdify(x1, ZA3a1b1s, modules=['numpy'])
fA3b1c1 = sp.lambdify(x1, ZA3b1c1s, modules=['numpy'])
fA3c1m = sp.lambdify(x1, ZA3c1ms, modules=['numpy'])
fA3mc2 = sp.lambdify(x1, ZA3mc2s, modules=['numpy'])
fA3c2b2 = sp.lambdify(x1, ZA3c2b2s, modules=['numpy'])
fA3b2a2 = sp.lambdify(x1, ZA3b2a2s, modules=['numpy'])

fA4a1b1 = sp.lambdify(x1, ZA4a1b1s, modules=['numpy'])
fA4b1c1 = sp.lambdify(x1, ZA4b1c1s, modules=['numpy'])
fA4c1m = sp.lambdify(x1, ZA4c1ms, modules=['numpy'])
fA4mc2 = sp.lambdify(x1, ZA4mc2s, modules=['numpy'])
fA4c2b2 = sp.lambdify(x1, ZA4c2b2s, modules=['numpy'])
fA4b2a2 = sp.lambdify(x1, ZA4b2a2s, modules=['numpy'])

fA5a1b1 = sp.lambdify(x1, ZA5a1b1s, modules=['numpy'])
fA5b1c1 = sp.lambdify(x1, ZA5b1c1s, modules=['numpy'])
fA5c1m = sp.lambdify(x1, ZA5c1ms, modules=['numpy'])

def A1(x1):
    if -1.5 <= x1 <= -1:
        return 1
    elif -1 <= x1 <= -0.75:
        return fA1mc2(x1)
    elif -0.75 <= x1 <= -0.5:
        return fA1c2b2(x1)
    elif -0.5 <= x1 <= -0.25:
        return fA1b2a2(x1)
    else:
        return 0

def A2(x1):
    if x1 <= -1.25:
        return 0
    elif -1.25 <= x1 <= -1:
        return fA2a1b1(x1)
    elif -1 <= x1 <= -0.75:
        return fA2b1c1(x1)
    elif -0.75 <= x1 <= -0.5:
        return fA2c1m(x1)
    elif -0.5 <= x1 <= -0.25:
        return fA2mc2(x1)
    elif -0.25 <= x1 <= 0:
        return fA2c2b2(x1)
    elif 0 <= x1 <= 0.25:
        return fA2b2a2(x1)
    else:
        return 0

def A3(x1):
    if x1 <= -0.75:
        return 0
    elif -0.75 <= x1 <= -0.5:
        return fA3a1b1(x1)
    elif -0.5 <= x1 <= -0.25:
        return fA3b1c1(x1)
    elif -0.25 <= x1 <= 0:
        return fA3c1m(x1)
    elif 0 <= x1 <= 0.25:
        return fA3mc2(x1)
    elif 0.25 <= x1 <= 0.5:
        return fA3c2b2(x1)
    elif 0.5 <= x1 <= 0.75:
        return fA3b2a2(x1)
    else:
        return 0

def A4(x1):
    if x1 <= -0.25:
        return 0
    elif -0.25 <= x1 <= 0:
        return fA4a1b1(x1)
    elif 0 <= x1 <= 0.25:
        return fA4b1c1(x1)
    elif 0.25 <= x1 <= 0.5:
        return fA4c1m(x1)
    elif 0.5 <= x1 <= 0.75:
        return fA4mc2(x1)
    elif 0.75 <= x1 <= 1:
        return fA4c2b2(x1)
    elif 1 <= x1 <= 1.25:
        return fA4b2a2(x1)
    else:
        return 0

def A5(x1):
    if x1 <= 0.25:
        return 0
    elif 0.25 <= x1 <= 0.5:
        return fA5a1b1(x1)
    elif 0.5 <= x1 <= 0.75:
        return fA5b1c1(x1)
    elif 0.75 <= x1 <= 1:
        return fA5c1m(x1)
    elif 1 <= x1 <= 1.25:
        return 1
    else:
        return 1

def safe(func, universe):
    vals = np.array([func(x) for x in universe])
    return np.clip(vals, 0, 1)


labels = ['NB', 'NS', 'ZE', 'PS', 'PB']

for var in [error, delta_error]:
    var['NB'] = safe(A1, var.universe)
    var['NS'] = safe(A2, var.universe)
    var['ZE'] = safe(A3, var.universe)
    var['PS'] = safe(A4, var.universe)
    var['PB'] = safe(A5, var.universe)

mf_dict = {
    'NB': A1,
    'NS': A2,
    'ZE': A3,
    'PS': A4,
    'PB': A5
}

singleton_output = {
    'NB': -1.0,
    'NS': -0.5,
    'ZE':  0.0,
    'PS':  0.5,
    'PB':  1.0
}

reduced_labels = ['NB', 'ZE', 'PB']
reduced_rule_table = [
    ['PB', 'PS', 'ZE'],
    ['PS', 'ZE', 'NS'],
    ['ZE', 'NS', 'NB']
]

def singleton_fuzzy_inference(e_val, de_val):
    num = 0.0
    den = 0.0

    for i, e_lbl in enumerate(reduced_labels):
        for j, de_lbl in enumerate(reduced_labels):
            out_lbl = reduced_rule_table[i][j]

            mu_e = float(np.clip(mf_dict[e_lbl](e_val), 0, 1))
            mu_de = float(np.clip(mf_dict[de_lbl](de_val), 0, 1))

            w = min(mu_e, mu_de)
            c = singleton_output[out_lbl]

            num += w * c
            den += w

    return 0.0 if den < 1e-12 else num / den

def road_input(t):
    if 1.0 <= t <= 1.2:
        return 0.05 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.0 <= t <= 5.5:
        return -0.01 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.5 <= t <= 6.5:
        return -0.07 * np.sin(np.pi * (t - 0.7) / 0.1)
    return 0


def spring_force(dx):
    return ks * dx + ks_nl * dx ** 3

def dynamics(t, state, fa):
    xs, dxs, xus, dxus = state
    xr = road_input(t)

    ddxs = (spring_force(xus - xs) + cs * (dxus - dxs) + fa) / ms
    ddxus = (spring_force(xs - xus) + cs * (dxs - dxus) + kt * (xr - xus) - fa) / mus

    return np.array([dxs, ddxs, dxus, ddxus])

dt = 0.01
T = 20
timeax = np.arange(0, T, dt)

state = np.zeros(4)
history = []
force_hist = []
acc_hist = []

start_time = time.perf_counter()
for t in timeax:
    e = state[0] / e_max
    de = (state[1] - state[3]) / de_max
    e_noisy = np.clip(e + np.random.normal(0, 0.08),-1, 1)
    de_noisy = np.clip(de + np.random.normal(0, 0.07)-1, 1)
    u = singleton_fuzzy_inference(e_noisy, de_noisy)
    fa = u * f_max
    k1 = dynamics(t, state, fa)
    k2 = dynamics(t+dt/2, state+dt*k1/2, fa)
    k3 = dynamics(t+dt/2, state+dt*k2/2, fa)
    k4 = dynamics(t+dt, state+dt*k3, fa)
    state = state + dt*(k1+2*k2+2*k3+k4)/6
    history.append(state.copy())
    force_hist.append(fa)
    acc_hist.append(k1[1])

history = np.array(history)
end_time = time.perf_counter()

total_simulation_time = end_time - start_time
print(f"\n RK4: {total_simulation_time:.4f} ")

def calculate_metrics_t2fs(acc_data, xs_data, force_data, t_axis):
    acc_arr = np.array(acc_data)
    xs_arr = np.array(xs_data)
    force_arr = np.array(force_data)

    rms_acc = np.sqrt(np.mean(np.square(acc_arr)))
    peak_acc = np.max(np.abs(acc_arr))
    iae_acc = _trapz(np.abs(acc_arr), t_axis)
    itae_acc = _trapz(t_axis * np.abs(acc_arr), t_axis)

    rms_disp = np.sqrt(np.mean(np.square(xs_arr)))
    peak_disp = np.max(np.abs(xs_arr))
    iae_disp = _trapz(np.abs(xs_arr), t_axis)
    itae_disp = _trapz(t_axis * np.abs(xs_arr), t_axis)

    rms_force = np.sqrt(np.mean(np.square(force_arr)))
    peak_force = np.max(np.abs(force_arr))

    return {
        'Acc': [rms_acc, peak_acc, iae_acc, itae_acc],
        'Disp': [rms_disp, peak_disp, iae_disp, itae_disp],
        'Force': [rms_force, peak_force]
    }

metrics_t2fs = calculate_metrics_t2fs(acc_hist, history[:, 0], force_hist, timeax)

print("First approach .")
print("\n" + "!" * 45)
print("   PERFORMANCE METRICS")
print("!" * 45)
print(f"Sprung Acc (RMS):    {metrics_t2fs['Acc'][0]:.4f} m/s²")
print(f"Sprung Acc (Peak):   {metrics_t2fs['Acc'][1]:.4f} m/s²")
print(f"Sprung Acc (IAE):    {metrics_t2fs['Acc'][2]:.4f}")
print(f"Sprung Acc (ITAE):   {metrics_t2fs['Acc'][3]:.4f}")
print("-" * 45)
print(f"Body Disp (RMS):     {metrics_t2fs['Disp'][0]:.4f} m")
print(f"Body Disp (Peak):    {metrics_t2fs['Disp'][1]:.4f} m")
print(f"Body Disp (IAE):     {metrics_t2fs['Disp'][2]:.4f}")
print(f"Body Disp (ITAE):    {metrics_t2fs['Disp'][3]:.4f}")
print("-" * 45)
print(f"Control Force (RMS):  {metrics_t2fs['Force'][0]:.2f} N")
print(f"Control Force (Peak): {metrics_t2fs['Force'][1]:.2f} N")
print("!" * 45)
