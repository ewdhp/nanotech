# lagrangian_demo.py
"""
Interactive demo:
 - Symbolically derive Euler-Lagrange for a mass-spring and a simple pendulum
 - Numerically integrate the nonlinear pendulum and plot theta(t) and phase space
Requires: sympy, numpy, scipy, matplotlib
Run: python lagrangian_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# -------------------------
# Symbolic derivation utils
# -------------------------
def deriv_euler_lagrange(L, qs, t):
    """
    Given L (sympy expr), qs = [q1(t), q2(t), ...] (sympy functions),
    return Euler-Lagrange equations as sympy expressions.
    """
    eqs = []
    for q in qs:
        dq = sp.diff(q, t)
        dL_dq = sp.diff(L, q)
        dL_ddq = sp.diff(L, dq)
        # total time derivative of dL_ddq
        ddt_dL_ddq = sp.diff(dL_ddq, t)
        # But sympy treats q(t) explicitly, so expand
        eq = sp.simplify(ddt_dL_ddq - dL_dq)
        eqs.append(sp.factor(eq))
    return eqs

# -------------------------
# Example 1: Mass-spring
# -------------------------
t = sp.symbols('t')
m, k = sp.symbols('m k', positive=True)
x = sp.Function('x')(t)
dx = sp.diff(x, t)

T_ms = sp.Rational(1,2) * m * dx**2
V_ms = sp.Rational(1,2) * k * x**2
L_ms = T_ms - V_ms

eqs_ms = deriv_euler_lagrange(L_ms, [x], t)
print("Mass-spring Euler-Lagrange (should be m x'' + k x = 0):")
sp.pprint(eqs_ms[0])   # shows the ODE

# -------------------------
# Example 2: Simple pendulum
# -------------------------
theta = sp.Function('theta')(t)
dtheta = sp.diff(theta, t)
l, g = sp.symbols('l g', positive=True)
# Kinetic energy: (1/2) m (l^2 theta_dot^2)
T_p = sp.Rational(1,2) * m * (l**2) * dtheta**2
# Potential energy: m g l (1 - cos(theta))  (zero at bottom)
V_p = m * g * l * (1 - sp.cos(theta))
L_p = T_p - V_p

eqs_p = deriv_euler_lagrange(L_p, [theta], t)
print("\nPendulum Euler-Lagrange (nonlinear):")
sp.pprint(eqs_p[0])  # will display m*l^2*theta'' + m*g*l*sin(theta) = 0

# Simplify to standard form (theta'')
# Solve eq for theta'':
theta_dd = sp.solve(sp.Eq(eqs_p[0], 0), sp.diff(theta, (t,2)) )[0]
print("\ntheta'' =")
sp.pprint(sp.simplify(theta_dd))

# -------------------------
# Numeric integration (pendulum)
# -------------------------
# Convert symbolic expression to numeric function for RHS
# We'll form system: y = [theta, omega]; theta' = omega; omega' = f(theta,omega)
f_rhs = sp.lambdify((theta, dtheta, m, l, g), theta_dd, 'numpy')

def pendulum_rhs(t, y, mval, lval, gval):
    th, w = y
    # here, theta_dd expression doesn't depend on omega for simple pendulum
    th_dd = f_rhs(th, w, mval, lval, gval)
    return [w, float(th_dd)]

# parameters & initial conditions
m_val = 1.0
l_val = 1.0
g_val = 9.81
y0 = [1.2, 0.0]       # initial theta (rad), initial omega

t_span = (0.0, 20.0)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

sol = solve_ivp(lambda tt, yy: pendulum_rhs(tt, yy, m_val, l_val, g_val),
                t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t [s]')
plt.ylabel(r'$\theta$ [rad]')
plt.title('Pendulum angle vs time')

plt.subplot(1,2,2)
plt.plot(sol.y[0], sol.y[1], linewidth=0.5)
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\dot\theta$ [rad/s]')
plt.title('Phase space (theta vs omega)')
plt.tight_layout()
plt.savefig('/home/ewd/github/ewdhp/nanotech/basics/euler_lagrange_pendulum.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved plot: euler_lagrange_pendulum.png")
plt.close()
