import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Given Parameters and Gains

# Gains
k = 1500
P = np.eye(3)*1500

# R matrix
R = np.eye(3)

# Final Time
t_f = 60 # s

# Inertia Tensor-Asymmetric (kg-m^2)
J = np.array([[2.336e9, 1.406e6, 5.1956],
             [1.406e6, 5.262e9 , 1.758e7],
             [5.1956, 1.758e7, 4.454e9]]) * 1e-6

# Inertia Tensor-Symmetric (kg-m^2)
Js = np.array([[2.371e9, 185.7917, 5.6952],
             [185.7917, 8.396e9 , 3.516e7],
             [5.6952, 3.516e7, 7.588e9]]) * 1e-6

# Pseudo Inverse Param
# Norm to faces of dodecahedron
alpha_inv = np.arctan2((3 +np.sqrt(5)), 4)
beta_inv = np.arctan2((3 - np.sqrt(5)), 4)
a = np.cos(alpha_inv)
b = np.sin(alpha_inv)
c = np.sin(beta_inv)
d = np.cos(beta_inv)

# NASA and Pyramid Config
a1 = b1 = c1 = d1 = 1/np.sqrt(2)
alpha = beta = gamma = 1/np.sqrt(3)

# Polyhedron 6 wheel
# b = c = 1/np.sqrt(3)
# a = d = np.sqrt(2/3)

# NASA Pseudo inverse
W_N_inv = np.array([[1 + beta**2 + gamma**2, -alpha*beta, -alpha*gamma],
             [-alpha*beta, 1 + alpha**2 + gamma**2, -alpha*beta],
             [-alpha*gamma, -beta*gamma, 1 + alpha**2 + beta**2],
             [alpha, beta, gamma]])*(1/(1 + alpha**2 + beta**2 + gamma**2))

# Pyramid Pseudo inverse
W_4_inv = np.array([[1/a1, b1/(b1**2 + c1**2), 0],
             [-1/a1,  b1/(b1**2 + c1**2), 0],
             [0,  c1/(b1**2 + c1**2), 1/d1],
             [0,  c1/(b1**2 + c1**2), -1/d1]])*0.5

# 6 RW Pseudo inverse
W_6 = np.array([[b, c, b, c, b, c],
             [0,  np.sqrt(3)*d/2, np.sqrt(3)*a/2, 0, -np.sqrt(3)*a/2, -np.sqrt(3)*d/2],
             [a,  d/2, -a/2, -d, -a/2, d/2]])
W_6_inv = np.linalg.pinv(W_6)

# Euler axis/angle
angle = 1.1205366743966416
e = np.array([1, 0, 0])


# Baseline Function Definitions

# Skew Matrix Function
def skew_mat(e):
    s = np.array([[0, -e[2], e[1]],
                 [e[2], 0, -e[0]],
                 [-e[1], e[0], 0]])
    return s

# A from Euler angle + axis
def A_euler(e, theta):
    A = np.eye(3) - np.sin(theta)*skew_mat(e) + (1 - np.cos(theta))*skew_mat(e)@skew_mat(e)
    return A

# Compute and print A matrix
A_f = A_euler(e, angle)
print('A_f = ', A_f)

# Attitude to Rodrigues Parameter
def A2g(A_rp):
    g = np.array([1 + 2*A_rp[0][0] - np.trace(A_rp),
                 A_rp[0][1] + A_rp[1][0],
                 A_rp[0][2] + A_rp[2][0],
                 ])/(A_rp[1][2] - A_rp[2][1])
    return g

# Rodrigues Paramter to Attitude Matrix
def g2A(g):
    A = np.eye(3) + 2*(skew_mat(g)@skew_mat(g) - skew_mat(g))/(1 + np.linalg.norm(g)**2)
    return A

# Polynomial Curves for Reference
def mu_poly(t, tf, mu_f):
    return mu_f * (10 * t**3 / tf**3 - 15 * t**4 / tf**4 + 6 * t**5 / tf**5)

def mu_dot_poly(t, tf, mu_f):
    return mu_f * (30 * t**2 / tf**3 - 60 * t**3 / tf**4 + 30 * t**4 / tf**5)

def mu_2dot_poly(t, tf, mu_f):
    return mu_f * (60 * t / tf**3 - 180 * t**2 / tf**4 + 120 * t**3 / tf**5)


# Initial Conditions

# Initial errors chosen, angle and w, -> crp
phi_0 = 1
mu_0 = np.tan(phi_0/2)*e
w0 = np.array([0.05, 0.05, -0.01])
h0 = np.array([0., 0., 0.])

# Initial State Vector for Integrator
x0 = np.hstack([mu_0, w0, h0])
