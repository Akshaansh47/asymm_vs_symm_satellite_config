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


# Integrator function
def func(t, x):
    mu = x[0:3]
    w = x[3:6]
    h = x[6:9]

    # Polynomial Curves
    phi = mu_poly(t, t_f, angle)
    phi_dot = mu_dot_poly(t, t_f, angle)
    phi_2dot = mu_2dot_poly(t, t_f, angle)

    # Reference state variables for trajectory
    mu_star = np.tan(phi / 2) * e
    w_star = phi_dot * e
    w_dot_star = phi_2dot * e
    w_cross = skew_mat(w)

    # Error
    A_error = g2A(mu) @ g2A(mu_star).T
    mu_error = A2g(A_error)

    # Control 
    u = -P @ (w - R @ w_star) + w_cross @ J @ w - J @ w_cross @ R @ w_star + J @ R @ w_dot_star - k * mu_error

    # mu_dot and w_dot, h_dot EOM
    mu_cross = skew_mat(mu)
    mu_dot = 0.5 * ((1 + np.linalg.norm(mu) ** 2) * np.eye(3) + mu_cross + mu_cross @ mu_cross) @ w
    w_dot = np.linalg.inv(J) @ (u - np.cross(w, J @ w))
    h_dot = -w_cross@h - u

    return list(np.concatenate((mu_dot, w_dot, h_dot)))

# Solve
X = solve_ivp(func, [0, t_f], x0,  method='RK45', rtol=1e-10, atol=1e-10)

# Extract results and initialize lists
time = X.t

mu1 = X.y[0]
mu2 = X.y[1]
mu3 = X.y[2]

w1 = X.y[3]
w2 = X.y[4]
w3 = X.y[5]

h1 = X.y[6]
h2 = X.y[7]
h3 = X.y[8]

w1star_list=[]
w2star_list=[]
w3star_list=[]

mu1s_list=[]
mu2s_list=[]
mu3s_list=[]

u1_list=[]
u2_list=[]
u3_list=[]

v_list=[]

h1_nasa = []
h2_nasa = []
h3_nasa = []
h4_nasa = []

h1_pyr = []
h2_pyr = []
h3_pyr = []
h4_pyr = []

h1_6w = []
h2_6w = []
h3_6w = []
h4_6w = []
h5_6w = []
h6_6w = []

# Get reference to graph
for i in range(len(X.t)):
    time = X.t[i]
    mu = np.array([X.y[0][i], X.y[1][i], X.y[2][i]])
    w = np.array([X.y[3][i], X.y[4][i], X.y[5][i]])
    h = np.array([X.y[6][i], X.y[7][i], X.y[8][i]])

    # Polynomial Curves
    phi = mu_poly(time, t_f, angle)
    phi_dot = mu_dot_poly(time, t_f, angle)
    phi_2dot= mu_2dot_poly(time, t_f, angle)

    # Ref variables for traj
    mu_star = np.tan(phi / 2) * e
    w_star = phi_dot * e
    w_dot_star = phi_2dot * e
    w_cross = skew_mat(w)

    # error
    A_error = g2A(mu) @ g2A(mu_star).T
    mu_error = A2g(A_error)

    # Control 
    u = -P @ (w - R @ w_star) + w_cross @ J @ w - J @ w_cross @ R @ w_star + J @ R @ w_dot_star - k * mu_error

    # Lyapunov function
    eps = w - w_star
    v = 0.5 * eps.T @ J @ eps + k*np.log(1 + np.inner(mu_error, mu_error))
    
    # Compute RW momenta in w.f.
    H_W_N = W_N_inv@h
    H_W_6 = W_6_inv@h
    H_W_4 = W_4_inv@h

    w1star_list.append(w_star[0])
    w2star_list.append(w_star[1])
    w3star_list.append(w_star[2])
    mu1s_list.append(mu_star[0])
    mu2s_list.append(mu_star[1])
    mu3s_list.append(mu_star[2])
    u1_list.append(u[0])
    u2_list.append(u[1])
    u3_list.append(u[2])
    v_list.append(v)
    h1_nasa.append(H_W_N[0]) 
    h2_nasa.append(H_W_N[1])  
    h3_nasa.append(H_W_N[2])  
    h4_nasa.append(H_W_N[3])  
    h1_6w.append(H_W_6[0]) 
    h2_6w.append(H_W_6[1])  
    h3_6w.append(H_W_6[2])  
    h4_6w.append(H_W_6[3]) 
    h5_6w.append(H_W_6[4])  
    h6_6w.append(H_W_6[5]) 
    h1_pyr.append(H_W_4[0]) 
    h2_pyr.append(H_W_4[1])  
    h3_pyr.append(H_W_4[2])  
    h4_pyr.append(H_W_4[3]) 
