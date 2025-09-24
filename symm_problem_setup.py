# Symmetric Integrator function
def func_s(t, x):
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
    u = -P @ (w - R @ w_star) + w_cross @ Js @ w - Js @ w_cross @ R @ w_star + Js @ R @ w_dot_star - k * mu_error

    # mu_dot and w_dot, h_dot EOM
    mu_cross = skew_mat(mu)
    mu_dot = 0.5 * ((1 + np.linalg.norm(mu) ** 2) * np.eye(3) + mu_cross + mu_cross @ mu_cross) @ w
    w_dot = np.linalg.inv(Js) @ (u - np.cross(w, Js @ w))
    h_dot = -w_cross@h - u

    return list(np.concatenate((mu_dot, w_dot, h_dot)))

# Solve
Xs = solve_ivp(func_s, [0, t_f], x0,  method='RK45', rtol=1e-10, atol=1e-10)

# Extract results and initialize lists
times = Xs.t

mu1s = Xs.y[0]
mu2s = Xs.y[1]
mu3s = Xs.y[2]

w1s = Xs.y[3]
w2s = Xs.y[4]
w3s = Xs.y[5]

h1s = Xs.y[6]
h2s = Xs.y[7]
h3s = Xs.y[8]

w1star_lists=[]
w2star_lists=[]
w3star_lists=[]

mu1s_lists=[]
mu2s_lists=[]
mu3s_lists=[]

u1_lists=[]
u2_lists=[]
u3_lists=[]

v_lists=[]

h1_nasas = []
h2_nasas = []
h3_nasas = []
h4_nasas = []

h1_pyrs = []
h2_pyrs = []
h3_pyrs = []
h4_pyrs = []

h1_6ws = []
h2_6ws = []
h3_6ws = []
h4_6ws = []
h5_6ws = []
h6_6ws = []

# Get reference to graph
for i in range(len(Xs.t)):
    times = Xs.t[i]
    mus = np.array([Xs.y[0][i], Xs.y[1][i], Xs.y[2][i]])
    ws = np.array([Xs.y[3][i], Xs.y[4][i], Xs.y[5][i]])
    hs = np.array([Xs.y[6][i], Xs.y[7][i], Xs.y[8][i]])

    # Polynomial Curves
    phi = mu_poly(times, t_f, angle)
    phi_dot = mu_dot_poly(times, t_f, angle)
    phi_2dot= mu_2dot_poly(times, t_f, angle)

    # Ref variables for traj
    mu_star = np.tan(phi / 2) * e
    w_star = phi_dot * e
    w_dot_star = phi_2dot * e
    w_cross_s = skew_mat(ws)

    # error
    A_errors = g2A(mus) @ g2A(mu_star).T
    mu_errors = A2g(A_errors)

    # Control 
    us = -P @ (ws - R @ w_star) + w_cross_s @ Js @ ws - Js @ w_cross_s @ R @ w_star + Js @ R @ w_dot_star - k * mu_errors

    # Lyapunov function
    eps_s = ws - w_star
    vs = 0.5 * eps_s.T @ Js @ eps_s + k*np.log(1 + np.inner(mu_errors, mu_errors))
    
    # Compute RW momenta in w.f.
    H_W_Ns = W_N_inv@hs
    H_W_6s = W_6_inv@hs
    H_W_4s = W_4_inv@hs

    w1star_lists.append(w_star[0])
    w2star_lists.append(w_star[1])
    w3star_lists.append(w_star[2])
    mu1s_lists.append(mu_star[0])
    mu2s_lists.append(mu_star[1])
    mu3s_lists.append(mu_star[2])
    u1_lists.append(us[0])
    u2_lists.append(us[1])
    u3_lists.append(us[2])
    v_lists.append(vs)
    h1_nasas.append(H_W_Ns[0]) 
    h2_nasas.append(H_W_Ns[1])  
    h3_nasas.append(H_W_Ns[2])  
    h4_nasas.append(H_W_Ns[3])  
    h1_6ws.append(H_W_6s[0]) 
    h2_6ws.append(H_W_6s[1])  
    h3_6ws.append(H_W_6s[2])  
    h4_6ws.append(H_W_6s[3]) 
    h5_6ws.append(H_W_6s[4])  
    h6_6ws.append(H_W_6s[5]) 
    h1_pyrs.append(H_W_4s[0]) 
    h2_pyrs.append(H_W_4s[1])  
    h3_pyrs.append(H_W_4s[2])  
    h4_pyrs.append(H_W_4s[3])
