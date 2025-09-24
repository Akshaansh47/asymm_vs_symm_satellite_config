# Asymmetric Integrator function
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
