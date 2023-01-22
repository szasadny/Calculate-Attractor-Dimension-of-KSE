import numpy as np
from tempfile import TemporaryFile

# Parameters
t0  = 0                  # t_0
tn  = 2000               # t_max 
eps = 0.00000001         # epsilon

# Parameters and initial conditions that depend on L and/or M.
def update_parameters(gridsize, current_domain):

    # Discretization
    global L, M, dt, nt, Q_dim, Dx, Dxx, Dxxxx, k_lin
    L      = current_domain
    M      = gridsize
    dt     = 0.0005                     # stepsize
    nt     = int((tn - t0) / dt)        # amount of steps 
    Q_dim  = M                          # length of Q
    D      = np.linspace(-M//2,M//2,M,endpoint=False)
    Dx     = np.fft.fftshift(np.multiply(D,(2*np.pi/L)))*np.sqrt(-1+0j)
    Dxx    = -np.fft.fftshift(np.power(np.multiply(D,(2*np.pi/L)),2))
    Dxxxx  = np.fft.fftshift(np.power(np.multiply(D,(2*np.pi/L)),4))
    k_lin  = -Dxx - Dxxxx # linear part of the KS equation
    
    # Solution space
    u = np.zeros((M,nt))
    Q = np.random.rand(M, Q_dim)
    
    # Initial condition
    x = np.linspace(start=0, stop=L-(L/M), num=M)
    u0 = np.cos((2 * np.pi * x)/ L) + 0.01 * np.cos((4 * np.pi * x)/L)
    u[:,0] = u0
    
    return u,Q,

"""
    Define function for each derivative.
"""
def u_x(u):
    return np.real(np.fft.ifft(np.multiply(Dx,np.fft.fft(u))))

def u_xx(u):
    return np.real(np.fft.ifft(np.multiply(Dxx,np.fft.fft(u))))

def u_xxxx(u):
    return np.real(np.fft.ifft(np.multiply(Dxxxx,np.fft.fft(u))))

def Lu(u):
    return (-u_xx(u) - u_xxxx(u))

def fu(u):
    return (-1)*np.multiply(u, u_x(u))

"""
    IMEX RK function for calculating u_{i+1}
"""
# Calculate the half step using the IMEX RK method.
def u_half(u):
    G = np.fft.fft(np.add(np.multiply((0.5*dt),fu(u)), u))
    Uhat = np.divide(G,(1-np.multiply((0.5*dt), k_lin)))
    
    return np.real(np.fft.ifft(Uhat))

# Calculate the full step using the IMEX RK method.
def u_plus1(u, u_half):
    return u + np.multiply(dt,Lu(u_half)) + np.multiply(dt,fu(u_half))

"""
    Functions needed for the calculation of B_ii
"""
# Explicitly evaluates f'(u) q.
def fu_dotq(u, q):
    return (fu(u+eps*q) - fu(u))/eps

# Define AQ.
def AQ(u, Q):
    
    # Solution space
    AQ = np.zeros(Q.shape)
    fuu = fu(u)
    Luu = Lu(u)
    
    for i in range(len(Q[0])):
        AQ[:,i] = (fu(u+eps*Q[:,i]) - fuu)/eps + Lu(Q[:,i])

    return AQ

"""
    IMEX RK function for calculating q_{i+1}\tilde{r}
"""
# Calculate the half step using the IMEX RK method.
def q_half(u, q):
    G = np.fft.fft(q + 0.5*dt*fu_dotq(u,q))
    Qhat = np.divide(G,(1-np.multiply((0.5*dt), k_lin)))
    
    return np.real(np.fft.ifft(Qhat))

# Calculate the full step using the IMEX RK method.
def q_plus1(u_half, q, q_half):
    return q + np.multiply(dt,fu_dotq(u_half, q_half)) + np.multiply(dt,Lu(q_half))

# Orthogonalize Y while applying QR factorization using the Modified Gramm Schmidt method.
def MGS(Q):
    for m in range(len(Y[0])):
        for j in range(0, m):
            R_mj =  np.vdot(Q[:,j].transpose(),Q[:,m])
            Q[:,m] = Q[:,m] - R_mj*Q[:,j]

        R_mm   = np.linalg.norm(Q[:,m])
        Q[:,m] = Q[:,m]/R_mm
    
    return Q

"""
    Functions that combine everything.
"""
# Solve the Kuramoto-Sivanchinsky Differential equation while calculating Q using the IMEX RK method.
def KS_Solve_and_get_Bii(u, Q):
    B_ii = np.zeros((M,nt-1))
    
    for i in range(nt-1):
        u_i = u[:,i]

        # IMEX RK for calculating u_{i+1}
        u_ihalf  = u_half(u_i)
        u[:,i+1] = u_plus1(u_i, u_ihalf)

        # Solution space
        A_iQ_i = np.zeros(Q.shape)
        
        # Iterate through each column in Q
        for j in range(len(Q[0])):
            q_j         = Q[:,j]
            q_ihalf     = q_half(u_i, q_j)
            A_iQ_i[:,j] = q_plus1(u_ihalf, q_j, q_ihalf)

        # Compute Q_{i+1} as the QR factorization of A_iQ_i using the modified Gram-Schmidt method
        Q = MGS(A_iQ_i)

        B_ii[:,i] = np.dot(Q.transpose(),AQ(u[:,i+1], Q)).diagonal()

    return u, B_ii

# Returns the Lyapunov exponents given the diagonal B_ii.
def calc_Lyapunov(B_ii):

    # Solution space
    lambda_i = np.zeros(len(B_ii[:,0]))

    # Calculate the integral as a Riemann sum.
    for i in range(len(B_ii[:,0])):
        for t in range(len(B_ii[0])):
            lambda_i[i] = lambda_i[i] + B_ii[i][t]*(dt)

    # Divide by t
    lambda_i = lambda_i / (tn-t0)
 
    return lambda_i

# Returns the Lyapunov dimension given the Lyapunov exponents.
def calc_D_L(lambda_i):

    # Sort the lyapunov exponents from the biggest to the smallest.
    lambda_sorted = -np.sort(-lambda_i)
    
    # Find the maximum k.
    k = 0
    for i in range(1, len(lambda_sorted)):
        if (sum(lambda_sorted[:i]) <= 0):
            break
        k=i

    # If the dimension can't be calculated, return -1.
    if (k == len(lambda_sorted)-1):
        D_L = -1
    else:
        # Calculate the Lyapunov dimension.
        D_L = k + (np.sum(lambda_sorted[:k]) / abs(lambda_sorted[k+1]))
    return D_L

# Main function that calls everything.
def main(gridsize, domain):

    u, Q           = update_parameters(gridsize, domain)
    u_result, B_ii = KS_Solve_and_get_Bii(u, Q)
    lambda_i       = calc_Lyapunov(B_ii)
    D_L            = calc_D_L(lambda_i)

    return B_ii, D_L

"""
    Input and output
"""

# Range of the gridsize
M_min    = 30
M_max    = 70
stepsize = 5

# Compute for L = 36
Dimension_values_L36 = np.zeros(int((M_max-M_min) / stepsize))

for m in range(M_min, M_max+stepsize, stepsize):
    B_ii, Dimension_values_L36[int((m-M_min) / stepsize)] = main(m, 36)

    # Save B_ii for  plots of Lyapunov exponent development.
    np.savetxt("B_ii_M" + str(m) + "_L36.csv", B_ii, delimiter=",")
    
    # Print D_L
    print("M = " + str(m) + ", D_L = " + str(Dimension_values_L36[int((m-M_min) / stepsize)]))

# Save D_L values of the devolopment of M
np.savetxt("D_L_M" + str(M_min) + "to" + str(M_max) + "_t2000_L36.csv", Dimension_values_L36, delimiter=",")