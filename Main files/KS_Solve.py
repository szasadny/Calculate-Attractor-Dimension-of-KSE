import numpy as np
import plotly.express as px
from tempfile import TemporaryFile

# Parameters
L = 100                  # domain length
M = 100                  # grid size
t0 = 0                   # t_0
tn = 200                 # t_max 
dt = 0.01                # stepsize
nt = int((tn - t0) / dt) # amount of steps
nu = 1                   # nu

# Discretization
global Dx, Dxx, Dxxxx
D     = np.linspace(-M//2,M//2,M,endpoint=False)
Dx    = np.fft.fftshift(np.multiply(D,(2*np.pi/L)))*np.sqrt(-1+0j)
Dxx   = -np.fft.fftshift(np.power(np.multiply(D,(2*np.pi/L)),2))
Dxxxx = np.fft.fftshift(np.power(np.multiply(D,(2*np.pi/L)),4))

# Solution space
u = np.zeros((M,nt))

# Initial condition
x = np.linspace(start=0, stop=L-(L/M), num=M)
u0 = np.cos((2 * np.pi * x)/ L) + 0.01 * np.cos((4 * np.pi * x)/L)
u[:,0] = u0

# Define function for each derivative.
def u_x(u):
    return np.real(np.fft.ifft(np.multiply(Dx,np.fft.fft(u))))

def u_xx(u):
    return np.real(np.fft.ifft(np.multiply(Dxx,np.fft.fft(u))))

def vu_xxxx(u):
    return nu*np.real(np.fft.ifft(np.multiply(Dxxxx,np.fft.fft(u))))

def Lu(u):
    return (-u_xx(u) - vu_xxxx(u))

def fu(u):
    return (-1)*np.multiply(u, u_x(u))

# Calculate the half step using the IMEX RK method.
def u_half(u):
    L    = -Dxx - Dxxxx
    g_i  = np.add(np.multiply((0.5*dt),fu(u)), u)
    G    = np.fft.fft(g_i)
    Uhat = np.divide(G,(1-np.multiply((0.5*dt),L)))
    
    return np.real(np.fft.ifft(Uhat))

# Calculate the full step using the IMEX RK method.
def u_plus1(u, u_half):
    return u + np.multiply(dt,Lu(u_half)) + np.multiply(dt,fu(u_half))

# Solve the Kuramoto-Sivanchinsky Differential equation .
def KS_Solve(u):
    for i in range(nt-1):
        u_i = u[:,i]

        u_ihalf  = u_half(u_i)
        u[:,i+1] = u_plus1(u_i, u_ihalf)

    return u

# Calculate and save the results.
u_result = KS_Solve(u)
np.savetxt("u_result.csv", u_result, delimiter=",")

# Take the values needed for the plot for every relevant t.
u_result_plot_relevant = np.zeros((len(u_result[:,0]),int(len(u_result[0])/(1/dt))))
j = 0
for i in range(len(u_result[0])):
    if i%(1/dt) == 0:
        u_result_plot_relevant[:,j] = u_result[:,i]
        j += 1

# Plot the results.
fig = px.imshow(u_result_plot_relevant, labels=dict(x="t", y="x", color = "u(x,t)"), title="KS solution for t between " + str(t0) + " and " + str(tn))
fig.show()