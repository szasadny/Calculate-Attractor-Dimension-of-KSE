# Calculate the attractor dimension of the Kuramoto-Sivashinsky equation

The python code accompanying my Bachelor Thesis about the scaling of of the attractor dimension in the discretized Kuramoto-Sivashinsky equation.


This repositery contains the following two files:

**1. KS_Solve.py**

Solves the KSE using the Implicit-Explicit Runge-Kutta (IMEX RK) and plots the result.

**2. KS_Lyapunov_Dimension.py**

Appoximates the Lyapunov dimension by approximating the Lyapunov exponents of the KSE for different values of the gridsize M. Within this file is the approximation of the solution A(t) of the KSE and the approximation of the solution of dot(Y) = A(t)Y done using the IMEX RK method.
