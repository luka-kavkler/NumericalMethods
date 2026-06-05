import numpy as np
import scipy
import pandas as pd
import math
from warnings import warn

TOL = 1.48e-08;
MAX_STEPS = 50;

def GaussNewton(F, x, maxSteps = MAX_STEPS, tol=TOL):
    """Takes in vector function F, and starting aproximation of x;
    returns back an aproximation for a zero of F function
    """
    F : function;
    x = np.asarray(x, dtype = float);

    steps = 0;
    dx = math.inf;

    while (steps < maxSteps):
        JF = scipy.differentiate.jacobian(F, x).df
        
        dx = np.linalg.lstsq(JF, -F(x))[0];
        x = x+dx;
        if (np.linalg.norm(dx, ord=2) < tol):
            return x;

        steps += 1;
    warn("Max steps reached");
    return x;


def HotelingReduction(A, lamb, x):
        """Returns an adjusted array for further eigen value analysis"""
        return A - lamb * np.outer(x,x);


def RaylK(A,x):
        """Returns raylleighs kvocient? for a normalised vector x"""
        return np.transpose(x) @ A @ x;


def powerMethod(A, x, max_steps = MAX_STEPS, tol = TOL):
    """Takes: square matrix A, aprox of dominant eigen vector x, n number of lambdas, inverse diferentiates between inverse or norma method
    Returns: n dominant lambda values (or n least dominant lambda values if inverse = True)"""
    x = np.asarray(x, dtype = float);

    for _ in range(max_steps):
        x = A @ x;
        x = x/np.linalg.norm(x);
        lamb = RaylK(A,x);
        if np.linalg.norm(A @ x - lamb * x) < tol:
             return (lamb, x)
        
    warn("Max steps reached")
    return (lamb, x)

A = np.array([[1, 0], [4, 8]]);
x = np.transpose(np.array([1, -2]));
print(powerMethod(A, x))



        



def f(x):
    return np.array([x[0]**2 +x[1]**2 -10*x[0] + x[1]-1, x[0]**2 - x[1]**2 -1*x[0] + 10*x[1]-25])

#F = f
#print(GaussNewton(F, np.array([9,-3])))


