import numpy as np
import scipy
import pandas as pd
import math
from warnings import warn

def GaussNewton(F, x, maxSteps = 50, tol=1.48e-08):
    """Takes in vector function F, and starting aproximation of x;
    returns back an aproximation for a zero of F function
    """
    F : function
    x = np.asarray(x, dtype = float);

    steps = 0;
    dx = math.inf;

    while (steps < maxSteps):
        JF = scipy.differentiate.jacobian(F, x).df
        
        dx = np.linalg.lstsq(JF, -F(x))[0];
        x = x+dx;
        if (np.linalg.norm(dx, ord=2) < tol):
            return x

        steps += 1;
    warn("Max steps reached")
    return x

def f(x):
    return np.array([x[0]**2 +x[1]**2 -10*x[0] + x[1]-1, x[0]**2 - x[1]**2 -1*x[0] + 10*x[1]-25])

F = f
print(GaussNewton(F, np.array([9,-3])))


