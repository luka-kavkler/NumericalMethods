import numpy as np
import scipy
import pandas as pd
import math
from warnings import warn
import itertools

TOL = 1.48e-08;
MAX_STEPS = 50;

def gaussNewton(F, x, maxSteps = MAX_STEPS, tol=TOL):
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


def hotelingReduction(A, lamb, x):
    """Returns an adjusted array for further eigen value analysis (meant for usage with powerMethod)"""
    return A - lamb * np.outer(x,x);


def raylK(A,x):
    """Returns raylleighs kvocient? for a normalised vector x"""
    return np.transpose(x) @ A @ x;


def powerMethod(A, x, max_steps = MAX_STEPS, tol = TOL):
    """Takes: square matrix A, aprox of dominant eigen vector x, n number of lambdas
    Returns: dominant lambda value"""
    x = np.asarray(x, dtype = float);

    for _ in range(max_steps):
        x = A @ x;
        x = x/np.linalg.norm(x);
        lamb = raylK(A,x);
        if np.linalg.norm(A @ x - lamb * x) < tol:
             return (lamb, x)
        
    warn("Max steps reached")
    return (lamb, x)

def extendedPowerMethod(A, x, n, max_steps = MAX_STEPS, tol = TOL):
    """Returns first n dominant pairs organised by size of lambda descending (lambda, v)"""
    pairs = list();
    B = A.copy();
    v = x;
    for _ in range(n):
        eigenPair = powerMethod(B, v, max_steps=max_steps, tol=tol);
        pairs.append(eigenPair);
        B = hotelingReduction(B, eigenPair[0], eigenPair[1]);
    return pairs



def inversePowerMethod(A, x, sigma, max_steps = MAX_STEPS, tol = TOL):
    """Takes: square matrix A, aprox of dominant eigen vector x, n number of lambdas
    Returns: lambda value closest to sigma aporksimation value"""
    x = np.asarray(x, dtype = float);
    x = x/np.linalg.norm(x)
    n = len(A);
    A_start = A.copy();
    A = (A - sigma * np.identity(n));
    lamb = sigma;

    for _ in range(max_steps):
        try:
            x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            warn("sigma was the exact eigen value -> x is not correct")
            return (sigma, x)
        
        x = x/np.linalg.norm(x);

        lamb = raylK(A_start,x);
        
        if np.linalg.norm(A_start @ x - lamb * x) < tol:
             return (lamb, x)
        
    warn("Max steps reached")
    return (lamb, x)


def singularMatrixAproximation(A, k):
    """Aproksimates matrix A, with the best possible matrix of rang k"""
    U, s, Vh = scipy.linalg.svd(A);
    A_ = np.zeros(np.shape(A));

    A_ = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :];

    return A_


def predeterminedSystemFunction(X, Y, F):
    """Finds a function that is a linear combination of functions in list of functions F,
    that aproximates the y values best by MLS and a vector of koeficients that represent the actual solution of the predetermined system"""
    X : list[int];
    Y : list[int];
    F : list[function];

    n = len(X);
    m = len(F);

    A = np.array(n*[m*[0]], dtype=float);
    for i in range(n):
        A[i, :] = np.array([x[0](x[1]) for x in itertools.product(F, [X[i]])]);

    Q, R = np.linalg.qr(A);
    koef = np.linalg.solve(R, np.transpose(Q) @ Y)

    def f(x):
        """Aproximating function for the overdeterminated system"""
        y = 0;
        for i in range(len(koef)):
            y += koef[i] * F[i](x);
        return y;

    return f, koef;

def predeterminedSystem(A,b):
    """Solves predetermined system and returns best solution vector by MLK"""
    Q, R = np.linalg.qr(A);
    z = np.linalg.solve(R, np.transpose(Q) @ b);
    return z;


X = [0.21, 0.66, 0.93, 1.25, 1.75, 2.03, 2.24, 2.57, 2.87, 2.98];
Y = [0.25, -0.27, -1.12, -0.45, 0.28, 0.13, -0.27, 0.26, 0.58, 1.03];
F = [math.exp, math.log, math.sin, math.cos];

print(predeterminedSystem(X,Y,F))

        


