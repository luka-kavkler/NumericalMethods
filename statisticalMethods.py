import numpy as np
import scipy
import pandas as pd
import math
from warnings import warn

def linearRegresion(X, Y):
    """Return function f(x) that aproksimates y = b*x + a + U and the R factor (R < 0.8 raises a warning)"""
    X : list[int]
    Y : list[int]

    n = len(X);
    Sx = sum(X);
    Sxx = sum(list(map(lambda x: x*x, X)));
    Sy = sum(Y);
    Sxy = sum(list(map(lambda x: x[0]*x[1], zip(X,Y))));
    xPovp = Sx/n;
    yPovp = Sy/n;

    b = (Sxy - (1/n) * Sx * Sy)/(Sxx - (1/n) * Sx*Sx);
    a = yPovp - b * xPovp;

    def f(x):
        """linear regresion function"""
        return b * x + a;
    
    R = math.sqrt(sum(list(map(lambda x: (f(x) - yPovp)*(f(x) - yPovp), X)))/sum(list(map(lambda y: (y - yPovp)*(y - yPovp), Y))))
    if R<0.8:
        warn("Bad aproksimation : R value bellow 0.8")
    return f, R;

X = [1, 2, 3, 4 , 5];
Y = [2, 6, 7, 9, 11];
f, R = linearRegresion(X, Y);
print(R);
for x in X:
    print(f(x))