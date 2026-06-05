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
        warn("Bad aproximation : R value bellow 0.8")
    return f, R;

#Parametric test
def T_Test(X, mi0, alpha, larger = True):
    """Students t test, tests H0(mi = mi0) : H1(mi > mi0) if larger = True, or H1(mi < mi0) if larger = False
    returns wether base hypothesis was rejected and p-value 
    """
    m = len(X);
    x_ = sum(X)/m;
    S = math.sqrt(sum([(x - x_)**2  for x in X]) / (m-1));

    T = (x_ - mi0)*math.sqrt(m)/S;
    
    dF = m - 1;

    if larger:
        tCritical = scipy.stats.t.ppf(1 - alpha, dF);
    else:
        tCritical = scipy.stats.t.ppf(alpha, dF);
    asumptionDenial = False;
    pValue = scipy.stats.t.sf(T, dF);
    if not larger:
        pValue = 1 -pValue;
    if larger and T > tCritical:
        asumptionDenial = True;
    elif not larger and T < tCritical:
        asumptionDenial = True;
    
    return asumptionDenial, pValue;


#Comparative parametric tests
def comparativeT_Test(X, Y, alpha):
    """Students comparative t test, 
    checks H0(miX = miY) : H1(miX > miY),
    with alfa level of trust
    returns true/false and p-value"""
    m = len(X);
    n = len(Y);
    x_ = sum(X)/m;
    y_ = sum(Y)/n;
    S = math.sqrt(
        (sum([(x - x_)**2  for x in X]) + sum([(y - y_)**2  for y in Y]))/(n + m -2)
                  );

    T = ((x_ - y_) / S)* math.sqrt(m*n/(m+n));
    dF = m + n - 2;
    tCritical = scipy.stats.t.ppf(1 - alpha, dF);
    asumptionDenial = False;
    pValue = scipy.stats.t.sf(T, dF);
    
    if T > tCritical:
        asumptionDenial = True;
    
    return asumptionDenial, pValue;



def comparativeF_Test(X, Y, alpha):
    """tests H0(sigmaX=sigmaY) : H1(sigmaX > sigmaY)
    with alfa level of trust
    returns true / false and pValue"""
    m = len(X);
    n = len(Y);
    x_ = sum(X)/m;
    y_ = sum(Y)/n;
    
    F = (sum([(x - x_)**2  for x in X]) / (m-1) ) / (sum([(y - y_)**2  for y in Y]) / (n-1));
    
    dF = (m-1, n-1);
    fCritical = scipy.stats.f.ppf(1 - alpha, dF[0], dF[1]);
    pValue = scipy.stats.f.sf(F,dF[0], dF[1]);
    asumptionDenial = False
    
    if F > fCritical:
        asumptionDenial = True;
    
    return asumptionDenial, pValue;


X = [1, 2, 3, 4 , 5];
Y = [2, 6, 7, 9, 11];
print(T_Test(X, 1, 0.05));
f, R = linearRegresion(X, Y);
