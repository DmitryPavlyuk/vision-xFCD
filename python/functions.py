# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings

# return(sum(hadamard.prod(omega,abs(Q-Qhat)))/sum(omega>0))
def omegaMAE(omega, Q, Qhat):
    return np.sum(np.array(np.multiply(omega,abs(Q-Qhat))))/np.count_nonzero(np.array(omega));

def omegaMAPE(omega, Q, Qhat):
    return np.sum(np.array(np.multiply(omega,abs(Q-Qhat)/(Q+1e-6))))/np.count_nonzero(np.array(omega));

def randomOmega(sparsity, Q):
    # df = pd.DataFrame(np.zero(len(Q), len(Q.columns)), columns=Q.columns, index=Q.index)
    df = np.random.binomial(n=Q, p=sparsity,size=(len(Q), len(Q.columns)))
    df[df>1] = 1
    return(pd.DataFrame(df, columns = Q.columns, index = Q.index))

def enhanceOmega(omega, gamma):
    res = omega.dot(gamma)
    res[res>1] = 1
    return(res)

def coverage(omega):
    return(np.count_nonzero(np.array(omega))/np.prod(np.array(omega).shape))
    
def TGMCS(Qdf,Lw,H, omega,accMask=np.nan, mu=1e-4, lambda1=1e-2, lambda2=5e-2, lambda3=5e-2, r=np.nan, maxIter=1e6, tol=1e-6, returnQhat=False):
    N = len(Qdf.columns)
    T = len(Qdf)
    Q = Qdf.T
    omega = omega.T
    if np.isnan(accMask): 
        accMask = omega
    if np.isnan(r): 
        r = N
    U = np.random.normal(size=(N, r))
    E = np.random.normal(size=(T, r))
    Qhat = U.dot(E.T)
    prevMAE = omegaMAE(omega, Q, Qhat)
    converged = False
    i = 1
    while i<maxIter:
        FR = np.multiply(omega, Qhat-Q)
        U = U - mu * (FR.dot(E)+2*lambda1*U+lambda2*(Lw+Lw.T).dot(U))
        E = E - mu * (FR.T.dot(U)+2*lambda1*E+2*lambda3*H.dot(H.T).dot(E))
        Qhat = U.dot(E.T)
        curMAE = omegaMAE(omega, Q, Qhat)
        if (abs(curMAE-prevMAE)<tol):
            converged = True
            break
        else: 
            prevMAE = curMAE
        i = i + 1
        if i==maxIter:
            warnings.warn("No convergence - maximum number of iterations reached")
            converged = False
        if (i % 1e4 == 0):
            print("curMAE",curMAE)
    unobsaccMask = accMask+1
    unobsaccMask[unobsaccMask==2] = 0
    res = {'observedMAE':omegaMAE(accMask,Q,Qhat),
           'unobservedMAE':omegaMAE(unobsaccMask,Q,Qhat),
           'observedMAPE':omegaMAPE(accMask,Q,Qhat),
           'unobservedMAPE':omegaMAPE(unobsaccMask,Q,Qhat),
           'converged':converged}
    if (returnQhat):
        res['Qhat'] = Qhat.T
    return(res)
            
    
    
    