import torch
import numpy as np

def cgd(A, b, x0=None, maxiter=None, tol=1e-10):
    """
    Conjugate Gradient Descent solver for Ax = b
    """
    b = b.view(-1)
    n = b.size(0)
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    
    if maxiter is None:
        maxiter = n
    
    r = b - A @ x
    p = r.clone()
    r_norm = r @ r
    
    for i in range(maxiter):
        Ap = A @ p
        alpha = r_norm / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_new = r @ r
        beta = r_norm_new / r_norm
        r_norm = r_norm_new
        if r_norm < tol:
            break
        p = r + beta * p
    
    return x 