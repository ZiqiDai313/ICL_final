import torch
import numpy as np
from data_solver import generate_linear_system_batch

def generate_quadratic_data(batch_size, dim, condition_number=5.0):
    """
    Generate synthetic quadratic function data.
    f(x) = x^T A x + b^T x + c
    where A is a positive definite matrix with given condition number
    """
    # Generate positive definite matrix A with controlled condition number
    U, _ = torch.linalg.qr(torch.randn(dim, dim))
    eigenvalues = torch.logspace(0, np.log10(condition_number), dim)
    A = U @ torch.diag(eigenvalues) @ U.T
    A = A.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Generate linear term b and constant term c
    b = torch.randn(batch_size, dim)
    c = torch.randn(batch_size, 1)
    
    # Generate input points
    X = torch.randn(batch_size, dim)
    
    # Calculate output
    quad_term = torch.sum(X.unsqueeze(1) @ A @ X.unsqueeze(-1), dim=1)
    linear_term = torch.sum(b * X, dim=1, keepdim=True)
    y = quad_term + linear_term + c
    
    return X, y, (A, b, c)

def prepare_icl_data(X, y, context_length, params=None):
    """Prepare data in ICL format with context points and target point"""
    batch_size = X.shape[0]
    dim = X.shape[1]
    
    # Generate context points
    context_X = torch.randn(batch_size, context_length, dim)
    
    if params is not None:  # For quadratic function
        A, b, c = params
        # Reshape context_X for batch matrix multiplication
        context_X_reshaped = context_X.view(batch_size * context_length, dim)
        A_expanded = A.repeat_interleave(context_length, dim=0)
        b_expanded = b.repeat_interleave(context_length, dim=0)
        c_expanded = c.repeat_interleave(context_length, dim=0)
        
        # Calculate quadratic term
        quad_term = torch.sum(context_X_reshaped.unsqueeze(1) @ A_expanded @ context_X_reshaped.unsqueeze(-1), dim=1)
        # Calculate linear term
        linear_term = torch.sum(b_expanded * context_X_reshaped, dim=1, keepdim=True)
        # Calculate full output
        context_y = (quad_term + linear_term + c_expanded).view(batch_size, context_length, 1)
    else:  # For linear function
        context_y = torch.randn(batch_size, context_length, 1)
    
    # Combine context and target
    full_X = torch.cat([context_X, X.unsqueeze(1)], dim=1)
    full_y = torch.cat([context_y, y.unsqueeze(1)], dim=1)
    
    # Create ICL format input [X; y]
    Z = torch.cat([full_X, full_y], dim=-1)
    
    return Z, y 