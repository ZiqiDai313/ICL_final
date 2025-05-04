import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

class Muon(Optimizer):
    """Implementation of Muon optimizer"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute momentum correction
                momentum_correction = torch.abs(grad - state['prev_grad'])
                state['prev_grad'] = grad.clone()
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] / (1 + momentum_correction.mean())
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class SOAP(Optimizer):
    """Implementation of SOAP optimizer"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(SOAP, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SOAP does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Compute gradient statistics
                grad_norm = grad.norm()
                grad_angle = torch.cosine_similarity(grad.view(-1), state['prev_grad'].view(-1), dim=0)
                state['prev_grad'] = grad.clone()

                # Adaptive momentum
                adaptive_beta1 = beta1 * (1 + grad_angle)
                
                # Update momentum
                exp_avg.mul_(adaptive_beta1).add_(grad, alpha=1 - adaptive_beta1)
                
                # Update second moment with gradient norm scaling
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2) / (1 + grad_norm))
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * (1 + grad_angle)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class SlimAdam(Optimizer):
    """Implementation of SlimAdam optimizer"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(SlimAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SlimAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment with gradient sparsification
                grad_sparse = grad.clone()
                grad_sparse[torch.abs(grad_sparse) < torch.quantile(torch.abs(grad_sparse), 0.9)] = 0
                exp_avg_sq.mul_(beta2).addcmul_(grad_sparse, grad_sparse, value=1 - beta2)
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss 