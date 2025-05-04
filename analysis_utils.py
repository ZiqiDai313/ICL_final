import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch.nn as nn
# from pyhessian import hessian
from local_hessian import hessian
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from copy import deepcopy

class ModelWrapper(nn.Module):
    def __init__(self, model, output_dim):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_dim = output_dim
        
    def forward(self, x):
        # Ensure input has correct dimensions [batch_size, seq_len, dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        output = self.model(x)
        return output[:, -1, :self.output_dim]  # Return only the relevant output

class HessianAnalyzer:
    """Analyzes the Hessian of a model's loss function."""
    
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
        self.hessian_comp = None
        
    def compute_hessian_trace(self, inputs, targets):
        """Compute Hessian trace using PyHessian."""
        try:
            # Safety check - use smaller batch
            if inputs.shape[0] > 4:
                inputs = inputs[:4]
                targets = targets[:4]
                
            try:
                # 直接使用原始模型而不是ModelWrapper
                self.hessian_comp = hessian(self.model, self.criterion, 
                                    data=(inputs.to(self.device), targets.to(self.device)),
                                    cuda=torch.cuda.is_available())
            
                # Get trace and convert to float with safety checks
                trace = self.hessian_comp.trace()
                
                # Numerical stability check
                if isinstance(trace, (list, tuple)):
                    trace_values = []
                    for t in trace:
                        if isinstance(t, torch.Tensor):
                            if torch.isfinite(t).all():
                                trace_values.append(float(t.item()))
                            else:
                                trace_values.append(0.0)
                        else:
                            trace_values.append(float(t) if isinstance(t, (int, float)) and np.isfinite(t) else 0.0)
                    trace = np.mean(trace_values) if trace_values else 0.0
                elif isinstance(trace, torch.Tensor):
                    trace = float(trace.item()) if torch.isfinite(trace).all() else 0.0
                else:
                    trace = float(trace) if isinstance(trace, (int, float)) and np.isfinite(trace) else 0.0
                    
                # Cap to a reasonable range to prevent extreme values
                trace = max(min(trace, 1e6), -1e6)
                    
                return trace
            except Exception as e:
                print(f"Error in hessian computation: {str(e)}")
                return 0.0
        except Exception as e:
            print(f"Error in compute_hessian_trace: {str(e)}")
            return 0.0
        
    def compute_eigenvalues(self, inputs, targets, top_n=5):
        """Compute top eigenvalues of Hessian using PyHessian."""
        try:
            # Safety check - use smaller batch
            if inputs.shape[0] > 4:
                inputs = inputs[:4]
                targets = targets[:4]
                
            try:
                if self.hessian_comp is None:
                    # 直接使用原始模型而不是ModelWrapper
                    self.hessian_comp = hessian(self.model, self.criterion, 
                                        data=(inputs.to(self.device), targets.to(self.device)),
                                        cuda=torch.cuda.is_available())
                
                # Get eigenvalues with safety checks
                eigenvalues = self.hessian_comp.eigenvalues(top_n=top_n)
                
                # Process and stabilize eigenvalues
                def process_eigenvalues_safely(e_vals):
                    if isinstance(e_vals, tuple):
                        e_vals = e_vals[0]  # Take only eigenvalues, not eigenvectors
                    
                    result = []
                    if isinstance(e_vals, torch.Tensor):
                        # Convert to numpy for easier processing
                        e_np = e_vals.detach().cpu().numpy()
                        # Filter out non-finite values
                        e_np = e_np[np.isfinite(e_np)]
                        # Cap extreme values
                        e_np = np.clip(e_np, -1e6, 1e6)
                        result = e_np.tolist() if e_np.size > 0 else [0.0] * top_n
                    elif isinstance(e_vals, list):
                        for e in e_vals:
                            if isinstance(e, torch.Tensor) and torch.isfinite(e).all():
                                result.append(float(e.item()))
                            elif isinstance(e, (int, float)) and np.isfinite(e):
                                result.append(float(e))
                            else:
                                result.append(0.0)
                    
                    # Ensure we have the requested number of eigenvalues
                    if len(result) < top_n:
                        result.extend([0.0] * (top_n - len(result)))
                    elif len(result) > top_n:
                        result = result[:top_n]
                        
                    return result
                
                return process_eigenvalues_safely(eigenvalues)
            except Exception as e:
                print(f"Error in eigenvalue computation: {str(e)}")
                return [0.0] * top_n
        except Exception as e:
            print(f"Error in compute_eigenvalues: {str(e)}")
            return [0.0] * top_n

    def compute_hessian(self, data, targets, top_n=5):
        """
        Compute Hessian eigenvalues and trace.
        
        Args:
            data: Input data tensor
            targets: Target tensor
            top_n: Number of top eigenvalues to return
            
        Returns:
            Dictionary with Hessian analysis results
        """
        hessian_comp = hessian(self.model, self.criterion, data=data, target=targets)
        
        # Get top eigenvalues and trace
        top_eigenvalues = hessian_comp.eigenvalues(top_n=top_n)
        trace = hessian_comp.trace()
        
        return {
            'top_eigenvalues': top_eigenvalues,
            'trace': trace
        }
    
    def plot_eigenvalue_distribution(self, eigenvalues, save_path):
        """Plot the distribution of eigenvalues."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(eigenvalues)), eigenvalues)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Top Hessian Eigenvalues')
        plt.savefig(save_path)
        plt.close()
        
    def save_hessian_data(self, hessian_data, save_path):
        """Save Hessian analysis data to disk."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'top_eigenvalues': hessian_data['top_eigenvalues'].tolist() if isinstance(
                hessian_data['top_eigenvalues'], np.ndarray) else hessian_data['top_eigenvalues'],
            'trace': float(hessian_data['trace']) if not isinstance(hessian_data['trace'], float) else hessian_data['trace']
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

class GradientTracker:
    """Tracks gradient statistics during training."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = {}
        self.hooks = []
        self.param_groups = {}
        self.gradient_stats = {
            'mean': [],
            'var': [],
            'norm': [],
            'layer_norms': {}
        }
        
    def _register_hooks(self):
        """Register hooks to track gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradients[name] = []
                hook = param.register_hook(
                    lambda grad, name=name: self._hook_fn(grad, name)
                )
                self.hooks.append(hook)
                
                # Group parameters by layer
                layer_name = name.split('.')[0]
                if layer_name not in self.param_groups:
                    self.param_groups[layer_name] = []
                self.param_groups[layer_name].append(name)
                
                # Initialize layer norm tracking
                if layer_name not in self.gradient_stats['layer_norms']:
                    self.gradient_stats['layer_norms'][layer_name] = []
                
    def _hook_fn(self, grad, name):
        """Hook function to store gradients."""
        if grad is not None:
            self.gradients[name].append(grad.detach().cpu().numpy())
            
    def get_gradient_stats(self):
        """Get gradient statistics."""
        stats = {}
        for name, grads in self.gradients.items():
            if len(grads) > 0:
                try:
                    grad_np = np.concatenate([g.flatten() for g in grads])
                    stats[name] = {
                        'mean': float(np.mean(grad_np)),
                        'std': float(np.std(grad_np)),
                        'max': float(np.max(np.abs(grad_np))),
                        'min': float(np.min(grad_np))
                    }
                except:
                    # Handle case where gradients might be incompatible
                    stats[name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'max': 0.0,
                        'min': 0.0
                    }
        return stats
    
    def collect_gradient_stats(self):
        """Collect gradient statistics after backward pass."""
        all_grads = []
        layer_norms = {}
        
        # Collect all gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Handle potential NaN/Inf values
                if not torch.isfinite(param.grad.data).all():
                    continue  # Skip non-finite gradients
                    
                grad_data = param.grad.data.cpu().numpy().flatten()
                
                # Apply clipping to extreme values
                grad_data = np.clip(grad_data, -1e4, 1e4)
                
                all_grads.append(grad_data)
                
                # Collect layer-specific gradients
                layer_name = name.split('.')[0]
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                
                try:
                    layer_norm = np.linalg.norm(grad_data)
                    # Check if norm is finite
                    if np.isfinite(layer_norm):
                        layer_norms[layer_name].append(layer_norm)
                except:
                    # Skip this layer if norm calculation fails
                    pass
        
        # Compute overall statistics with safety checks
        if all_grads:
            try:
                all_grads = np.concatenate(all_grads)
                # Remove any lingering non-finite values
                all_grads = all_grads[np.isfinite(all_grads)]
                
                if all_grads.size > 0:
                    mean_grad = float(np.mean(all_grads))
                    var_grad = float(np.var(all_grads))
                    norm_grad = float(np.linalg.norm(all_grads))
                    
                    # Cap extreme values
                    mean_grad = max(min(mean_grad, 1e4), -1e4)
                    var_grad = min(var_grad, 1e8)
                    norm_grad = min(norm_grad, 1e6)
                else:
                    mean_grad = 0.0
                    var_grad = 0.0
                    norm_grad = 0.0
            except:
                mean_grad = 0.0
                var_grad = 0.0
                norm_grad = 0.0
        else:
            mean_grad = 0.0
            var_grad = 0.0
            norm_grad = 0.0
        
        # Update statistics
        self.gradient_stats['mean'].append(mean_grad)
        self.gradient_stats['var'].append(var_grad)
        self.gradient_stats['norm'].append(norm_grad)
        
        # Update layer norms
        for layer_name, norms in layer_norms.items():
            if layer_name not in self.gradient_stats['layer_norms']:
                self.gradient_stats['layer_norms'][layer_name] = []
            
            if norms:
                norm_value = np.mean(norms)
                # Cap extreme values
                norm_value = min(norm_value, 1e6)
                self.gradient_stats['layer_norms'][layer_name].append(norm_value)
            else:
                self.gradient_stats['layer_norms'][layer_name].append(0.0)
        
        return {
            'mean': mean_grad,
            'var': var_grad,
            'norm': norm_grad,
            'layer_norms': {k: v[-1] for k, v in self.gradient_stats['layer_norms'].items()}
        }
    
    def save_gradient_stats(self, save_path):
        """Save gradient statistics to disk."""
        with open(save_path, 'w') as f:
            json.dump(self.gradient_stats, f, indent=4)

class AttentionVisualizer:
    """Visualizes attention patterns in transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.attention_weights = {}
        self.hooks = []  # Add hooks attribute
        
    def _register_hooks(self):
        """Register hooks to collect attention weights."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)  # Save hook reference
                
    def _attention_hook(self, module, input, output):
        """Hook function to store attention weights."""
        if hasattr(module, 'attn'):
            self.attention_weights[module] = module.attn.detach().cpu()
        # Add a hook for the attention_double function
        if output[1] is not None and isinstance(output[1], torch.Tensor):
            layer_idx = len(self.attention_weights)
            head_idx = 0
            self.attention_weights[f'layer_{layer_idx}_head_{head_idx}'] = output[1].detach().cpu()
            
    def plot_attention_heatmap(self, weights, save_path):
        """Plot attention weights as a heatmap."""
        if not weights:
            return False
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis')
        plt.title('Attention Weights')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return True
    
    def plot_attention_maps(self, save_path):
        """Plot all attention maps and save to file."""
        if not self.attention_weights:
            print("No attention weights collected")
            return False
            
        num_maps = len(self.attention_weights)
        if num_maps == 0:
            return False
            
        fig, axes = plt.subplots(nrows=1, ncols=num_maps, figsize=(6*num_maps, 6))
        if num_maps == 1:
            axes = [axes]
            
        for i, (key, attn) in enumerate(self.attention_weights.items()):
            if isinstance(attn, torch.Tensor):
                attn_np = attn.numpy() if not torch.is_tensor(attn) else attn.detach().cpu().numpy()
                if len(attn_np.shape) > 2:
                    attn_np = attn_np[0]  # Take the first batch
                sns.heatmap(attn_np, cmap='viridis', ax=axes[i])
                axes[i].set_title(f'Attention Map {key}')
                
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return True
    
    def collect_attention_weights(self):
        """Collect attention weights after forward pass."""
        return self.attention_weights
    
    def save_attention_data(self, save_path):
        """Save attention data to disk."""
        # Convert to serializable format
        serializable_data = {}
        for key, value in self.attention_weights.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
                
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
            
    def get_attn_map(self):
        """Return attention map data for visualization."""
        return self.attention_weights
        
    def get_attention_maps(self, model, input):
        """Get attention maps directly from model forward pass with the given input."""
        if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
            with torch.no_grad():
                _, attn_dict = model(input, record_attn=True)
            self.attention_weights = attn_dict
            return attn_dict
        else:
            print("Model doesn't have a compatible forward method")
            return {}

class HiddenStateAnalyzer:
    """Analyzes hidden state representations."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.hidden_states = {}
        self.hooks = []  # Add hooks attribute
        
    def _register_hooks(self):
        """Register hooks to collect hidden states."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._hidden_hook(m, i, o, name)
                )
                self.hooks.append(hook)  # Save hook reference
                
    def _hidden_hook(self, module, input, output, name):
        """Hook function to store hidden states."""
        self.hidden_states[name] = {
            'input': input[0].detach().cpu(),
            'output': output.detach().cpu()
        }
        
    def compare_with_solver(self, solver_outputs):
        """Compare model outputs with solver outputs."""
        comparisons = {}
        try:
            for state in self.hidden_states.values():
                if state['name'] == 'output_layer':
                    model_output = state['output'].numpy()
                    solver_outputs_np = np.array(solver_outputs)
                    
                    # Make sure shapes are compatible
                    if model_output.shape != solver_outputs_np.shape:
                        print(f"Shape mismatch: model_output {model_output.shape}, solver_outputs {solver_outputs_np.shape}")
                        
                        # Try to make them compatible
                        model_flat = model_output.flatten()[:min(model_output.size, solver_outputs_np.size)]
                        solver_flat = solver_outputs_np.flatten()[:min(model_output.size, solver_outputs_np.size)]
                        
                        cosine_sim = float(cosine_similarity([model_flat], [solver_flat])[0][0])
                        l2_dist = float(np.linalg.norm(model_flat - solver_flat))
                    else:
                        cosine_sim = float(
                            np.mean([cosine_similarity(m.reshape(1, -1), s.reshape(1, -1))[0][0] 
                                    for m, s in zip(model_output, solver_outputs_np)])
                        )
                        l2_dist = float(
                            np.mean([np.linalg.norm(m - s) for m, s in zip(model_output, solver_outputs_np)])
                        )
                    
                    comparisons['cosine_similarity'] = cosine_sim
                    comparisons['l2_distance'] = l2_dist
        except Exception as e:
            print(f"Error in compare_with_solver: {str(e)}")
            comparisons['cosine_similarity'] = 0.0
            comparisons['l2_distance'] = 0.0
            
        return comparisons 