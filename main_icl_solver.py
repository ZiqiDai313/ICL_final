"""
Main ICL solver for comparing optimizer effects on different regression tasks.

This module implements the core training and evaluation loop for studying how different
optimizers affect in-context learning (ICL) abilities of transformer models. Currently
supports both linear and quadratic regression tasks, with extensibility for other
function classes.

Key Features:
- Modular task support (linear/quadratic regression)
- Configurable optimizer choice (Adam, SGD, Adagrad)
- Comprehensive logging and visualization
- Easy extension to new function classes
"""

from data_generator import generate_quadratic_data, prepare_icl_data
from data_solver import generate_linear_system_batch, convert_data_solver_to_tf
from analysis_utils import HessianAnalyzer, GradientTracker, AttentionVisualizer, HiddenStateAnalyzer
import torch
import sys
import numpy as np
from baseline_solver import cgd
from scipy.sparse.linalg import cg
from collections import defaultdict
import argparse
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from linear_transformer import in_context_loss, Transformer_F, Transformer_F_w_embed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_experiment_dir(args):
    """Create experiment directory and save config"""
    exp_name = f"{args.function_type}_{args.optimizer}_dim{args.dim}_layers{args.n_layer}_heads{args.n_head}_cond{args.condition_number}"
    exp_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create analysis subdirectories
    os.makedirs(os.path.join(exp_dir, 'analysis'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'analysis/hessian'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'analysis/gradients'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'analysis/attention'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'analysis/hidden_states'), exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(exp_dir, 'experiment.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log experiment configuration
    logging.info(f"Starting experiment with {args.function_type} regression task")
    logging.info(f"Optimizer: {args.optimizer}")
    logging.info(f"Model configuration: {args.n_layer} layers, {args.n_head} heads, dimension {args.dim}")
    
    # Save experiment config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    return exp_dir

def plot_learning_curves(train_losses, test_losses, save_path):
    """Plot and save training and test loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses['iter_list'], train_losses['loss'], label='Train Loss')
    plt.plot(test_losses['iter_list'], test_losses['loss'], label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(save_path)
    plt.close()

parser = argparse.ArgumentParser(description='ICL Optimizer Effects Study')
parser.add_argument('--log_dir', type=str, default='results', help='Directory for saving results')
parser.add_argument('--only_eval', type=str2bool, default=False)
parser.add_argument('--model', type=str, default='linear_w_embed')
parser.add_argument('--optimizer', type=str, default='adam', 
                   choices=['adam', 'sgd', 'adagrad'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dim', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--condition_number', type=int, default=5)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--function_type', type=str, default='linear', choices=['linear', 'quadratic'])
parser.add_argument('--context_length', type=int, default=8)
parser.add_argument('--super_large', type=str2bool, default=False)
args = parser.parse_args()

# Setup device and experiment directory
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Explicitly specify second GPU (NVIDIA RTX 5080)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Setup CUDA optimization
    torch.cuda.set_device(0)
    # Parallelize memory transfers and computation
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Use NVIDIA optimizer (if available)
    try:
        import apex
        print("APEX is available for mixed precision training")
    except ImportError:
        print("APEX not available, using PyTorch native mixed precision")
else:
    print("Using CPU")

# GPU warmup
if torch.cuda.is_available():
    # GPU intensive warmup - more aggressive but with smaller memory footprint
    print("Performing GPU intensive warmup...")
    # Create large matrices and perform matrix multiplication
    batch_size = 128   # Smaller batch size
    seq_len = 512      # Smaller sequence length
    hidden_dim = 1024  # Smaller hidden dimension
    
    # Create several large tensors and perform operations
    print(f"Creating large test tensor: {batch_size}x{seq_len}x{hidden_dim}...")
    
    # Batch create large random tensors
    for i in range(10):  # Create 10 large test batches
        print(f"Warmup batch {i+1}/10")
        a = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        b = torch.randn(batch_size, hidden_dim, seq_len, device=device)
        # Perform some compute-intensive operations
        c = torch.bmm(a, b)  # Batch matrix multiplication
        # Add some random operations
        d = torch.relu(c)
        e = torch.softmax(d, dim=-1)
        f = torch.mean(e, dim=1)
        # Ensure computation completes
        torch.cuda.synchronize()
        
        # Report memory usage
        used_mem = torch.cuda.memory_allocated() / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory usage: {used_mem:.2f}GB / {total_mem:.2f}GB ({used_mem/total_mem*100:.1f}%)")
        
        # Free some memory but keep GPU active
        del a, b, c, d, e, f
        torch.cuda.empty_cache()
    
    print("GPU intensive warmup complete")

# Set random seeds
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Create experiment directory
exp_dir = setup_experiment_dir(args)

# Initialize model
var = 0.0001  # initialization scale
if args.model == 'linear_w_embed':
    # Check if super large model parameters already exist
    if hasattr(args, 'super_large') and args.super_large:
        # Create super large model to maximize GPU usage
        print("Creating super large model to maximize GPU usage...")
        # Force extremely large dimensions regardless of user settings
        large_dim = 512
        large_layers = 36
        large_heads = 32
        model = Transformer_F_w_embed(large_layers, large_heads, large_dim, var, 
                                    hidden_dim=large_dim*4, device=device)
        print(f"Created model with {large_layers} layers, {large_heads} heads, {large_dim} dimensions")
        
        # Summarize model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    else:
        model = Transformer_F_w_embed(args.n_layer, args.n_head, args.dim, var, 
                                    hidden_dim=args.dim*4, device=device)
else:
    model = Transformer_F(args.n_layer, args.n_head, args.dim, var)
# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()
model = model.to(device)

# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 0.1,
        total_steps=args.max_iters,
        pct_start=0.1,  # 10% warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1e4  # final_lr = initial_lr/1e4
    )
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

criterion = nn.MSELoss()

# Training metrics
metrics = {
    'train_loss': [],
    'test_loss': [],
    'iter_list': [],
    'grad_norms': []
}

if not args.only_eval:
    logging.info(f"Beginning training on {args.function_type} regression task with {args.optimizer} optimizer")
    
    # Initialize analysis tools
    hessian_analyzer = HessianAnalyzer(model, criterion)
    gradient_tracker = GradientTracker(model)
    attention_visualizer = AttentionVisualizer(model)
    hidden_state_analyzer = HiddenStateAnalyzer(model)
    
    # Register hooks for tracking
    gradient_tracker._register_hooks()
    attention_visualizer._register_hooks()
    hidden_state_analyzer._register_hooks()
    
    for t in range(args.max_iters):
        start = time.time()
        
        # Create multiple batches for parallel processing
        batch_multiplier = 4  # Process 4 batches simultaneously
        total_loss = 0.0
        
        for b in range(batch_multiplier):
            # Generate data
            if args.function_type == 'linear':
                Z, x_batch = generate_linear_system_batch(args.batch_size, args.dim, 
                                                        args.condition_number, mode='tf')
                task_type = "Linear Regression"
                # Ensure Z has correct dimensions [batch_size, seq_len, dim]
                if len(Z.shape) == 2:
                    Z = Z.unsqueeze(0)  # Add batch dimension
                if len(x_batch.shape) == 1:
                    x_batch = x_batch.unsqueeze(0)  # Add batch dimension
            else:  # quadratic
                X, y, params = generate_quadratic_data(args.batch_size, args.dim, 
                                                     args.condition_number)
                Z, y = prepare_icl_data(X, y, args.context_length, params)
                x_batch = y
                task_type = "Quadratic Regression"
                # Ensure Z has correct dimensions [batch_size, seq_len, dim]
                if len(Z.shape) == 2:
                    Z = Z.unsqueeze(0)  # Add batch dimension
                if len(x_batch.shape) == 1:
                    x_batch = x_batch.unsqueeze(0)  # Add batch dimension
            
            # Pre-allocate CUDA memory and non-blockingly transfer data to GPU
            with torch.cuda.stream(torch.cuda.Stream()):
                Z = Z.to(device, non_blocking=True)
                x_batch = x_batch.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                output = model(Z)
                if args.function_type == 'linear':
                    # Extract the relevant part of the output for loss calculation
                    output_for_loss = output[:, -1, :args.dim]  # [batch_size, dim]
                    loss = criterion(output_for_loss, x_batch)
                else:  # quadratic
                    # Extract the relevant part of the output for loss calculation
                    output_for_loss = output[:, -1, :args.dim]  # [batch_size, dim]
                    loss = criterion(output_for_loss, x_batch)
                
                # Accumulate loss from multiple batches
                total_loss += loss.item()
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
        
        # Average loss from multiple batches
        loss = total_loss / batch_multiplier
        
        # Gradient clipping with different thresholds for different optimizers
        if args.optimizer == 'sgd':
            # Clip scaled gradients
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), min(args.clip * 0.01, 1.0))
            # Scale gradients for stability
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data / (1 + grad_norm)
            
            # Ensure weights are updated every iteration
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        else:
            # Clip scaled gradients - more aggressive for Adam/Adagrad
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), min(args.clip * 0.05, 1.0))
            # Additional stability measure for Adam/Adagrad
            if args.function_type == 'quadratic':
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        param.grad.data.fill_(0)  # Zero out non-finite gradients
                    elif param.grad is not None:
                        param.grad.data = param.grad.data / (1 + grad_norm)
                        
            scaler.step(optimizer)
            scaler.update()
        
        # Pre-fetch next batch of data
        torch.cuda.current_stream().synchronize()
        
        end = time.time()
        
        # Monitor GPU usage
        if torch.cuda.is_available() and t % 10 == 0:
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Move loss to CPU for logging
        loss_item = loss
        grad_norm_item = grad_norm.item()
        
        # Log metrics
        metrics['train_loss'].append(loss_item)
        metrics['iter_list'].append(t)
        metrics['grad_norms'].append(grad_norm_item)
        
        if t % 100 == 0 or t < 5:
            log_msg = (f"{task_type} - {args.optimizer} optimizer | "
                      f"iter {t} | Loss: {loss_item:.6f} | "
                      f"time: {end-start:.3f} | grad_norm: {grad_norm_item:.6f}")
            logging.info(log_msg)
            print(log_msg)
            
            # Sparse execution of expensive analysis
            do_heavy_analysis = (t % 500 == 0)
            
            # Use separate CUDA stream asynchronously execute analysis
            if do_heavy_analysis:
                # Allocate separate CUDA stream for analysis without blocking main training stream
                with torch.cuda.stream(torch.cuda.Stream()):
                    try:
                        # Move data to CPU for analysis
                        Z_cpu = Z.cpu()
                        x_batch_cpu = x_batch.cpu()
                        
                        # Hessian analysis with error handling
                        subset_size = min(16, Z_cpu.shape[0])  # Smaller subset size
                        Z_subset = Z_cpu[:subset_size]
                        x_batch_subset = x_batch_cpu[:subset_size]
                        
                        # Asynchronously compute Hessian analysis
                        torch.cuda.synchronize()
                        
                        # Compute Hessian trace and eigenvalues
                        print(f"Computing Hessian trace and eigenvalues at iteration {t}...")
                        hessian_trace = hessian_analyzer.compute_hessian_trace(Z_subset, x_batch_subset)
                        eigenvalues = hessian_analyzer.compute_eigenvalues(Z_subset, x_batch_subset, top_n=3)
                        
                        # Process eigenvalues to floats
                        def process_eigenvalues(e):
                            if isinstance(e, (list, tuple)):
                                result = []
                                for v in e:
                                    result.extend(process_eigenvalues(v))
                                return result
                            elif isinstance(e, torch.Tensor):
                                return [float(e.item())]
                            elif isinstance(e, (int, float)):
                                return [float(e)]
                            else:
                                return [0.0]  # Default for invalid values
                        
                        processed_eigenvalues = process_eigenvalues(eigenvalues)
                        
                        # Save Hessian data
                        hessian_data = {
                            'iter': t,
                            'trace': float(hessian_trace),
                            'eigenvalues': processed_eigenvalues
                        }
                        np.save(os.path.join(exp_dir, f'analysis/hessian/hessian_{t}.npy'), hessian_data)
                        print(f"Saved Hessian data for iteration {t}")
                        
                        # Gradient analysis
                        print(f"Computing gradient statistics at iteration {t}...")
                        grad_stats = gradient_tracker.get_gradient_stats()
                        np.save(os.path.join(exp_dir, f'analysis/gradients/grad_stats_{t}.npy'), grad_stats)
                        print(f"Saved gradient statistics for iteration {t}")
                        
                        # Attention visualization
                        print(f"Generating attention maps at iteration {t}...")
                        attention_visualizer.plot_attention_maps(
                            os.path.join(exp_dir, f'analysis/attention/attention_{t}.png')
                        )
                        print(f"Saved attention maps for iteration {t}")
                        
                        # Hidden state analysis
                        if args.function_type == 'linear':
                            print(f"Computing hidden state comparisons at iteration {t}...")
                            # Compare with conjugate gradient solver
                            solver_outputs = cgd(Z_subset[:, :-1, :-1].mean(0), Z_subset[:, :-1, -1].mean(0))
                            hidden_comparisons = hidden_state_analyzer.compare_with_solver(solver_outputs)
                            np.save(os.path.join(exp_dir, f'analysis/hidden_states/comparisons_{t}.npy'), 
                                   hidden_comparisons)
                            print(f"Saved hidden state comparisons for iteration {t}")
                    except Exception as e:
                        print(f"Error during analysis at iteration {t}: {str(e)}")
            
            # Lightweight save that doesn't block GPU
            if do_heavy_analysis:
                # Asynchronously save checkpoint
                torch.cuda.synchronize()  # Ensure previous operations complete
                
                # Save checkpoint and metrics
                checkpoint = {
                    'iter': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_item,
                    'task_type': task_type,
                    'optimizer': args.optimizer
                }
                
                # Asynchronously save to disk
                torch.save(checkpoint, os.path.join(exp_dir, f'checkpoint_{t}.pt'))
                np.save(os.path.join(exp_dir, 'metrics.npy'), metrics)
                
                # Main thread continues training, avoiding batch processing bottleneck
                torch.cuda.current_stream().synchronize()
            
            # Plot learning curves
            plot_learning_curves(
                {'iter_list': metrics['iter_list'], 'loss': metrics['train_loss']},
                {'iter_list': metrics['iter_list'], 'loss': metrics['train_loss']},
                os.path.join(exp_dir, 'learning_curves.png')
            )
    
    # Remove hooks after training
    for hook in gradient_tracker.hooks:
        hook.remove()

    # Fix AttentionVisualizer missing hooks attribute issue
    try:
        for hook in attention_visualizer.hooks:
            hook.remove()
    except AttributeError:
        # If hooks attribute doesn't exist, it may be stored elsewhere or not initialized
        print("Note: AttentionVisualizer has no hooks attribute, skipping cleanup")

    # Fix HiddenStateAnalyzer missing hooks attribute issue
    try:
        for hook in hidden_state_analyzer.hooks:
            hook.remove()
    except AttributeError:
        # If hooks attribute doesn't exist, it may be stored elsewhere or not initialized
        print("Note: HiddenStateAnalyzer has no hooks attribute, skipping cleanup")

# Final evaluation
model.eval()
with torch.no_grad():
    if args.function_type == 'linear':
        test_data = torch.load('results/data/test_data.pt')['dim_9']
        A_batch, x_batch, b_batch = test_data['A_batch'], test_data['x_batch'], test_data['b_batch']
        Z, x_batch = convert_data_solver_to_tf(A_batch, x_batch, b_batch)
    else:
        X, y, params = generate_quadratic_data(100, args.dim, args.condition_number)  # Test set
        Z, y = prepare_icl_data(X, y, args.context_length, params)
        x_batch = y
        
    Z, x_batch = Z.to(device), x_batch.to(device)
    output = model(Z)
    test_loss = criterion(output[:, -1, :args.dim], x_batch).item()

# Save final results
final_results = {
    'optimizer': args.optimizer,
    'function_type': args.function_type,
    'final_train_loss': metrics['train_loss'][-1],
    'final_test_loss': test_loss,
    'generalization_gap': abs(metrics['train_loss'][-1] - test_loss),
}

with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
    json.dump(final_results, f, indent=4)

print("\nTraining completed! Final results:")
print(f"Function type: {args.function_type}")
print(f"Optimizer: {args.optimizer}")
print(f"Final train loss: {metrics['train_loss'][-1]:.6f}")
print(f"Final test loss: {test_loss:.6f}")
print(f"Generalization gap: {abs(metrics['train_loss'][-1] - test_loss):.6f}")