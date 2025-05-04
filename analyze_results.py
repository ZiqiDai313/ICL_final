import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import glob

def load_experiment_data(exp_dir):
    """Load all analysis data from an experiment directory"""
    data = {
        'hessian': {},
        'gradients': {},
        'attention': {},
        'hidden_states': {},
        'metrics': None
    }
    
    # Load metrics
    metrics_path = os.path.join(exp_dir, 'metrics.npy')
    if os.path.exists(metrics_path):
        data['metrics'] = np.load(metrics_path, allow_pickle=True).item()
    
    # Load Hessian data
    hessian_files = glob.glob(os.path.join(exp_dir, 'analysis/hessian/*.npy'))
    for f in hessian_files:
        iter_num = int(os.path.basename(f).split('_')[1].split('.')[0])
        data['hessian'][iter_num] = np.load(f, allow_pickle=True).item()
    
    # Load gradient data
    grad_files = glob.glob(os.path.join(exp_dir, 'analysis/gradients/*.npy'))
    for f in grad_files:
        iter_num = int(os.path.basename(f).split('_')[2].split('.')[0])
        data['gradients'][iter_num] = np.load(f, allow_pickle=True).item()
    
    # Load hidden state comparisons
    hidden_files = glob.glob(os.path.join(exp_dir, 'analysis/hidden_states/*.npy'))
    for f in hidden_files:
        iter_num = int(os.path.basename(f).split('_')[1].split('.')[0])
        data['hidden_states'][iter_num] = np.load(f, allow_pickle=True).item()
    
    return data

def plot_hessian_analysis(data, save_dir):
    """Plot Hessian trace and eigenvalues over time"""
    if not data['hessian']:
        return
    
    # Extract data
    iterations = sorted(data['hessian'].keys())
    traces = [data['hessian'][i]['trace'] for i in iterations]
    eigenvalues = [data['hessian'][i]['eigenvalues'] for i in iterations]
    
    # Plot trace
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, traces)
    plt.xlabel('Iteration')
    plt.ylabel('Hessian Trace')
    plt.title('Hessian Trace Over Time')
    plt.savefig(os.path.join(save_dir, 'hessian_trace.png'))
    plt.close()
    
    # Plot eigenvalues
    plt.figure(figsize=(10, 5))
    for i in range(len(eigenvalues[0])):
        plt.plot(iterations, [e[i] for e in eigenvalues], label=f'Eigenvalue {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue')
    plt.title('Top Hessian Eigenvalues Over Time')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'hessian_eigenvalues.png'))
    plt.close()

def plot_gradient_analysis(data, save_dir):
    """Plot gradient statistics over time"""
    if not data['gradients']:
        return
    
    # Extract data
    iterations = sorted(data['gradients'].keys())
    param_names = list(data['gradients'][iterations[0]].keys())
    
    # Plot mean gradient norms for each parameter
    plt.figure(figsize=(12, 6))
    for param in param_names:
        means = [data['gradients'][i][param]['mean'] for i in iterations]
        plt.plot(iterations, means, label=param)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Gradient Norm')
    plt.title('Mean Gradient Norms Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_norms.png'))
    plt.close()

def plot_hidden_state_analysis(data, save_dir):
    """Plot hidden state comparisons with solver"""
    if not data['hidden_states']:
        return
    
    # Extract data
    iterations = sorted(data['hidden_states'].keys())
    layer_names = list(data['hidden_states'][iterations[0]].keys())
    
    # Plot MSE between hidden states and solver outputs
    plt.figure(figsize=(12, 6))
    for layer in layer_names:
        mses = [data['hidden_states'][i][layer] for i in iterations]
        plt.plot(iterations, mses, label=layer)
    plt.xlabel('Iteration')
    plt.ylabel('MSE with Solver')
    plt.title('Hidden State Comparison with Solver')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hidden_state_comparison.png'))
    plt.close()

def analyze_experiment(exp_dir):
    """Analyze a single experiment"""
    print(f"Analyzing experiment: {exp_dir}")
    
    # Create analysis directory
    analysis_dir = os.path.join(exp_dir, 'analysis_plots')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load data
    data = load_experiment_data(exp_dir)
    
    # Generate plots
    plot_hessian_analysis(data, analysis_dir)
    plot_gradient_analysis(data, analysis_dir)
    plot_hidden_state_analysis(data, analysis_dir)
    
    # Save summary statistics
    if data['metrics']:
        summary = {
            'final_train_loss': data['metrics']['train_loss'][-1],
            'final_test_loss': data['metrics']['test_loss'][-1] if 'test_loss' in data['metrics'] else None,
            'final_grad_norm': data['metrics']['grad_norms'][-1],
            'hessian_trace': data['hessian'][max(data['hessian'].keys())]['trace'] if data['hessian'] else None,
            'top_eigenvalue': data['hessian'][max(data['hessian'].keys())]['eigenvalues'][0] if data['hessian'] else None
        }
        
        with open(os.path.join(analysis_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

def main():
    """Analyze all experiments in the results directory"""
    results_dir = 'results'
    
    # Find all experiment directories
    exp_dirs = []
    for root, dirs, files in os.walk(results_dir):
        if 'config.json' in files:
            exp_dirs.append(root)
    
    # Analyze each experiment
    for exp_dir in exp_dirs:
        try:
            analyze_experiment(exp_dir)
        except Exception as e:
            print(f"Error analyzing {exp_dir}: {e}")

if __name__ == '__main__':
    main() 