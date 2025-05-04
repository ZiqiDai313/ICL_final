import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_experiment_results(results_dir):
    """Load results from all experiments in the results directory"""
    results = {}
    for exp_dir in Path(results_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        # Load metrics
        metrics_path = exp_dir / 'metrics.npy'
        if not metrics_path.exists():
            continue
            
        metrics = np.load(metrics_path, allow_pickle=True).item()
        
        # Load config and final results
        with open(exp_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        final_results_path = exp_dir / 'final_results.json'
        if final_results_path.exists():
            with open(final_results_path, 'r') as f:
                final_results = json.load(f)
        else:
            final_results = {}
            
        key = f"{config['optimizer']}_{config['function_type']}"
        results[key] = {
            'metrics': metrics,
            'config': config,
            'final_results': final_results
        }
    
    return results

def plot_optimizer_comparison(results, save_dir):
    """Create detailed comparison plots for different optimizers and tasks"""
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    
    # Separate results by task type
    linear_results = {k: v for k, v in results.items() if 'linear' in k}
    quadratic_results = {k: v for k, v in results.items() if 'quadratic' in k}
    
    # Plot training loss curves by task type
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Linear regression tasks
    for key, data in linear_results.items():
        optimizer = key.split('_')[0]
        metrics = data['metrics']
        ax1.plot(metrics['iter_list'], metrics['train_loss'], 
                label=f'{optimizer}', linewidth=2)
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Training Loss (MSE)')
    ax1.set_title('Linear Regression - Optimizer Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Quadratic regression tasks
    for key, data in quadratic_results.items():
        optimizer = key.split('_')[0]
        metrics = data['metrics']
        ax2.plot(metrics['iter_list'], metrics['train_loss'], 
                label=f'{optimizer}', linewidth=2)
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Training Loss (MSE)')
    ax2.set_title('Quadratic Regression - Optimizer Comparison')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_specific_comparison.png'))
    plt.close()
    
    # Create summary tables
    summary_data = []
    for key, data in results.items():
        optimizer, func_type = key.split('_')
        metrics = data['metrics']
        final_results = data['final_results']
        
        summary = {
            'Optimizer': optimizer,
            'Function Type': func_type,
            'Final Train Loss': f"{metrics['train_loss'][-1]:.6f}",
            'Final Test Loss': final_results.get('final_test_loss', 'N/A'),
            'Generalization Gap': final_results.get('generalization_gap', 'N/A'),
            'Convergence Iter': np.argmin(metrics['train_loss']),
            'Mean Gradient Norm': f"{np.mean(metrics['grad_norms']):.6f}"
        }
        summary_data.append(summary)
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(summary_data)
    
    # Save summary as both JSON and CSV
    df.to_json(os.path.join(save_dir, 'optimizer_comparison_summary.json'), orient='records', indent=4)
    df.to_csv(os.path.join(save_dir, 'optimizer_comparison_summary.csv'), index=False)
    
    # Create heatmap of final losses
    plt.figure(figsize=(10, 6))
    pivot_df = df.pivot(index='Function Type', columns='Optimizer', values='Final Train Loss')
    pivot_df = pivot_df.astype(float)
    sns.heatmap(pivot_df, annot=True, fmt='.2e', cmap='YlOrRd')
    plt.title('Final Training Loss Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_heatmap.png'))
    plt.close()

if __name__ == "__main__":
    results_dir = "results"
    results = load_experiment_results(results_dir)
    plot_optimizer_comparison(results, results_dir) 