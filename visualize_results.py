import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from collections import defaultdict
from matplotlib.colors import LogNorm
import re

def load_experiment_data(experiment_dir):
    """加载实验数据"""
    results = {}
    
    # 查找所有实验结果文件夹
    if os.path.exists(experiment_dir):
        exp_folders = sorted(glob.glob(os.path.join(experiment_dir, "*")))
        latest_folder = exp_folders[-1]  # 使用最新的实验结果
        print(f"使用最新实验结果: {latest_folder}")
        
        # 查找所有结果文件
        experiment_files = glob.glob(os.path.join(latest_folder, "*.json"))
        
        for exp_file in experiment_files:
            with open(exp_file, 'r') as f:
                data = json.load(f)
                task_name = f"{data['function_type']}_{data['optimizer']}"
                results[task_name] = data
                
        return results, latest_folder
    else:
        print(f"实验目录 {experiment_dir} 不存在")
        return None, None

def plot_results(results, output_dir):
    """可视化实验结果的对比"""
    if not results:
        print("没有结果可以可视化")
        return
    
    # 准备数据
    tasks = list(results.keys())
    train_losses = [results[task].get('final_train_loss', float('nan')) for task in tasks]
    test_losses = [results[task].get('final_test_loss', float('nan')) for task in tasks]
    gen_gaps = [results[task].get('generalization_gap', float('nan')) for task in tasks]
    
    # 对非数值结果进行处理
    train_losses = [float(x) if (isinstance(x, (int, float)) and not np.isnan(x) and x != float('inf')) else np.nan for x in train_losses]
    test_losses = [float(x) if (isinstance(x, (int, float)) and not np.isnan(x) and x != float('inf')) else np.nan for x in test_losses]
    gen_gaps = [float(x) if (isinstance(x, (int, float)) and not np.isnan(x) and x != float('inf')) else np.nan for x in gen_gaps]
    
    # 绘制三种指标的柱状图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 训练损失
    axes[0].bar(tasks, train_losses)
    axes[0].set_title('训练损失对比')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticklabels(tasks, rotation=45)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 测试损失
    axes[1].bar(tasks, test_losses)
    axes[1].set_title('测试损失对比')
    axes[1].set_ylabel('Loss')
    axes[1].set_xticklabels(tasks, rotation=45)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 泛化间隙
    axes[2].bar(tasks, gen_gaps)
    axes[2].set_title('泛化间隙对比')
    axes[2].set_ylabel('Gap')
    axes[2].set_xticklabels(tasks, rotation=45)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_comparison.png'), dpi=300)
    print(f"结果对比图已保存到 {os.path.join(output_dir, 'results_comparison.png')}")
    
    # 绘制学习曲线
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task in enumerate(tasks):
        # 获取训练损失历史
        losses = results[task].get('train_loss_history', [])
        if losses:
            iterations = list(range(len(losses)))
            axes[i].plot(iterations, losses)
            axes[i].set_title(f'{task} 学习曲线')
            axes[i].set_xlabel('迭代次数')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)
    print(f"学习曲线已保存到 {os.path.join(output_dir, 'learning_curves.png')}")

def find_latest_experiment():
    """Find the most recent experiment directory"""
    latest_exp = None
    latest_time = 0
    
    for dir_name in os.listdir('results'):
        if dir_name.startswith('experiment_'):
            exp_path = os.path.join('results', dir_name)
            if os.path.isdir(exp_path):
                create_time = os.path.getctime(exp_path)
                if create_time > latest_time:
                    latest_time = create_time
                    latest_exp = exp_path
    
    return latest_exp

def find_experiment_directories(base_dir):
    """Find all experiment directories for different optimizer/task combinations"""
    experiment_dirs = {}
    
    for function_type in ['linear', 'quadratic']:
        experiment_dirs[function_type] = {}
        for optimizer in ['adam', 'sgd', 'adagrad']:
            # Find matching directory
            for root, dirs, _ in os.walk(base_dir):
                for d in dirs:
                    if f"{function_type}_{optimizer}_" in d:
                        exp_dir = os.path.join(root, d)
                        experiment_dirs[function_type][optimizer] = exp_dir
                        break
    
    return experiment_dirs

def load_metrics(exp_dir):
    """Load training metrics from an experiment directory"""
    metrics_file = os.path.join(exp_dir, 'metrics.json')
    final_results_file = os.path.join(exp_dir, 'final_results.json')
    
    metrics = {'train_loss': [], 'test_loss': [], 'iter_list': []}
    final_results = {'final_train_loss': None, 'final_test_loss': None}
    
    # Load metrics if available
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    # Load final results if available
    if os.path.exists(final_results_file):
        with open(final_results_file, 'r') as f:
            final_results = json.load(f)
    
    return metrics, final_results

def plot_loss_curves(experiment_dirs, output_dir):
    """Plot loss curves for all experiments"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    for i, function_type in enumerate(['linear', 'quadratic']):
        ax = axes[i]
        
        for optimizer in ['adam', 'sgd', 'adagrad']:
            if optimizer in experiment_dirs[function_type]:
                exp_dir = experiment_dirs[function_type][optimizer]
                metrics, _ = load_metrics(exp_dir)
                
                if metrics['iter_list'] and metrics['train_loss']:
                    # Plot training loss
                    ax.plot(metrics['iter_list'], metrics['train_loss'], 
                           label=f"{optimizer.upper()} (train)", 
                           linewidth=2)
                    
                    # Plot test loss at the end if available
                    if metrics['iter_list'] and metrics['test_loss']:
                        ax.plot(metrics['iter_list'], metrics['test_loss'], 
                               label=f"{optimizer.upper()} (test)", 
                               linestyle='--', linewidth=2)
        
        ax.set_title(f"{function_type.capitalize()} Regression Task", fontsize=16)
        ax.set_ylabel("Loss (MSE)", fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
    
    axes[1].set_xlabel("Iterations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.close()

def create_heatmap(experiment_dirs, output_dir):
    """Create heatmap of final test losses"""
    function_types = ['linear', 'quadratic']
    optimizers = ['adam', 'sgd', 'adagrad']
    
    # Create data matrix for heatmap
    data = np.zeros((len(function_types), len(optimizers)))
    
    # Fill in the data
    for i, function_type in enumerate(function_types):
        for j, optimizer in enumerate(optimizers):
            if optimizer in experiment_dirs[function_type]:
                exp_dir = experiment_dirs[function_type][optimizer]
                _, final_results = load_metrics(exp_dir)
                
                if final_results['final_test_loss'] is not None:
                    data[i, j] = final_results['final_test_loss']
                else:
                    data[i, j] = np.nan
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis',
                    xticklabels=[opt.upper() for opt in optimizers],
                    yticklabels=[f"{ft.capitalize()}" for ft in function_types])
    
    plt.title('Final Test Loss by Task and Optimizer', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_loss_heatmap.png'), dpi=300)
    plt.close()

def create_generalization_gap_heatmap(experiment_dirs, output_dir):
    """Create heatmap of generalization gaps"""
    function_types = ['linear', 'quadratic']
    optimizers = ['adam', 'sgd', 'adagrad']
    
    # Create data matrix for heatmap
    data = np.zeros((len(function_types), len(optimizers)))
    
    # Fill in the data
    for i, function_type in enumerate(function_types):
        for j, optimizer in enumerate(optimizers):
            if optimizer in experiment_dirs[function_type]:
                exp_dir = experiment_dirs[function_type][optimizer]
                _, final_results = load_metrics(exp_dir)
                
                if (final_results['final_train_loss'] is not None and 
                    final_results['final_test_loss'] is not None):
                    
                    # Handle infinity and NaN values
                    train_loss = final_results['final_train_loss']
                    test_loss = final_results['final_test_loss']
                    
                    if isinstance(train_loss, str) and train_loss.lower() == 'inf':
                        data[i, j] = np.nan
                    else:
                        try:
                            gap = float(test_loss) - float(train_loss)
                            data[i, j] = gap
                        except (ValueError, TypeError):
                            data[i, j] = np.nan
                else:
                    data[i, j] = np.nan
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data, annot=True, fmt='.3f', cmap='coolwarm',
                    xticklabels=[opt.upper() for opt in optimizers],
                    yticklabels=[f"{ft.capitalize()}" for ft in function_types])
    
    plt.title('Generalization Gap by Task and Optimizer', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_gap_heatmap.png'), dpi=300)
    plt.close()

def create_summary_table(experiment_dirs, output_dir):
    """Create a summary table of results"""
    table_data = []
    
    # Header row
    header = ["Task", "Optimizer", "Train Loss", "Test Loss", "Gen. Gap"]
    table_data.append(header)
    
    # Add separator row
    table_data.append(["-" * len(h) for h in header])
    
    # Data rows
    for function_type in ['linear', 'quadratic']:
        for optimizer in ['adam', 'sgd', 'adagrad']:
            if optimizer in experiment_dirs[function_type]:
                exp_dir = experiment_dirs[function_type][optimizer]
                _, final_results = load_metrics(exp_dir)
                
                train_loss = final_results.get('final_train_loss', "N/A")
                test_loss = final_results.get('final_test_loss', "N/A")
                
                # Calculate generalization gap
                if train_loss != "N/A" and test_loss != "N/A":
                    if isinstance(train_loss, str) and train_loss.lower() == 'inf':
                        gen_gap = "N/A"
                    else:
                        try:
                            gen_gap = float(test_loss) - float(train_loss)
                            gen_gap = f"{gen_gap:.6f}"
                        except (ValueError, TypeError):
                            gen_gap = "N/A"
                else:
                    gen_gap = "N/A"
                
                # Format losses
                if isinstance(train_loss, (int, float)):
                    train_loss = f"{train_loss:.6f}"
                if isinstance(test_loss, (int, float)):
                    test_loss = f"{test_loss:.6f}"
                
                row = [
                    function_type.capitalize(),
                    optimizer.upper(),
                    train_loss,
                    test_loss,
                    gen_gap
                ]
                
                table_data.append(row)
    
    # Write table to file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        for row in table_data:
            f.write("{:<12} {:<10} {:<15} {:<15} {:<15}\n".format(*row))

def parse_logs_for_metrics(experiment_dirs):
    """Parse log files to extract metrics when JSON files aren't available"""
    for function_type in experiment_dirs:
        for optimizer in experiment_dirs[function_type]:
            exp_dir = experiment_dirs[function_type][optimizer]
            log_file = os.path.join(exp_dir, 'experiment.log')
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                # Extract final train and test loss
                train_loss_match = re.search(r'Final train loss: ([0-9.]+|inf)', log_content)
                test_loss_match = re.search(r'Final test loss: ([0-9.]+|inf)', log_content)
                
                if train_loss_match and test_loss_match:
                    train_loss = train_loss_match.group(1)
                    test_loss = test_loss_match.group(1)
                    
                    # Save as final_results.json
                    final_results = {
                        'final_train_loss': train_loss,
                        'final_test_loss': test_loss
                    }
                    
                    with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
                        json.dump(final_results, f)

def main():
    # Find latest experiment directory
    latest_exp = find_latest_experiment()
    if not latest_exp:
        print("No experiment directories found.")
        return
    
    print(f"Analyzing results from: {latest_exp}")
    
    # Find all experiment directories
    experiment_dirs = find_experiment_directories(latest_exp)
    
    # Parse logs for metrics if needed
    parse_logs_for_metrics(experiment_dirs)
    
    # Create visualization output directory
    output_dir = os.path.join(latest_exp, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    plot_loss_curves(experiment_dirs, output_dir)
    create_heatmap(experiment_dirs, output_dir)
    create_generalization_gap_heatmap(experiment_dirs, output_dir)
    create_summary_table(experiment_dirs, output_dir)
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 