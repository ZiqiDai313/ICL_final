import os
import subprocess
import time
from datetime import datetime
import json

def main():
    print("Starting optimized quadratic regression experiments...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"stable_quadratic_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Only focus on quadratic problems
    function_type = "quadratic"
    
    # Optimizers to test
    optimizers = ["adam", "sgd", "adagrad"]
    
    # Record start time
    start_time = time.time()
    
    # Create experiment list with different learning rates
    experiments = []
    
    # Adam optimizer experiments with different learning rates
    adam_lrs = [1e-6, 5e-6, 1e-5]
    for lr in adam_lrs:
        experiments.append(("adam", lr))
    
    # SGD optimizer experiments with different learning rates
    sgd_lrs = [5e-6, 1e-5, 5e-5]
    for lr in sgd_lrs:
        experiments.append(("sgd", lr))
    
    # Adagrad optimizer experiments with different learning rates
    adagrad_lrs = [1e-5, 5e-5, 1e-4]
    for lr in adagrad_lrs:
        experiments.append(("adagrad", lr))
    
    total = len(experiments)
    print(f"Total experiments to run: {total}")
    
    # Run each experiment
    results = []
    for i, (optimizer, lr) in enumerate(experiments):
        print(f"\n[{i+1}/{total}] Running quadratic + {optimizer} with lr={lr} experiment...")
        
        # Create experiment subdir
        exp_name = f"quadratic_{optimizer}_lr{lr:.7f}"
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "main_icl_solver.py",
            f"--function_type={function_type}",
            f"--optimizer={optimizer}",
            f"--lr={lr}",
            "--n_layer=3",
            "--n_head=1",
            "--dim=9",
            "--condition_number=5",
            "--batch_size=64",  # Smaller batch for stability
            "--clip=0.05",      # Smaller clip value 
            "--max_iters=1000"  # Increased iterations to ensure convergence
        ]
        
        # Save experiment parameters
        with open(os.path.join(exp_dir, "experiment_params.json"), "w") as f:
            json.dump({
                "function_type": function_type,
                "optimizer": optimizer,
                "learning_rate": lr,
                "full_command": " ".join(cmd)
            }, f, indent=4)
        
        # Run command
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        try:
            process = subprocess.run(cmd, env=env, check=True)
            success = True
            print(f"✅ quadratic + {optimizer} (lr={lr}) experiment completed")
        except subprocess.CalledProcessError:
            success = False
            print(f"❌ quadratic + {optimizer} (lr={lr}) experiment failed")
        
        results.append({
            "function_type": function_type,
            "optimizer": optimizer,
            "learning_rate": lr,
            "success": success
        })
        
        # Show progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i+1)
        remaining = avg_time * (total - (i+1))
        
        print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        print(f"Time elapsed: {elapsed/60:.1f} minutes, estimated remaining: {remaining/60:.1f} minutes")
    
    # Collect results
    print("\nExperiments completed! Summary:")
    success_count = sum(1 for r in results if r["success"])
    print(f"Success: {success_count}/{total} ({success_count/total*100:.1f}%)")
    
    # Save summary results
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved in {results_dir}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 