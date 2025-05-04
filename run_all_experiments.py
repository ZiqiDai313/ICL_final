import os
import subprocess
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Starting all optimizer experiments...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"experiment_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Function types
    function_types = ["linear", "quadratic"]
    
    # Optimizers - removed three custom optimizers
    optimizers = ["adam", "sgd", "adagrad"]
    
    # Record start time
    start_time = time.time()
    
    # Create experiment list
    experiments = []
    for function_type in function_types:
        for optimizer in optimizers:
            experiments.append((function_type, optimizer))
    
    total = len(experiments)
    print(f"Total experiments to run: {total}")
    
    # Run each experiment
    results = []
    for i, (function_type, optimizer) in enumerate(experiments):
        print(f"\n[{i+1}/{total}] Running {function_type} + {optimizer} experiment...")
        
        # Set optimizer-specific parameters with separate settings for each function type
        if function_type == "linear":
            # Linear regression learning rates (original settings)
            if optimizer == "sgd":
                lr = 0.0001  # SGD uses smaller learning rate
            elif optimizer == "adagrad":
                lr = 0.01    # Adagrad uses larger learning rate
            else:  # Adam default
                lr = 0.001
        else:  # quadratic
            # Quadratic regression learning rates (modified for stability)
            if optimizer == "adam":
                lr = 0.000001  # Very small learning rate for Adam
            elif optimizer == "sgd":
                lr = 0.000005  # Very small learning rate for SGD
            elif optimizer == "adagrad":
                lr = 0.00001   # Very small learning rate for Adagrad
        
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
            "--batch_size=128",
            "--clip=0.1",  # Reduced clip value for all optimizers
            "--max_iters=1000"  # Increased iterations to ensure convergence
        ]
        
        # Run command
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        try:
            process = subprocess.run(cmd, env=env, check=True)
            success = True
            print(f"✅ {function_type} + {optimizer} experiment completed")
        except subprocess.CalledProcessError:
            success = False
            print(f"❌ {function_type} + {optimizer} experiment failed")
        
        results.append({
            "function_type": function_type,
            "optimizer": optimizer,
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
    
    # Analyze data in results directory
    print("\nAnalyzing final performance:")
    for function_type in function_types:
        print(f"\n{function_type.upper()} function results:")
        print("Optimizer\tTrain Loss\tTest Loss")
        print("-"*50)
        
        for optimizer in optimizers:
            # Find experiment directory
            exp_dir = None
            for root, dirs, _ in os.walk("results"):
                for d in dirs:
                    if f"{function_type}_{optimizer}_" in d:
                        exp_dir = os.path.join(root, d)
                        break
                if exp_dir:
                    break
            
            if not exp_dir:
                print(f"{optimizer.upper()}\t\tfailed\t\tfailed")
                continue
                
            # Read final results
            try:
                with open(os.path.join(exp_dir, "final_results.json"), "r") as f:
                    import json
                    results_data = json.load(f)
                    train_loss = results_data.get("final_train_loss", "N/A")
                    test_loss = results_data.get("final_test_loss", "N/A")
                    print(f"{optimizer.upper()}\t\t{train_loss:.6f}\t{test_loss:.6f}")
            except:
                print(f"{optimizer.upper()}\t\tnot found\tnot found")
    
    # Runtime statistics
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Average per experiment: {total_time/total/60:.1f} minutes")
    
    print(f"\nAll results saved in {results_dir}")

if __name__ == "__main__":
    main() 