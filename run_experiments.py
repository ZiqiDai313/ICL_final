import os
import subprocess
import sys

def main():
    print("Running ICL experiments with optimized learning rates for quadratic tasks")
    
    # Check if Python is in the path and accessible
    try:
        subprocess.run([sys.executable, "--version"], check=True)
        python_cmd = sys.executable
    except:
        print("Warning: Unable to determine Python executable path. Using 'python' command.")
        python_cmd = "python"
    
    # Run the updated experiments
    try:
        cmd = [python_cmd, "run_all_experiments.py"]
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("\nAll experiments completed successfully!")
            print("The experiments used the following learning rates:")
            print("\nLinear regression tasks:")
            print("- Adam: 0.001")
            print("- SGD: 0.0001")
            print("- Adagrad: 0.01")
            
            print("\nQuadratic regression tasks:")
            print("- Adam: 0.00001")
            print("- SGD: 0.00001")
            print("- Adagrad: 0.0001")
        else:
            print("Experiments failed with return code:", result.returncode)
    
    except subprocess.CalledProcessError as e:
        print("Error running experiments:", e)
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == "__main__":
    main() 