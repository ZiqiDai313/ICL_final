import torch
import time
import threading

# Global variable to track whether to continue running
keep_running = True

# Continuously monitor GPU usage
def monitor_gpu():
    while keep_running:
        if torch.cuda.is_available():
            used_mem = torch.cuda.memory_allocated() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {used_mem:.2f} GB / {total_mem:.2f} GB ({used_mem/total_mem*100:.1f}%)")
        time.sleep(2)  # Update every 2 seconds

# Run multiple matrix computation tasks in parallel
def task(task_id, size):
    if torch.cuda.is_available():
        # Create large matrices
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Keep GPU busy with continuous computation
        count = 0
        while keep_running and count < 1000:  # Limit loop iterations to prevent infinite loops
            # Perform multiple matrix multiplications to ensure full GPU utilization
            c = torch.matmul(a, b)
            # Add more compute-intensive operations
            d = torch.sin(c)
            e = torch.relu(d)
            f = torch.matmul(e, a)
            g = torch.softmax(f, dim=1)
            h = torch.log_softmax(g, dim=0)
            
            # Perform more operations to keep GPU active
            i = torch.matmul(h, b)
            j = torch.nn.functional.normalize(i, dim=0)
            k = torch.matmul(j, a.t())
            
            # Ensure all operations complete
            torch.cuda.synchronize()
            
            # Update result to continue computation
            a = k[:size, :size] / torch.norm(k)
            
            count += 1
            
            if count % 10 == 0:
                print(f"Task {task_id} completed {count} iterations")
    
    print(f"Task {task_id} finished, executed {count} iterations")

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a GPU-enabled system.")
        return

    # Turn off gradient calculation to avoid memory growth
    torch.set_grad_enabled(False)
    
    # Start GPU monitoring thread
    global keep_running
    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Create multiple task threads to maximize GPU usage
    tasks = []
    sizes = [1024, 2048, 3072, 4096]
    
    try:
        for i, size in enumerate(sizes):
            # Delay starting next thread slightly to spread memory allocation
            time.sleep(0.5)
            t = threading.Thread(target=task, args=(i, size))
            t.daemon = True
            t.start()
            tasks.append(t)
            print(f"Started task {i} with matrix size {size}x{size}")
        
        # Let tasks run for 30 seconds
        time.sleep(30)
    finally:
        # Signal threads to stop
        print("Stopping test...")
        keep_running = False
        
        # Wait for all tasks to finish
        for t in tasks:
            t.join(timeout=5.0)
        
        print("GPU stress test completed")

if __name__ == "__main__":
    main() 