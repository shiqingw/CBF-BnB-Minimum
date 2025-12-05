import torch
import torch.nn as nn
import time
import numpy as np
import gc

# Try importing auto_LiRPA
try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
except ImportError:
    raise ImportError("Please install auto_LiRPA: pip install auto_LiRPA")

# ==========================================
# 1. MODEL DEFINITION
# ==========================================

class BenchmarkMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, activation='relu'):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_dim, width))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(self._get_activation(activation))
            
        # Output layer
        layers.append(nn.Linear(width, out_dim))
        
        self.net = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == 'relu': return nn.ReLU()
        if name == 'tanh': return nn.Tanh()
        if name == 'sigmoid': return nn.Sigmoid()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. BENCHMARK UTILITIES
# ==========================================

def measure_performance(func, name, n_repeats=10):
    """
    Measures wall-clock time and peak CUDA memory.
    """
    # 1. Warmup
    func() 
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # 2. Memory Reset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 3. Timing Loop
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    t0 = time.time()
    if start_event: start_event.record()
    
    for _ in range(n_repeats):
        func()
        
    if end_event: 
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0 # Convert ms to s
    else:
        elapsed = time.time() - t0

    avg_time = elapsed / n_repeats
    
    # 4. Memory Check
    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
    return avg_time, peak_mem_mb

def run_benchmark():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Benchmark on: {device}")
    
    batch_size = 16
    in_dim = 64
    out_dim = 10
    depth = 5
    
    # We will vary the width of the layers to see scaling
    test_widths = [64, 128, 256, 512, 1024]
    epsilon = 0.1
    
    print(f"{'Width':<10} | {'Method':<10} | {'Avg Time (s)':<15} | {'Peak Mem (MB)':<15} | {'Bound Range':<15}")
    print("-" * 80)

    for width in test_widths:
        # 1. Setup Model and Input
        model = BenchmarkMLP(in_dim, out_dim, width, depth).to(device)
        model.eval()
        
        x0 = torch.randn(batch_size, in_dim, device=device)
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        my_input = BoundedTensor(x0, ptb)
        
        # 2. Initialize Auto_LiRPA
        # We assume standard bound options. For ReLU networks, 'adaptive' is good but adds overhead.
        # We stick to standard settings for raw throughput comparison.
        try:
            bounded_model = BoundedModule(
                model, torch.zeros_like(x0), 
                device=device, verbose=False
            )
        except Exception as e:
            print(f"Skipping Width {width}: Init failed ({e})")
            continue

        # --- Benchmark IBP ---
        def run_ibp():
            return bounded_model.compute_bounds(x=(my_input,), method='IBP')

        ibp_time, ibp_mem = measure_performance(run_ibp, "IBP")
        
        # Get one result for sanity check (range of bounds)
        lb, ub = run_ibp()
        ibp_range = (ub - lb).mean().item()

        print(f"{width:<10} | {'IBP':<10} | {ibp_time:<15.5f} | {ibp_mem:<15.2f} | {ibp_range:<15.4f}")

        # --- Benchmark CROWN (Backward) ---
        # Note: 'CROWN' in compute_bounds maps to method='backward' (Backward LiRPA)
        def run_crown():
            return bounded_model.compute_bounds(x=(my_input,), method='CROWN')

        crown_time, crown_mem = measure_performance(run_crown, "CROWN")
        
        lb, ub = run_crown()
        crown_range = (ub - lb).mean().item()

        print(f"{width:<10} | {'CROWN':<10} | {crown_time:<15.5f} | {crown_mem:<15.2f} | {crown_range:<15.4f}")
        
        # Optional: Add separator
        print("-" * 80)
        
        # Cleanup to prevent VRAM accumulation
        del bounded_model
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_benchmark()