import torch
import torch.nn as nn
import time
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from cores.models.fcn import FullyConnectedNetwork

class HessianWrapper(nn.Module):
    """
    Wraps the network to expose the Hessian computation.
    """
    def __init__(self, net: FullyConnectedNetwork):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.hessian(x)

def benchmark_hessian():
    # 0. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Benchmarking Hessian (IBP vs CROWN) on {device} ---")

    # 1. Configuration
    # Smaller dims than Jacobian benchmark because Hessian is O(N^2) output
    input_dim = 5 
    output_dim = 2
    widths = [input_dim, 16, 16, output_dim] 
    activations = ['tanh', 'sigmoid']
    batch_size = 1
    
    # 2. Instantiate Network
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths
    ).to(device)

    # 3. Setup Auto_LiRPA Wrapper
    hessian_model = HessianWrapper(net).to(device)
    
    # 4. Inputs and Perturbations
    x0 = torch.randn(batch_size, input_dim, device=device)
    eps = 0.05
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    my_input = BoundedTensor(x0, ptb).to(device)
    
    # 5. Initialize BoundedModule (Tracing)
    print("Step 1: Tracing graph...")
    try:
        bounded_model = BoundedModule(
            hessian_model, 
            torch.zeros_like(x0), 
            bound_opts={'relu': 'adaptive', 'conv_mode': 'matrix'},
            device=device
        )
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # Helper to run and time a method
    def run_benchmark(method_name):
        # Warmup
        try:
            bounded_model.compute_bounds(x=(my_input,), method=method_name)
        except Exception:
            return None, None, 0.0
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        # Run actual computation
        lb, ub = bounded_model.compute_bounds(x=(my_input,), method=method_name)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        return lb, ub, (end_time - start_time)

    # 6. Run Benchmarks
    print("\nStep 2: Running IBP...")
    lb_ibp, ub_ibp, time_ibp = run_benchmark('IBP')
    
    print("Step 3: Running CROWN...")
    lb_crown, ub_crown, time_crown = run_benchmark('CROWN')

    if lb_ibp is None or lb_crown is None:
        print("❌ Benchmark failed during bound computation.")
        return

    # 7. Compute Statistics
    width_ibp = (ub_ibp - lb_ibp).mean().item()
    width_crown = (ub_crown - lb_crown).mean().item()
    
    # Compute Empirical Range for Reference (1000 samples)
    print("Step 4: Computing Empirical Baseline (Monte Carlo)...")
    samples = []
    with torch.no_grad():
        for _ in range(1000):
            noise = (torch.rand_like(x0) * 2 - 1) * eps
            samples.append(net.hessian(x0 + noise))
    all_samples = torch.cat(samples, dim=0)
    emp_min = all_samples.min(dim=0)[0]
    emp_max = all_samples.max(dim=0)[0]
    width_emp = (emp_max - emp_min).mean().item()

    # 8. Comparison Table
    print("\n" + "="*90)
    print(f"{'HESSIAN BENCHMARK RESULTS':^90}")
    print("="*90)
    print(f"{'Metric':<20} | {'IBP':<15} | {'CROWN':<15} | {'Empirical':<15}")
    print("-" * 90)
    print(f"{'Time (sec)':<20} | {time_ibp:<15.4f} | {time_crown:<15.4f} | {'-':<15}")
    print(f"{'Avg Width':<20} | {width_ibp:<15.6f} | {width_crown:<15.6f} | {width_emp:<15.6f}")
    
    # Ratios
    ratio_ibp = width_ibp / width_emp if width_emp > 0 else float('inf')
    ratio_crown = width_crown / width_emp if width_emp > 0 else float('inf')
    
    print("-" * 90)
    print(f"{'Looseness (x Emp)':<20} | {ratio_ibp:<15.2f}x | {ratio_crown:<15.2f}x | {'1.00x':<15}")
    print("="*90)
    
    speedup = time_crown / time_ibp if time_ibp > 0 else 0
    tightness = (width_ibp - width_crown) / width_ibp * 100 if width_ibp > 0 else 0
    
    print(f"\nSummary:")
    print(f"1. Speed: IBP is {speedup:.1f}x faster than CROWN.")
    print(f"2. Precision: CROWN bounds are {tightness:.1f}% tighter than IBP.")

if __name__ == "__main__":
    benchmark_hessian()