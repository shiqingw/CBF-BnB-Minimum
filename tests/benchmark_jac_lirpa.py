import torch
import torch.nn as nn
import time
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from cores.models.fcn import FullyConnectedNetwork

class JacobianCrownWrapper(nn.Module):
    """
    Wraps the network to expose the CROWN-optimized Jacobian computation.
    """
    def __init__(self, net: FullyConnectedNetwork):
        super().__init__()
        self.net = net

    def forward(self, x):
        # Uses the transposed propagation method optimized for CROWN
        return self.net.jacobian(x)

def compare_ibp_vs_crown():
    # 0. Setup Device
    device = torch.device("cpu")
    print(f"--- Benchmarking Auto_LiRPA: IBP vs CROWN on {device} ---")

    # 1. Configuration
    input_dim = 10 
    output_dim = 5
    # Slightly larger network to make timing differences measurable
    widths = [input_dim, 32, 32, output_dim] 
    activations = ['tanh', 'sigmoid']
    batch_size = 1 # Verification usually done per sample
    
    # 2. Instantiate Network
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths
    ).to(device)

    # 3. Setup Auto_LiRPA Wrapper
    jacobian_model = JacobianCrownWrapper(net).to(device)
    
    # 4. Inputs and Perturbations
    x0 = torch.randn(batch_size, input_dim, device=device)
    eps = 0.05
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    my_input = BoundedTensor(x0, ptb).to(device)
    
    # 5. Initialize BoundedModule (Tracing)
    print("Step 1: Tracing graph...")
    # We use bound_opts to enable optimizations if available
    bounded_model = BoundedModule(
        jacobian_model, 
        torch.zeros_like(x0), 
        bound_opts={'relu': 'adaptive', 'conv_mode': 'matrix'},
        device=device
    )

    # Helper to run and time a method
    def run_benchmark(method_name):
        # Warmup
        bounded_model.compute_bounds(x=(my_input,), method=method_name)
        
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

    # 7. Compute Statistics
    width_ibp = (ub_ibp - lb_ibp).mean().item()
    width_crown = (ub_crown - lb_crown).mean().item()
    
    # Compute Empirical Range for Reference (1000 samples)
    print("Step 4: Computing Empirical Baseline (Monte Carlo)...")
    samples = []
    with torch.no_grad():
        for _ in range(1000):
            noise = (torch.rand_like(x0) * 2 - 1) * eps
            # Use standard jacobian for empirical sampling
            samples.append(net.jacobian(x0 + noise))
    all_samples = torch.cat(samples, dim=0)
    emp_min = all_samples.min(dim=0)[0]
    emp_max = all_samples.max(dim=0)[0]
    width_emp = (emp_max - emp_min).mean().item()

    # 8. Comparison Table
    print("\n" + "="*80)
    print(f"{'COMPARISON RESULTS':^80}")
    print("="*80)
    print(f"{'Metric':<20} | {'IBP':<15} | {'CROWN':<15} | {'Empirical':<15}")
    print("-" * 80)
    print(f"{'Time (sec)':<20} | {time_ibp:<15.4f} | {time_crown:<15.4f} | {'-':<15}")
    print(f"{'Avg Width':<20} | {width_ibp:<15.6f} | {width_crown:<15.6f} | {width_emp:<15.6f}")
    
    # Ratios
    ratio_ibp = width_ibp / width_emp if width_emp > 0 else float('inf')
    ratio_crown = width_crown / width_emp if width_emp > 0 else float('inf')
    improvement = (width_ibp - width_crown) / width_ibp * 100
    
    print("-" * 80)
    print(f"{'Looseness (x Emp)':<20} | {ratio_ibp:<15.2f}x | {ratio_crown:<15.2f}x | {'1.00x':<15}")
    print("="*80)
    
    print(f"\nAnalysis:")
    print(f"1. Speed: IBP is {time_crown / time_ibp:.1f}x faster than CROWN.")
    print(f"2. Tightness: CROWN reduced bound width by {improvement:.1f}% compared to IBP.")
    
    if ratio_crown < ratio_ibp:
        print("   ✅ CROWN provided tighter bounds as expected.")
    else:
        print("   ⚠️ CROWN did not provide tighter bounds (likely due to derivative decomposition limitations).")

if __name__ == "__main__":
    compare_ibp_vs_crown()