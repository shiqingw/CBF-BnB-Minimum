import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from cores.models.fcn import FullyConnectedNetwork
from cores.models.activation_function import get_activation_der
from cores.models.interval_utils import (
    ibp_linear, ibp_jacobian_linear,
    ibp_jacobian_activation, get_derivative_bounds
)

def naive_ibp_jacobian(net: FullyConnectedNetwork, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Manual implementation of IBP for Jacobian bounds on the specific FCN architecture.
    """
    device = x_lb.device
    batch_size = x_lb.shape[0]
    in_features = net.in_features
    
    # 1. Input Transform Layer: z = (x + b) * t
    t = net.input_transform
    b = net.input_bias
    
    # Value Propagation (Interval Arithmetic for Affine)
    # zp = x + b
    zp_lb = x_lb + b
    zp_ub = x_ub + b
    
    # z = zp * t (t is constant, but can be negative)
    p1 = zp_lb * t
    p2 = zp_ub * t
    z_lb = torch.min(p1, p2)
    z_ub = torch.max(p1, p2)
    
    # Jacobian Initialization: J_0 = diag(t)
    # J is constant w.r.t input (derivative of affine is constant scale)
    eye = torch.eye(in_features, device=device)
    J_base = eye * t.unsqueeze(1) # Scale rows
    J_lb = J_base.unsqueeze(0).expand(batch_size, -1, -1)
    J_ub = J_lb.clone()
    
    # 2. Iterate Layers
    act_index = 0
    curr_lb, curr_ub = z_lb, z_ub
    
    for layer in net.model:
        if isinstance(layer, nn.Linear):
            # --- Linear Layer ---
            # Value Prop
            curr_lb, curr_ub = ibp_linear(layer, curr_lb, curr_ub)
            # Jacobian Prop
            J_lb, J_ub = ibp_jacobian_linear(layer.weight, J_lb, J_ub)
            
        else:
            # --- Activation Layer ---
            act_name = net.activations[act_index]
            dphi_fn = get_activation_der(act_name)
            
            # Derivative Bounds
            D_lb, D_ub = get_derivative_bounds(curr_lb, curr_ub, act_name, dphi_fn=dphi_fn)
            
            # Jacobian Prop (Scaling)
            J_lb, J_ub = ibp_jacobian_activation(D_lb, D_ub, J_lb, J_ub)
            
            # Value Prop (Monotonic Activation)
            # Assuming monotonic increasing for standard activations (ReLU, Tanh, Sigmoid)
            curr_lb = layer(curr_lb)
            curr_ub = layer(curr_ub)
            
            act_index += 1
            
    return J_lb, J_ub

class JacobianWrapper(nn.Module):
    """
    Wraps the network so that 'forward' returns the Jacobian.
    This allows auto_LiRPA to treat the Jacobian computation as the 
    neural network to be verified.
    """
    def __init__(self, net: FullyConnectedNetwork):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.jacobian(x)

def test_auto_lirpa_jacobian():
    # 0. Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Auto_LiRPA (IBP & CROWN) vs Naive IBP Verification on {device} ---")

    # 1. Configuration
    input_dim = 10 
    output_dim = 2
    widths = [input_dim, 16, 16, output_dim]
    activations = ['tanh', 'tanh']
    batch_size = 1 
    
    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    # 2. Instantiate the Core Network
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths,
        layer_bias=True,
        input_bias=rand_input_bias,
        input_transform=rand_input_transform
    ).to(device)

    # 3. Setup Auto_LiRPA
    jacobian_model = JacobianWrapper(net).to(device)
    
    x0 = torch.randn(batch_size, input_dim, device=device)
    eps = 0.1
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    my_input = BoundedTensor(x0, ptb).to(device)
    
    # Initialize BoundedModule once
    try:
        bounded_model = BoundedModule(
            jacobian_model, 
            torch.zeros_like(x0), 
            bound_opts={'relu': 'adaptive'}, 
            device=device
        )
    except Exception as e:
        print(f"❌ Auto_LiRPA Init Failed: {e}")
        return

    # --- METHOD A: AUTO_LIRPA IBP ---
    print("\n[Method A] Auto_LiRPA IBP")
    try:
        lb_lirpa_ibp, ub_lirpa_ibp = bounded_model.compute_bounds(x=(my_input,), method='IBP')
        print(f"  Shape: {lb_lirpa_ibp.shape}")
    except Exception as e:
        print(f"  ❌ Auto_LiRPA IBP Failed: {e}")
        return

    # --- METHOD B: AUTO_LIRPA CROWN ---
    print("\n[Method B] Auto_LiRPA CROWN")
    try:
        lb_lirpa_crown, ub_lirpa_crown = bounded_model.compute_bounds(x=(my_input,), method='CROWN')
        print(f"  Shape: {lb_lirpa_crown.shape}")
    except Exception as e:
        print(f"  ❌ Auto_LiRPA CROWN Failed: {e}")
        return

    # --- METHOD C: NAIVE IBP ---
    print("\n[Method C] Naive Manual IBP")
    x_lb = x0 - eps
    x_ub = x0 + eps
    try:
        lb_naive, ub_naive = naive_ibp_jacobian(net, x_lb, x_ub)
        print(f"  Shape: {lb_naive.shape}")
    except Exception as e:
        print(f"  ❌ Naive IBP Failed: {e}")
        raise e

    # --- EMPIRICAL VERIFICATION ---
    print("\n[Verification] Monte Carlo Sampling (2000 samples)")
    num_samples = 2000
    sampled_jacobians = []
    
    for _ in range(num_samples):
        noise = (torch.rand_like(x0) * 2 - 1) * eps
        x_perturbed = x0 + noise
        J_actual = net.jacobian(x_perturbed)
        sampled_jacobians.append(J_actual)

    all_J = torch.cat(sampled_jacobians, dim=0)
    empirical_min = all_J.min(dim=0)[0].unsqueeze(0)
    empirical_max = all_J.max(dim=0)[0].unsqueeze(0)

    # --- STATISTICAL COMPARISON ---
    
    # Helper to compute metrics
    def get_metrics(lb, ub, emp_min, emp_max):
        width = ub - lb
        avg_width = width.mean().item()
        # Violation check
        viol = ((all_J < lb - 1e-5) | (all_J > ub + 1e-5)).any().item()
        return avg_width, viol

    avg_width_ibp, viol_ibp = get_metrics(lb_lirpa_ibp, ub_lirpa_ibp, empirical_min, empirical_max)
    avg_width_crown, viol_crown = get_metrics(lb_lirpa_crown, ub_lirpa_crown, empirical_min, empirical_max)
    avg_width_naive, viol_naive = get_metrics(lb_naive, ub_naive, empirical_min, empirical_max)
    
    width_emp = empirical_max - empirical_min
    avg_width_emp = width_emp.mean().item()

    print("\n" + "="*110)
    print(f"{'STATISTICS COMPARISON':^110}")
    print("="*110)
    print(f"{'Metric':<20} | {'LiRPA IBP':<15} | {'LiRPA CROWN':<15} | {'Naive IBP':<15} | {'Empirical':<15}")
    print("-" * 110)
    print(f"{'Avg Width':<20} | {avg_width_ibp:<15.6f} | {avg_width_crown:<15.6f} | {avg_width_naive:<15.6f} | {avg_width_emp:<15.6f}")
    
    ratio_ibp = avg_width_ibp / avg_width_emp if avg_width_emp > 0 else 0
    ratio_crown = avg_width_crown / avg_width_emp if avg_width_emp > 0 else 0
    ratio_naive = avg_width_naive / avg_width_emp if avg_width_emp > 0 else 0
    
    print(f"{'Looseness (x Emp)':<20} | {ratio_ibp:<15.2f}x | {ratio_crown:<15.2f}x | {ratio_naive:<15.2f}x | {'1.00x':<15}")
    print("-" * 110)
    print(f"{'Violations':<20} | {str(viol_ibp):<15} | {str(viol_crown):<15} | {str(viol_naive):<15} | {'N/A':<15}")
    print("="*110)

    if viol_ibp or viol_crown or viol_naive:
        print("⚠️  WARNING: Some bounds were violated by empirical samples!")
    
    print("\nSample Element [0,0,0]:")
    print(f"  LiRPA IBP:   [{lb_lirpa_ibp[0,0,0]:.4f}, {ub_lirpa_ibp[0,0,0]:.4f}]")
    print(f"  LiRPA CROWN: [{lb_lirpa_crown[0,0,0]:.4f}, {ub_lirpa_crown[0,0,0]:.4f}]")
    print(f"  Naive IBP:   [{lb_naive[0,0,0]:.4f}, {ub_naive[0,0,0]:.4f}]")
    print(f"  Empirical:   [{empirical_min[0,0,0]:.4f}, {empirical_max[0,0,0]:.4f}]")

if __name__ == "__main__":
    test_auto_lirpa_jacobian()