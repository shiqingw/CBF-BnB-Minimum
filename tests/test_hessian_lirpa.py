import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from cores.models.fcn import FullyConnectedNetwork

from cores.models.activation_function import get_activation_der, get_activation_second_der
from cores.models.interval_utils import (
    ibp_linear, ibp_jacobian_linear, ibp_product,
    ibp_jacobian_activation, get_derivative_bounds, get_second_derivative_bounds
)

def naive_ibp_hessian(net: FullyConnectedNetwork, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Manual IBP for Hessian H (B, Out, N, N).
    Recursing: v (Value), J (Jacobian), H (Hessian).
    """
    device = x_lb.device
    batch_size = x_lb.shape[0]
    n_in = net.in_features
    
    # 1. Init z, J, H
    # z = (x + b) * t
    t = net.input_transform
    b = net.input_bias
    zp_lb = x_lb + b; zp_ub = x_ub + b
    p1 = zp_lb * t; p2 = zp_ub * t
    z_lb = torch.min(p1, p2); z_ub = torch.max(p1, p2)
    
    # J = diag(t) (Constant)
    eye = torch.eye(n_in, device=device)
    J_base = eye * t.unsqueeze(1) # Scale rows
    J_lb = J_base.unsqueeze(0).expand(batch_size, -1, -1)
    J_ub = J_lb.clone()
    
    # H = 0 (Constant)
    H_lb = torch.zeros(batch_size, n_in, n_in, n_in, device=device)
    H_ub = torch.zeros(batch_size, n_in, n_in, n_in, device=device)
    
    act_index = 0
    
    for layer in net.model:
        if isinstance(layer, nn.Linear):
            # --- Linear Step ---
            # z = W z + b
            z_lb, z_ub = ibp_linear(layer, z_lb, z_ub)
            
            # J = W J
            J_lb, J_ub = ibp_jacobian_linear(layer.weight, J_lb, J_ub)
            
            # H = W H
            # H is (B, LayerIn, N, N). W is (LayerOut, LayerIn)
            # Reshape H to (B, LayerIn, N*N) to use generic linear prop
            h_shape = H_lb.shape # (B, Lin, N, N)
            H_flat_lb = H_lb.reshape(batch_size, h_shape[1], -1)
            H_flat_ub = H_ub.reshape(batch_size, h_shape[1], -1)
            
            H_new_flat_lb, H_new_flat_ub = ibp_jacobian_linear(layer.weight, H_flat_lb, H_flat_ub)
            
            # Reshape back: (B, Lout, N, N)
            out_dim = layer.weight.shape[0]
            H_lb = H_new_flat_lb.reshape(batch_size, out_dim, n_in, n_in)
            H_ub = H_new_flat_ub.reshape(batch_size, out_dim, n_in, n_in)
            
        else:
            # --- Activation Step ---
            act_name = net.activations[act_index]
            d1_fn = get_activation_der(act_name)
            d2_fn = get_activation_second_der(act_name)
            
            # 1. Bounds for derivatives (based on PRE-activation z)
            D1_lb, D1_ub = get_derivative_bounds(z_lb, z_ub, act_name, d1_fn)
            D2_lb, D2_ub = get_second_derivative_bounds(z_lb, z_ub, act_name, d2_fn)
            
            # 2. Outer Product Bounds: J_outer = J @ J.T (broadly speaking)
            # We want Outer_bijk = J_bij * J_bik
            # J_lb is (B, W, N)
            J_rows_lb = J_lb.unsqueeze(-1) # (B, W, N, 1)
            J_rows_ub = J_ub.unsqueeze(-1)
            J_cols_lb = J_lb.unsqueeze(-2) # (B, W, 1, N)
            J_cols_ub = J_ub.unsqueeze(-2)
            
            Outer_lb, Outer_ub = ibp_product(J_rows_lb, J_rows_ub, J_cols_lb, J_cols_ub)
            # Outer is (B, W, N, N)
            
            # 3. Hessian Update terms
            # Term 1: D1 * H
            # Flatten H for scalar mult: (B, W, N*N)
            H_flat_lb = H_lb.reshape(batch_size, -1, n_in*n_in)
            H_flat_ub = H_ub.reshape(batch_size, -1, n_in*n_in)
            
            T1_flat_lb, T1_flat_ub = ibp_jacobian_activation(D1_lb, D1_ub, H_flat_lb, H_flat_ub)
            
            # Term 2: D2 * Outer
            Outer_flat_lb = Outer_lb.reshape(batch_size, -1, n_in*n_in)
            Outer_flat_ub = Outer_ub.reshape(batch_size, -1, n_in*n_in)
            
            T2_flat_lb, T2_flat_ub = ibp_jacobian_activation(D2_lb, D2_ub, Outer_flat_lb, Outer_flat_ub)
            
            # Sum
            H_flat_lb = T1_flat_lb + T2_flat_lb
            H_flat_ub = T1_flat_ub + T2_flat_ub
            
            # Reshape back
            H_lb = H_flat_lb.reshape(batch_size, -1, n_in, n_in)
            H_ub = H_flat_ub.reshape(batch_size, -1, n_in, n_in)
            
            # 4. Jacobian Update: J = D1 * J
            J_lb, J_ub = ibp_jacobian_activation(D1_lb, D1_ub, J_lb, J_ub)
            
            # 5. Value Update
            z_lb = layer(z_lb)
            z_ub = layer(z_ub)
            
            act_index += 1
            
    return H_lb, H_ub

# --- MAIN TEST WRAPPER ---

class HessianWrapper(nn.Module):
    """Wraps network to return Hessian."""
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        return self.net.hessian(x)

def test_auto_lirpa_hessian():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Auto_LiRPA vs Naive IBP Hessian Verification on {device} ---")

    # Config
    input_dim = 5
    output_dim = 2
    widths = [input_dim, 8, 8, output_dim]
    activations = ['tanh', 'sigmoid']
    batch_size = 1

    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    # Net
    net = FullyConnectedNetwork(
        in_features=input_dim, out_features=output_dim,
        activations=activations, widths=widths,
        layer_bias=True,
        input_bias=rand_input_bias,
        input_transform=rand_input_transform
    ).to(device)
    
    # Wrapper
    hessian_model = HessianWrapper(net).to(device)
    
    # Input
    x0 = torch.randn(batch_size, input_dim, device=device)
    eps = 0.05
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    my_input = BoundedTensor(x0, ptb).to(device)
    
    # Initialize Auto_LiRPA
    print("\n[Setup] Initializing Auto_LiRPA BoundedModule...")
    try:
        bounded_model = BoundedModule(
            hessian_model, torch.zeros_like(x0), 
            bound_opts={'relu': 'adaptive'}, device=device
        )
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # --- 1. Auto_LiRPA IBP ---
    print("\n[Method A] Auto_LiRPA IBP")
    try:
        lb_lirpa, ub_lirpa = bounded_model.compute_bounds(x=(my_input,), method='IBP')
        print(f"  Shape: {lb_lirpa.shape}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        lb_lirpa, ub_lirpa = None, None

    # --- 2. Auto_LiRPA CROWN ---
    print("\n[Method B] Auto_LiRPA CROWN")
    try:
        # Expecting potential failure due to einsum/quadratic ops
        lb_crown, ub_crown = bounded_model.compute_bounds(x=(my_input,), method='CROWN')
        print(f"  Shape: {lb_crown.shape}")
    except Exception as e:
        print(f"  ❌ Failed (Expected): {e}")
        lb_crown, ub_crown = None, None

    # --- 3. Naive IBP ---
    print("\n[Method C] Naive Manual IBP")
    try:
        lb_naive, ub_naive = naive_ibp_hessian(net, x0 - eps, x0 + eps)
        print(f"  Shape: {lb_naive.shape}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        lb_naive, ub_naive = None, None

    # --- Empirical Sampling ---
    print("\n[Verification] Monte Carlo Sampling (1000 samples)")
    samples = []
    with torch.no_grad():
        for _ in range(1000):
            noise = (torch.rand_like(x0) * 2 - 1) * eps
            samples.append(net.hessian(x0 + noise))
    
    all_H = torch.cat(samples, dim=0)
    emp_min = all_H.min(dim=0)[0].unsqueeze(0)
    emp_max = all_H.max(dim=0)[0].unsqueeze(0)
    width_emp = (emp_max - emp_min).mean().item()

    # --- Stats ---
    def get_stats(lb, ub, name):
        if lb is None: return "N/A", "N/A"
        width = (ub - lb).mean().item()
        ratio = width / width_emp if width_emp > 0 else 0
        viol = ((all_H < lb - 1e-5) | (all_H > ub + 1e-5)).any().item()
        return f"{width:.6f}", f"{ratio:.2f}x", str(viol)

    w_lirpa, r_lirpa, v_lirpa = get_stats(lb_lirpa, ub_lirpa, "LiRPA IBP")
    w_crown, r_crown, v_crown = get_stats(lb_crown, ub_crown, "LiRPA CROWN")
    w_naive, r_naive, v_naive = get_stats(lb_naive, ub_naive, "Naive IBP")

    print("\n" + "="*100)
    print(f"{'HESSIAN BOUND STATISTICS':^100}")
    print("="*100)
    print(f"{'Method':<20} | {'Avg Width':<15} | {'Looseness':<15} | {'Violations':<15}")
    print("-" * 100)
    print(f"{'LiRPA IBP':<20} | {w_lirpa:<15} | {r_lirpa:<15} | {v_lirpa:<15}")
    print(f"{'LiRPA CROWN':<20} | {w_crown:<15} | {r_crown:<15} | {v_crown:<15}")
    print(f"{'Naive IBP':<20} | {w_naive:<15} | {r_naive:<15} | {v_naive:<15}")
    print(f"{'Empirical':<20} | {width_emp:.6f}{'':<9} | 1.00x{'':<10} | N/A")
    print("="*100)

if __name__ == "__main__":
    test_auto_lirpa_hessian()