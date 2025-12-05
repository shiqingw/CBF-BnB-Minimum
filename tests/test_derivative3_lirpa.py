import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from cores.models.fcn import FullyConnectedNetwork
from cores.models.activation_function import get_activation_der, get_activation_second_der, get_activation_third_der
# --- HELPER FUNCTIONS ---

def ibp_product(l1, u1, l2, u2):
    """Interval Product: [a,b] * [c,d]"""
    p1 = l1 * l2; p2 = l1 * u2; p3 = u1 * l2; p4 = u1 * u2
    lb = torch.min(torch.min(p1, p2), torch.min(p3, p4))
    ub = torch.max(torch.max(p1, p2), torch.max(p3, p4))
    return lb, ub

def ibp_linear(layer: nn.Linear, x_lb: torch.Tensor, x_ub: torch.Tensor):
    w_pos = layer.weight.clamp(min=0)
    w_neg = layer.weight.clamp(max=0)
    lb = F.linear(x_lb, w_pos) + F.linear(x_ub, w_neg)
    ub = F.linear(x_ub, w_pos) + F.linear(x_lb, w_neg)
    if layer.bias is not None:
        lb += layer.bias; ub += layer.bias
    return lb, ub

def ibp_jacobian_linear(W: torch.Tensor, J_lb: torch.Tensor, J_ub: torch.Tensor):
    """
    Propagate generic tensor J (B, In, ...) through linear weight W (Out, In).
    Uses einsum 'kp, bpn... -> bkn...' logic.
    """
    W_pos = W.clamp(min=0); W_neg = W.clamp(max=0)
    # Explicit broadcasting for shape (B, In, N...) -> (B, Out, N...)
    # Flatten trailing dimensions
    b_sz = J_lb.shape[0]
    in_dim = W.shape[1]
    
    J_flat_lb = J_lb.reshape(b_sz, in_dim, -1)
    J_flat_ub = J_ub.reshape(b_sz, in_dim, -1)
    
    # 'kp, bpn -> bkn'
    new_lb = torch.einsum('kp, bpn -> bkn', W_pos, J_flat_lb) + torch.einsum('kp, bpn -> bkn', W_neg, J_flat_ub)
    new_ub = torch.einsum('kp, bpn -> bkn', W_pos, J_flat_ub) + torch.einsum('kp, bpn -> bkn', W_neg, J_flat_lb)
    
    # Restore shape
    out_shape = list(J_lb.shape)
    out_shape[1] = W.shape[0]
    return new_lb.reshape(out_shape), new_ub.reshape(out_shape)

def ibp_jacobian_activation(D_lb, D_ub, J_lb, J_ub):
    """Elementwise scaling J_new = D * J"""
    # D is (B, Out). J is (B, Out, ...). Broadcast D.
    target_shape = [-1] + [1] * (J_lb.ndim - 2) # (B, Out, 1, 1...)
    D_lb_exp = D_lb.view(D_lb.shape[0], D_lb.shape[1], *([1]*(J_lb.ndim-2)))
    D_ub_exp = D_ub.view(D_ub.shape[0], D_ub.shape[1], *([1]*(J_ub.ndim-2)))
    return ibp_product(D_lb_exp, D_ub_exp, J_lb, J_ub)

def get_derivative_bounds(u_lb, u_ub, act_name, dphi_fn):
    # Simplified for Tanh/Sigmoid/ReLU
    if act_name in ['tanh', 'sigmoid']:
        d_lb = dphi_fn(u_lb); d_ub = dphi_fn(u_ub)
        d_zero = dphi_fn(torch.zeros_like(u_lb))
        contains_zero = (u_lb < 0) & (u_ub > 0)
        return torch.min(d_lb, d_ub), torch.where(contains_zero, d_zero, torch.max(d_lb, d_ub))
    if act_name == 'relu':
        return dphi_fn(u_lb), dphi_fn(u_ub) # Simplified, strict handling needed for 0 crossing
    return torch.ones_like(u_lb), torch.ones_like(u_ub) # Identity

def get_second_derivative_bounds(u_lb, u_ub, act_name, d2phi_fn):
    if act_name == 'tanh':
        # Approx range [-0.77, 0.77]
        v_lb = d2phi_fn(u_lb); v_ub = d2phi_fn(u_ub)
        has_pos_peak = (u_lb < 0.658) & (u_ub > 0.658)
        has_neg_peak = (u_lb < -0.658) & (u_ub > -0.658)
        lb = torch.where(has_pos_peak, torch.full_like(u_lb, -0.7698), torch.min(v_lb, v_ub))
        ub = torch.where(has_neg_peak, torch.full_like(u_ub, 0.7698), torch.max(v_lb, v_ub))
        return lb, ub
    return d2phi_fn(u_lb), d2phi_fn(u_ub) # Fallback

def get_third_derivative_bounds(u_lb, u_ub, act_name, d3phi_fn):
    if act_name == 'tanh':
        # Range [-2, 2/3]. Min at 0.
        v_lb = d3phi_fn(u_lb); v_ub = d3phi_fn(u_ub)
        has_min = (u_lb < 0) & (u_ub > 0) # Peak at 0 (value -2)
        lb = torch.where(has_min, torch.full_like(u_lb, -2.0), torch.min(v_lb, v_ub))
        ub = torch.max(v_lb, v_ub) # Approx
        return lb, ub
    return d3phi_fn(u_lb), d3phi_fn(u_ub)

# --- NAIVE IBP IMPLEMENTATION ---

def naive_ibp_derivative3(net: FullyConnectedNetwork, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Computes bounds for 3rd Derivative T: (B, Out, N, N, N)
    """
    device = x_lb.device
    batch_size = x_lb.shape[0]
    n_in = net.in_features
    
    # 1. Initialization
    t = net.input_transform
    b = net.input_bias
    z_lb = (x_lb + b) * t if (t>=0).all() else torch.min((x_lb+b)*t, (x_ub+b)*t) # Simplified affine
    z_ub = (x_ub + b) * t if (t>=0).all() else torch.max((x_lb+b)*t, (x_ub+b)*t)
    
    # J = diag(t)
    J_base = torch.eye(n_in, device=device) * t.unsqueeze(1)
    J_lb = J_base.unsqueeze(0).expand(batch_size, -1, -1)
    J_ub = J_lb.clone()
    
    # H = 0, T = 0
    H_lb = torch.zeros(batch_size, n_in, n_in, n_in, device=device)
    H_ub = torch.zeros(batch_size, n_in, n_in, n_in, device=device)
    T_lb = torch.zeros(batch_size, n_in, n_in, n_in, n_in, device=device)
    T_ub = torch.zeros(batch_size, n_in, n_in, n_in, n_in, device=device)
    
    act_index = 0
    
    for layer in net.model:
        if isinstance(layer, nn.Linear):
            # --- Linear Step ---
            z_lb, z_ub = ibp_linear(layer, z_lb, z_ub)
            J_lb, J_ub = ibp_jacobian_linear(layer.weight, J_lb, J_ub)
            H_lb, H_ub = ibp_jacobian_linear(layer.weight, H_lb, H_ub)
            T_lb, T_ub = ibp_jacobian_linear(layer.weight, T_lb, T_ub)
            
        else:
            # --- Activation Step ---
            act_name = net.activations[act_index]
            d1_fn = get_activation_der(act_name)
            d2_fn = get_activation_second_der(act_name)
            d3_fn = get_activation_third_der(act_name)
            
            # Derivatives
            D1_lb, D1_ub = get_derivative_bounds(z_lb, z_ub, act_name, d1_fn)
            D2_lb, D2_ub = get_second_derivative_bounds(z_lb, z_ub, act_name, d2_fn)
            D3_lb, D3_ub = get_third_derivative_bounds(z_lb, z_ub, act_name, d3_fn)
            
            # --- T Update ---
            # Term A: s' * T
            TA_lb, TA_ub = ibp_jacobian_activation(D1_lb, D1_ub, T_lb, T_ub)
            
            # Term B: s'' * (Sum J x H)
            # Perm 1: J_i H_jk -> J(.., N, 1, 1) * H(.., 1, N, N)
            J_e1_lb = J_lb.unsqueeze(-1).unsqueeze(-1); J_e1_ub = J_ub.unsqueeze(-1).unsqueeze(-1)
            H_e1_lb = H_lb.unsqueeze(-3); H_e1_ub = H_ub.unsqueeze(-3)
            C1_lb, C1_ub = ibp_product(J_e1_lb, J_e1_ub, H_e1_lb, H_e1_ub)
            
            # Perm 2: J_j H_ik -> J(.., 1, N, 1) * H(.., N, 1, N)
            J_e2_lb = J_lb.unsqueeze(-1).unsqueeze(-3); J_e2_ub = J_ub.unsqueeze(-1).unsqueeze(-3)
            H_e2_lb = H_lb.unsqueeze(-2); H_e2_ub = H_ub.unsqueeze(-2)
            C2_lb, C2_ub = ibp_product(J_e2_lb, J_e2_ub, H_e2_lb, H_e2_ub)
            
            # Perm 3: J_k H_ij -> J(.., 1, 1, N) * H(.., N, N, 1)
            J_e3_lb = J_lb.unsqueeze(-2).unsqueeze(-3); J_e3_ub = J_ub.unsqueeze(-2).unsqueeze(-3)
            H_e3_lb = H_lb.unsqueeze(-1); H_e3_ub = H_ub.unsqueeze(-1)
            C3_lb, C3_ub = ibp_product(J_e3_lb, J_e3_ub, H_e3_lb, H_e3_ub)
            
            Sum_Mix_lb = C1_lb + C2_lb + C3_lb
            Sum_Mix_ub = C1_ub + C2_ub + C3_ub
            TB_lb, TB_ub = ibp_jacobian_activation(D2_lb, D2_ub, Sum_Mix_lb, Sum_Mix_ub)
            
            # Term C: s''' * (J x J x J)
            # J_i J_j J_k -> (.., N, 1, 1) * (.., 1, N, 1) * (.., 1, 1, N)
            JJ_lb, JJ_ub = ibp_product(J_e1_lb, J_e1_ub, J_e2_lb, J_e2_ub) # J_i * J_j
            JJJ_lb, JJJ_ub = ibp_product(JJ_lb, JJ_ub, J_e3_lb, J_e3_ub)   # * J_k
            
            TC_lb, TC_ub = ibp_jacobian_activation(D3_lb, D3_ub, JJJ_lb, JJJ_ub)
            
            T_lb = TA_lb + TB_lb + TC_lb
            T_ub = TA_ub + TB_ub + TC_ub
            
            # --- H Update ---
            # H = s' H + s'' (J x J)
            # J_i J_j
            Outer_lb, Outer_ub = ibp_product(J_e1_lb.squeeze(-1), J_e1_ub.squeeze(-1), 
                                             J_e2_lb.squeeze(-1), J_e2_ub.squeeze(-1))
            
            H_term1_lb, H_term1_ub = ibp_jacobian_activation(D1_lb, D1_ub, H_lb, H_ub)
            H_term2_lb, H_term2_ub = ibp_jacobian_activation(D2_lb, D2_ub, Outer_lb, Outer_ub)
            H_lb = H_term1_lb + H_term2_lb
            H_ub = H_term1_ub + H_term2_ub
            
            # --- J Update ---
            J_lb, J_ub = ibp_jacobian_activation(D1_lb, D1_ub, J_lb, J_ub)
            
            # --- Value Update ---
            z_lb = layer(z_lb); z_ub = layer(z_ub)
            
            act_index += 1
            
    return T_lb, T_ub
# --- TEST WRAPPER ---

class D3Wrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        # Calls the auto_lirpa friendly derivative3 method
        return self.net.derivative3(x)

def test_auto_lirpa_d3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Auto_LiRPA vs Naive IBP 3rd Derivative Verification on {device} ---")

    # 1. Configuration
    # Very small dims because D3 tensor is (B, Out, N, N, N) -> O(N^3)
    input_dim = 3
    output_dim = 2
    widths = [input_dim, 8, 8, output_dim]
    activations = ['tanh', 'tanh']
    batch_size = 1
    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    # 2. Setup
    net = FullyConnectedNetwork(
        in_features=input_dim, out_features=output_dim,
        activations=activations, widths=widths,
        input_bias=rand_input_bias, input_transform=rand_input_transform,
        layer_bias=True
    ).to(device)
    
    d3_model = D3Wrapper(net).to(device)
    
    x0 = torch.randn(batch_size, input_dim, device=device)
    eps = 0.05
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    my_input = BoundedTensor(x0, ptb).to(device)
    
    print("\n[Setup] Initializing Auto_LiRPA BoundedModule...")
    try:
        bounded_model = BoundedModule(
            d3_model, torch.zeros_like(x0), 
            bound_opts={'relu': 'adaptive'}, device=device
        )
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # --- A. Auto_LiRPA IBP ---
    print("\n[Method A] Auto_LiRPA IBP")
    try:
        lb_lirpa, ub_lirpa = bounded_model.compute_bounds(x=(my_input,), method='IBP')
        print(f"  Shape: {lb_lirpa.shape}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        lb_lirpa, ub_lirpa = None, None

    # --- B. Auto_LiRPA CROWN ---
    print("\n[Method B] Auto_LiRPA CROWN")
    try:
        # Expected to fail or be very slow due to broadcasting complexity
        lb_crown, ub_crown = bounded_model.compute_bounds(x=(my_input,), method='CROWN')
        print(f"  Shape: {lb_crown.shape}")
    except Exception as e:
        print(f"  ❌ Failed (Expected for high order ops): {e}")
        lb_crown, ub_crown = None, None

    # --- C. Naive IBP ---
    print("\n[Method C] Naive IBP")
    try:
        x_lb, x_ub = x0 - eps, x0 + eps
        lb_naive, ub_naive = naive_ibp_derivative3(net, x_lb, x_ub)
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
            samples.append(net.derivative3(x0 + noise))
    
    all_T = torch.cat(samples, dim=0)
    emp_min = all_T.min(dim=0)[0].unsqueeze(0)
    emp_max = all_T.max(dim=0)[0].unsqueeze(0)
    width_emp = (emp_max - emp_min).mean().item()

    # --- Stats ---
    def get_stats(lb, ub):
        if lb is None: return "N/A", "N/A", "N/A"
        width = (ub - lb).mean().item()
        ratio = width / width_emp if width_emp > 0 else 0
        viol = ((all_T < lb - 1e-5) | (all_T > ub + 1e-5)).any().item()
        return f"{width:.6f}", f"{ratio:.2f}x", str(viol)

    w_lirpa, r_lirpa, v_lirpa = get_stats(lb_lirpa, ub_lirpa)
    w_crown, r_crown, v_crown = get_stats(lb_crown, ub_crown)
    w_naive, r_naive, v_naive = get_stats(lb_naive, ub_naive)

    print("\n" + "="*100)
    print(f"{'3RD DERIVATIVE BOUND STATISTICS':^100}")
    print("="*100)
    print(f"{'Method':<20} | {'Avg Width':<15} | {'Looseness':<15} | {'Violations':<15}")
    print("-" * 100)
    print(f"{'LiRPA IBP':<20} | {w_lirpa:<15} | {r_lirpa:<15} | {v_lirpa:<15}")
    print(f"{'LiRPA CROWN':<20} | {w_crown:<15} | {r_crown:<15} | {v_crown:<15}")
    print(f"{'Naive IBP':<20} | {w_naive:<15} | {r_naive:<15} | {v_naive:<15}")
    print(f"{'Empirical':<20} | {width_emp:.6f}{'':<9} | 1.00x{'':<10} | N/A")
    print("="*100)

if __name__ == "__main__":
    test_auto_lirpa_d3()