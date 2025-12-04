import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_function import get_activation_der

def ibp_product(l1, u1, l2, u2):
    """
    Standard Interval Product: [a,b] * [c,d]
    """
    p1 = l1 * l2
    p2 = l1 * u2
    p3 = u1 * l2
    p4 = u1 * u2
    lb = torch.min(torch.min(p1, p2), torch.min(p3, p4))
    ub = torch.max(torch.max(p1, p2), torch.max(p3, p4))
    return lb, ub

def ibp_square(l, u):
    """
    Tighter bounds for x^2 where x in [l, u].
    If 0 in [l, u], min is 0. Else min(l^2, u^2).
    Max is max(l^2, u^2).
    """
    sq_l = l * l
    sq_u = u * u
    contains_zero = (l <= 0) & (u >= 0)
    lb = torch.where(contains_zero, torch.zeros_like(l), torch.min(sq_l, sq_u))
    ub = torch.max(sq_l, sq_u)
    return lb, ub

def ibp_linear(layer: nn.Linear, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Propagate bounds through a Linear layer using the standard IBP formula.
    """
    weight = layer.weight
    bias = layer.bias

    # Split weights into positive and negative components, running on the GPU
    w_pos = weight.clamp(min=0)
    w_neg = weight.clamp(max=0)

    # Calculate bounds using matrix multiplication (F.linear)
    lb = F.linear(x_lb, w_pos) + F.linear(x_ub, w_neg)
    ub = F.linear(x_ub, w_pos) + F.linear(x_lb, w_neg)

    if bias is not None:
        lb = lb + bias
        ub = ub + bias

    return lb, ub

def ibp_activation(layer: nn.Module, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Propagate bounds through monotonic activations (all supported FCN activations).
    """
    # For monotonically increasing functions, simply apply the function to the bounds.
    return layer(x_lb), layer(x_ub)

def ibp_generic(layer: nn.Module, x_lb: torch.Tensor, x_ub: torch.Tensor):
    """
    Dispatcher for interval bound propagation.
    """
    if isinstance(layer, nn.Linear):
        return ibp_linear(layer, x_lb, x_ub)
    
    # Check for supported monotonic activations
    if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.Identity, nn.LeakyReLU)):
        return ibp_activation(layer, x_lb, x_ub)
    
    raise NotImplementedError(f"Interval arithmetic not implemented for layer type: {type(layer)}")

def ibp_jacobian_linear(W: torch.Tensor, J_lb: torch.Tensor, J_ub: torch.Tensor):

    W_pos = W.clamp(min=0); W_neg = W.clamp(max=0)
    J_new_lb = torch.einsum('kp, bpn -> bkn', W_pos, J_lb) + torch.einsum('kp, bpn -> bkn', W_neg, J_ub)
    J_new_ub = torch.einsum('kp, bpn -> bkn', W_pos, J_ub) + torch.einsum('kp, bpn -> bkn', W_neg, J_lb)

    return J_new_lb, J_new_ub

def ibp_jacobian_activation(D_lb: torch.Tensor, D_ub: torch.Tensor, J_lb: torch.Tensor, J_ub: torch.Tensor):
    
    D_lb_exp = D_lb.unsqueeze(-1); D_ub_exp = D_ub.unsqueeze(-1)
    P1 = D_lb_exp * J_lb; P2 = D_lb_exp * J_ub
    P3 = D_ub_exp * J_lb; P4 = D_ub_exp * J_ub
    J_new_lb = torch.min(torch.min(P1, P2), torch.min(P3, P4))
    J_new_ub = torch.max(torch.max(P1, P2), torch.max(P3, P4))

    return J_new_lb, J_new_ub

def get_derivative_bounds(u_lb: torch.Tensor, u_ub: torch.Tensor, act_name: str, dphi_fn: callable = None, leak: float = 0.01):
    
    if act_name == 'identity': return torch.ones_like(u_lb), torch.ones_like(u_ub)
    if act_name == 'relu':
        fully_neg = u_ub <= 0; fully_pos = u_lb >= 0; crosses = (~fully_neg) & (~fully_pos)
        d_lb = torch.where(fully_neg, torch.zeros_like(u_lb), torch.ones_like(u_lb))
        d_ub = torch.where(fully_pos, torch.ones_like(u_ub), torch.ones_like(u_ub))
        d_lb = torch.where(crosses, torch.zeros_like(u_lb), d_lb)
        d_ub = torch.where(crosses, torch.ones_like(u_ub), d_ub)
        return d_lb, d_ub
    if act_name == 'leaky_relu':
        a = leak; min_der = min(1.0, a); max_der = max(1.0, a)
        fully_neg = u_ub <= 0; fully_pos = u_lb >= 0; crosses = (~fully_neg) & (~fully_pos)
        d_lb = torch.where(fully_neg, torch.full_like(u_lb, a), torch.ones_like(u_lb))
        d_ub = torch.where(fully_neg, torch.full_like(u_ub, a), torch.ones_like(u_ub))
        d_lb = torch.where(crosses, torch.full_like(u_lb, min_der), d_lb)
        d_ub = torch.where(crosses, torch.full_like(u_ub, max_der), d_ub)
        return d_lb, d_ub
    
    # --- Logic for Smooth Unimodal Derivatives (Tanh, Sigmoid) ---
    if act_name in ['tanh', 'sigmoid']:
        # 1. Evaluate derivatives at the boundaries
        d_at_lb = dphi_fn(u_lb)
        d_at_ub = dphi_fn(u_ub)
        
        # 2. Evaluate derivative at the peak (u=0)
        d_at_zero = dphi_fn(torch.zeros_like(u_lb)) # This will be 1.0 for Tanh, 0.25 for Sigmoid
        
        # 3. Check if the interval contains the peak (u=0)
        contains_zero = (u_lb < 0) & (u_ub > 0)
        
        # The true lower bound is always the minimum of the two boundary values.
        # This corresponds to the point furthest from zero.
        D_lb = torch.min(d_at_lb, d_at_ub)
        
        # The true upper bound depends on whether the peak is included:
        # If the interval contains zero, the max derivative is at zero (d_at_zero).
        # Otherwise, the max derivative is the max of the two boundary values 
        # (the boundary closest to zero).
        max_d_at_boundaries = torch.max(d_at_lb, d_at_ub)
        D_ub = torch.where(contains_zero, d_at_zero, max_d_at_boundaries)
        
        return D_lb, D_ub

    if act_name == 'softplus':
        # Softplus derivative (sigmoid) is monotonically increasing.
        d_at_lb = dphi_fn(u_lb)
        d_at_ub = dphi_fn(u_ub)
        return d_at_lb, d_at_ub
        
    raise ValueError(f"Unsupported activation for derivative bounds: {act_name}")

def get_second_derivative_bounds(u_lb: torch.Tensor, u_ub: torch.Tensor, act_name: str, d2phi_fn: callable = None):
    """
    Computes bounds for the second derivative (D2) of the activation function.
    """
    # Piecewise linear activations have 0 second derivative (almost everywhere).
    if act_name in ['relu', 'leaky_relu', 'identity']:
        return torch.zeros_like(u_lb), torch.zeros_like(u_ub)

    # Softplus second derivative is Sigmoid derivative (phi'' = sigmoid').
    # We can reuse get_derivative_bounds logic for sigmoid.
    if act_name == 'softplus':
        # We fake the dphi/d2phi calls since we just need the bounds logic of Sigmoid's 1st deriv
        # The "derivative of softplus" is sigmoid. So "second derivative of softplus" is "derivative of sigmoid".
        sigma_prime = get_activation_der('sigmoid')
        # Use the logic for sigmoid derivative bounds
        return get_derivative_bounds(u_lb, u_ub, 'sigmoid', dphi_fn=sigma_prime)

    if act_name == 'tanh':
        # Tanh second derivative: phi''(x) = -2 tanh(x) (1 - tanh^2(x))
        # It is an odd function.
        # max = 0.7698004 at x approx -0.6584789
        # min = -0.7698004 at x approx 0.6584789
        
        # Constants
        c_pos = 0.6584789       # Location of the valley (local min)
        c_neg = -0.6584789      # Location of the peak (local max)
        val_max = 0.7698004     # Value at c_neg
        val_min = -0.7698004    # Value at c_pos
        
        # 1. Evaluate at boundaries
        v_lb = d2phi_fn(u_lb)
        v_ub = d2phi_fn(u_ub)
        
        # 2. Check inclusion of critical points
        # Does the interval contain the negative critical point (Global Max)?
        has_neg_peak = (u_lb < c_neg) & (u_ub > c_neg)
        # Does the interval contain the positive critical point (Global Min)?
        has_pos_peak = (u_lb < c_pos) & (u_ub > c_pos)
        
        # 3. Determine bounds
        # The min is the minimum of boundaries, unless we cross the positive critical point (valley)
        current_min = torch.min(v_lb, v_ub)
        D2_lb = torch.where(has_pos_peak, torch.full_like(u_lb, val_min), current_min)
        
        # The max is the maximum of boundaries, unless we cross the negative critical point (peak)
        current_max = torch.max(v_lb, v_ub)
        D2_ub = torch.where(has_neg_peak, torch.full_like(u_ub, val_max), current_max)
        
        return D2_lb, D2_ub

    if act_name == 'sigmoid':
        # Sigmoid second derivative: phi''(x) = sigma(x)(1-sigma(x))(1-2sigma(x))
        # It is an odd function (can be proven).
        # Roots at x=0.
        # max = 0.096225 at x approx -1.31696
        # min = -0.096225 at x approx 1.31696
        
        # Constants
        c_pos = 1.3169579       # Location of the valley (local min)
        c_neg = -1.3169579      # Location of the peak (local max)
        val_max = 0.096225      # Value at c_neg
        val_min = -0.096225     # Value at c_pos
        
        v_lb = d2phi_fn(u_lb)
        v_ub = d2phi_fn(u_ub)
        
        has_neg_peak = (u_lb < c_neg) & (u_ub > c_neg)
        has_pos_peak = (u_lb < c_pos) & (u_ub > c_pos)
        
        current_min = torch.min(v_lb, v_ub)
        D2_lb = torch.where(has_pos_peak, torch.full_like(u_lb, val_min), current_min)
        
        current_max = torch.max(v_lb, v_ub)
        D2_ub = torch.where(has_neg_peak, torch.full_like(u_ub, val_max), current_max)
        
        return D2_lb, D2_ub

    raise ValueError(f"Unsupported activation for second derivative bounds: {act_name}")

def get_third_derivative_bounds(u_lb: torch.Tensor, u_ub: torch.Tensor, act_name: str, d3phi_fn: callable = None):
    """
    Computes bounds for the third derivative (D3) of the activation function.
    """
    if act_name in ['relu', 'leaky_relu', 'identity']:
        return torch.zeros_like(u_lb), torch.zeros_like(u_ub)

    # Softplus''' is Sigmoid''
    if act_name == 'softplus':
        from .activation_function import get_activation_second_der
        sig_d2 = get_activation_second_der('sigmoid')
        return get_second_derivative_bounds(u_lb, u_ub, 'sigmoid', d2phi_fn=sig_d2)
        
    # Evaluate at boundaries
    v_lb = d3phi_fn(u_lb)
    v_ub = d3phi_fn(u_ub)
    current_min = torch.min(v_lb, v_ub)
    current_max = torch.max(v_lb, v_ub)

    if act_name == 'tanh':
        # Tanh'''(x) = (1-t^2)(6t^2-2). Even function.
        # Min at x=0 (t=0), val = -2.
        # Max at x=+/-1.146 (t^2=2/3), val = 2/3.
        
        val_min = -2.0
        val_max = 2.0/3.0
        c_max = 1.146
        
        # Check inclusions
        has_min = (u_lb < 0) & (u_ub > 0)
        has_pos_max = (u_lb < c_max) & (u_ub > c_max)
        has_neg_max = (u_lb < -c_max) & (u_ub > -c_max)
        has_max = has_pos_max | has_neg_max
        
        D3_lb = torch.where(has_min, torch.full_like(u_lb, val_min), current_min)
        D3_ub = torch.where(has_max, torch.full_like(u_ub, val_max), current_max)
        return D3_lb, D3_ub

    if act_name == 'sigmoid':
        # Sigmoid''' is Even.
        # Roots of S''' derivative are at x=0, x=+/-2.29
        # x=0 is Local Min (-1/8 = -0.125)
        # x=+/-2.29 is Max (approx 1/24 = 0.04166)
        
        val_min = -1.0/8
        val_max = 1.0/24 # Approx
        c_max = 2.292431
        
        has_min = (u_lb < 0) & (u_ub > 0)
        has_pos_max = (u_lb < c_max) & (u_ub > c_max)
        has_neg_max = (u_lb < -c_max) & (u_ub > -c_max)
        has_max = has_pos_max | has_neg_max

        D3_lb = torch.where(has_min, torch.full_like(u_lb, val_min), current_min)
        D3_ub = torch.where(has_max, torch.full_like(u_ub, val_max), current_max)
        return D3_lb, D3_ub

    # Default fallback if not specifically handled (though covers common ones)
    return current_min, current_max