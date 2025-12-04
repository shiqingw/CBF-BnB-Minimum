import torch
import torch.nn as nn
import time
from cores.models.fcn import FullyConnectedNetwork
from torch.autograd.functional import jacobian as torch_jacobian

def benchmark_jacobian_options():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Benchmarking Jacobian Implementations on {device} ---")
    
    # Setup Large Model to make timing differences obvious
    input_dim = 100
    output_dim = 50
    widths = [input_dim, 256, 256, 128, output_dim]
    activations = ['tanh', 'softplus', 'sigmoid']
    
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths
    ).to(device)
    
    batch_size = 64
    x = torch.randn(batch_size, input_dim, device=device)
    
    # --- 1. Define Alternative Implementations ---
    
    def implementation_broadcasting(model, x):
        """The optimized version currently in class"""
        return model.jacobian(x)
        
    def implementation_einsum(model, x):
        """Alternative using Einsum"""
        if x.dim() == 1: x = x.unsqueeze(0)
        batch_size = x.shape[0]
        z = (x + model.input_bias) * model.input_transform
        
        eye = torch.eye(model.in_features, device=x.device, dtype=x.dtype)
        J = torch.einsum('i, ij -> ij', model.input_transform, eye).unsqueeze(0).expand(batch_size, -1, -1)
        
        act_index = 0
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                z = layer(z)
                J = torch.matmul(layer.weight, J)
            else:
                der_func = model.activation_derivatives[act_index]
                d_sigma = der_func(z)
                z = layer(z)
                # Einsum scaling
                J = torch.einsum('bi, bij -> bij', d_sigma, J)
                act_index += 1
        return J

    def implementation_loop(model, x):
        """The slow loop-based 'safe' version"""
        if x.dim() == 1: x = x.unsqueeze(0)
        batch_size = x.shape[0]
        z = (x + model.input_bias) * model.input_transform
        
        eye = torch.eye(model.in_features, device=x.device, dtype=x.dtype)
        J_base = eye * model.input_transform.unsqueeze(1)
        J = J_base.unsqueeze(0).expand(batch_size, -1, -1)

        act_index = 0
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                z = layer(z)
                J = torch.matmul(layer.weight, J)
            else:
                der_func = model.activation_derivatives[act_index]
                d_sigma = der_func(z)
                z = layer(z)
                
                # SLOW LOOP
                J_slices = []
                for i in range(model.in_features):
                    j_slice = J.select(2, i)
                    j_new_slice = j_slice * d_sigma
                    J_slices.append(j_new_slice)
                J = torch.stack(J_slices, dim=2)
                act_index += 1
        return J
    
    def implementation_autograd(model, x):
        """Standard Autograd (Ground Truth)"""
        # Note: computes full cross-batch Jacobian, extremely expensive
        def func(inp):
            return model(inp)
        full_jac = torch_jacobian(func, x)
        return torch.einsum('bibj->bij', full_jac)

    # --- 2. Run Benchmarks ---
    
    methods = [
        ("Broadcasting (Current)", implementation_broadcasting),
        ("Einsum", implementation_einsum),
        ("Loop (Slow Safe)", implementation_loop),
        # ("Autograd", implementation_autograd) # Uncomment if you want to wait, usually 10x slower
    ]
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        implementation_broadcasting(net, x)
        
    iterations = 50
    print(f"Running {iterations} iterations per method...")
    
    for name, func in methods:
        # Sync CUDA if needed
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            func(net, x)
            
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000 # ms
        print(f"{name:<25}: {avg_time:.4f} ms per batch")

if __name__ == "__main__":
    benchmark_jacobian_options()