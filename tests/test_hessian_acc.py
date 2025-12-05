import torch
import torch.nn as nn
from cores.models.fcn import FullyConnectedNetwork

def test_hessian_correctness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing Recursive Hessian on {device} ---")
    
    # 1. Setup Network
    input_dim = 10
    output_dim = 2
    widths = [input_dim, 8, 8, output_dim]
    activations = ['tanh', 'sigmoid']
    batch_size = 3

    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths,
        input_bias=rand_input_bias,
        input_transform=rand_input_transform
    ).to(device)
    
    # 2. Input
    x = torch.randn(batch_size, input_dim, device=device, requires_grad=True)
    
    # 3. Compute Manual Hessian
    # Shape: (Batch, Out, In, In)
    H_manual = net.hessian(x)
    
    # 4. Compute Autograd Hessian
    # PyTorch's hessian function works on scalar outputs. 
    # We must compute it for every output element of every batch item.
    
    H_autograd = torch.zeros_like(H_manual)
    
    print("Computing Autograd Hessian (Slow loop)...")
    for b in range(batch_size):
        for out_idx in range(output_dim):
            
            def scalar_func(inp_vec):
                # Run forward pass for specific batch item
                # inp_vec is (Input_Dim)
                out = net(inp_vec.unsqueeze(0)) # Returns (1, Out_Dim)
                return out[0, out_idx]
            
            # Compute Hessian for this specific output scalar
            # x[b] is (Input_Dim)
            h_element = torch.autograd.functional.hessian(scalar_func, x[b])
            
            H_autograd[b, out_idx] = h_element

    # 5. Compare
    diff = (H_manual - H_autograd).abs().max().item()
    print(f"Hessian Shape: {H_manual.shape}")
    print(f"Max Difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: Manual Hessian matches Autograd.")
    else:
        print("❌ FAILURE: Significant difference detected.")
        print("Manual[0,0]:\n", H_manual[0,0])
        print("Autograd[0,0]:\n", H_autograd[0,0])

if __name__ == "__main__":
    test_hessian_correctness()