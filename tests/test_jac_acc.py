from cores.models.fcn import FullyConnectedNetwork
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian as torch_jacobian
import numpy as np

def test_jacobian_accuracy():
    print("--- Starting Jacobian Verification ---")
    device = torch.device("cuda")
    
    # Configuration
    input_dim = 10
    output_dim = 10
    widths = [input_dim, 5, 9, output_dim]
    activations = ['tanh', 'sigmoid'] # Matches len(widths) - 2
    batch_size = 5

    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    
    # Instantiate Model
    # Using your JacobiaNetwork wrapper logic or the method directly
    net = FullyConnectedNetwork(
        in_features=input_dim,
        out_features=output_dim,
        activations=activations,
        widths=widths,
        layer_bias=True,
        input_bias=rand_input_bias,
        input_transform=rand_input_transform
    ).to(device)
    
    # Create random inputs
    # Ensure requires_grad=True is NOT set for your manual method (usually), 
    # but IS required for autograd if we were doing backward(). 
    # torch.autograd.functional.jacobian handles the grad tracking internally.
    x_batch = torch.randn(batch_size, input_dim, device=device)
    
    # --- A. Compute Your Manual Jacobian ---
    # Expected Shape: (Batch, Out, In)
    manual_jac = net.jacobian(x_batch)
    
    # --- B. Compute PyTorch Autograd Jacobian ---
    # torch.autograd.functional.jacobian computes J for a function func(x).
    # If x is (Batch, In), and y is (Batch, Out), 
    # PyTorch's jacobian returns shape (Batch, Out, Batch, In).
    # It calculates dy[b1]/dx[b2], which includes cross-batch terms (usually zero).
    # We need to extract the diagonal blocks (where b1 == b2).

    def forward_fn(x):
        return net(x)

    # Note: functional.jacobian can be slow for large batches
    autograd_full_jac = torch_jacobian(forward_fn, x_batch)
    
    # Extract only the relevant (batch_i, :, batch_i, :) slices
    # Auto_jac shape: (Batch, Out, Batch, In) -> We want (Batch, Out, In)
    autograd_jac = torch.einsum('bibj->bij', autograd_full_jac)
    
    # --- C. Comparison ---
    print(f"Manual Shape:   {manual_jac.shape}")
    print(f"Autograd Shape: {autograd_jac.shape}")
    
    # Check max difference
    diff = (manual_jac - autograd_jac).abs().max().item()
    print(f"Max Difference: {diff:.8f}")
    
    # Assertions
    if diff < 1e-5:
        print("✅ SUCCESS: Manual Jacobian matches Autograd Jacobian.")
    else:
        print("❌ FAILURE: Significant difference detected.")
        # Debug print
        print("\nFirst Batch Sample - Manual:\n", manual_jac[0])
        print("\nFirst Batch Sample - Autograd:\n", autograd_jac[0])

if __name__ == "__main__":
    test_jacobian_accuracy()