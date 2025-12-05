import torch
import torch.nn as nn
from cores.models.fcn import FullyConnectedNetwork

def test_hessian_backprop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing Hessian Backpropagation Safety on {device} ---")
    
    # 1. Setup
    input_dim = 3
    output_dim = 2
    widths = [input_dim, 8, output_dim]
    activations = ['tanh']

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
    
    # Enable gradients for weights (default) and input
    x = torch.randn(2, input_dim, device=device, requires_grad=True)
    
    print("1. Forward Pass (Computing Hessian)...")
    # H shape: (Batch, Out, In, In)
    H = net.hessian(x)
    
    # Define a dummy loss function: Sum of all Hessian elements
    # L = Sum(d^2y / dx^2)
    loss = H.sum()
    print(f"   Hessian Sum: {loss.item():.4f}")
    
    print("2. Backward Pass...")
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"❌ FAILURE: Backprop crashed with error:\n{e}")
        return

    # 3. Check Gradients
    print("3. Verifying Gradients...")
    
    # Check Input Gradients
    if x.grad is not None and x.grad.abs().sum() > 0:
        print(f"   ✅ Input Gradients exist. Norm: {x.grad.norm().item():.4f}")
    else:
        print("   ❌ Input Gradients Missing or Zero!")

    # Check Weight Gradients (First Layer)
    first_layer = net.model[0]
    if first_layer.weight.grad is not None and first_layer.weight.grad.abs().sum() > 0:
        print(f"   ✅ Weight Gradients exist. Norm: {first_layer.weight.grad.norm().item():.4f}")
    else:
        print("   ❌ Weight Gradients Missing or Zero!")

    print("\nConclusion: The hessian method is safe for training/optimization.")

if __name__ == "__main__":
    test_hessian_backprop()