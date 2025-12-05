import torch
from cores.models.fcn import FullyConnectedNetwork

def test_derivative3_einsum():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing 3rd Order Derivative (Einsum Version) on {device} ---")
    
    # 1. Setup
    input_dim = 10
    output_dim = 2
    widths = [input_dim, 8, output_dim]
    activations = ['tanh'] 
    rand_input_bias = torch.randn(input_dim).tolist()
    rand_input_transform = torch.randn(input_dim).tolist()
    
    dtype = torch.float32 
    
    net = FullyConnectedNetwork(
        in_features=input_dim, 
        out_features=output_dim, 
        activations=activations, 
        widths=widths,
        input_bias=rand_input_bias,
        input_transform=rand_input_transform,
        dtype=dtype
    ).to(device)
    
    x = torch.randn(2, input_dim, device=device, dtype=dtype, requires_grad=True)
    
    # 2. Manual D3 
    print("Computing Manual D3 ...")
    D3_manual = net.derivative3(x)
    
    # 3. Autograd Verification (Jacobian of Hessian)
    print("Computing Autograd D3...")
    D3_autograd = torch.zeros_like(D3_manual)
    
    for b in range(x.shape[0]):
        def hessian_func(inp):
            hs = []
            for o in range(output_dim):
                def scalar_out(i): return net(i.unsqueeze(0))[0, o]
                
                # CRITICAL FIX: create_graph=True allows gradients to flow 
                # through the Hessian calculation itself!
                h = torch.autograd.functional.hessian(scalar_out, inp, create_graph=True)
                hs.append(h)
            return torch.stack(hs) 

        # Jacobian of Hessian = D3
        d3_elem = torch.autograd.functional.jacobian(hessian_func, x[b])
        D3_autograd[b] = d3_elem

    # 4. Analysis
    diff = (D3_manual - D3_autograd).abs().max().item()
    print(f"D3 Shape: {D3_manual.shape}")
    print(f"Max Difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: Manual D3 matches Autograd.")
    else:
        print("❌ FAILURE: Mismatch detected.")
        print("Manual Sample [0,0,0,0]:\n", D3_manual[0,0,0,0])
        print("Auto   Sample [0,0,0,0]:\n", D3_autograd[0,0,0,0])

if __name__ == "__main__":
    test_derivative3_einsum()