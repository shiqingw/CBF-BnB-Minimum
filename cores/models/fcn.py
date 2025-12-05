import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import List, Union
from .activation_function import get_activation, get_activation_der, get_activation_second_der, get_activation_third_der

class FullyConnectedNetwork(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 activations: List[str], 
                 widths: List[int], 
                 layer_bias: bool = True,
                 input_bias: Union[None, List[float]] = None,
                 input_transform: Union[None, List[float]] = None, 
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()

        # If widths includes input and output, length is num_layers + 1
        if len(widths) < 2:
            raise ValueError("Widths must include at least input and output dimensions.")
        
        # We expect activations for hidden layers. 
        # Usually: Input -> [Linear -> Activation] ... -> [Linear] -> Output
        # So len(activations) should act on the hidden layers.
        # This implementation assumes the user provided activations for every layer *except* the last linear layer.
        if len(activations) != len(widths) - 2: 
             raise ValueError(f"Expected {len(widths) - 2} activations, got {len(activations)}.")

        if widths[-1] != out_features:
            raise ValueError("Last width must match number of output channels.")
        if widths[0] != in_features:
            raise ValueError("First width must match number of input channels.")

        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations
        self.widths = widths
        self.dtype = dtype
        
        if input_bias is None:
            input_bias_t = torch.zeros(in_features, dtype=self.dtype, requires_grad=False)
        else:
            input_bias_t = torch.tensor(input_bias, dtype=self.dtype, requires_grad=False)
            
        if input_transform is None:
            input_transform_t = torch.ones(in_features, dtype=self.dtype, requires_grad=False)
        else:
            input_transform_t = torch.tensor(input_transform, dtype=self.dtype, requires_grad=False)
            
        self.register_buffer('input_bias', input_bias_t)
        self.register_buffer('input_transform', input_transform_t)
        
        # Build Layers
        layers = []
        self.activation_derivatives = []
        self.activation_second_derivatives = []
        self.activation_third_derivatives = []
        # Loop over the number of transitions (Linear layers needed)
        # widths: [d_in, h1, h2, d_out] -> 3 transitions
        for i in range(len(widths) - 1):
            # Add Linear Layer
            layers.append(nn.Linear(widths[i], widths[i+1], bias=layer_bias))
            
            # Add Activation (if not the last layer)
            if i < len(activations):
                layers.append(get_activation(activations[i]))
                self.activation_derivatives.append(get_activation_der(activations[i]))
                self.activation_second_derivatives.append(get_activation_second_der(activations[i]))
                self.activation_third_derivatives.append(get_activation_third_der(activations[i]))

        self.model = nn.Sequential(*layers)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        # Apply input transform
        x = (x + self.input_bias) * self.input_transform
        # Apply model
        y = self.model(x)       
        return y
    
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian using optimized Transposed Propagation.
        
        Strategy: Propagate the TRANSPOSE of the Jacobian (J^T).
        J^T shape: (Batch, Network_Input_Dim, Layer_Input_Dim)
        
        Benefits:
        1. Compatible with auto_LiRPA (Standard Matmul/Linear structure).
        2. Fast (Contiguous memory access).
        3. Autograd Safe (Out-of-place operations).
        
        Returns:
            J: (Batch, Out_Features, In_Features)
        """

        batch_size = x.shape[0]

        # 1. State Initialization
        z = (x + self.input_bias) * self.input_transform
        
        # 2. Jacobian Transpose Initialization
        # J_0 = diag(t)  =>  J_0^T = diag(t)
        # Create Identity and scale to get diag(t)
        eye = torch.eye(self.in_features, device=x.device, dtype=x.dtype)
        
        # Shape: (1, In, In)
        J_T = eye.unsqueeze(0) * self.input_transform.view(1, -1, 1)
        
        # Expand to batch size and Clone to ensure independent gradients
        J_T = J_T.expand(batch_size, -1, -1).clone()

        act_index = 0
        
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # Update State: z_{l} = W z_{l-1} + b
                z = layer(z)
                
                # Update Jacobian Transpose:
                # Rule: J_{l} = W @ J_{l-1}
                # Transpose: J_{l}^T = J_{l-1}^T @ W^T
                J_T = torch.matmul(J_T, layer.weight.T)

            else:
                # Activation Layer
                der_func = self.activation_derivatives[act_index]
                d_sigma = der_func(z) # Shape: (Batch, Width)
                
                # Update State: z_{l} = sigma(z_{l})
                z = layer(z)
                
                # Update Jacobian Transpose:    
                # Rule: J_{l} = diag(d_sigma) @ J_{l-1}
                # Transpose: J_{l}^T = J_{l-1}^T @ diag(d_sigma)
                # Equivalent to scaling the columns of J^T
                J_T = J_T * d_sigma.unsqueeze(1)
                
                act_index += 1

        # Transpose back to (Batch, Out_Features, In_Features)
        return J_T.transpose(1, 2)
    
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian matrix (Second Derivative) recursively.
        Designed for Auto_LiRPA compatibility (Avoids einsum, uses permute+matmul).
        
        Args:
            x: Input tensor (Batch, In_Features)
        Returns:
            H: Hessian tensor (Batch, Out_Features, In_Features, In_Features)
        """

        batch_size = x.shape[0]

        # --- 1. Initialization ---
        z = (x + self.input_bias) * self.input_transform
        
        # J: (Batch, Layer_Dim, Input_Dim)
        eye = torch.eye(self.in_features, device=x.device, dtype=x.dtype)
        J = eye.unsqueeze(0) * self.input_transform.view(1, -1, 1)
        J = J.expand(batch_size, -1, -1).clone()

        # H: (Batch, Layer_Dim, Input_Dim, Input_Dim)
        H = torch.zeros(batch_size, self.in_features, self.in_features, self.in_features, 
                        device=x.device, dtype=x.dtype)

        act_index = 0

        # --- 2. Recursion ---
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # --- Linear Step ---
                # 1. Update Jacobian: J_new = W @ J
                # Permute J to (B, N, In_Layer) to multiply W on the right
                J_perm = J.permute(0, 2, 1) 
                # (B, N, In_Layer) @ (In_Layer, Out_Layer) -> (B, N, Out_Layer)
                J = torch.matmul(J_perm, layer.weight.T).permute(0, 2, 1)
                
                # 2. Update Hessian: H_new = W @ H
                # We need to contract W with the channel dim (dim 1) of H.
                # H: (B, C_in, N, N) -> Permute to (B, N, N, C_in)
                H_perm = H.permute(0, 2, 3, 1)
                # (B, N, N, C_in) @ (C_in, C_out) -> (B, N, N, C_out)
                H_new = torch.matmul(H_perm, layer.weight.T)
                # Permute back to (B, C_out, N, N)
                H = H_new.permute(0, 3, 1, 2)
                
                # 3. Update Value
                z = layer(z)
                
            else:
                # --- Activation Step ---
                d1_func = self.activation_derivatives[act_index]
                d2_func = self.activation_second_derivatives[act_index]
                
                sigma_1 = d1_func(z)
                sigma_2 = d2_func(z) if d2_func else None
                
                z = layer(z)
                
                # Update Hessian
                # Term 1: Propagated Curvature -> sigma' * H
                term1 = sigma_1.view(batch_size, -1, 1, 1) * H
                
                term2 = 0
                if sigma_2 is not None:
                    # Term 2: Generated Curvature -> sigma'' * (J outer J)
                    # J is (B, L, N).
                    # J.unsqueeze(-1) is (B, L, N, 1)
                    # J.unsqueeze(-2) is (B, L, 1, N)
                    # Broadcasting creates Outer Product (B, L, N, N)
                    outer = J.unsqueeze(-1) * J.unsqueeze(-2)
                    term2 = sigma_2.view(batch_size, -1, 1, 1) * outer
                
                H = term1 + term2
                
                # Update Jacobian: J_new = sigma' * J
                J = sigma_1.unsqueeze(-1) * J
                
                act_index += 1
                
        return H

    def derivative3(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes 3rd Order Derivative D3 (B, Out, N, N, N).
        """
        # NO DIM CHECKS - Assume (Batch, N)
        batch_size = x.shape[0]
        n_in = self.in_features

        z = (x + self.input_bias) * self.input_transform
        
        eye = torch.eye(n_in, device=x.device, dtype=x.dtype)
        J = eye.unsqueeze(0) * self.input_transform.view(1, -1, 1)
        J = J.expand(batch_size, -1, -1).clone()

        H = torch.zeros(batch_size, n_in, n_in, n_in, device=x.device, dtype=x.dtype)
        D3 = torch.zeros(batch_size, n_in, n_in, n_in, n_in, device=x.device, dtype=x.dtype)

        act_index = 0

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # --- Linear Step ---
                
                # J: (B, L, N)
                J_perm = J.permute(0, 2, 1)
                J = torch.matmul(J_perm, layer.weight.T).permute(0, 2, 1)
                
                # H: (B, L, N, N) -> Permute L to last
                H_perm = H.permute(0, 2, 3, 1)
                H = torch.matmul(H_perm, layer.weight.T).permute(0, 3, 1, 2)
                
                # D3: (B, L, N, N, N) -> Permute L to last
                D3_perm = D3.permute(0, 2, 3, 4, 1)
                D3 = torch.matmul(D3_perm, layer.weight.T).permute(0, 4, 1, 2, 3)
                
                z = layer(z)
                
            else:
                # --- Activation Step ---
                d1 = self.activation_derivatives[act_index]
                d2 = self.activation_second_derivatives[act_index]
                d3 = self.activation_third_derivatives[act_index]
                
                s1 = d1(z)
                s2 = d2(z)
                s3 = d3(z)
                z = layer(z) 
                
                # Broadcasting Views
                # s: (B, L)
                s_view_3d = s1.view(batch_size, -1, 1)          # For J: (B, L, N)
                s_view_4d = s1.view(batch_size, -1, 1, 1)       # For H: (B, L, N, N)
                s_view_5d = s1.view(batch_size, -1, 1, 1, 1)    # For D3: (B, L, N, N, N)
                
                s2_view_4d = s2.view(batch_size, -1, 1, 1)
                s2_view_5d = s2.view(batch_size, -1, 1, 1, 1)
                s3_view_5d = s3.view(batch_size, -1, 1, 1, 1)
                
                # --- Update D3 ---
                # Term 1: s' * D3
                term1 = s_view_5d * D3
                
                # Term 2: s'' * (J_i H_jk + J_j H_ik + J_k H_ij)
                c1 = J.unsqueeze(-1).unsqueeze(-1) * H.unsqueeze(-3)
                c2 = J.unsqueeze(-1).unsqueeze(-3) * H.unsqueeze(-2)
                c3 = J.unsqueeze(-2).unsqueeze(-3) * H.unsqueeze(-1)
                term2 = s2_view_5d * (c1 + c2 + c3)
                    
                # Term 3: s''' * (J_i J_j J_k)
                j_outer_3 = J.unsqueeze(-1).unsqueeze(-1) * \
                            J.unsqueeze(-1).unsqueeze(-3) * \
                            J.unsqueeze(-2).unsqueeze(-3)
                
                term3 = s3_view_5d * j_outer_3
                    
                D3 = term1 + term2 + term3
                
                # --- Update H (Post-update) ---
                # H_new = s' H + s'' (J outer J)
                outer_j = J.unsqueeze(-1) * J.unsqueeze(-2) # (B, L, N, N)
                h_term2 = s2_view_4d * outer_j
                H = (s_view_4d * H) + h_term2
                
                # --- Update J ---
                J = s_view_3d * J
                
                act_index += 1
                
        return D3
