import torch
import torch.nn as nn 
from typing import Callable

def get_activation(activation_name: str) -> nn.Module:
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'softplus':
        return nn.Softplus()
    elif activation_name == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported activation function: {activation_name}")

def get_activation_der(activation_name: str) -> Callable:
    if activation_name == 'sigmoid':
        def sigmoid_derivative(x):
            return 0.25 * (1 - torch.tanh(x / 2)**2)
        return sigmoid_derivative

    elif activation_name == 'tanh':
        def tanh_derivative(x):
            return 1 - torch.tanh(x) ** 2
        return tanh_derivative
    
    elif activation_name == 'softplus':
        def softplus_derivative(x):
            return torch.sigmoid(x)
        return softplus_derivative

    elif activation_name == 'identity':
        def identity_derivative(x):
            return torch.ones_like(x)
        return identity_derivative
    
    else:
        raise NotImplementedError(f"Derivative not implemented for {activation_name}")
    
def get_activation_second_der(activation_name: str) -> Callable:
    if activation_name == 'sigmoid':
        def sigmoid_second_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig) * (1 - 2 * sig)
        return sigmoid_second_derivative

    elif activation_name == 'tanh':
        def tanh_second_derivative(x):
            return -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)
        return tanh_second_derivative
    
    elif activation_name == 'softplus':
        def softplus_second_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig)
        return softplus_second_derivative

    elif activation_name == 'identity':
        def identity_second_derivative(x):
            return torch.zeros_like(x)
        return identity_second_derivative
    
    else:
        raise NotImplementedError(f"Second derivative not implemented for {activation_name}")
    
def get_activation_third_der(activation_name: str) -> Callable:
    if activation_name == 'identity':
        def identity_third_derivative(x):
            return torch.zeros_like(x)
        return identity_third_derivative
    
    elif activation_name == 'sigmoid':
        # s' = s(1-s)
        # s'' = s'(1-2s)
        # s''' = s''(1-2s) - 2(s')^2  <-- Chain rule on s''
        # Alternatively: s''' = s(1-s)(1 - 6s + 6s^2)
        def sigmoid_third_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig) * (1 - 6 * sig + 6 * sig**2)
        return sigmoid_third_derivative
    
    elif activation_name == 'tanh':
        # t' = 1 - t^2
        # t'' = -2t(1-t^2)
        # t''' = -2(1-t^2) + 6t^2(1-t^2) = (1-t^2)(-2 + 6t^2)
        # t''' = -2(1-t^2)(1 - 3t^2)
        def tanh_third_derivative(x):
            t = torch.tanh(x)
            return -2 * (1 - t**2) * (1 - 3 * t**2)
        return tanh_third_derivative
    
    elif activation_name == 'softplus':
        # Softplus''' is Sigmoid''
        def softplus_third_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig) * (1 - 2 * sig)
        return softplus_third_derivative
    
    else:
        raise NotImplementedError(f"Third derivative not implemented for {activation_name}")