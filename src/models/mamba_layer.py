"""
Simplified Mamba layer for hyperbolic models.
Mamba always operates in Euclidean space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MambaLayer(nn.Module):
    """
    Simplified Mamba-like selective state space layer.
    Always operates in Euclidean space.
    
    This is a simplified version focused on the key Mamba concepts:
    - Selective state space mechanism
    - Linear time complexity
    - Data-dependent filtering
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Selective mechanism
        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space parameters (data-independent initialization)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Mamba layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, state) where:
                - output has shape (batch, seq_len, d_model)
                - state is None (simplified version)
        """
        batch, seq_len, _ = x.shape
        
        # Apply normalization
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # Selective SSM
        y = self.selective_scan(x)
        
        # Gating mechanism
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection
        output = output + residual
        
        return output, None
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan mechanism.
        
        Args:
            x: Input of shape (batch, seq_len, d_inner)
            
        Returns:
            Output of shape (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        
        # Data-dependent parameters
        delta = F.softplus(self.dt_proj(x))  # (batch, seq_len, d_inner)
        
        # Simplified state update (using cumsum as approximation)
        # In full Mamba, this would be a parallel scan
        A = -torch.exp(self.A.float())  # (d_inner, d_state)
        
        # Project input to state space
        B = self.x_proj(x)  # (batch, seq_len, d_state)
        C = self.x_proj(x)  # (batch, seq_len, d_state)
        
        # Simplified SSM computation
        # y = C @ (discretized_A @ state + discretized_B @ x)
        y = torch.einsum('bld,de,ble->bld', delta, A, B) * x
        
        # Add direct path
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba block with FFN.
    """
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.mamba = MambaLayer(d_model, d_state)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, state)
        """
        # Mamba layer
        x, state = self.mamba(x)
        
        # FFN with residual
        x = x + self.ffn(x)
        
        return x, state
