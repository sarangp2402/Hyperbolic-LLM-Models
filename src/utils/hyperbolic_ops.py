"""
Hyperbolic geometry operations with safety checks.
Based on Poincaré ball model with MIT-style safety mechanisms.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PoincareManifold:
    """Poincaré ball manifold with curvature c."""
    
    def __init__(self, c=1.0):
        self.c = c
        self.eps = 1e-8
        
    def dist(self, x, y):
        """
        Compute hyperbolic distance between points x and y.
        
        Args:
            x: Tensor of shape (..., dim)
            y: Tensor of shape (..., dim)
            
        Returns:
            Distance tensor of shape (...)
        """
        sqrt_c = math.sqrt(self.c)
        
        # Compute Möbius addition: -x ⊕ y
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        # Möbius addition formula
        num = (1 + 2 * self.c * xy + self.c * y_norm_sq) * x - (1 - self.c * x_norm_sq) * y
        denom = 1 + 2 * self.c * xy + self.c * self.c * x_norm_sq * y_norm_sq
        
        diff = num / (denom + self.eps)
        diff_norm = torch.norm(diff, dim=-1)
        
        # Distance formula
        dist = (2.0 / sqrt_c) * torch.arctanh(sqrt_c * diff_norm + self.eps)
        
        return dist
    
    def projx(self, x):
        """
        Project point x onto the Poincaré ball.
        
        Args:
            x: Tensor of shape (..., dim)
            
        Returns:
            Projected tensor on the manifold
        """
        max_norm = (1.0 - self.eps) / math.sqrt(self.c)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Project if outside the ball
        projected = torch.where(
            x_norm > max_norm,
            x * (max_norm / (x_norm + self.eps)),
            x
        )
        
        return projected
    
    def expmap0(self, v):
        """
        Exponential map at origin.
        
        Args:
            v: Tangent vector at origin
            
        Returns:
            Point on manifold
        """
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        
        result = torch.tanh(sqrt_c * v_norm / 2.0) * v / (sqrt_c * v_norm + self.eps)
        
        return self.projx(result)
    
    def logmap0(self, p):
        """
        Logarithmic map at origin.
        
        Args:
            p: Point on manifold
            
        Returns:
            Tangent vector at origin
        """
        sqrt_c = math.sqrt(self.c)
        p_norm = torch.norm(p, dim=-1, keepdim=True)
        
        result = (2.0 / sqrt_c) * torch.arctanh(sqrt_c * p_norm + self.eps) * p / (p_norm + self.eps)
        
        return result


def safe_project(v, manifold, scale=0.01):
    """
    Safely project Euclidean vectors to hyperbolic space.
    Uses MIT-style safety mechanisms to prevent numerical instability.
    
    Args:
        v: Euclidean vectors of shape (..., dim)
        manifold: PoincareManifold instance
        scale: Scaling factor (default 0.01)
        
    Returns:
        Projected points on Poincaré ball
    """
    # Step 1: Scale down with tanh to ensure < 1
    effective_scale = torch.tanh(torch.tensor(scale))
    v_scaled = v * effective_scale
    
    # Step 2: Clamp norm to prevent explosion
    v_norm = torch.norm(v_scaled, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, max=5.0)
    v_scaled = v_scaled * torch.minimum(
        torch.ones_like(v_norm),
        5.0 / (v_norm + manifold.eps)
    )
    
    # Step 3: Apply exponential map with safety
    c_sqrt = math.sqrt(manifold.c)
    v_norm = torch.norm(v_scaled, dim=-1, keepdim=True)
    factor = torch.tanh(c_sqrt * v_norm) / (c_sqrt * v_norm + manifold.eps)
    p = v_scaled * factor
    
    # Step 4: Project onto manifold
    p = manifold.projx(p)
    
    # Step 5: Final safety check
    max_norm = (1.0 - 1e-5) / math.sqrt(manifold.c)
    p_norm = torch.norm(p, dim=-1, keepdim=True)
    p = torch.where(
        p_norm > max_norm,
        p * (max_norm / (p_norm + manifold.eps)),
        p
    )
    
    return p


def safe_expmap(v, manifold):
    """
    Safe exponential map at origin with clamping.
    
    Args:
        v: Tangent vectors
        manifold: PoincareManifold instance
        
    Returns:
        Points on manifold
    """
    # Clamp input norm
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_clamped = v * torch.minimum(
        torch.ones_like(v_norm),
        3.0 / (v_norm + manifold.eps)
    )
    
    return manifold.expmap0(v_clamped)


def safe_logmap(p, manifold):
    """
    Safe logarithmic map at origin with clamping.
    
    Args:
        p: Points on manifold
        manifold: PoincareManifold instance
        
    Returns:
        Tangent vectors
    """
    # Ensure point is on manifold
    p = manifold.projx(p)
    
    return manifold.logmap0(p)


class HyperbolicLayer(nn.Module):
    """
    Base class for hyperbolic neural network layers.
    """
    
    def __init__(self, manifold: PoincareManifold):
        super().__init__()
        self.manifold = manifold
    
    def forward(self, x):
        raise NotImplementedError


class EuclideanToHyperbolic(nn.Module):
    """
    Projects Euclidean embeddings to hyperbolic space safely.
    """
    
    def __init__(self, manifold: PoincareManifold, scale: float = 0.01):
        super().__init__()
        self.manifold = manifold
        self.scale = nn.Parameter(torch.tensor(scale))
    
    def forward(self, x):
        """
        Args:
            x: Euclidean embeddings (..., dim)
            
        Returns:
            Hyperbolic embeddings on Poincaré ball
        """
        return safe_project(x, self.manifold, self.scale)


class HyperbolicToEuclidean(nn.Module):
    """
    Projects hyperbolic embeddings back to Euclidean space.
    """
    
    def __init__(self, manifold: PoincareManifold):
        super().__init__()
        self.manifold = manifold
    
    def forward(self, p):
        """
        Args:
            p: Hyperbolic points (..., dim)
            
        Returns:
            Euclidean vectors
        """
        return safe_logmap(p, self.manifold)


def kl_divergence_hyperbolic(mu_q, logvar_q, mu_p, logvar_p, manifold):
    """
    Hyperbolic KL divergence for VAE.
    Based on APO-VAE Equation 7.
    
    Args:
        mu_q: Mean of q (posterior) on hyperbolic space
        logvar_q: Log variance of q
        mu_p: Mean of p (prior) on hyperbolic space
        logvar_p: Log variance of p
        manifold: PoincareManifold instance
        
    Returns:
        KL divergence per sample
    """
    # Hyperbolic distance term
    d = manifold.dist(mu_q, mu_p)
    
    # Variance terms
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    # APO-VAE Eq. 7
    # KL = log(σ_p/σ_q) + (σ_q² + d²)/(2σ_p²) - 1/2
    kl = torch.log(var_p / (var_q + manifold.eps)) + \
         (var_q + d.unsqueeze(-1)**2) / (2 * var_p + manifold.eps) - 0.5
    
    return kl.sum(dim=-1)
