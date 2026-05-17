"""
Model 1: VAE Faithful (APO-VAE Architecture)

Architecture:
    Input Tokens → Encoder (Euclidean) → Pool → Project to Hyperbolic μ, σ
                                                  ↓
                                               Sample z (hyperbolic)
                                                  ↓
                                        Project back to Euclidean
                                                  ↓
                                        Decoder (Euclidean) → Tokens
    
    Loss = Reconstruction + β * KL_hyperbolic(z || prior)

Key points:
- Encoder/Decoder operate in Euclidean space
- Only latent z lives in hyperbolic space
- Uses correct hyperbolic KL formula (Eq. 7 from APO-VAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mamba_layer import MambaBlock
from utils.hyperbolic_ops import (
    PoincareManifold, 
    EuclideanToHyperbolic, 
    HyperbolicToEuclidean,
    kl_divergence_hyperbolic,
    safe_project
)


class HyperbolicVAEMamba(nn.Module):
    """
    VAE with Mamba encoder/decoder and hyperbolic latent space.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        max_seq_len: int = 512,
        curvature: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.beta = beta
        
        # Initialize Poincaré manifold
        self.manifold = PoincareManifold(c=curvature)
        
        # Token embeddings (Euclidean)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Encoder (Euclidean Mamba layers)
        self.encoder_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Projection to hyperbolic latent space
        self.to_mu = nn.Linear(d_model, d_model)
        self.to_logvar = nn.Linear(d_model, d_model)
        
        # Projection from hyperbolic to Euclidean for decoder
        self.from_latent = nn.Linear(d_model, d_model)
        
        # Decoder (Euclidean Mamba layers)
        self.decoder_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embeddings.weight
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Encode input tokens to hyperbolic latent representation.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Dictionary with mu (hyperbolic), logvar, and pooled representation
        """
        batch, seq_len = input_ids.shape
        
        # Euclidean embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        # Encode with Mamba (Euclidean)
        for layer in self.encoder_layers:
            hidden, _ = layer(hidden)
        
        hidden = self.encoder_norm(hidden)
        
        # Pool: mean over sequence (with masking if provided)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden.mean(dim=1)
        
        # Project to get mu and logvar (still Euclidean)
        mu_euclidean = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        
        # Project mu to hyperbolic space
        mu_hyperbolic = safe_project(mu_euclidean, self.manifold, scale=0.01)
        
        return {
            'mu': mu_hyperbolic,
            'logvar': logvar,
            'pooled': pooled
        }
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick in hyperbolic space.
        
        Args:
            mu: Mean on hyperbolic space (batch, d_model)
            logvar: Log variance (batch, d_model)
            
        Returns:
            Sampled point on hyperbolic space
        """
        std = torch.exp(0.5 * logvar)
        
        # Sample in Euclidean tangent space at mu
        eps = torch.randn_like(std)
        
        # Map mu to Euclidean
        mu_euclidean = self.manifold.logmap0(mu)
        
        # Add noise in Euclidean space
        z_euclidean = mu_euclidean + eps * std
        
        # Map back to hyperbolic
        z = safe_project(z_euclidean, self.manifold, scale=0.01)
        
        return z
    
    def decode(self, z: torch.Tensor, target_len: int):
        """
        Decode from hyperbolic latent to tokens.
        
        Args:
            z: Latent representation on hyperbolic space (batch, d_model)
            target_len: Target sequence length
            
        Returns:
            Logits (batch, target_len, vocab_size)
        """
        batch = z.shape[0]
        
        # Project from hyperbolic to Euclidean
        z_euclidean = self.manifold.logmap0(z)
        hidden = self.from_latent(z_euclidean)
        
        # Expand to sequence
        hidden = hidden.unsqueeze(1).expand(batch, target_len, self.d_model)
        
        # Add positional embeddings
        positions = torch.arange(target_len, device=z.device).unsqueeze(0)
        hidden = hidden + self.position_embeddings(positions)
        
        # Decode with Mamba (Euclidean)
        for layer in self.decoder_layers:
            hidden, _ = layer(hidden)
        
        hidden = self.decoder_norm(hidden)
        
        # LM head
        logits = self.lm_head(hidden)
        
        return logits
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target tokens (batch, seq_len)
            
        Returns:
            Dictionary with loss, recon_loss, kl_loss, and logits
        """
        # Encode
        encoder_output = self.encode(input_ids, attention_mask)
        mu = encoder_output['mu']
        logvar = encoder_output['logvar']
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        seq_len = input_ids.shape[1]
        logits = self.decode(z, seq_len)
        
        # Compute losses
        outputs = {'logits': logits}
        
        if labels is not None:
            # Reconstruction loss
            recon_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                reduction='mean'
            )
            
            # KL divergence (hyperbolic)
            # Prior: mu_p = 0 (origin), logvar_p = 0 (unit variance)
            mu_p = torch.zeros_like(mu)
            logvar_p = torch.zeros_like(logvar)
            
            kl_loss = kl_divergence_hyperbolic(mu, logvar, mu_p, logvar_p, self.manifold)
            kl_loss = kl_loss.mean()
            
            # Clamp KL to prevent collapse
            kl_loss = torch.clamp(kl_loss, min=0.0, max=50.0)
            
            # Total loss
            total_loss = recon_loss + self.beta * kl_loss
            
            outputs.update({
                'loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'elbo': -(recon_loss + kl_loss)
            })
        
        return outputs
