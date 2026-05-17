"""
Model 2: Simple Encoder-Decoder with Separated Gradient Flow

Architecture:
    Sentence A → Encoder (Euclidean) → Pool → Project to Hyperbolic → z_A
    Sentence B → Encoder (Euclidean) → Pool → Project to Hyperbolic → z_B
    
    Hierarchical Loss = clustering(z_A, z_B) + centripetal(z_A, z_B)
    
    For generation:
    z_A → Project to Euclidean → Decoder (Euclidean) → Tokens

Key points:
- Stop gradients from decoder to encoder (use .detach())
- Encoder should only see hierarchical loss
- Decoder should only see LM loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mamba_layer import MambaBlock
from utils.hyperbolic_ops import (
    PoincareManifold,
    safe_project,
    safe_logmap
)


class HyperbolicEncoderDecoder(nn.Module):
    """
    Encoder-Decoder model with hyperbolic sentence embeddings.
    Encoder and decoder gradients are separated.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        max_seq_len: int = 512,
        curvature: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Initialize Poincaré manifold
        self.manifold = PoincareManifold(c=curvature)
        
        # Shared token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Encoder (Euclidean)
        self.encoder_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Project to hyperbolic
        self.to_hyperbolic = nn.Linear(d_model, d_model)
        
        # Decoder (Euclidean)
        self.from_hyperbolic = nn.Linear(d_model, d_model)
        self.decoder_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embeddings.weight
    
    def encoder(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input to hyperbolic sentence embedding.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Hyperbolic sentence embeddings (batch, d_model)
        """
        batch, seq_len = input_ids.shape
        
        # Euclidean embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        # Encode with Mamba (Euclidean)
        for layer in self.encoder_layers:
            hidden, _ = layer(hidden)
        
        hidden = self.encoder_norm(hidden)
        
        # Pool: mean over sequence
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            pooled = hidden.mean(dim=1)
        
        # Project to hyperbolic space
        pooled_proj = self.to_hyperbolic(pooled)
        z_hyperbolic = safe_project(pooled_proj, self.manifold, scale=0.01)
        
        return z_hyperbolic
    
    def decoder(
        self,
        input_ids: torch.Tensor,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from hyperbolic embedding to tokens.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            z: Hyperbolic sentence embeddings (batch, d_model)
            labels: Target tokens (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Dictionary with logits and optionally loss
        """
        batch, seq_len = input_ids.shape
        
        # Project from hyperbolic to Euclidean
        z_euclidean = safe_logmap(z, self.manifold)
        z_proj = self.from_hyperbolic(z_euclidean)
        
        # Get token embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        # Add sentence context
        hidden = hidden + z_proj.unsqueeze(1)
        
        # Decode with Mamba (Euclidean)
        for layer in self.decoder_layers:
            hidden, _ = layer(hidden)
        
        hidden = self.decoder_norm(hidden)
        
        # LM head
        logits = self.lm_head(hidden)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Reconstruction loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                reduction='mean'
            )
            outputs['loss'] = loss
            outputs['recon_loss'] = loss
        
        return outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        child_input_ids: Optional[torch.Tensor] = None,
        child_mask: Optional[torch.Tensor] = None,
        parent_input_ids: Optional[torch.Tensor] = None,
        parent_mask: Optional[torch.Tensor] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional hierarchical training.
        
        Args:
            input_ids: Token IDs for decoding (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target tokens (batch, seq_len)
            child_input_ids: Child concept tokens for hierarchy (batch, seq_len)
            child_mask: Child attention mask
            parent_input_ids: Parent concept tokens for hierarchy (batch, seq_len)
            parent_mask: Parent attention mask
            negative_input_ids: Negative samples for contrastive learning
            negative_mask: Negative attention mask
            
        Returns:
            Dictionary with losses and embeddings
        """
        outputs = {}
        
        # If hierarchical inputs provided, compute hierarchical loss
        if child_input_ids is not None and parent_input_ids is not None:
            # Encode child and parent (gradients flow to encoder)
            z_child = self.encoder(child_input_ids, child_mask)
            z_parent = self.encoder(parent_input_ids, parent_mask)
            
            # Compute hierarchical loss
            hierarchical_loss = self.compute_hierarchical_loss(
                z_child, z_parent, negative_input_ids, negative_mask
            )
            
            outputs['hierarchical_loss'] = hierarchical_loss
            outputs['z_child'] = z_child
            outputs['z_parent'] = z_parent
        
        # If labels provided, compute LM loss
        if labels is not None:
            # Encode input
            z = self.encoder(input_ids, attention_mask)
            
            # IMPORTANT: Detach z to stop gradients from decoder to encoder
            z_detached = z.detach()
            
            # Decode (gradients only flow to decoder)
            decoder_outputs = self.decoder(input_ids, z_detached, labels, attention_mask)
            
            outputs.update(decoder_outputs)
            outputs['z'] = z  # Keep undetached for logging
        
        return outputs
    
    def compute_hierarchical_loss(
        self,
        z_child: torch.Tensor,
        z_parent: torch.Tensor,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
        margin: float = 1.0
    ) -> torch.Tensor:
        """
        Compute hierarchical loss (clustering + centripetal).
        
        Args:
            z_child: Child hyperbolic embeddings (batch, d_model)
            z_parent: Parent hyperbolic embeddings (batch, d_model)
            negative_input_ids: Negative samples
            negative_mask: Negative attention mask
            margin: Margin for contrastive loss
            
        Returns:
            Hierarchical loss
        """
        # Distance between child and parent (should be small)
        pos_dist = self.manifold.dist(z_child, z_parent)
        
        # Clustering loss: minimize distance to parent
        clustering_loss = pos_dist.mean()
        
        # Centripetal loss: push toward origin
        origin = torch.zeros_like(z_child)
        dist_to_origin = self.manifold.dist(z_child, origin)
        centripetal_loss = torch.relu(dist_to_origin - 2.0).mean()
        
        # Contrastive loss with negatives if provided
        contrastive_loss = 0.0
        if negative_input_ids is not None:
            z_neg = self.encoder(negative_input_ids, negative_mask)
            neg_dist = self.manifold.dist(z_child.unsqueeze(1), z_neg.unsqueeze(0))
            contrastive_loss = torch.relu(margin - neg_dist).mean()
        
        total_loss = clustering_loss + 0.1 * centripetal_loss + 0.5 * contrastive_loss
        
        return total_loss
