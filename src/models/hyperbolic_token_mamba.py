"""
Model 3: Token-Level Hyperbolic Mamba

Architecture:
    Tokens → Euclidean Embeddings → Mamba (Euclidean) → Hidden States
                                                              ↓
                                                  Project to Hyperbolic
                                                              ↓
                                                  Token Hyperbolic Embeddings
                                                              ↓
                                        Pool (in hyperbolic) → Sentence Embedding

Key points:
- Mamba always operates in Euclidean space
- Project to hyperbolic AFTER Mamba processing
- Token-level hyperbolic embeddings used for hierarchy losses
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
    safe_project,
    safe_logmap
)


class TokenLevelHyperbolicMamba(nn.Module):
    """
    Mamba model with token-level hyperbolic embeddings.
    Mamba operates in Euclidean space, projection happens after.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        max_seq_len: int = 512,
        curvature: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Initialize Poincaré manifold
        self.manifold = PoincareManifold(c=curvature)
        
        # Token embeddings (Euclidean)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Mamba layers (Euclidean)
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Project to hyperbolic (after Mamba processing)
        self.euclidean_to_poincare = nn.Linear(d_model, d_model)
        
        # LM head operates on hyperbolic via log map
        self.poincare_to_euclidean = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embeddings.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hyperbolic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target tokens (batch, seq_len)
            return_hyperbolic: Whether to return hyperbolic embeddings
            
        Returns:
            Dictionary with loss, logits, and optionally hyperbolic embeddings
        """
        batch, seq_len = input_ids.shape
        
        # 1. Euclidean embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        # 2. Mamba processing (Euclidean)
        for layer in self.layers:
            hidden, _ = layer(hidden)
        
        hidden = self.norm(hidden)
        
        # 3. Project to hyperbolic AFTER Mamba
        hidden_flat = hidden.view(-1, self.d_model)
        hidden_proj = self.euclidean_to_poincare(hidden_flat)
        hidden_hyp = safe_project(hidden_proj, self.manifold, scale=0.01)
        hidden_hyp = hidden_hyp.view(batch, seq_len, self.d_model)
        
        # 4. LM head operates on hyperbolic (via log map)
        hidden_hyp_flat = hidden_hyp.view(-1, self.d_model)
        hidden_euclidean = safe_logmap(hidden_hyp_flat, self.manifold)
        hidden_euclidean = self.poincare_to_euclidean(hidden_euclidean)
        hidden_euclidean = hidden_euclidean.view(batch, seq_len, self.d_model)
        
        logits = self.lm_head(hidden_euclidean)
        
        outputs = {'logits': logits}
        
        if return_hyperbolic:
            outputs['hidden_hyperbolic'] = hidden_hyp
            
            # Pool hyperbolic embeddings for sentence-level representation
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                # Pool in hyperbolic space (using Fréchet mean approximation)
                pooled_hyp = self.hyperbolic_mean(hidden_hyp, mask_expanded)
            else:
                pooled_hyp = self.hyperbolic_mean(hidden_hyp)
            
            outputs['sentence_embedding'] = pooled_hyp
        
        if labels is not None:
            # Language modeling loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                reduction='mean'
            )
            outputs['loss'] = loss
        
        return outputs
    
    def hyperbolic_mean(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Fréchet mean (approximation) in hyperbolic space.
        
        Args:
            embeddings: Hyperbolic embeddings (batch, seq_len, d_model)
            mask: Optional mask (batch, seq_len, 1)
            
        Returns:
            Mean embeddings (batch, d_model)
        """
        # Simple approximation: map to Euclidean, average, map back
        batch, seq_len, d_model = embeddings.shape
        
        # Map to Euclidean
        embeddings_flat = embeddings.view(-1, d_model)
        embeddings_euclidean = safe_logmap(embeddings_flat, self.manifold)
        embeddings_euclidean = embeddings_euclidean.view(batch, seq_len, d_model)
        
        # Average in Euclidean space
        if mask is not None:
            pooled = (embeddings_euclidean * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            pooled = embeddings_euclidean.mean(dim=1)
        
        # Map back to hyperbolic
        pooled_hyp = safe_project(pooled, self.manifold, scale=0.01)
        
        return pooled_hyp
    
    def compute_hierarchical_loss(
        self,
        child_input_ids: torch.Tensor,
        parent_input_ids: torch.Tensor,
        child_mask: Optional[torch.Tensor] = None,
        parent_mask: Optional[torch.Tensor] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hierarchical loss on sentence embeddings.
        
        Args:
            child_input_ids: Child tokens (batch, seq_len)
            parent_input_ids: Parent tokens (batch, seq_len)
            child_mask: Child mask
            parent_mask: Parent mask
            negative_input_ids: Negative samples
            negative_mask: Negative mask
            
        Returns:
            Hierarchical loss
        """
        # Get hyperbolic sentence embeddings
        child_out = self.forward(child_input_ids, child_mask, return_hyperbolic=True)
        parent_out = self.forward(parent_input_ids, parent_mask, return_hyperbolic=True)
        
        z_child = child_out['sentence_embedding']
        z_parent = parent_out['sentence_embedding']
        
        # Distance between child and parent
        pos_dist = self.manifold.dist(z_child, z_parent)
        
        # Clustering loss
        clustering_loss = pos_dist.mean()
        
        # Centripetal loss
        origin = torch.zeros_like(z_child)
        dist_to_origin = self.manifold.dist(z_child, origin)
        centripetal_loss = torch.relu(dist_to_origin - 2.0).mean()
        
        # Contrastive loss
        contrastive_loss = 0.0
        if negative_input_ids is not None:
            neg_out = self.forward(negative_input_ids, negative_mask, return_hyperbolic=True)
            z_neg = neg_out['sentence_embedding']
            
            # Distance to negatives
            neg_dist = self.manifold.dist(z_child.unsqueeze(1), z_neg.unsqueeze(0))
            contrastive_loss = torch.relu(1.0 - neg_dist).mean()
        
        total_loss = clustering_loss + 0.1 * centripetal_loss + 0.5 * contrastive_loss
        
        return total_loss
