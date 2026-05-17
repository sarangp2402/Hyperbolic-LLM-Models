"""
Evaluation utilities for hyperbolic models.
Includes model-specific perplexity calculation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import math
from pathlib import Path


class HyperbolicModelEvaluator:
    """
    Evaluator for hyperbolic LLM models.
    Handles different model architectures appropriately.
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def calculate_perplexity(
        self,
        triplet_file: Optional[str] = None,
        data: Optional[List[str]] = None,
        max_samples: int = 20,
        max_length: int = 128
    ) -> Dict[str, float]:
        """
        Calculate perplexity for the model.
        Automatically detects model type and uses appropriate method.
        
        Args:
            triplet_file: Path to triplet file (if using hierarchical data)
            data: List of text strings to evaluate
            max_samples: Maximum number of samples to use
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity and loss metrics
        """
        if data is None:
            # Use some default test data if not provided
            data = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand text.",
            ]
        
        total_loss = 0.0
        total_tokens = 0
        
        # Detect model type
        model_type = self._detect_model_type()
        
        with torch.no_grad():
            for i, text in enumerate(data[:max_samples]):
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Calculate loss based on model type
                if model_type == 'vae':
                    loss = self._calculate_vae_loss(input_ids, attention_mask)
                elif model_type == 'encoder_decoder':
                    loss = self._calculate_enc_dec_loss(input_ids, attention_mask)
                elif model_type == 'decoder_only':
                    loss = self._calculate_decoder_only_loss(input_ids, attention_mask)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Accumulate
                num_tokens = attention_mask.sum().item()
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': total_tokens
        }
    
    def _detect_model_type(self) -> str:
        """
        Detect the type of model.
        
        Returns:
            Model type: 'vae', 'encoder_decoder', or 'decoder_only'
        """
        # Check for VAE (has encode/decode methods)
        if hasattr(self.model, 'encode') and hasattr(self.model, 'reparameterize'):
            return 'vae'
        
        # Check for encoder-decoder (has separate encoder and decoder)
        if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            return 'encoder_decoder'
        
        # Default to decoder-only
        return 'decoder_only'
    
    def _calculate_vae_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Calculate loss for VAE model.
        Uses only reconstruction loss for perplexity (not ELBO).
        """
        outputs = self.model(input_ids, attention_mask, labels=input_ids)
        
        # Use reconstruction loss, not total ELBO
        loss = outputs['recon_loss'].item()
        
        return loss
    
    def _calculate_enc_dec_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Calculate loss for encoder-decoder model.
        Uses decoder reconstruction loss.
        """
        # Encode
        z = self.model.encoder(input_ids, attention_mask)
        
        # Decode
        decoder_outputs = self.model.decoder(
            input_ids,
            z,
            labels=input_ids,
            attention_mask=attention_mask
        )
        
        loss = decoder_outputs['recon_loss'].item()
        
        return loss
    
    def _calculate_decoder_only_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Calculate loss for decoder-only model.
        """
        outputs = self.model(input_ids, attention_mask, labels=input_ids)
        loss = outputs['loss'].item()
        
        return loss
    
    def evaluate_hierarchical(
        self,
        child_texts: List[str],
        parent_texts: List[str],
        max_samples: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate hierarchical structure quality.
        
        Args:
            child_texts: List of child concept texts
            parent_texts: List of parent concept texts
            max_samples: Maximum samples to evaluate
            
        Returns:
            Dictionary with hierarchical metrics
        """
        assert len(child_texts) == len(parent_texts)
        
        distances = []
        
        with torch.no_grad():
            for i in range(min(len(child_texts), max_samples)):
                # Tokenize
                child_enc = self.tokenizer(
                    child_texts[i],
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                parent_enc = self.tokenizer(
                    parent_texts[i],
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                child_ids = child_enc['input_ids'].to(self.device)
                parent_ids = parent_enc['input_ids'].to(self.device)
                child_mask = child_enc['attention_mask'].to(self.device)
                parent_mask = parent_enc['attention_mask'].to(self.device)
                
                # Get embeddings
                if hasattr(self.model, 'encoder'):
                    z_child = self.model.encoder(child_ids, child_mask)
                    z_parent = self.model.encoder(parent_ids, parent_mask)
                elif hasattr(self.model, 'encode'):
                    z_child = self.model.encode(child_ids, child_mask)['mu']
                    z_parent = self.model.encode(parent_ids, parent_mask)['mu']
                else:
                    # For token-level model
                    child_out = self.model(child_ids, child_mask, return_hyperbolic=True)
                    parent_out = self.model(parent_ids, parent_mask, return_hyperbolic=True)
                    z_child = child_out['sentence_embedding']
                    z_parent = parent_out['sentence_embedding']
                
                # Calculate distance
                dist = self.model.manifold.dist(z_child, z_parent)
                distances.append(dist.mean().item())
        
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        return {
            'avg_child_parent_distance': avg_distance,
            'num_pairs': len(distances)
        }


def evaluate_model(
    model: nn.Module,
    tokenizer,
    test_texts: List[str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Hyperbolic model
        tokenizer: Tokenizer
        test_texts: Test texts
        device: Device to use
        
    Returns:
        Evaluation metrics
    """
    evaluator = HyperbolicModelEvaluator(model, tokenizer, device)
    metrics = evaluator.calculate_perplexity(data=test_texts)
    
    return metrics
