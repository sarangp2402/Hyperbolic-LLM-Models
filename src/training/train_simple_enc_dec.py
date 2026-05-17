"""
Training script for Simple Encoder-Decoder model.
Implements separate gradient flow between encoder and decoder.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hyperbolic_mamba_enc_dec_simple import HyperbolicEncoderDecoder
from utils.evaluation_utils import HyperbolicModelEvaluator


class HierarchicalDataset(Dataset):
    """Dataset for hierarchical training."""
    
    def __init__(self, triplets, tokenizer, max_length=128):
        """
        Args:
            triplets: List of (child, parent, negative) text tuples
            tokenizer: Tokenizer
            max_length: Max sequence length
        """
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        child, parent, negative = self.triplets[idx]
        
        child_enc = self.tokenizer(
            child, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        parent_enc = self.tokenizer(
            parent, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        neg_enc = self.tokenizer(
            negative, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'child_input_ids': child_enc['input_ids'].squeeze(0),
            'child_mask': child_enc['attention_mask'].squeeze(0),
            'parent_input_ids': parent_enc['input_ids'].squeeze(0),
            'parent_mask': parent_enc['attention_mask'].squeeze(0),
            'negative_input_ids': neg_enc['input_ids'].squeeze(0),
            'negative_mask': neg_enc['attention_mask'].squeeze(0),
        }


def train_encoder_decoder(
    train_triplets,
    val_texts=None,
    vocab_size=32000,
    d_model=256,
    n_encoder_layers=4,
    n_decoder_layers=4,
    max_seq_len=512,
    curvature=1.0,
    hierarchical_weight=1.0,
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    device='cuda',
    save_dir='checkpoints/enc_dec'
):
    """
    Train encoder-decoder model with separated gradients.
    
    Args:
        train_triplets: List of (child, parent, negative) tuples
        val_texts: Validation texts
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        max_seq_len: Maximum sequence length
        curvature: Hyperbolic curvature
        hierarchical_weight: Weight for hierarchical loss
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device
        save_dir: Checkpoint directory
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    train_dataset = HierarchicalDataset(train_triplets, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = HyperbolicEncoderDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        max_seq_len=max_seq_len,
        curvature=curvature
    )
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_hier_loss = 0.0
        total_lm_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            child_ids = batch['child_input_ids'].to(device)
            child_mask = batch['child_mask'].to(device)
            parent_ids = batch['parent_input_ids'].to(device)
            parent_mask = batch['parent_mask'].to(device)
            neg_ids = batch['negative_input_ids'].to(device)
            neg_mask = batch['negative_mask'].to(device)
            
            # CRITICAL: Separate gradient flow
            # Encoder sees hierarchical loss
            z_child = model.encoder(child_ids, child_mask)
            z_parent = model.encoder(parent_ids, parent_mask)
            hierarchical_loss = model.compute_hierarchical_loss(
                z_child, z_parent, neg_ids, neg_mask
            )
            
            # Decoder sees LM loss (with detached z!)
            z_detached = z_child.detach()  # Stop gradient!
            decoder_outputs = model.decoder(child_ids, z_detached, labels=child_ids)
            lm_loss = decoder_outputs['loss']
            
            # Total loss
            loss = lm_loss + hierarchical_weight * hierarchical_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Stats
            total_hier_loss += hierarchical_loss.item()
            total_lm_loss += lm_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        # Print epoch stats
        avg_hier = total_hier_loss / num_batches
        avg_lm = total_lm_loss / num_batches
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  LM Loss: {avg_lm:.4f}")
        print(f"  Hierarchical Loss: {avg_hier:.4f}")
        
        # Validation
        if val_texts is not None and (epoch + 1) % 5 == 0:
            evaluator = HyperbolicModelEvaluator(model, tokenizer, device)
            metrics = evaluator.calculate_perplexity(data=val_texts, max_samples=10)
            print(f"  Val Perplexity: {metrics['perplexity']:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Perplexity should be decreasing (from ~100 to ~30-50)")
    print(f"Hierarchical loss should be decreasing")
    
    return model


if __name__ == "__main__":
    # Example hierarchical triplets
    train_triplets = [
        ("A poodle is a type of dog", "A dog is an animal", "A cat is an animal"),
        ("An apple is a fruit", "Fruit comes from plants", "Vegetables are healthy"),
        ("Python is a programming language", "Programming languages are used in software", "Hardware is physical"),
        ("A laptop is a computer", "Computers process information", "Books contain information"),
    ] * 10
    
    val_texts = [
        "A car is a vehicle",
        "Birds can fly",
    ]
    
    model = train_encoder_decoder(
        train_triplets=train_triplets,
        val_texts=val_texts,
        d_model=128,
        n_encoder_layers=2,
        n_decoder_layers=2,
        batch_size=4,
        num_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
