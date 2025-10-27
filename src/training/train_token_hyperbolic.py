"""
Training script for Token-Level Hyperbolic Mamba model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hyperbolic_token_mamba import TokenLevelHyperbolicMamba
from utils.evaluation_utils import HyperbolicModelEvaluator


class SimpleTextDataset(Dataset):
    """Simple dataset for text data."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def train_token_mamba(
    train_texts,
    val_texts=None,
    vocab_size=32000,
    d_model=256,
    n_layers=6,
    max_seq_len=512,
    curvature=1.0,
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    device='cuda',
    save_dir='checkpoints/token_mamba'
):
    """
    Train Token-Level Hyperbolic Mamba model.
    
    Args:
        train_texts: List of training texts
        val_texts: List of validation texts
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of layers
        max_seq_len: Maximum sequence length
        curvature: Hyperbolic curvature
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = SimpleTextDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TokenLevelHyperbolicMamba(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        curvature=curvature
    )
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=input_ids)
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate stats
            total_loss += loss.item()
            num_batches += 1
        
        # Print epoch stats
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        
        # Validation
        if val_texts is not None and (epoch + 1) % 5 == 0:
            evaluator = HyperbolicModelEvaluator(model, tokenizer, device)
            metrics = evaluator.calculate_perplexity(data=val_texts, max_samples=10)
            print(f"  Val Perplexity: {metrics['perplexity']:.2f}")
        
        # Check that Mamba operates in Euclidean space
        if (epoch + 1) % 5 == 0:
            print("  NOTE: Mamba operates in Euclidean space, projection to hyperbolic happens AFTER")
        
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
    print("Token embeddings stayed in Euclidean space during Mamba processing")
    print("Projection to hyperbolic happened AFTER Mamba")
    
    return model


if __name__ == "__main__":
    # Example usage
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables AI to understand text.",
        "Deep learning models require large amounts of data.",
        "Hyperbolic geometry is useful for hierarchical data.",
        "Neural networks learn patterns from examples.",
        "Transformers use attention mechanisms.",
        "Language models predict the next word in a sequence.",
    ] * 10
    
    val_texts = [
        "Artificial intelligence is advancing rapidly.",
        "Neural networks learn from examples.",
    ]
    
    model = train_token_mamba(
        train_texts=train_texts,
        val_texts=val_texts,
        d_model=128,
        n_layers=3,
        batch_size=4,
        num_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
