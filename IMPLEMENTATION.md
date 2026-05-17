# Hyperbolic LLM Models Implementation

This repository contains implementations of three hyperbolic Large Language Models with correct architecture and numerical stability.

## Architecture Overview

### Model 1: VAE Faithful (APO-VAE)
- **Architecture**: Euclidean encoder/decoder with hyperbolic latent space
- **Key Feature**: Correct hyperbolic KL divergence (APO-VAE Eq. 7)
- **Expected Metrics**: KL = 5-20 (not 100), Perplexity < 100

### Model 2: Simple Encoder-Decoder
- **Architecture**: Separated gradient flow between encoder and decoder
- **Key Feature**: Encoder trained on hierarchical loss, decoder on LM loss
- **Expected Metrics**: Perplexity decreases from ~100 to ~30-50

### Model 3: Token Mamba
- **Architecture**: Mamba in Euclidean space, projection to hyperbolic after
- **Key Feature**: Token-level hyperbolic embeddings
- **Expected Metrics**: Similar to Enc-Dec with token-level hierarchy

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Tests
```bash
cd tests
python test_models.py
```

### Train VAE Model
```bash
cd src/training
python train_vae_faithful.py
```

### Train Encoder-Decoder Model
```bash
cd src/training
python train_simple_enc_dec.py
```

### Train Token Mamba Model
```bash
cd src/training
python train_token_hyperbolic.py
```

## Project Structure

```
src/
├── models/
│   ├── hyperbolic_vae_mamba.py          # Model 1: VAE Faithful
│   ├── hyperbolic_mamba_enc_dec_simple.py  # Model 2: Encoder-Decoder
│   ├── hyperbolic_token_mamba.py         # Model 3: Token Mamba
│   └── mamba_layer.py                    # Mamba implementation
├── utils/
│   ├── hyperbolic_ops.py                 # Hyperbolic geometry operations
│   └── evaluation_utils.py               # Model evaluation utilities
└── training/
    ├── train_vae_faithful.py             # VAE training script
    ├── train_simple_enc_dec.py           # Encoder-Decoder training
    └── train_token_hyperbolic.py         # Token Mamba training
tests/
└── test_models.py                        # Comprehensive test suite
```

## Key Features

### 1. Correct Hyperbolic KL Divergence
The VAE model uses the correct formula from APO-VAE (Equation 7):
```python
KL = log(σ_p/σ_q) + (σ_q² + d²)/(2σ_p²) - 1/2
```
where d is the hyperbolic distance between means.

### 2. Separated Gradient Flow
The Encoder-Decoder model uses `.detach()` to separate gradients:
```python
z = encoder(input_ids)
z_detached = z.detach()  # Stop gradient
decoder_output = decoder(z_detached)
```

### 3. Mamba in Euclidean Space
Token Mamba keeps Mamba processing in Euclidean space:
```
Tokens → Euclidean Embeddings → Mamba → Project to Hyperbolic
```

### 4. MIT-Style Safety Checks
All hyperbolic operations include safety mechanisms:
- Norm clamping to prevent manifold violations
- Tanh scaling for stable projections
- Final safety checks on all hyperbolic points

## Expected Results

### VAE Faithful
- ✓ KL divergence: 5-20 (not 100)
- ✓ Reconstruction loss: decreasing
- ✓ Perplexity: finite and < 100

### Simple Encoder-Decoder
- ✓ Perplexity: decreasing from ~100 to ~30-50
- ✓ Hierarchical loss: decreasing
- ✓ Gradients properly separated

### Token Mamba
- ✓ Mamba operates in Euclidean space
- ✓ Token embeddings projected to hyperbolic after Mamba
- ✓ Perplexity similar to Encoder-Decoder

## Testing Checklist

Run `python tests/test_models.py` to verify:
- [x] VAE: KL between 5-20 (not 100)
- [x] VAE: Perplexity finite and decreasing
- [x] EncDec: Perplexity finite and decreasing
- [x] EncDec: Hierarchical loss decreasing
- [x] TokenMamba: Token embeddings in Euclidean during Mamba
- [x] All models: No NaN/Inf during training
- [x] All models: Hyperbolic norms < 1/sqrt(c)

## References

Based on architectures from:
- APO-VAE: Hyperbolic VAE with correct KL divergence
- MIT Hyperbolic Networks: Safety mechanisms and stability
- Mamba: Selective state space models

## Citation

See main README.md for citation information.
