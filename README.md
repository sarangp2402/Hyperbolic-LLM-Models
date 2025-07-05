# Hyperbolic-LLM-Models
A comprehensive collection of research papers on Hyperbolic Large Language Models with paper titles, publication years, code links, and dataset links. This repository organizes 74 core research papers according to the taxonomy presented in our survey paper.

A comprehensive collection of research papers on Hyperbolic Large Language Models with paper titles, publication years, code links, and dataset links. This repository organizes **74 core research papers** according to the taxonomy presented in our survey paper.

## Table of Contents
- [HypLLMs](#hypllms)
- [Foundations and Theory](#foundations-and-theory)
- [Applications](#applications)  
- [Core Infrastructure](#core-infrastructure)

---
## HypLLMs

### Hybrid Hyperbolic-Euclidean Models
Models that integrate hyperbolic geometry with traditional Euclidean operations, typically using exponential/logarithmic mappings.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Hyperbolic pre-trained language model | 2024 | [Code](https://github.com/thunlp/hyperbolic_llm) | WordNet, SNLI |
| Language models as hierarchy encoders | 2024 | [Code](https://github.com/KRR-Oxford/HierarchyTransformers) | WordNet, FoodOn, DOID |
| Vision-language understanding in hyperbolic space | 2024 | [Code]() | [Dataset]() |
| Hyperbolic Learning with Multimodal Large Language Models | 2024 | [Code]() | [Dataset]() |
| Coneheads: Hierarchy aware attention | 2023 | [Code](https://github.com/tsengalb99/coneheads) | [Dataset]() |
| Large Language Models Enhanced Hyperbolic Space Recommender Systems | 2025 | [Code]() | [Dataset]() |
| Hyperbolic graph-llm alignment for exploration and exploitation in recommender systems | 2024 | [Code]() | Amazon-Books, Yelp |
| Enhancing Multimodal Survival Prediction with Pathology Reports in Hyperbolic Space | 2024 | [Code]() | TCGA |
| HySurvPred: Multimodal Hyperbolic Embedding with Angle-Aware Hierarchical Contrastive Learning and Uncertainty Constraints for Survival Prediction | 2025 | [Code]() | TCGA-BLCA, TCGA-BRCA, TCGA-UCEC |
| ANTHEM: Attentive hyperbolic entity model for product search | 2022 | [Code]() | [Dataset]() |

### Hyperbolic Fine-tuned Models
Parameter-efficient fine-tuning methods that adapt pre-trained LLMs to hyperbolic space.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Hyperbolic Fine-tuning for Large Language Models | 2024 | [Code](https://github.com/marlin-codes/HypLLM) | MAWPS, SVAMP, GSM8K |
| Enhancing llm complex reasoning capability through hyperbolic geometry | 2024 | [Code]() | [Dataset]() |

### Fully Hyperbolic Models
Models that operate entirely within hyperbolic space without requiring exponential/logarithmic mappings.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Hypformer: Exploring efficient transformer fully in hyperbolic space | 2024 | [Code](https://github.com/marlin-codes/hyperbolicTransformer) | Amazon2M, ogbn-proteins, ogbn-arxiv |
| HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts | 2025 | [Code]() | [Dataset]() |
| Hypercore: The core framework for building hyperbolic foundation models with comprehensive modules | 2025 | [Code](https://github.com/Graph-and-Geometric-Learning/HyperCore) | [Dataset]() |
| Fully hyperbolic neural networks | 2021 | [Code]() | [Dataset]() |
| HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition | 2025 | [Code]() | [Dataset]() |

### Hyperbolic State-Space Models
State-space models (like Mamba) extended to hyperbolic geometry for efficient sequence modeling.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings | 2025 | [Code]() | WordNet, FoodOn, DOID |
| SHMamba: Structured Hyperbolic State Space Model for Audio-Visual Question Answering | 2024 | [Code]() | MUSIC-AVQA, AVQA |
| HMamba: Hyperbolic Mamba for Sequential Recommendation | 2025 | [Code]() | MovieLens-1M |

---

## Foundations and Theory

### Hyperbolic Geometry Foundations
Core mathematical foundations and geometric properties of hyperbolic space.

| Paper Title | Year |
|-------------|------|
| Hyperbolic entailment cones for learning hierarchical embeddings | 2018 |
| Hyperbolic neural networks | 2018 | [Code](https://github.com/dalab/hyperbolic_nn) | [Dataset]() |
| Poincaré embeddings for learning hierarchical representations | 2017 | 
| Learning continuous hierarchies in the lorentz model of hyperbolic geometry | 2018 | 
| Hyperbolic groups | 1987 | 
| Network geometry | 2021 | 


### Hyperbolic Graph Neural Networks
Graph neural networks adapted for hyperbolic space representations.

| Paper Title | Year | 
|-------------|------|
| Hyperbolic graph neural networks | 2019 | 
| Hyperbolic graph convolutional neural networks | 2019 | 
| Low-Dimensional Hyperbolic Knowledge Graph Embeddings | 2020 | 
| Fully hyperbolic graph convolution network for recommendation | 2021 | 
| Knowledge graph representation via hierarchical hyperbolic neural graph embedding | 2021 | 
| Hyperkg: Hyperbolic knowledge graph embeddings for knowledge base completion | 2019 | 
| Complex hyperbolic knowledge graph embeddings with fast fourier transform | 2022 |

### Riemannian Optimization Techniques
Optimization methods for training models on Riemannian manifolds.

| Paper Title | Year |
|-------------|------|
| Stochastic gradient descent on Riemannian manifolds | 2013 |
| Advances in Neural Information Processing Systems | 2016 | 
| Riemannian Accelerated Gradient Methods via Extrapolation | 2023 |
| Riemannian Stein variational gradient descent for Bayesian inference | 2018 |

### Numerical Stability and Challenges
Research addressing computational challenges in hyperbolic representation learning.

| Paper Title | Year |
|-------------|------|
| The Numerical Stability of Hyperbolic Representation Learning | 2023 |
| Representation tradeoffs for hyperbolic embeddings | 2018 |


### Specialized Techniques and Methods
Specialized approaches and technical innovations for hyperbolic learning.

| Paper Title | Year | 
|-------------|------|
| The hyperbolic theory of special relativity | 2011 |
| Representing hyperbolic space accurately using multi-component floats | 2021 | 
| Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks | 2024 |
| Adaptive data embedding for curved spaces | 2024 |
| Poincaré GloVe | 2019 | 
| Probing BERT in hyperbolic spaces | 2021 |
| Hyperbolic geometry of complex networks | 2010 | 
| Hyperbolic graph neural networks: A review of methods and applications | 2022 |
| Hyperbolic deep neural networks: A survey | 2021 | 
| Hyperbolic attention networks | 2018 |
| Neural embeddings of graphs in hyperbolic space | 2017 |
| Hierarchical image classification using entailment cone embeddings | 2020 | 
| Llms are good action recognizers | 2024 | 

### Core LLM and Transformer Foundations
Foundational papers on large language models and transformer architectures.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Attention is all you need | 2017 | 
| GPT | 2023 | 
| Gemini: a family of highly capable multimodal models | 2023 | 
| The Llama | 2024 | 
| BERT | 2019 |

### Background

| Paper Title | Year | 
|-------------|------|
| Zipf’s word frequency law in natural language: A critical review and future directions | 2014 |
| Hyperbolic sentence representations for solving Textual Entailment | 2024 | 
| Large language models could make natural language again the universal interface of healthcare | 2024 | 
| Lora: Low-rank adaptation of large language models. | 2022 |
| Mamba: Linear-time sequence modeling with selective state spaces | 2024 |
| A Comparative Study on Dynamic Graph Embedding based on Mamba and Transformers | 2024 |

---
## Applications

### Brain Networks and Neuroscience Applications
Hyperbolic models applied to brain network analysis and neuroscience research.

| Paper Title | Year |
|-------------|------|
| Hyperbolic graph embedding of MEG brain networks to study brain alterations in individuals with subjective cognitive decline | 2024 |
| Fully Hyperbolic Neural Networks: A Novel Approach to Studying Aging Trajectories | 2024 | 
| Navigating Memorability Landscapes: Hyperbolic Geometry Reveals Hierarchical Structures in Object Concept Memory | 2024 |
| Hyperbolic geometry of the olfactory space | 2018 |
| Neural manifolds for the control of movement | 2017 | 
| Nonlinear manifolds underlie neural population activity during behaviour | 2024 | 

### Biomedical and Healthcare Applications
Applications of hyperbolic geometry in biomedical and healthcare domains.

| Paper Title | Year |
|-------------|------|
| Hig2vec: hierarchical representations of gene ontology and genes in the poincar\'e | 2021 |
| Hyperbolic hierarchical knowledge graph embeddings for biological entities | 2023 | 
| Accurate structure prediction of biomolecular interactions with AlphaFold 3 | 2024 |

---

