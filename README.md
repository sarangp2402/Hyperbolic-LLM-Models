# Hyperbolic-LLM-Models
A comprehensive collection of research papers on Hyperbolic Large Language Models with paper titles, publication years, code links, and dataset links.

## Table of Contents
- [HypLLMs](#hypllms)
- [Foundations, Theory and Applications](#foundations-and-theory)

---
## HypLLMs

### Hybrid Hyperbolic-Euclidean Models
Models that integrate hyperbolic geometry with traditional Euclidean operations, typically using exponential/logarithmic mappings.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| [Hyperbolic pre-trained language model](https://ieeexplore.ieee.org/document/10542420) | 2024 | [Code](https://github.com/thunlp/hyperbolic_llm) | GAP, DPR, WSC, WG, PDP, FIGER, Open Entity, CoNLL-2003, Few-NERD, TACRED, TACREV, Re-TACRED, MRQA|
| [Language models as hierarchy encoders](https://openreview.net/forum?id=GJMYvWzjE1&referrer=%5Bthe%20profile%20of%20Yuan%20He%5D(%2Fprofile%3Fid%3D~Yuan_He5)) | 2024 | [Code](https://github.com/KRR-Oxford/HierarchyTransformers) | WordNet, FoodOn, DOID, SNOMED |
| [Vision-language understanding in hyperbolic space](https://www.amazon.science/publications/vision-language-understanding-in-hyperbolic-space) | 2024 | - | Food-101, CIFAR-10, CIFAR-100, CUB-200-2011, SUN397, Aircraft, DTD, Pets, Caltech-101, Flowers, STL-10, EuroSAT, RESISC45, Country211, MNIST, CLEVR, PCam, SST2|
| [Hyperbolic Learning with Multimodal Large Language Models](https://arxiv.org/abs/2408.05097) | 2024 | - | MS COCO |
| [Coneheads: Hierarchy aware attention](https://arxiv.org/abs/2306.00392) | 2023 | [Code](https://github.com/tsengalb99/coneheads) | IWSLT, ImageNet, Cora, PPI|
| [Large Language Models Enhanced Hyperbolic Space Recommender Systems](https://arxiv.org/abs/2504.05694) | 2025 | [Code](https://github.com/Qin-lab-code/HyperLLM) | Amazon-Toys, Amazon-Sports, Amazon-Beauty|
| [Hyperbolic graph-llm alignment for exploration and exploitation in recommender systems](https://arxiv.org/abs/2411.13865) | 2024 | [Code](https://github.com/Martin-qyma/HERec) | Amazon-Books, Yelp, Google-Reviews|
| [Enhancing Multimodal Survival Prediction with Pathology Reports in Hyperbolic Space](https://openreview.net/forum?id=PbC786k7qc) | 2024 | - | TCGA |
| [HySurvPred: Multimodal Hyperbolic Embedding with Angle-Aware Hierarchical Contrastive Learning and Uncertainty Constraints for Survival Prediction](https://arxiv.org/abs/2503.13862) | 2025 | - | TCGA |
| [ANTHEM: Attentive hyperbolic entity model for product search](https://dl.acm.org/doi/abs/10.1145/3488560.3498456) | 2022 | [Code](https://github.com/amazon-science/hyperbolic-embeddings) | E-commerce search data |

### Hyperbolic Fine-tuned Models
Parameter-efficient fine-tuning methods that adapt pre-trained LLMs to hyperbolic space.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| [Hyperbolic Fine-tuning for Large Language Models](https://arxiv.org/abs/2410.04010) | 2024 | [Code](https://github.com/marlin-codes/HypLLM) | MAWPS, SVAMP, GSM8K, AQuA|
| [Enhancing llm complex reasoning capability through hyperbolic geometry](https://openreview.net/forum?id=5lFiIVza6x&referrer=%5Bthe%20profile%20of%20Rex%20Ying%5D(%2Fprofile%3Fid%3D~Rex_Ying1)) | 2024 | - | MAWPS, SVAMP, GSM8K, AQuA|

### Fully Hyperbolic Models
Models that operate entirely within hyperbolic space.
| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| [Hypformer: Exploring efficient transformer fully in hyperbolic space](https://arxiv.org/abs/2407.01290) | 2024 | [Code](https://github.com/marlin-codes/hyperbolicTransformer) | Amazon2M, ogbn-proteins, ogbn-arxiv, ogbn-papers100M|
| [HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts](https://arxiv.org/abs/2505.24722) | 2025 | [Code](https://github.com/graph-and-geometric-learning/helm) | MMLU, ARC-Challenging, CommonsenseQA, HellaSwag, OpenBookQA |
| [Hypercore: The core framework for building hyperbolic foundation models with comprehensive modules](https://arxiv.org/abs/2504.08912) | 2025 | [Code](https://github.com/Graph-and-Geometric-Learning/HyperCore) | CIFAR-10, CIFAR-100, ImageNet, RedCaps, WebQSP|


### Hyperbolic State-Space Models
State-space models (like Mamba) extended to hyperbolic geometry for efficient sequence modeling.

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| [Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings](https://arxiv.org/abs/2505.18973) | 2025 | [Code](https://github.com/BerryByte/HiM) | WordNet, FoodOn, DOID, SNOMED |
| [SHMamba: Structured Hyperbolic State Space Model for Audio-Visual Question Answering](https://arxiv.org/abs/2406.09833) | 2024 | - | MUSIC-AVQA, AVQA |
| [HMamba: Hyperbolic Mamba for Sequential Recommendation](https://arxiv.org/abs/2505.09205) | 2025 | - | MovieLens-1M, Texas, California, New York |

---

## Foundations, Theory and Applications

### Hyperbolic Geometry Foundations
Core mathematical foundations and geometric properties of hyperbolic space.

| Paper Title | Year |
|-------------|------|
| [Hyperbolic entailment cones for learning hierarchical embeddings](https://proceedings.mlr.press/v80/ganea18a.html) | 2018 |
| [Hyperbolic neural networks](https://papers.nips.cc/paper_files/paper/2018/hash/dbab2adc8f9d078009ee3fa810bea142-Abstract.html) | 2018 | 
| [Poincaré embeddings for learning hierarchical representations](https://papers.nips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) | 2017 | 
| [Learning continuous hierarchies in the lorentz model of hyperbolic geometry](https://arxiv.org/abs/1806.03417) | 2018 | 
| [Hyperbolic groups](https://link.springer.com/chapter/10.1007/978-1-4613-9586-7_3) | 1987 | 
| [Network geometry](https://www.nature.com/articles/s42254-020-00264-4) | 2021 | 

### Background/Motivation

| Paper Title | Year | 
|-------------|------|
| [Zipf’s word frequency law in natural language: A critical review and future directions](https://link.springer.com/article/10.3758/s13423-014-0585-6) | 2014 |
| [Hyperbolic sentence representations for solving Textual Entailment](https://arxiv.org/abs/2406.15472) | 2024 | 
| [Large language models could make natural language again the universal interface of healthcare](https://www.nature.com/articles/s41591-024-03199-w) | 2024 | 

### Hyperbolic Graph Neural Networks
Graph neural networks adapted for hyperbolic space representations.

| Paper Title | Year | 
|-------------|------|
| [Hyperbolic graph neural networks](https://arxiv.org/abs/1910.12892) | 2019 | 
| [Hyperbolic graph convolutional neural networks](https://papers.nips.cc/paper_files/paper/2019/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html) | 2019 | 
| [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://aclanthology.org/2020.acl-main.617/) | 2020 | 
| [Fully hyperbolic graph convolution network for recommendation](https://dl.acm.org/doi/10.1145/3459637.3482109) | 2021 | 
| [Knowledge graph representation via hierarchical hyperbolic neural graph embedding](https://ieeexplore.ieee.org/document/9671651) | 2021 | 
| [Hyperkg: Hyperbolic knowledge graph embeddings for knowledge base completion](https://arxiv.org/abs/1908.04895) | 2019 | 
| [Complex hyperbolic knowledge graph embeddings with fast fourier transform](https://arxiv.org/abs/2211.03635) | 2022 |
| [Hyperbolic graph neural networks: A review of methods and applications](https://arxiv.org/abs/2202.13852) | 2022 |


### Riemannian Optimization Techniques
Optimization methods for training models on Riemannian manifolds.

| Paper Title | Year |
|-------------|------|
| [Stochastic gradient descent on Riemannian manifolds](https://ieeexplore.ieee.org/document/6487381) | 2013 |
| [Riemannian SVRG: Fast Stochastic Optimisation on Riemannian Manifolds](https://papers.nips.cc/paper_files/paper/2016/hash/98e6f17209029f4ae6dc9d88ec8eac2c-Abstract.html) | 2016 | 
| [Riemannian Accelerated Gradient Methods via Extrapolation](https://proceedings.mlr.press/v206/han23a.html) | 2023 |
| [Riemannian Stein variational gradient descent for Bayesian inference](https://ojs.aaai.org/index.php/AAAI/article/view/11810) | 2018 |

### Numerical Stability and Challenges
Research addressing computational challenges in hyperbolic representation learning.

| Paper Title | Year |
|-------------|------|
| [The Numerical Stability of Hyperbolic Representation Learning](https://proceedings.mlr.press/v202/mishne23a.html) | 2023 |
| [Representation tradeoffs for hyperbolic embeddings](https://proceedings.mlr.press/v80/sala18a.html) | 2018 |


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
| Hyperbolic deep neural networks: A survey | 2021 | 
| Hyperbolic attention networks | 2018 |
| Fully hyperbolic neural networks | 2021 | 
| HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition | 2025 | 
| Neural embeddings of graphs in hyperbolic space | 2017 |
| Hierarchical image classification using entailment cone embeddings | 2020 | 

### Core LLM, Transformer and Mamba Foundations

| Paper Title | Year | Code Link | Dataset Link |
|-------------|------|-----------|--------------|
| Attention is all you need | 2017 | 
| GPT-4 Technical Report | 2023 | 
| Gemini: a family of highly capable multimodal models | 2023 | 
| The Llama 3 Herd of Models | 2024 | 
| BERT: Pre-training of deep bidirectional transformers for language understanding | 2019 |
| Lora: Low-rank adaptation of large language models. | 2022 |
| Mamba: Linear-time sequence modeling with selective state spaces | 2024 |
| A Comparative Study on Dynamic Graph Embedding based on Mamba and Transformers | 2024 |

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

