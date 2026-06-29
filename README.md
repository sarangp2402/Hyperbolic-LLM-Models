# Hyperbolic-LLM-Models
A comprehensive collection of research papers on Hyperbolic Large Language Models with paper titles, publication years, code links, and dataset links.

## Table of Contents
- [HypLLMs](#hypllms)
  - [Hybrid Hyperbolic-Euclidean Models](#hybrid-hyperbolic-euclidean-models)
  - [Hyperbolic Fine-tuned Models](#hyperbolic-fine-tuned-models)
  - [Fully Hyperbolic Models](#fully-hyperbolic-models)
  - [Hyperbolic State-Space Models](#hyperbolic-state-space-models)
- [Foundations, Theory and Applications](#foundations-theory-and-applications)
  - [Hyperbolic Geometry Foundations](#hyperbolic-geometry-foundations)
  - [Background/Motivation](#backgroundmotivation)
  - [Hyperbolic Graph Neural Networks](#hyperbolic-graph-neural-networks)
  - [Riemannian Optimization Techniques](#riemannian-optimization-techniques)
  - [Numerical Stability and Challenges](#numerical-stability-and-challenges)
  - [Specialized Techniques and Methods](#specialized-techniques-and-methods)
  - [Core LLM, Transformer and Mamba Foundations](#core-llm-transformer-and-mamba-foundations)
  - [Brain Networks and Neuroscience Applications](#brain-networks-and-neuroscience-applications)
  - [Biomedical and Healthcare Applications](#biomedical-and-healthcare-applications)
---
## HypLLMs

### Hybrid Hyperbolic-Euclidean Models
Models that integrate hyperbolic geometry with traditional Euclidean operations, typically using exponential/logarithmic mappings.

| Paper Title | Year | Journal/ Conference | Code Link | Dataset Link |
|-------------|------|---------|-----------|--------------|
| [ANTHEM: Attentive hyperbolic entity model for product search](https://dl.acm.org/doi/abs/10.1145/3488560.3498456) | 2022 | WSDM | [Code](https://github.com/amazon-science/hyperbolic-embeddings) | E-commerce search data |
| [Coneheads: Hierarchy aware attention](https://arxiv.org/abs/2306.00392) | 2023 | NeurIPS | [Code](https://github.com/tsengalb99/coneheads) | [IWSLT](https://huggingface.co/datasets/bbaaaa/iwslt14-de-en), [ImageNet](https://image-net.org/index.php), [Cora](https://graphsandnetworks.com/the-cora-dataset/), [PPI](https://snap.stanford.edu/graphsage/#datasets)|
| [Hyperbolic pre-trained language model](https://ieeexplore.ieee.org/document/10542420) | 2024 | IEEE | [Code](https://github.com/thunlp/hyperbolic_llm) | [GAP](https://github.com/google-research-datasets/gap-coreference), [DPR](https://huggingface.co/datasets/coref-data/dpr_raw), [WSC](https://huggingface.co/datasets/ErnestSDavis/winograd_wsc), [WG](https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino), PDP, [FIGER](https://github.com/xiaoling/figer), [Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html), [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/), [Few-NERD](https://ningding97.github.io/fewnerd/), [TACRED](https://nlp.stanford.edu/projects/tacred/), [TACREV](https://github.com/DFKI-NLP/tacrev), [Re-TACRED](https://github.com/gstoica27/Re-TACRED), [MRQA](https://mrqa.github.io/2019/shared.html)|
| [Language models as hierarchy encoders](https://openreview.net/forum?id=GJMYvWzjE1&referrer=%5Bthe%20profile%20of%20Yuan%20He%5D(%2Fprofile%3Fid%3D~Yuan_He5)) | 2024 | NeurIPS | [Code](https://github.com/KRR-Oxford/HierarchyTransformers) | [WordNet](https://zenodo.org/records/14036213), [FoodOn](https://zenodo.org/records/14036213), [DOID](https://zenodo.org/records/14036213), [SNOMED](https://zenodo.org/records/14036213) |
| [Vision-language understanding in hyperbolic space](https://www.amazon.science/publications/vision-language-understanding-in-hyperbolic-space) | 2024 | SIGIR |- | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [SUN397](https://vision.princeton.edu/projects/2010/SUN/), [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [STL-10](https://cs.stanford.edu/~acoates/stl10/), [EuroSAT](https://github.com/phelber/eurosat), [RESISC45](https://huggingface.co/datasets/timm/resisc45), [Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [PCam](https://github.com/basveeling/pcam), [SST2](https://github.com/YJiangcm/SST-2-sentiment-analysis)|
| [Hyperbolic Learning with Multimodal Large Language Models](https://arxiv.org/abs/2408.05097) | 2024 | arXiv | - | [MS COCO](https://cocodataset.org/#home) |
| [Hyperbolic graph-llm alignment for exploration and exploitation in recommender systems](https://arxiv.org/abs/2411.13865) | 2024 | arXiv | [Code](https://github.com/Martin-qyma/HERec) | [Amazon-Books](https://drive.google.com/drive/folders/18RmXeLmc3mJVkgMn8v9sOcxmffXWOU5C), [Yelp](https://drive.google.com/drive/folders/18RmXeLmc3mJVkgMn8v9sOcxmffXWOU5C), [Google-Reviews](https://drive.google.com/drive/folders/18RmXeLmc3mJVkgMn8v9sOcxmffXWOU5C)|
| [Enhancing Multimodal Survival Prediction with Pathology Reports in Hyperbolic Space](https://openreview.net/forum?id=PbC786k7qc) | 2024 | ICLR | - | [TCGA](https://portal.gdc.cancer.gov/analysis_page?app=CohortBuilder&tab=general) |
| [Large Language Models Enhanced Hyperbolic Space Recommender Systems](https://arxiv.org/abs/2504.05694) | 2025 | arXiv | [Code](https://github.com/Qin-lab-code/HyperLLM) | [Amazon-Toys](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), [Amazon-Sports](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), [Amazon-Beauty](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)|
| [HySurvPred: Multimodal Hyperbolic Embedding with Angle-Aware Hierarchical Contrastive Learning and Uncertainty Constraints for Survival Prediction](https://arxiv.org/abs/2503.13862) | 2025 | arXiv | - | [TCGA](https://portal.gdc.cancer.gov/analysis_page?app=CohortBuilder&tab=general) |
| [HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models](https://arxiv.org/abs/2510.20322) | 2025 | NeurIPS (Oral) | [Code](https://github.com/godlin-sjtu/HyperET) | [LLaVA-1.5 benchmarks](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf) |
| [PHyCLIP: ℓ1-Product of Hyperbolic Factors Unifies Hierarchy and Compositionality in Vision-Language Representation Learning](https://openreview.net/forum?id=I3Ct1eDmVI) | 2026 | ICLR | [Code](https://github.com/tksmatsubara/PHyCLIP) | [GRIT](https://arxiv.org/html/2505.15879v2)  |
| [HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation](https://openreview.net/pdf?id=w6FXg6d6My) | 2026 | ICML (Poster) | [Code](https://github.com/Graph-and-Geometric-Learning/HypRAG) | RAG/QA benchmarks |
| [Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tuning](https://icml.cc/virtual/2026/poster/66585) | 2026 | ICML | - | multimodal instruction-tuning benchmarks |
| [Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning](https://icml.cc/virtual/2026/poster/61337) | 2026 | ICML | - | multimodal in-context learning benchmarks |

### Hyperbolic Fine-tuned Models
Parameter-efficient fine-tuning methods that adapt pre-trained LLMs to hyperbolic space.

| Paper Title | Year | Journal/ Conference | Code Link | Dataset Link |
|-------------|------|---------|-----------|--------------|
| [Hyperbolic Fine-tuning for Large Language Models](https://arxiv.org/abs/2410.04010) | 2024 | arXiv | [Code](https://github.com/marlin-codes/HypLLM) | [MAWPS](https://github.com/sroy9/mawps), [SVAMP](https://github.com/arkilpatel/SVAMP), [GSM8K](https://github.com/openai/grade-school-math), [AQuA](https://github.com/google-deepmind/AQuA)|
| [Enhancing llm complex reasoning capability through hyperbolic geometry](https://openreview.net/forum?id=5lFiIVza6x&referrer=%5Bthe%20profile%20of%20Rex%20Ying%5D(%2Fprofile%3Fid%3D~Rex_Ying1)) | 2024 | PMLR | - | [MAWPS](https://github.com/sroy9/mawps), [SVAMP](https://github.com/arkilpatel/SVAMP), [GSM8K](https://github.com/openai/grade-school-math), [AQuA](https://github.com/google-deepmind/AQuA)|

### Fully Hyperbolic Models
Models that operate entirely within hyperbolic space.
| Paper Title | Year | Journal/ Conference | Code Link | Dataset Link |
|-------------|------|---------|-----------|--------------|
| [Hypformer: Exploring efficient transformer fully in hyperbolic space](https://arxiv.org/abs/2407.01290) | 2024 | KDD | [Code](https://github.com/marlin-codes/hyperbolicTransformer) | [Amazon2M](https://ogb.stanford.edu/docs/nodeprop/), [ogbn-proteins](https://ogb.stanford.edu/docs/nodeprop/), [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/), [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/)|
| [HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts](https://arxiv.org/abs/2505.24722) | 2025 | arXiv | [Code](https://github.com/graph-and-geometric-learning/helm) | [MMLU](https://github.com/hendrycks/test?tab=readme-ov-file), [ARC-Challenging](https://huggingface.co/datasets/allenai/ai2_arc), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [HellaSwag](https://rowanzellers.com/hellaswag/), [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) |
| [Hypercore: The core framework for building hyperbolic foundation models with comprehensive modules](https://arxiv.org/abs/2504.08912) | 2025 | arXiv | [Code](https://github.com/Graph-and-Geometric-Learning/HyperCore) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet](https://image-net.org/index.php), [RedCaps](https://redcaps.xyz/), [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763)|
| [Lorentzian Residual Neural Networks](https://dl.acm.org/doi/10.1145/3690624.3709292) | 2025 | KDD | [Code](https://github.com/heneil/LRN) | standard graph/image benchmarks |

### Hyperbolic State-Space Models
State-space models (like Mamba) extended to hyperbolic geometry for efficient sequence modeling.

| Paper Title | Year | Journal/ Conference | Code Link | Dataset Link |
|-------------|------|---------|-----------|--------------|
| [SHMamba: Structured Hyperbolic State Space Model for Audio-Visual Question Answering](https://arxiv.org/abs/2406.09833) | 2024 | arXiv | - | [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/), [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) |
| [Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings](https://arxiv.org/abs/2505.18973) | 2025 | arXiv | [Code](https://github.com/BerryByte/HiM) | [WordNet](https://zenodo.org/records/14036213), [FoodOn](https://zenodo.org/records/14036213), [DOID](https://zenodo.org/records/14036213), [SNOMED](https://zenodo.org/records/14036213) |
| [HMamba: Hyperbolic Mamba for Sequential Recommendation](https://arxiv.org/abs/2505.09205) | 2025 | arXiv | - | [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), Texas, California, New York |
| [Hyperbolic-Enhanced Mixture-of-Experts Mamba for Sequential Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/38567) | 2026 | AAAI | - | [Books] [Toys] [NYC] [TKY] |

---

## Foundations, Theory and Applications

### Hyperbolic Geometry Foundations
Core mathematical foundations and geometric properties of hyperbolic space.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Hyperbolic groups](https://link.springer.com/chapter/10.1007/978-1-4613-9586-7_3) | 1987 |  Essays in group theory |
| [Poincaré embeddings for learning hierarchical representations](https://papers.nips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) | 2017 | NeurIPS |
| [Hyperbolic entailment cones for learning hierarchical embeddings](https://proceedings.mlr.press/v80/ganea18a.html) | 2018 | ICML |
| [Hyperbolic neural networks](https://papers.nips.cc/paper_files/paper/2018/hash/dbab2adc8f9d078009ee3fa810bea142-Abstract.html) | 2018 | NeurIPS | 
| [Learning continuous hierarchies in the lorentz model of hyperbolic geometry](https://proceedings.mlr.press/v80/nickel18a/nickel18a.pdf) | 2018 | ICML |
| [Hyperbolic Neural Networks++](https://openreview.net/pdf?id=Ec85b0tUwbA) | 2021 | ICLR |

### Background/Motivation

| Paper Title | Year | Journal/ Conference | 
|-------------|------|---------|
| [Zipf’s word frequency law in natural language: A critical review and future directions](https://link.springer.com/article/10.3758/s13423-014-0585-6) | 2014 | Psychon. Bull. Rev. |
| [Hyperbolic sentence representations for solving Textual Entailment](https://arxiv.org/abs/2406.15472) | 2024 |  arXiv |
| [Large language models could make natural language again the universal interface of healthcare](https://www.nature.com/articles/s41591-024-03199-w) | 2024 | Nat. Med. |

### Hyperbolic Graph Neural Networks
Graph neural networks adapted for hyperbolic space representations.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Hyperbolic graph neural networks](https://papers.nips.cc/paper_files/paper/2019/hash/103303dd56a731e377d01f6a37badae3-Abstract.html) | 2019 | NeurIPS |
| [Hyperkg: Hyperbolic knowledge graph embeddings for knowledge base completion](https://arxiv.org/abs/1908.04895) | 2019 | arXiv | 
| [Hyperbolic graph convolutional neural networks](https://papers.nips.cc/paper_files/paper/2019/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html) | 2019 | NeurIPS |
| [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://aclanthology.org/2020.acl-main.617/) | 2020 | ACL |
| [Fully hyperbolic graph convolution network for recommendation](https://dl.acm.org/doi/10.1145/3459637.3482109) | 2021 | CIKM |
| [Knowledge graph representation via hierarchical hyperbolic neural graph embedding](https://ieeexplore.ieee.org/document/9671651) | 2021 | IEEE |
| [Complex hyperbolic knowledge graph embeddings with fast fourier transform](https://arxiv.org/abs/2211.03635) | 2022 | arXiv |
| [Hyperbolic graph neural networks: A review of methods and applications](https://arxiv.org/abs/2202.13852) | 2022 | arXiv |
| [Spectro-Riemannian Graph Neural Networks](https://arxiv.org/abs/2502.00401) | 2025 | ICLR |

### Riemannian Optimization Techniques
Optimization methods for training models on Riemannian manifolds.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Stochastic gradient descent on Riemannian manifolds](https://ieeexplore.ieee.org/document/6487381) | 2013 | IEEE |
| [Riemannian SVRG: Fast Stochastic Optimisation on Riemannian Manifolds](https://papers.nips.cc/paper_files/paper/2016/hash/98e6f17209029f4ae6dc9d88ec8eac2c-Abstract.html) | 2016 | NeurIPS |
| [Riemannian Stein variational gradient descent for Bayesian inference](https://ojs.aaai.org/index.php/AAAI/article/view/11810) | 2018 | AAAI |
| [Riemannian Accelerated Gradient Methods via Extrapolation](https://proceedings.mlr.press/v206/han23a.html) | 2023 | AISTATS |

### Numerical Stability and Challenges
Research addressing computational challenges in hyperbolic representation learning.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Representation tradeoffs for hyperbolic embeddings](https://proceedings.mlr.press/v80/sala18a.html) | 2018 | ICML |
| [The Numerical Stability of Hyperbolic Representation Learning](https://proceedings.mlr.press/v202/mishne23a.html) | 2023 | ICML |
| [Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis](https://dl.acm.org/doi/pdf/10.1145/3637528.3671875) | 2024 | KDD |

### Specialized Techniques and Methods
Specialized approaches and technical innovations for hyperbolic learning.

| Paper Title | Year | Journal/ Conference | 
|-------------|------|---------|
| [Hyperbolic geometry of complex networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.036106) | 2010 |Physical Review E |
| [The hyperbolic theory of special relativity](https://arxiv.org/abs/1102.0462) | 2011 | arXiv |
| [Neural embeddings of graphs in hyperbolic space](https://arxiv.org/abs/1705.10359) | 2017 | arXiv |
| [Hyperbolic attention networks](https://openreview.net/forum?id=rJxHsjRqFQ) | 2018 | ICLR |
| [Poincaré GloVe: Hyperbolic word embedding](https://openreview.net/forum?id=Ske5r3AqK7) | 2019 | ICLR |
| [Hierarchical image classification using entailment cone embeddings](https://openaccess.thecvf.com/content_CVPRW_2020/html/w50/Dhall_Hierarchical_Image_Classification_Using_Entailment_Cone_Embeddings_CVPRW_2020_paper.html) | 2020 | CVPR |
| [Representing hyperbolic space accurately using multi-component floats](https://proceedings.neurips.cc/paper/2021/hash/832353270aacb6e3322f493a66aaf5b9-Abstract.html) | 2021 | NeurIPS |
| [Probing BERT in hyperbolic spaces](https://openreview.net/forum?id=17VnwXYZyhH) | 2021 | ICLR |
| [Fully hyperbolic neural networks](https://arxiv.org/abs/2105.14686) | 2021 | arXiv | 
| [Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](https://arxiv.org/abs/2405.15481) | 2024 | arXiv |
| [Adaptive data embedding for curved spaces](https://www.sciencedirect.com/science/article/pii/S258900422402491X) | 2024 | iScience |
| [HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition](https://arxiv.org/abs/2502.05869) | 2025 | arXiv |

### Core LLM, Transformer and Mamba Foundations

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Attention is all you need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | 2017 | NeurIPS |
| [BERT: Pre-training of deep bidirectional transformers for language understanding](https://aclanthology.org/N19-1423/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_m9bbH_7ECE1h3lZ3D61TYg52rKpifVNjL4fvJ85uqggrXsWDBTB7YooFLJeNXHWqhvOyC) | 2019 | NAACL-HLT |
| [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023 | arxIv |
| [Gemini: a family of highly capable multimodal models](https://arxiv.org/abs/2312.11805) | 2023 | arXiv |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 2024 | arXiv |
| [Mamba: Linear-time sequence modeling with selective state spaces](https://openreview.net/forum?id=tEYskw1VY2) | 2024 | COLM |
| [A Comparative Study on Dynamic Graph Embedding based on Mamba and Transformers](https://arxiv.org/abs/2412.11293) | 2024 | arXiv |

### Brain Networks and Neuroscience Applications
Hyperbolic models applied to brain network analysis and neuroscience research.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Neural manifolds for the control of movement](https://www.sciencedirect.com/science/article/pii/S0896627317304634) | 2017 | Neuron |
| [Hyperbolic geometry of the olfactory space](https://www.science.org/doi/10.1126/sciadv.aaq1458) | 2018 | Science Advances |
| [Navigating Memorability Landscapes: Hyperbolic Geometry Reveals Hierarchical Structures in Object Concept Memory](https://www.biorxiv.org/content/10.1101/2024.09.22.614329v1) | 2024 | bioRxiv |
| [Nonlinear manifolds underlie neural population activity during behaviour](https://pubmed.ncbi.nlm.nih.gov/37503015/) | 2024 | bioRxiv |
| [Hyperbolic graph embedding of MEG brain networks to study brain alterations in individuals with subjective cognitive decline](https://ieeexplore.ieee.org/abstract/document/10564006) | 2024 | IEEE JHBI |
| [Fully Hyperbolic Neural Networks: A Novel Approach to Studying Aging Trajectories](https://ieeexplore.ieee.org/abstract/document/10916497) | 2025 | IEEE JBHI |
| [Brain-HGCN: A Hyperbolic Graph Convolutional Network for Brain Functional Network Analysis](https://ieeexplore.ieee.org/document/11464016) | 2026 | ICASSP |
| [HyFI: Hyperbolic Feature Interpolation for Brain-Vision Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/37476) | 2026 | AAAI |

### Biomedical and Healthcare Applications
Applications of hyperbolic geometry in biomedical and healthcare domains.

| Paper Title | Year | Journal/ Conference |
|-------------|------|---------|
| [Hig2vec: hierarchical representations of gene ontology and genes in the poincar\'e](https://academic.oup.com/bioinformatics/article/37/18/2971/6184857) | 2021 | Bioinformatics |
| [Hyperbolic hierarchical knowledge graph embeddings for biological entities](https://www.sciencedirect.com/science/article/pii/S1532046423002241) | 2023 | JBI |
| [Hyperbolic Genome Embeddings](https://proceedings.iclr.cc/paper_files/paper/2025/file/b63ad8c24354b0e5bcb7aea16490beab-Paper-Conference.pdf) | 2025 | ICLR |
| [Learning Protein–Ligand Binding in Hyperbolic Space](https://ojs.aaai.org/index.php/AAAI/article/view/37086) | 2026 | AAAI |

---

## Citation

If you find this repository and our paper useful, please cite our publication:

*Sarang Patil, Zeyong Zhang, Yiran Huang, Tengfei Ma, Mengjia Xu. Hyperbolic Large Language Models. arXiv preprint arXiv:2509.05757 (2025). [arxiv](https://arxiv.org/abs/2509.05757)*

```bibtex
@article{patil2025hyperbolic,
  title={Hyperbolic Large Language Models},
  author={Patil, Sarang and Zhang, Zeyong and Huang, Yiran and Ma, Tengfei and Xu, Mengjia},
  journal={arXiv preprint arXiv:2509.05757},
  year={2025}
}
