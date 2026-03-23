# SpaAGMF: Adaptive Gated Multi-Scale Fusion of Histology and Spatial Transcriptomics for Cancer Region Classification

# Introduction
Accurate classification of cancer tissue regions (CTR) in spatial transcriptomics is critical for understanding tumor heterogeneity and informing clinical treatment decisions. 
However, current computational methods struggle to bridge the semantic gap between high-dimensional transcriptomic profiles and complex histological images, while also being hindered by inherent noise and significant batch effects across clinical samples and sequencing platforms, thereby limiting their generalizability in real-world applications.
To address these challenges, we propose SpaAGMF, a novel adaptive gated multi-scale fusion framework. SpaAGMF integrates a pre-trained pathology foundation model to extract semantic histological features and employs a bidirectional contrastive learning strategy to align these visual patterns with transcriptomic profiles in a shared latent space. A core component of SpaAGMF is the Multi-Scale Gated Representation Extractor (MS-GRE), which uses a gated multi-head attention mechanism to adaptively fuse micro-scale details with macro-scale neighborhood contexts. 
We evaluated SpaAGMF on five spatial transcriptomics datasets encompassing various cancer types and sequencing platforms. Extensive experiments demonstrate that SpaAGMF consistently outperforms state-of-the-art methods in cross-sample, cross-platform and cross-batch classification tasks.

# Model Structure
![Schematic of the proposed framework](./resource/Fig1_version_3.png)
**Figure 1.** (A) Multimodal input processing from whole slide images and spatial transcriptomics. (B) Stage 1: Cross-modal alignment using bidirectional InfoNCE loss. (C) Stage 2: The MS-GRE network, featuring gene-guided gated cross-attention for micro-scale features and neighborhood-aware gated self-attention for macro-scale context. (D) Detailed architecture of the GMHA module.

# Data
| Dataset | Data Type | Tumor Type | Spots | Tumor Ratio | Platform | Link| 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CRC | Multi-slice | Colorectal Cancer | 2203 | 27.01 | Visium | [Link]() |
| STHBC | Multi-slice | Breast Cancer | 346 | 87.57 | ST | [Link]() |
| XeHBC | Single-slice | Breast Cancer | 4050 | 34.96 | Xenium | [Link]() |
| ViHBC | Single-slice | Breast Cancer | 2518 | 42.77 | Visium | [Link]() |
| IDC | Single-slice | Breast Cancer | 3798 | 65.01 | Visium | [Link]() |

# Environment
conda create -n SpaAGMF python==3.10.20
conda activate SpaAGMF
pip install -r requirements.txt
