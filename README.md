<div align="center">
  
## Liver Predictor: Self-supervised Model Predicting Liver Allocation Based On Transplant Donor Medical Data

University of Zurich, University Hospital of Zurich, ETH AI Center

ICML Jan, NeurIPS May, ECCV Mar 2026

**Yiru Yang, (), Wei Wei, Yong Wang, (prof. Davide Scaramuzza)**


</div>

<br>

## Data Availability

Due to strict clinical governance and patient confidentiality, raw donor data and PDF-derived JSON files cannot be shared. This repository provides:

- complete preprocessing and modeling code
- full data schema from `json`
- synthetic example donors<br>
- allocation + synetic data + biomarker<br>

All experimental results reported in the paper were obtained using confidential hospital-internal data under approved protocols

<p align="left">
  <img src="https://github.com/yiruyang2025/Liver_Predictor/blob/main/assets/encoder.png" alt="Project 1 Visualization" width="60%">
</p>


## Overview

- Our work follows the DeepMind line of representation-first learning, where structure is learned prior to task supervision, and labels are used only to minimally align representations to downstream decisions under extreme data scarcity
- We learnt A SSL Donor Representation Encoder, and then use the binary transplantability classification (TX vs. NTX) to test

**Key Words**: Representation Learning • Data Scarcity • Self-supervised Learning • Liver Organ suitability • Extreme Low-data Regime

<br>

## Structure - Zip file for conference submission

```
Liver_predictor/
├── data/            
│   ├── schema.json             
│   ├── example_donor.json
│   └── private/
│       ├── donor_001.json
│       ├── donor_002.json
│       ├── ...
│       └── donor_039.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # JSON → feature vector
│   ├── ssl_encoder.py           # SSL backbone
│   ├── ssl_objectives.py        # contrastive / masking
│   ├── classifier.py            # TX / NTX head
│   ├── train_ssl.py             # self-supervised pretraining
│   ├── train_classifier.py      # downstream prediction
│   ├── evaluate.py              # LOOCV / metrics
│   └── utils.py                 # common helpers
+   ├── ablation.py           
+   └── compare_results.py 
│
├── cross_validation/
│   ├── run_loocv.sh             # LOOCV = gold standard when N < 100
+   └── run_ablation.sh
│
├── requirements.txt
├── README.md
└── LICENSE
```

<br>

## Abstract

> LLM stands for Discrete Symbolic Model, but images / depth / point clouds / continuous signals are data in continuous space.<br>

> Forcibly tokenizing continuous space into LLM is incorrect from an information theory perspective. The traditional transformer can only deal with softmax(QKᵀ)V for the information output. <br>

> Fusion Transformer in latent-to-latent for ViT / other cv feature extractor to language tokens alignment. Use Continuous Encoder → Fake Token → Attention Trick to make LLM see the failure of continuous information but still effectively hack it.<br>

> It's like turning an image into 6 random tokens and telling the LLM, ‘This is image embedding, please pretend you understand it’. This is not multimodal understanding.<br>

<br>

> To address the above fundamental Drawback, we propose ★ Liver Predictor, a SSL Model for liver transplantability prediction under extreme data scarcity.<br>

<br>


<p align="left">
  <img src="https://github.com/yiruyang2025/Liver_Predictor/blob/main/assets/workflow.png" alt="Project 1 Visualization" width="60%">
</p>

<br>

**Clustering 39 raw json files**

<p align="left">
  <img src="https://github.com/yiruyang2025/Liver_Predictor/blob/main/assets/visual_39.png" alt="Project 1 Visualization" width="60%">
</p>


<br>

## Get Started

### 1. Prerequisites

  - Install CMake (version ≥ 3.18)
  - Install dependencies: OpenCV, LibTorch (C++ API for PyTorch)


### 2. Build

```
git clone https://github.com/your-username/Multimodal_Cpp.git  
cd Multimodal_Cpp  
mkdir build && cd build  
cmake ..  
make -j4  
```


## Readings

- [1] 📍 ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021 Oral.
- [2] 📍 BYOL: Bootstrap your own latent: A new approach to self-supervised Learning, Google Deepmind, NeurIPS 2020.
- [3] 📍 CLIP: Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.
- [4] 📍 AlexNet: ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012.
- [5] DINOv2: Learning Robust Visual Features without Supervision, 2024.
- [6] Large-scale pancreatic cancer detection via non-contrast CT and deep learning, Nature 2023.
- [7] SimCLR: A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- [8] Debiasing Real-World Data to Enable Causal Inference As If From A Random Trail, arvix 2026.
- [9] Revisiting Deep Learning Models for Tabular Data, NeurIPS 2021.
- [10] Multimodal LLMs for health grounded in individual-specific data, 2023.
- [11] ImageNet: A large-scale hierarchical image database, 2009.
- [12] Teras: A Unified Deep Learning Library for Tabular Data, https://github.com/KhawajaAbaid/teras.



<br><br><br>


