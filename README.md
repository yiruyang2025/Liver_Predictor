<div align="center">
  
## Learning in Latent: Self-supervised Model Predicting Liver Transplantability Based On Transplant Donor Medical Data

University of Zurich, University Hospital of Zurich, ETH AI Center

ICML Jan, NeurIPS May, ECCV Mar 2026

**Wei Wei, Yiru Yang, Yong Wang, (prof. Davide Scaramuzza)**


</div>

<br>

## Data Availability

Due to strict clinical governance and patient confidentiality, raw donor data and PDF-derived JSON files cannot be shared. This repository provides:
- complete preprocessing and modeling code
- full data schema
- synthetic example donors<br>

All experimental results reported in the paper were obtained using confidential hospital-internal data under approved protocols.

<br>

## Overview

- Our work follows the DeepMind line of representation-first learning, where structure is learned prior to task supervision, and labels are used only to minimally align representations to downstream decisions under extreme data scarcity.
- We learn a donor state manifold using 39 DIFs, and then align it to the real transplantation decision with minimal supervision.


<br>

## Structure - Zip file for conference submission

```
Liver_transplantability/
├── data/            
│   ├── schema.json             
│   └── example_donor.json
│
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
│
├── cross_validation/
│   └── run_loocv.sh             # LOOCV = gold standard when N < 100
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

> To address the above fundamental wrongness, we propose ★ Liver Donor, a SSL model for liver transplantability prediction under extreme data scarcity.<br>


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
- [1] Multimodal LLMs for health grounded in individual-specific data, 2023.
- [2] A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- [3] What Makes for Good Views for Contrastive Learning? NeurIPS 2020.
- [4] Data-Efficient Reinforcement Learning with Self-Supervised Predictive Representations. NeurIPS 2021.
- [5] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR 2022.
- [6] Teras: A Unified Deep Learning Library for Tabular Data, https://github.com/KhawajaAbaid/teras


<br><br><br>


