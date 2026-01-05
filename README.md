<div align="center">
  
## Self-supervised Model Predicting Liver Transplantability Based On Transplant Donor Medical Data

University of Zurich, University Hospital of Zurich, ETH AI Center

ICML Jan, NeurIPS May, ECCV Mar 2026

**Wei Wei, Yiru Yang, Yong Wang, (prof. Davide Scaramuzza)**


</div>

<br>

## Data Availability

Due to strict clinical governance and patient confidentiality, raw donor data and PDF-derived JSON files cannot be shared. This repository provides:
- complete preprocessing and modeling code
- full data schema
- synthetic example donors
All experimental results reported in the paper were obtained using confidential hospital-internal data under approved protocols.

<br>

## Structure - Zip file for conference submission

```
Liver_transplantability/
│
├── data/
│   ├── README.md               
│   ├── schema.json             
│   ├── example_donor.json    
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loader.py           # JSON → feature vector
│   ├── ssl_encoder.py           # SSL backbone
│   ├── ssl_objectives.py        # contrastive / masking
│   ├── classifier.py            # TX / NTX head
│   ├── train_ssl.py             # self-supervised pretraining
│   ├── train_classifier.py      # downstream prediction
│   ├── evaluate.py              # LOOCV / metrics
│   └── utils.py                 # common helpers
│
├── experiments/
│   └── train.sh         
│
├── requirements.txt
├── README.md
└── LICENSE
```

<br>

## Abstract

> LLM stands for Discrete Symbolic Model, but images/depth/point clouds/continuous signals are data in continuous space.<br>

> Forcibly tokenizing continuous space into LLM is incorrect from an information theory perspective. The traditional transformer can only deal with softmax(QKᵀ)V for the information output. <br>

> Fusion Transformer in latent-to-latent for ViT/other cv feature extractor to language tokens alignment. Use Continuous Encoder → Fake Token → Attention Trick to make LLM see the failure of continuous information but still effectively hack it.<br>

> It's like turning an image into 6 random tokens and telling the LLM, ‘This is image embedding, please pretend you understand it’. This is not multimodal understanding.<br>

<br>

> To address the above fundamental wrongness, we propose ★MedNet as multimodal ViT backbone for real-world bio signal processing.<br>


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
[1] Multimodal LLMs for health grounded in individual-specific data, 2023. https://arxiv.org/pdf/2307.09018



<br><br><br>


