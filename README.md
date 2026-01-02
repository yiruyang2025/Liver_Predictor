<div align="center">
  
# 3D_LLM: A Large-scale Multi-modal LLM encoding 3D Clinic Data

A Tiny ViT for multi-modal medical data classification

University of Zurich, University Hospital of Zurich, ETH AI Center ｜ ICML Jan, NeurIPS May, ECCV Mar 2026

Yiru Yang, Wei Wei, Yong Wang, (dhia, songyou peng if available), ViT author, (prof. Davide Scaramuzza)


</div>

<br>

## Structure - Zip file for conference submission

```
Liver_transplantability/
├── data/
│   └── donors.csv
│
├── src/
│   ├── data_loader.py
│   ├── data.json
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
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

## Datasets in Use by Nov 2025<br>

> Google / Meta’s Aria gen 2 research glasses<br>
> UKB-RAP, https://www.ukbiobank.ac.uk/use-our-data/research-analysis-platform/, ukbiobank tier 1<br>
> Cardiac MRI using the Swiss Heart Study<br>

## Training Goal

The goals of MedNet are:
- To ensure each bio-signal exists in a continuous space (ℝⁿ)
- To process it using CNN + ViT + KNN (local manifold)
- To preserve the original signal's geometry, topology, and frequency structure
- To output a semantic latent, not a fake token
- Ultimately, to allow LLM to see true continuous semantics instead of random tokens


High-dimensional continuous clinical/wearable signals, including:
- EEG / MEG / neural biosignals
- ECG / PPG / Doppler
- depth / infrared
- wearable motion trajectories
- Audio (diagnostic audio like cough/breath)
- High-frequency physiological waveforms
- Google / Meta Research glasses sensor streams
- AR medical sensor data
- UK Biobank continuous measurements (spirometry, accelerometer)


MedNet is specifically designed for continuous real-world bio-signals, including:
- high-frequency physiological waveforms
- neural biosignals
- audio / Doppler / PPG
- depth / infrared
- motion trajectories
- wearable sensor streams (health trackers, AR glasses, medical devices)
  
Instead of forcing all signals into text tokens, MedNet processes each modality in its native continuous space.



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


