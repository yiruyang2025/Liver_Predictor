<div align="center">
  
# MedNet: Multi-modal ViT on Large-scale 3D Clinic Data

Nov 2025 - Mar 2026 ｜ AI Center, UZH AI, CVG, ETHz SCAI Lab ｜ NeurIPS May 2026, ECCV Mar 26, master thesis

co-first authors: Yiru Yang, (dhia, songyou peng学长 if available),(Taein Kwon)<br>
authors: xxx, <br>
supervisors: ViT author Xiaohua Zhai学长 if available, prof. Davide Scaramuzza, postdoc. Wen Guo, postdoc.   <br>


</div>

<br>

## Structure

```

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


