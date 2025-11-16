<div align="center">
  
# ★ MedNet: Multi-modal ViT on Large-scale 3D Clinic Data

Nov 2025 - Feb 2026 ｜ AI Center, UZH AI, USZ | NeurIPS May 2026, ECCV Mar 26, master thesis

co-first authors: Yiru Yang, (dhia, songyou peng学长 & Xiaohua Zhai学长 if available),<br>
authors: xxx, <br>
supervisors: prof. Davide Scaramuzza, Dr. Wei Wei, Dr. Yong Wang<br>


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

> To address the above fundamental wrongness, we propose MedNet as multimodal ViT backbone for real-world bio signal processing. Dataset from `either google / meta’s research glasses`.<br>
> Dataset, UKB-RAP, https://www.ukbiobank.ac.uk/use-our-data/research-analysis-platform/, ukbiobank tier 1, 2D<br>

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


<br><br><br>


