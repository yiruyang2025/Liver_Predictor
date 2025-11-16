# Multi-modal ViT on Large-scale 3D Clinic Data

<br>

## Structure

```
2D-UKBiobank/
├── src/                
│   ├── main.cpp
│   ├── preprocess.cpp
│   ├── model.cpp
│   ├── utils.hpp
│   └── CMakeLists.txt
├── data/              
├── docs/               # documentation
├── results/           
└── README.md
```

<br>

## Folder & File Descriptions

  - `src/` – Source Code Folder
    - Contains all core C++ source files and the build configuration

  - `main.cpp` – Main program entry
    - Orchestrates the entire pipeline: loads data, calls preprocessing and model modules, runs inference, and saves outputs

  - `preprocess.cpp` – Image preprocessing module
    - Handles image reading and normalization, resizing, denoising, augmentation, and other basic data preparation

  - `model.cpp` – Model logic module
    - Loads or defines the neural network (often via LibTorch)
    - Handles forward passes, feature extraction, and prediction

  - `utils.hpp` – Utility header file
    - Stores helper functions such as logging, timing, or path handling used across modules

  - `CMakeLists.txt` – Build configuration script
    - Defines compilation rules and dependencies (e.g., OpenCV, Torch)
    - Used by CMake to generate a Makefile and compile the project

  - `data/` – Data Folder
    - Holds sample or example image files (not full UK Biobank data, due to privacy)
    - Can also include download instructions or preprocessing notes


  - `docs/` – Documentation Folder
    - Contains research documentation: diagrams, methodology notes, references, and citation files
    - Used for project reports or paper appendices


  - `results/` – Results Folder
    - Stores outputs such as segmentation masks, metrics (CSV), or visualization figures (heatmaps, ROC curves, etc.)


  - `README.md` – Project Description
    - Top-level markdown file that introduces the project purpose, dependencies, build and run instructions, and citation information.
    - It is displayed automatically on the GitHub repository homepage

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


