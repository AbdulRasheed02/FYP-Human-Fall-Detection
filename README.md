<div align="center">

# FYP-Human-Fall-Detection

## Privacy-Preserving Fall Detection for Elderly Using Vision-Based Deep Learning

</div>

### Description

To Do

### Dataset

To Do

https://github.com/MUVIM/FallDetection

### Setup

#### Installation

1. Clone the repository.

   ```
   git clone https://github.com/AbdulRasheed02/FYP-Human-Fall-Detection
   ```

2. [Install CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

3. Create and activate conda environment

   ```
   conda env create -f ./Environment_Configs/environment2.yml
   conda activate fyp_base_paper_2
   ```

#### Dataset Directory Layout

1. Download the zip file(s) corresponding to each modality.
2. Extract the downloaded zip file(s) of each modality to their corresponding folders within the Dataset\Fall-Data directory. (This can be done on your local storage or an external hard drive.)

### How to Run

1. Run Dataset\dataset_creator.py for each modality to generate a compressed h5py file.
2. Run Code\single_modality.ipynb for training and testing a model on any modality.
3. Adjust parameters in Code\parameters.py
