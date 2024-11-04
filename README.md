# [BMVC 2024] Toward Highly Efficient Semantic-Guided Machine Vision for Low-Light Object Detection

# Abstract

![intro_figure2](E:\EMV-YOLO-main\figures\intro_figure2.png)

Detectors trained on well-lit data often experience significant performance degradation when applied to low-light conditions. To address this challenge, low-light enhancement methods are commonly employed to improve detection performance. However, existing human vision-oriented enhancement methods have shown limited effectiveness, which overlooks the semantic information for detection and achieves high computation costs. To overcome these limitations, we introduce a machine vision-oriented highly efficient low-light object detection method with the Efficient semantic-guided Machine Vision-oriented module (EMV). EMV can dynamically adapt to the object detection part based on end-to-end training and emphasize the semantic information for the detection. Besides, by lightening the network for feature decomposition and generating the enhanced image on latent space, EMV is a highly lightweight network for image enhancement, which contains only 27K parameters and achieves high inference speed. Extensive experiments conducted on ExDark and DarkFace datasets demonstrate that our method significantly improves detector performance in low-light environments.

# Getting Started

## Dependencies

1. Create conda environment

   ```
   conda create -n EMV-YOLO python=3.8 -y
   conda activate EMV-YOLO
   ```

2. Install PyTorch. This repo is tested with PyTorch==1.10.0

   ​	for OSX:

   ```
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
   ```

   ​	for Linux and Windows:

   ```
   # CUDA 10.2
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
   
   # CUDA 11.3
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
   
   # CPU Only
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
   ```

   

3. Install python packages using following command:	

   (1) Download mmcv 1.4.0, and download adapte to your own cuda version and torch version:

   ```
   pip install mmcv-full==1.4.0 https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   ```

   (2) Then set up mmdet (2.15.1):

   ```
   pip install opencv-python scipy
   pip install -r requirements/build.txt
   pip install -v -e .
   ```

   

![final_vis](E:\EMV-YOLO-main\figures\final_vis.png)
