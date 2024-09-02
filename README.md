# Repository for "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines" Paper
This repository contains curated datasets, implementation of methods experimented and introduced in the paper titled "NS-HyMAP: Neurosymbolic Multimodal Hybrid Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines".

# 1. Data Preprocessing #

The final preprocessed Image dataset(YOLO-FF) available at: https://drive.google.com/drive/folders/1VdIsSouurlVAFRLaZnPuemsDXLyKRN-2?usp=drive_link


# 2. Baselines # 

This folder is including the baseline models developed.
Three baseline models are:

## Custom ViT ##
To run py .Baselines/custom_vit.py

## CNN ##
To run py .Baselines/image_with_seg_cnn.py

## VIT ##
To run py .Baselines/image_with_segmentation_vit.py

# 3. Proposed  Approach #

This folder includes the models for the proposed approach.

## EfficentNet ##
To run py .Proposed Method/image_with_segmentation_pretrainedcnn.py



# 4. Model #
This folder includes the pretrained efficient-net model pretrained for YOLO-FF dataste.