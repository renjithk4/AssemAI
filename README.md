# Repository for "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines" Paper
This repository contains derived datasets, implementation of methods experimented and introduced in the paper titled "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines".

# 1. Data Preprocessing #
This folder includes the steps for object detection from the images in the Future Factories dataset.
The final preprocessed image dataset available at: https://drive.google.com/drive/folders/1VdIsSouurlVAFRLaZnPuemsDXLyKRN-2?usp=drive_link
To train the YOLO model run, -> py 1. Data Preprocessing/YOLO-FF.py
The folder "YOLO-FF Model for Object Detection", includes the results of the model training.
The YOLO-FF model is saved at "1. Data Preprocessing/YOLO-FF Model for Object Detection/YOLO-FF_train/weights/YOLO-FF.pt"

# 2. Baselines # 

This folder is including the baseline models developed.
Three baseline models are:

## Custom ViT ##
To run py .2. Baselines/custom_vit.py

## CNN ##
To run py .2. Baselines/image_with_seg_cnn.py

## VIT ##
To run py .2. Baselines/image_with_segmentation_vit.py

# 3. Proposed Anomaly Detection Model #

This folder includes the models for the proposed approach.

## EfficentNet ##
To run py .3. Proposed Anomaly Detection Model/image_with_segmentation_pretrainedcnn.py
The best model is saved at "3. Proposed Anomaly Detection Model/efficientnet_model.pth"