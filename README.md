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

# 4. AssemAI Interface #

This folder includes the codes and supplementary files on model deployment.


# 5. Future Factories Setup #
Below figure includes an image of a rocket assembled by Future Factories Lab. 

![A rocket Assembled by the Future Factories Lab](/Users/chathurangishyalika/AssemAI/rocket.png "A rocket Assembled by the Future Factories Lab")

This is a rocket Assembled by the Future Factories Lab. Any missing part is considered an anomaly: for example, the absence of Rocket body 1 is labeled as ”NoBody1,” while the absence of both Rocket body 1 and body 2 is labeled as ”NoBody2, NoBody1.”


Some visual representations of the lab setup are included in below Figure.

![FF assembly cell](/Users/chathurangishyalika/AssemAI/assembly.png "FF assembly cell")


Some images from the FF assembly cell. The top image (Image I) is for cycle state four and bottom image (Image II) represents cycle state nine