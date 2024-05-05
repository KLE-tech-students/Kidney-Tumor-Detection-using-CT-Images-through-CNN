# Kidney Tumor Detection using CT Images through CNN

## Overview
This project aims to develop a Convolutional Neural Network (CNN) model for detecting kidney tumors in Computed Tomography (CT) images. The model utilizes deep learning techniques to analyze medical images and identify potential abnormalities indicative of kidney tumors.

## Table of Contents
+ [Introduction](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#introduction)
+ [Dataset Description](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#dependencies)
+ [Dependencies](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#dependencies)
+ [Implementation Steps](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#implementation-steps)




## Introduction
Kidney tumor is a significant health concern worldwide, and early detection plays a crucial role in improving patient outcomes. Medical imaging techniques, such as Computed Tomography (CT) is commonly used for the diagnosis and monitoring of kidney tumors. However, manual interpretation of these images by healthcare professionals is time-consuming and subject to human error. CNNs have demonstrated remarkable success in image recognition tasks, making them well-suited for medical image analysis. The goal of this project is to develop a CNN-based model capable of automatically identifying kidney tumors in medical images, thereby aiding healthcare professionals in diagnosis.

## Dataset Description
The dataset used in this project was obtained from Kaggle, a platform for data science competitions and datasets.
The dataset was collected from PACS (Picture Archiving and Communication System) from different hospitals in Dhaka, Bangladesh, where patients were already diagnosed with having a kidney tumor, cyst, normal or stone findings. Since this project is mainly focusing on the detection of kidney tumors we are going to consider only the kidney tumor and normal classes of the dataset. The dataset is of 764 MB in size and the images present in it are of '.jpg' format.
- The dataset consists two classes of images. One class consisting the images of kidneys with tumors and another class consisting the images of normal kidneys.
1. Kidney-Tumors: 2283 Images

2. Normal: 5077 Images

   [Link to the dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

**Sample images**
</br></br>
Kidney Normal images
</br><img src="/Images/Kidney_normal.png" alt="Logo" width="400"/>
</br></br>
Kidney Tumor images
</br><img src="/Images/Kidney_tumor.png" alt="Logo" width="400"/>

## Implementation Steps
To replicate and build upon this project, follow these implementation steps:
1. **Data Loading**:
   - Download the dataset from the provided link.
   - Load the dataset into project environment.
2. **Data Preprocessing**:
   - Preprocess the images, including resizing, normalization, and augmentation.
3. **Model Development**:
   - Design and implement the CNN architecture.
   - Train the model using the preprocessed dataset.
4. **Model Evaluation**:
   - Evaluate the trained model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Model
We have developed two distinct models for detecting kidney tumors: Convolutional Neural Network (CNN) and ResNet (Residual Neural Network). These models leverage state-of-the-art deep learning techniques to analyze CT images and identify potential tumor regions.
### Convolutional Neural Network(CNN)
The dataset is trained with CNN architecture which consists of several convolutional layers followed by max-pooling layers. The final layers include fully connected (dense) layers with dropout regularization to prevent overfitting. \
![Screenshot 2024-04-28 120343](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN/assets/105357853/661f5edf-9de7-4a39-aeb1-ff817698dedd)

Convolutional Layers: \
Layer 1 (Conv2D): \
Filters: 32 \
Filter Size: 3x3 \
<br>
Layer 2 (Conv2D): \
Filters: 64 \
Filter Size: 3x3 \
<br>
Layer 3 (Conv2D): \
Filters: 64 \
Filter Size: 3x3 \
<br>
Layer 4 (Conv2D): \
Filters: 64 \
Filter Size: 3x3 \
<br>
Layer 5 (Conv2D): \
Filters: 128 \
Filter Size: 3x3 \
<br>
MaxPooling Layers: \
Pool Size: 2x2 applied after each Conv2D layer. \
<br>
Additional Layers: \
Flatten Layer: Converts the 2D feature maps into a 1D vector. \
Dropout Layer: Regularization with dropout rate of 0.5. \
Dense Layers: \
Layer 1 (Dense): 128 neurons with ReLU activation. \
Output Layer (Dense): 2 neurons with softmax activation for binary classification. 


###  Residual Neural Network (ResNet50)
Input (512x512x2)  \
       |  \
    ResNet50 \
       |  \
 GlobalAveragePooling2D  \
       |  \
      Dropout    
       | \
      Dense (128, relu)  \
       | \
      Dense (2, softmax)  
       |   
    Output   \
    <br>
ResNet50: This is the pre-trained ResNet50 model with weights frozen. \
GlobalAveragePooling2D: Averages the spatial dimensions of the ResNet50 output to create a fixed-length vector. \
Dropout Layer: Helps in reducing overfitting by randomly dropping a fraction of the neurons during training. \
Dense Layer(128, relu): Fully connected layer with 128 units and ReLU activation function. \
Dense Layer(2, softmax): Final fully connected layer with 2 units (one for each class) and softmax activation function for classification. \
Output: Represents the final output with 2 classes (normal or tumor).






