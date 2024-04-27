# Kidney Tumor Detection using CT Images through CNN

## Overview
This project aims to develop a convolutional Neural Network (CNN) model for detecting kidney tumors in Computed Tomography (CT) images. The model utilizes deep learning techniques to analyze medical images and identify potential abnormalities indicative of kidney tumors.

## Table of Contents
+ [Introduction](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#introduction)
+ [Dataset Description](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#dependencies)
+ [Dependencies](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#dependencies)
+ [Implementation Steps](https://github.com/KLE-tech-students/Kidney-Tumor-Detection-using-CT-Images-through-CNN?tab=readme-ov-file#implementation-steps)




## Introduction
Kidney tumor is a significant health concern worldwide, and early detection plays a crucial role in improving patient outcomes. Medical imaging techniques, such as computed tomography (CT) is commonly used for the diagnosis and monitoring of kidney tumors. However, manual interpretation of these images by healthcare professionals is time-consuming and subject to human error. CNNs have demonstrated remarkable success in image recognition tasks, making them well-suited for medical image analysis. The goal of this project is to develop a CNN-based model capable of automatically identifying kidney tumors in medical images, thereby aiding healthcare professionals in diagnosis.

## Dataset Description
The dataset used in this project was obtained from Kaggle, a platform for data science competitions and datasets.
The dataset was collected from PACS (Picture Archiving and Communication System) from different hospitals in Dhaka, Bangladesh, where patients were already diagnosed with having a kidney tumor, cyst, normal or stone findings. Since this project is mainly focusing on the detection of kidney tumors we are going to consider only the kidney tumor and normal classes of the dataset. The dataset is of 764 MB in size and the images present in it are of '.jpg' format.
- The dataset consists two classes of images. One class consisting the images of kidneys with tumors and another class consisting the images of normal kidneys.
1. Kidney-Tumors: 2283 Images

2. Normal: 5077 Images

   [Link to the dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

**Sample images**
</br><img src="/Images/Picture1.png" alt="Logo" width="500"/>



## Dependencies
+ TensorFlow
+ NumPy
+ Pandas
+ Keras
+ OpenCv

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
