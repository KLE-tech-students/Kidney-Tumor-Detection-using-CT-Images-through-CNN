# Kidney Tumor Detection using CT Images through CNN

## Overview
This project aims to develop a convolutional Neural Network (CNN) model for detecting kidney tumors in Computed Tomography (CT) images. The model utilizes deep learning techniques to analyze medical images and identify potential abnormalities indicative of kidney tumors.

## Introduction
Kidney tumor is a significant health concern worldwide, and early detection plays a crucial role in improving patient outcomes. Medical imaging techniques, such as computed tomography (CT) is commonly used for the diagnosis and monitoring of kidney tumors. However, manual interpretation of these images by healthcare professionals is time-consuming and subject to human error. CNNs have demonstrated remarkable success in image recognition tasks, making them well-suited for medical image analysis. The goal of this project is to develop a CNN-based model capable of automatically identifying kidney tumors in medical images, thereby aiding healthcare professionals in diagnosis.

## Dataset Description
The dataset was collected from PACS (Picture archiving and communication system) from different hospitals in Dhaka, Bangladesh where patients were already diagnosed with having a kidney tumor, cyst, normal or stone findings. Both the Coronal and Axial cuts were selected from both contrast and non-contrast studies with protocol for the whole abdomen and urogram.

1. Kidney -Tumours: 2283 Images

2. Normal: 5077 image

   [Link to the dataset](https://www.kaggle.com/code/osinachichibuor/kidney-diseases-0-999) 

## Dependencies
+ TensorFlow
+ NumPy
+ Pandas
+ Keras
+ OpenCv
