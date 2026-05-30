# Car-Image-Classifier-Using-Deep-Learning-and-Streamlit
This project applies Deep Learning and Computer Vision to classify vehicle images. Built with TensorFlow and Streamlit, it provides real-time car recognition, confidence scores, and interactive visualizations. The system offers an accurate, scalable, and user-friendly solution for intelligent vehicle identification and analysis.-
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f1a70087-e9a6-4960-bdf1-a53685619ead" />
## Problem Statement

Manual identification of vehicle types from images is time-consuming, error-prone, and not scalable for modern intelligent systems such as traffic monitoring, smart parking, and automated surveillance. There is a need for an automated, accurate, and efficient system capable of classifying vehicle images into predefined categories using computer vision techniques.

---

## Proposed Solution

To address this problem, a deep learning-based image classification system is developed using a Convolutional Neural Network (CNN). The system automatically processes input vehicle images, extracts relevant visual features, and predicts the corresponding vehicle category in real time.

A Streamlit-based web application is integrated to provide an interactive interface where users can upload images and receive instant predictions along with confidence scores and top-3 class probabilities.

---

## Model Used

The solution is built using a Convolutional Neural Network (CNN), implemented with TensorFlow and Keras.

### Model Pipeline:

* Input: Vehicle images
* Preprocessing: Resizing, normalization, and format standardization
* Feature Extraction: Convolutional layers for spatial feature learning
* Dimensionality Reduction: Pooling layers
* Classification: Fully connected dense layers
* Output: Softmax layer producing probability distribution across 7 classes

### Output Classes:

* Audi
* Hyundai Creta
* Mahindra Scorpio
* Rolls Royce
* Swift
* Tata Safari
* Toyota Innova

The model outputs the highest probability class as the final prediction along with confidence scores for interpretability.

---

## System Outcome

The final system provides a fast, scalable, and accurate solution for vehicle image classification. It demonstrates the effectiveness of CNN-based architectures for real-world computer vision tasks and integrates machine learning with a user-friendly deployment interface for practical usability.
