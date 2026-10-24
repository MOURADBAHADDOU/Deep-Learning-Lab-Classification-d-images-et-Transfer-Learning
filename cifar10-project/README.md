# CIFAR-10 Model Studio

An interactive deep learning project for image classification using the CIFAR-10 dataset. This project compares five neural network models and provides a Streamlit application to explore their predictions.

## Project Overview

The goal of this project is to explore image classification with deep learning, compare different model architectures, and apply transfer learning using ResNet50 pretrained on ImageNet.

## Models

Five models are trained and evaluated:

- **MLP** — Multilayer Perceptron baseline
- **CNN A** — Convolutional Neural Network baseline
- **CNN B** — Deeper CNN architecture
- **CNN C** — CNN with Batch Normalization and data augmentation
- **ResNet50** — Transfer learning using ImageNet pretrained weights

## Test Results

| Model | Test Accuracy |
|---|---:|
| MLP | 43.36% |
| CNN A | 72.26% |
| CNN B | 77.33% |
| CNN C | 81.91% |
| ResNet50 | 90.17% |

ResNet50 achieved the highest test accuracy among the five evaluated models in our experiments.

## Application Features

- Upload an image and obtain a predicted class
- Select a model or compare predictions across models
- Display predicted classes and confidence scores
- Compare the test accuracy of the five models

## Dataset

CIFAR-10 contains 60,000 color images divided into 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The dataset is divided into 50,000 training images and 10,000 test images.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Project Structure

```text
cifar10-model-studio/
├── app.py
├── cifar10_model_training.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/MOURADBAHADDOU/cifar10-model-studio.git
cd cifar10-model-studio
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

**Important:** The trained model files must be available at the locations expected by `app.py`. They are not included in this repository by default. The application will need access to those files to run predictions.

## What I Learned

- Image preprocessing and data preparation
- Building and training neural networks with TensorFlow/Keras
- Comparing MLP and CNN architectures
- Using Batch Normalization and data augmentation
- Applying transfer learning with ResNet50
- Evaluating classification models on a test dataset
- Building an interactive application with Streamlit

## Future Improvements

- Add model performance visualizations
- Improve the deployment workflow
- Provide access to trained model weights
- Add more evaluation metrics and error analysis

---

Developed as a deep learning portfolio project focused on image classification and model comparison.
