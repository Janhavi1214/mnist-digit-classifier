# 🔢 MNIST Digit Classifier

A simple feedforward neural network built with TensorFlow/Keras to classify
handwritten digits from the MNIST dataset with ~98% test accuracy.

## 📌 Overview

This project trains a fully connected neural network on the classic MNIST dataset
— 70,000 grayscale images of handwritten digits (0–9) — and evaluates its
performance on unseen test data.

## 🧠 Model Architecture

| Layer  | Type    | Units | Activation |
|--------|---------|-------|------------|
| Input  | Flatten | 784   | —          |
| Hidden | Dense   | 128   | ReLU       |
| Output | Dense   | 10    | Softmax    |

- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Metric:** Accuracy

- ## 📉 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The matrix shows per-digit classification performance across all 10,000 test samples.

## 📁 Project Structure

mnist-digit-classifier/
├── mnist_classifier.py   # Main training & evaluation script
├── requirements.txt      # Dependencies
└── README.md

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

# Clone the repository
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier

# Install dependencies
pip install -r requirements.txt

### Run

python mnist_classifier.py

## 📊 Results

| Metric         | Value   |
|----------------|---------|
| Test Accuracy  | ~98%    |
| Epochs Trained | 5       |
| Training Size  | 60,000  |
| Test Size      | 10,000  |

## 📦 Requirements

tensorflow>=2.x
numpy

## 📄 License

MIT License — feel free to use and modify.
```

---

## `requirements.txt`
```
tensorflow>=2.10.0
numpy>=1.21.0
