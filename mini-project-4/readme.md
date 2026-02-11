# COMP 9130 – Mini Project 4  
## Deep Learning Classifier for Fashion Product Categorization

This project implements a deep learning image classification system for **StyleSort**, an online fashion retailer.  
The goal is to automatically categorize product images into **10 clothing categories** using **PyTorch**, while providing confidence scores to support human-in-the-loop review.

The project uses the **Fashion-MNIST** dataset and compares multiple neural network architectures, including MLPs and a CNN, to evaluate performance.

## Model Architecture
- Multi-Layer Perceptron (Baseline)
  - 2 linear layers
  - Activates with ReLU
  - Run for 10 epochs
- Multi-Layer Perceptron (Dropout and Batch Normalization)
  - 2 linear layers
  - Dropout rate of 0.3 after each layer
  - Batch normalization after each layer
  - Activates with Leaky ReLU
  - Runs for 12 epochs
- Convolutional Neural Network
  - 3 Convolutional 2D layers, 1 Linear layer to convert for output
  - Batch normalization after every layer
  - Pooling to half the size of the input
  - Dropout rate of 0.25
  - Runs for 10 epochs

Optimized used: AdamW

---

## Project Structure
```
mini-project-4/
├── README.md
├── requirements.txt
├── data/ # Dataset (auto-downloaded, gitignored)
├── notebooks/
│ └── fashion_classifier.ipynb
├── src/
│ ├── model.py # Neural network architectures (nn.Module)
│ ├── train.py # Custom training loop
│ └── utils.py # Helper functions (metrics, analysis)
└── results/
├── training_curves.png
├── confusion_matrix.png
├── confidence_threshold.png
└── misclassified_examples.png
```


---

## Setup Instructions

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

```

## Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Summary of Results
- Multi-Layer Perceptron (Baseline)
  - Accuracy after 10 epochs: 0.8764
  - Loss after 10 epochs: 0.3464
- Multi-Layer Perceptron (Dropout and Batch Normalization)
  - Accuracy after 10 epochs: 0.8842
  - Loss after 10 epochs: 0.3151
- Convolutional Neural Network
  - Accuracy after 10 epochs: 0.9236
  - Loss after 10 epochs: 0.2234
  - Cost-weighted accuracy: 0.9464

### Per-class Analysis
T-shirt/top   acc=0.811
Trouser       acc=0.983
Pullover      acc=0.827
Dress         acc=0.923
Coat          acc=0.918
Sandal        acc=0.984
Shirt         acc=0.809
Sneaker       acc=0.981
Bag           acc=0.986
Ankle boot    acc=0.962

Most confused pairs (top 10 off-diagonal):
T-shirt/top -> Shirt : 156
Pullover -> Coat : 94
Shirt -> Coat : 71
Pullover -> Shirt : 60
Shirt -> T-shirt/top : 49
Shirt -> Pullover : 44
Coat -> Shirt : 44
Dress -> Shirt : 38
Ankle boot -> Sneaker : 33
Dress -> Coat : 27

### Confidence Threshold
- % of items needing review with threshold of 0.8: 0.1177
- Accuracy on accepted: 0.9632

## Business Recommendations
We recommend for StyleSort to use a CNN model, focus on coat -> pullover and shirt -> t-shirt/top pairs, and use a confidence threshold of 0.70.

## Team Members

- **Tanishq Rawat**: Neural network design, results
- **Nicky Cheng**: Report, README

---

## Notes

- This project uses **PyTorch `nn.Module`** (not Keras or TensorFlow).
- A **custom training loop** is implemented.
- The notebook orchestrates experiments and analysis, while core logic is modularized in the `src/` directory.
- Detailed methodology, results, and business analysis are provided in the submitted report.
