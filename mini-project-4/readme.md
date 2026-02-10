# COMP 9130 – Mini Project 4  
## Deep Learning Classifier for Fashion Product Categorization

This project implements a deep learning image classification system for **StyleSort**, an online fashion retailer.  
The goal is to automatically categorize product images into **10 clothing categories** using **PyTorch**, while providing confidence scores to support human-in-the-loop review.

The project uses the **Fashion-MNIST** dataset and compares multiple neural network architectures, including MLPs and a CNN, to evaluate performance.

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

## Team Members

- **Tanishq Rawat**  
- **Nicky Cheng**

---

## Notes

- This project uses **PyTorch `nn.Module`** (not Keras or TensorFlow).
- A **custom training loop** is implemented.
- The notebook orchestrates experiments and analysis, while core logic is modularized in the `src/` directory.
- Detailed methodology, results, and business analysis are provided in the submitted report.
