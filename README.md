# DNA Sequence Classification Using a Simple CNN
A toy deep learning project demonstrating CNN-based classification of DNA sequences using synthetic motif data.

### A lightweight deep learning project for sequence-based prediction from DNA

---

## Overview

This project demonstrates a minimal deep learning workflow for DNA sequence classification using a convolutional neural network (CNN).

Synthetic DNA sequences are generated, with a subset containing a regulatory motif (e.g., **TATAAA**). The model is trained to classify whether a given sequence contains the motif.

Although this is a toy dataset, the workflow reflects core ideas used in modern functional genomics and regulatory sequence modeling.

---

## Motivation

This project was designed to demonstrate:

- Deep learning applied to biological sequences  
- Sequence-based feature learning using CNNs  
- One-hot encoding of DNA sequences  
- Binary classification and model evaluation  
- A simplified version of regulatory motif detection  

These concepts are directly relevant to problems such as:

- Sequence determinants of gene expression  
- Regulatory element prediction  
- Variant effect prediction  

---

## Dataset

- Synthetic DNA sequences (length = 100 bp)
- Motif inserted randomly in ~50% of sequences
- Labels:
  - `1` → motif present  
  - `0` → motif absent  

This controlled setup allows clear demonstration of model behavior.

---

## Methodology

### 1. Data Generation
- Random DNA sequences generated from {A, C, G, T}
- Motif inserted at random positions

### 2. Encoding
Each nucleotide is one-hot encoded:

| Base | Encoding |
|------|--------|
| A    | [1, 0, 0, 0] |
| C    | [0, 1, 0, 0] |
| G    | [0, 0, 1, 0] |
| T    | [0, 0, 0, 1] |

### 3. Model Architecture
The convolutional layer learns sequence patterns similar to motif recognition.
A simple 1D CNN architecture is used: Conv1D → ReLU → MaxPooling → Fully Connected → Sigmoid output.

---

## Results
The model demonstrates the ability to learn sequence patterns associated with motif presence, although performance is modest due to the simplicity of the dataset and model.

### Generated outputs:

- Training loss curve  
- Training accuracy curve  
- Confusion matrix  
- ROC curve  

Example metrics (from one run):

- Accuracy: ~0.60  
- ROC-AUC: ~0.64

Note: Performance varies due to the synthetic dataset and simple model architecture. This project is intended as a proof-of-concept demonstrating the workflow rather than achieving optimal predictive performance.

---

## Example Outputs

All visualizations are saved in the `outputs/` directory:

outputs/
├── training_loss_curve.png
├── training_accuracy_curve.png
├── confusion_matrix.png
└── roc_curve.png

---

## Limitations

- Uses synthetic data rather than real genomic datasets  
- Motif detection task is simplified  
- Model performance is limited due to small dataset and simple architecture  

This project is intended as a minimal demonstration of a deep learning workflow for sequence data.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
