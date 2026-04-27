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

---

## Results

The model successfully learns to detect the motif with high accuracy.

### Generated outputs:

- Training loss curve  
- Training accuracy curve  
- Confusion matrix  
- ROC curve  

Example metrics (will vary slightly per run):

- Accuracy: ~0.95–1.00  
- ROC-AUC: ~0.98–1.00  

---

## Example Outputs

All visualizations are saved in the `outputs/` directory:


---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
