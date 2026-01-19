# Intel Natural Scenes Image Classification (Vanilla CNN vs. AlexNet Transfer Learning)

This project trains and compares:
- **custom “Vanilla” CNN** built from scratch, and **pretrained AlexNet (ImageNet) fine-tuned** for the Intel Natural Scenes dataset, to classify images into **6 scene categories**: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Data Loading & Transforms](#data-loading--transforms)
- [Modeling Approach](#modeling-approach)
  - [1) Vanilla CNN](#1-vanilla-cnn)
  - [2) Transfer Learning: AlexNet](#2-transfer-learning-alexnet)
  - [3) Transfer Learning: AlexNet (2-layer unfreeze)](#3-transfer-learning-alexnet-2-layer-unfreeze)
- [Training](#training)
- [Results](#results)

---

## Project Overview

Goal: **fine-tune a neural network on a new dataset** for multi-class image classification.

We implement a full training pipeline in PyTorch:
- dataset download & extraction
- transforms + DataLoaders
- model training and evaluation
- experiment tracking with **Weights & Biases (wandb)**

---

## Motivation

Why compare these approaches?

- **Vanilla CNN** is a strong baseline to validate the pipeline and build intuition about capacity vs. generalization.
- **Transfer learning** (AlexNet pretrained on ImageNet) typically converges faster and achieves higher accuracy on small/medium datasets by leveraging learned visual features (edges → textures → object parts).

---

## Dataset

[**Intel Image Classification (Natural Scenes)**](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

Directory layout (as used in this project):
- `seg_train/seg_train/` — training images
- `seg_test/seg_test/` — test images
- `seg_pred/seg_pred/` — unlabeled images (optional, for inference)

Classes (6):
- `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

Image size:
- Images are resized to **150×150** during training.

---

## Tech Stack

- **PyTorch / torchvision** — modeling & pretrained AlexNet
- **NumPy**, **matplotlib** — utilities & visualization
- **scikit-learn** — accuracy metric
- **Weights & Biases (wandb)** — experiment tracking


---

## Data Loading & Transforms

Transforms used:
- `Resize((150, 150))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet-style normalization)

DataLoaders:
- Train loader: shuffled, batch size
- Test loader: not shuffled

---

## Modeling Approach

### 1) Vanilla CNN

A compact ConvNet:
- 3 convolution blocks (Conv → ReLU), with 2 MaxPool layers
- Flatten → FC(256) → FC(num_classes)

Target classes: **6**.

---

### 2) Transfer Learning: AlexNet

We load **AlexNet pretrained on ImageNet** and replace the final classifier layer:

- Replace `classifier[6]` with `Linear(4096 → 6)`
- Freeze **all layers** except the classifier head
- Train a few epochs (fast convergence)

This is the classic “feature extractor + new head” approach.

---

### 3) Transfer Learning: AlexNet (2-layer unfreeze)

Variant: train more capacity by unfreezing **two** classifier layers:
- unfreeze `classifier[4]` and `classifier[6]`
- keep the rest frozen

This increases trainable parameters and can overfit if the dataset is not large enough or regularization/augmentation is weak.

---

## Training

Optimization setup:
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `1e-4`
- Epochs:
  - Vanilla CNN: 20
  - AlexNet variants: 5
- Logging:
  - batch-level and epoch-level metrics tracked with **wandb**

---

## Results

Test accuracy (single-batch quick check, as implemented in the notebook):

| Model | Train Accuracy | Test Accuracy |
|------|---------------:|--------------:|
| Vanilla CNN | ~0.7608 | **0.75** |
| AlexNet (freeze backbone, train head) | ~0.9555 | **0.84375** |
| AlexNet (unfreeze 2 classifier layers) | ~0.9426 | **0.78125** |

Key takeaway:
- **AlexNet transfer learning performs best** on test (0.84375) despite fewer epochs.
- Unfreezing extra layers improved train accuracy but **hurt test accuracy**, consistent with overfitting.

Project's experiments are stored in [wandb](https://wandb.ai/vonexel0/Intel%20Image%20Classification%20Vailla-ConvNet%20vs%20AlexNet?nw=nwuservonexel)




