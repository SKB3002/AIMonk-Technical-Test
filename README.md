# AIMonk Multilabel Image Classification – Technical Assignment

## 🔍 Problem Statement

This project addresses a multilabel image classification problem where each image may contain multiple attributes. The dataset includes missing labels (NA values) and exhibits class imbalance across attributes.

## 🧠 Approach Overview

Framework: TensorFlow (Keras)

Backbone: ImageNet-pretrained ResNet50

Task: Multi-label classification (4 attributes)

Output: Sigmoid-based multilabel predictions

The pretrained backbone is fine-tuned for the target dataset, avoiding training from scratch.

## 🏗 Model Architecture

ResNet50 (pretrained on ImageNet, include_top=False)

Global Average Pooling

Dense classification head with 4 logits (one per attribute)

## 🔧 Data Preprocessing

Image resizing to 224×224

ImageNet normalization

Handling missing images by filtering invalid file paths

## ⚠️ Handling Missing Labels (NA)

Some samples contain missing annotations (NA).
To avoid discarding such samples, a masked binary cross-entropy loss was implemented, which:

Includes all samples during training

Ignores NA positions in loss computation

This allows the model to learn from partially labeled data without introducing incorrect gradients.

## ⚖️ Handling Class Imbalance

The dataset is skewed across attributes.
To address this:

Per-label positive class weights were computed

Weighted binary cross-entropy was used to penalize minority class errors more strongly

This improves recall on underrepresented attributes.

## 📈 Training Strategy

Optimizer: Adam

Initial learning rate: 1e-4

Learning rate decay: ReduceLROnPlateau

Early stopping on validation loss

This stabilized training and improved generalization.

Best validation performance achieved:

F1-score ≈ 0.71

Accuracy ≈ 0.71

## 🧪 Experiments & Observations

Several techniques were explored:

Technique	Effect on F1
Baseline fine-tuning	~0.68
Data augmentation	No significant change
Two-phase fine-tuning	Degraded performance
LR decay + early stopping	Improved to ~0.71

This suggests the primary bottleneck is dataset size and label noise rather than model capacity.

## 📊 Evaluation

The model was evaluated on a held-out validation set.
Metrics reported:

| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0.0** | 0.73 | 0.64 | 0.68 | 378 |
| **Class 1.0** | 0.70 | 0.78 | 0.74 | 402 |
| **Accuracy** | | | **0.71** | **780** |
| **Macro Avg** | 0.71 | 0.71 | 0.71 | 780 |
| **Weighted Avg** | 0.71 | 0.71 | 0.71 | 780 |

Validation F1 ≈ 0.71, indicating reasonable generalization given class imbalance and missing labels.

## 🚀 Inference

An inference script is provided to load the trained model and predict attributes for a single input image.
