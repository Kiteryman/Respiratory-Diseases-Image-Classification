# Respiratory Diseases Radiography Classification with Improved ResNet-34

This repository contains a high-performance deep learning pipeline for classifying radiographic images into four categories: **COVID-19**, **Lung Opacity**, **Normal**, and **Viral Pneumonia**. The model utilizes a custom **Improved ResNet-34** architecture incorporating GeM Pooling, Squeeze-and-Excitation (SE) blocks, and advanced data augmentation.

## 📂 Data Attribution

The dataset used for this project is the **Balanced Augmented COVID CXR Dataset**, curated and provided by **Mrinal Tyagi** via Kaggle.

*   **Dataset Link:** [Balanced Augmented COVID CXR Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/balanced-augmented-covid-cxr-dataset)
*   **Description:** This dataset is a balanced and augmented version of the original Chest X-Ray dataset, designed to address class imbalance and improve model generalization for medical imaging tasks.

---

## 🚀 Key Features

*   **Architecture**: ResNet-34 backbone modified with:
    *   **GeM (Generalized Mean) Pooling**: Superior feature extraction for medical textures compared to standard Global Average Pooling.
    *   **SE (Squeeze-and-Excitation) Block**: Channel-wise attention mechanism to focus on critical radiographic features.
    *   **SiLU Activation**: Smoother non-linearity for better gradient flow.
*   **Training Pipeline**:
    *   **MixUp & CutMix**: Regularization techniques to improve generalization.
    *   **OneCycleLR**: Optimized learning rate scheduling for faster convergence.
    *   **Automatic Mixed Precision (AMP)**: Faster training and lower memory footprint.
    *   **TenCrop Evaluation**: Test-time augmentation (TTA) for robust inference.

---

## 📊 Performance Metrics

Based on the test set evaluation of **6,004 samples**, the model achieves an **Overall Accuracy of 97.75%**.

### Confusion Matrix Analysis

| Class | Samples | Correct | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **COVID** | 1750 | 1721 | 99.59% | 98.34% | 98.96% |
| **Lung Opacity** | 1536 | 1469 | 98.00% | 95.64% | 96.81% |
| **Normal** | 1644 | 1611 | 94.49% | 97.99% | 96.21% |
| **Viral Pneumonia** | 1074 | 1068 | 99.63% | 99.44% | 99.53% |
| **TOTAL / AVG** | **6004** | **5869** | **97.93%** | **97.75%** | **97.83%** |

---

## 🛠️ Installation & Usage

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Kiteryman/Respiratory-Diseases-Image-Classification.git
   cd Respiratory-Diseases-Image-Classification
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tr1gg3rtrash/balanced-augmented-covid-cxr-dataset) and organize it:
   ```text
   Output/
   ├── train/
   ├── val/
   └── test/
   ```

### Training & Testing
The script handles both training (with checkpoint resumption) and final evaluation:
```bash
python code.py
```
