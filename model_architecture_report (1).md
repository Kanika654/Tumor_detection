# Medical Image Classification: cGAN-Balanced Architecture Specification

This document details the architectural design of the machine learning pipeline used for medical image classification (BraTS2020 / Lungs). The architecture heavily utilizes a Conditional Generative Adversarial Network (cGAN) to handle class imbalance, combined with deep transfer learning using ResNet/ViT architectures to maintain **90%+ predictive accuracy**.

---

## 1. Pipeline Overview

The pipeline consists of three major phases:
1.  **Baseline Training**: A baseline classifier is trained on the naturally imbalanced dataset.
2.  **cGAN Augmentation**: A Conditional GAN is trained to synthesize realistic medical images corresponding specifically to the minority class.
3.  **Balanced Fine-Tuning**: A fresh deep learning classifier utilizes ImageNet pre-trained weights and is fine-tuned on the merged dataset (real + synthetically generated minority class images) with specialized augmentations to prevent overfitting to GAN artifacts.

---

## 2. Generator Architecture (cGAN)

The Generator synthesizes artificial medical images from random Gaussian noise, conditioned on a specific class label (e.g., Tumor Positive).

- **Inputs**: 
  - `z` vector (Gaussian noise): Dimension 100
  - `label`: Class condition (0 or 1)
- **Embedding Space**: 
  - The class label is mapped to an embedding and concatenated with the noise vector.
- **Network Layers**:
  - **Dense Layer**: Projects the concatenated input into a spatial block of `128 x (img_size/4) x (img_size/4)`.
  - **Upsampling Block 1**: 
    - `nn.Upsample(scale_factor=2)`
    - `nn.Conv2d(128, 128, kernel_size=3, padding=1)`
    - `nn.BatchNorm2d(128)` + `nn.ReLU()`
  - **Upsampling Block 2**: 
    - `nn.Upsample(scale_factor=2)`
    - `nn.Conv2d(128, 64, kernel_size=3, padding=1)`
    - `nn.BatchNorm2d(64)` + `nn.ReLU()`
  - **Output Layer**: 
    - `nn.Conv2d(64, img_channels, kernel_size=3, padding=1)`
    - `nn.Tanh()` activation mapping pixels to the range `[-1, 1]`.

---

## 3. Discriminator Architecture (cGAN)

The Discriminator acts as a critique, evaluating whether an image is real (from the dataset) or fake (from the Generator), taking the class label into account.

- **Inputs**: 
  - `img`: Tensor of shape `(channels, img_size, img_size)`
  - `label`: Corresponding class condition
- **Label Integration**: 
  - The class label is passed through an embedding layer, expanded to the spatial dimensions of the image, and concatenated as an extra channel map.
- **Network Layers (4 Downsampling Blocks)**:
  - Each block utilizes:
    - `nn.Conv2d(in, out, kernel_size=3, stride=2, padding=1)`
    - `nn.LeakyReLU(0.2)`
    - `nn.Dropout2d(0.25)`
    - `nn.BatchNorm2d()` (Skipped on the first block to preserve initial spatial statistics)
  - The filter sequence progresses as: `(img_channels + num_classes) -> 32 -> 64 -> 128 -> 256`
- **Output Layer**: 
  - Flattened projection through an `nn.Linear` layer.
  - `nn.Sigmoid()` activation to output a probability between `0` (Fake) and `1` (Real).

---

## 4. Balanced Classifier Architecture (ResNet-18)

To achieve the targeted **90%+ accuracy**, we utilize deep transfer learning via a modified ResNet-18.

- **Base Architecture**: `torchvision.models.resnet18`
- **Pre-trained Weights**: initialized with `ResNet18_Weights.DEFAULT` (ImageNet).
- **Modifications**:
  - **Input Conv1 Layer**: The standard 3-channel input layer is dynamically replaced with an `nn.Conv2d` layer scaled to accept `img_channels` (e.g., 4 channels for BraTS sequences). The pre-trained weights from the first 3 channels are preserved, while the 4th channel is initialized with the mean of the RGB weights.
  - **Classification Head**: The default 1000-class fully connected layer is replaced:
    - `nn.Linear(num_features, 1)`
    - `nn.Sigmoid()` outputting a probability of the positive diagnosis.

---

## 5. System Optimizations for 90% Accuracy

To prevent the `cGAN Balanced` classifier from underperforming and ensure it exceeds a strict 90% accuracy boundary, three core optimizations are enforced within the pipeline:

### A. Prevention of Synthetic Artifact Overfitting
The classifier will naturally "cheat" by memorizing GAN-induced artifacts in the synthesized minority images. 
- **Solution**: The `MergedDataset` class explicitly wraps all synthetic (`synth_images`) in randomized geometric transformations (`transforms.RandomHorizontalFlip()`, `transforms.RandomRotation(10)`). This forces the ResNet to ignore GAN noise and learn generalized anatomical features.

### B. Anti-Catastrophic Forgetting
When fine-tuning a pre-trained ImageNet model on a merged medical dataset, standard learning rates will completely destroy the robust initialized weights.
- **Solution**: The `Adam` optimizer is restricted to a very slow learning rate of `1e-4` (down from 1e-3).
- **Solution**: `weight_decay=1e-4` is applied for strict L2 regularization.

### C. Learning Rate Plateau Scheduling
- **Solution**: An `lr_scheduler.ReduceLROnPlateau` tracks `avg_loss` and halves the learning rate (`factor=0.5`) whenever training plateaus, ensuring the model flawlessly converges on the complex boundary between synthetic and real spatial features.
