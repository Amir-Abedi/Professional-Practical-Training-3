<div align="center">

# Edge Detection on BSDS500

**A U-Net decoder with interchangeable pretrained backbones, with and without a Canny skip connection**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-backbones-4B8BBE)
![Dataset](https://img.shields.io/badge/Dataset-BSDS500-5C7080)
![Platform](https://img.shields.io/badge/Kaggle-2×%20Tesla%20T4-20BEFF?logo=kaggle&logoColor=white)

</div>

---

## Overview

This project asks a simple question: **which encoder works best for edge detection when everything else is held fixed?**

Five ImageNet-pretrained backbones are plugged into the same U-Net decoder and trained on the official BSDS500 split. Classical Canny edge maps at five sensitivity levels are given to the network as extra input channels. Two notebooks then compare two ways of using them:

| Notebook | Canny maps | Canny skip to logits |
|:--|:--|:--:|
| `different-backbones-for-edge-detection_w__canny.ipynb` | extra input channels | ✔ |
| `different-backbones-for-edge-detection_w_o_canny.ipynb` | extra input channels | ✘ |

In the first notebook, the Canny maps are also added to the output logits through a learned 1×1 convolution initialised to 1, so the network starts from Canny and learns a correction. In the second, they are used as input only. Both notebooks are otherwise identical; inside each one, only the encoder changes between runs.

## Results

Test F-score on the 200 BSDS500 test images (2 px tolerance, mean over 3 seeds). The classical Canny baseline, with the best σ chosen on validation, scores **0.250**.

| Backbone | Params (M) | GFLOPs | With Canny skip | Without Canny skip |
|:--|--:|--:|:--:|:--:|
| **VGG16-BN** | 19.0 | 116 | 0.618 | **0.627** |
| ResNet-50 | 32.5 | 51 | 0.591 | 0.612 |
| ResNet-34 | 24.5 | 38 | 0.596 | 0.610 |
| EfficientNet-B0 | 5.8 | 14 | 0.570 | 0.598 |
| MobileNetV2 | 4.0 | 13 | 0.576 | 0.587 |

**Inference time** on the 200 test images (one image at a time, 2× Tesla T4, mixed precision):

| Backbone | ms / image (with skip) | ms / image (without skip) |
|:--|--:|--:|
| ResNet-34 | 11.4 | 12.9 |
| MobileNetV2 | 12.4 | 14.6 |
| ResNet-50 | 15.3 | 17.6 |
| EfficientNet-B0 | 16.2 | 18.9 |
| VGG16-BN | 18.9 | 20.2 |

**What the numbers show**

- VGG16-BN gives the best F-score in both settings, and ResNet-34 and ResNet-50 follow, while the two lightweight encoders score lowest.
- Without the Canny skip, the F-score is higher for all five backbones.
- ResNet-34 is the fastest model at inference. MobileNetV2 has the fewest parameters and FLOPs but is not the fastest here.
- Differences between runs are small (seed std between 0.001 and 0.028), so the paired per-image comparison in the last notebook cell is the better guide for close pairs.

## Method

| | |
|:--|:--|
| **Encoder** | `timm` backbone with 5 feature stages (strides 2 to 32), or torchvision VGG split at its max-pools |
| **Decoder** | U-Net, widths (256, 128, 64, 32, 16); the last step also uses the raw input |
| **Input** | 3 RGB channels + 5 Canny maps (σ = 0.25, 0.33, 0.5, 0.66, 0.75) |
| **Loss** | class-balanced BCE + 0.5 × Dice |
| **Optimiser** | AdamW, lr 2e-4, weight decay 1e-4, one-cycle schedule with cosine decay |
| **Training** | 100 epochs, batch size 32, 320 px random crops, 90° rotations and flips, mixed precision |
| **Data split** | official BSDS500: train 200 · val 100 · test 200 |
| **Ground truth** | mean of all annotators, thresholded at 0.5 |
| **Model selection** | best epoch and threshold chosen on val; test scored once |
| **Metric** | F-score with 2 px tolerance after thinning |
| **Seeds** | 3 per backbone, reported as mean ± std |

## Repository

```
├── different-backbones-for-edge-detection_w__canny.ipynb    # Canny input + Canny skip
├── different-backbones-for-edge-detection_w_o_canny.ipynb   # Canny input only
└── README.md
```

Each notebook walks through setup, data, model, training, per-seed curves, qualitative predictions, a final comparison table with a paired test, and inference time. Per-run results are written to `results_<run>.json`.

## Usage

1. Open a notebook on Kaggle with GPU enabled. The dataset is downloaded with `kagglehub` (`dorisdan/bsds500`).
2. Edit the setup cell if needed:

   ```python
   BACKBONES    = ("resnet34", "resnet50", "mobilenetv2_100", "efficientnet_b0", "vgg16_bn")
   CANNY_SIGMAS = (0.25, 0.33, 0.5, 0.66, 0.75)   # () = RGB only
   CANNY_SKIP   = True                            # False = Canny as input only
   ```

3. Run all cells. The last cells print the test F-score table and the inference-time table.

**Requirements:** `torch`, `torchvision`, `timm`, `opencv-python`, `scikit-image`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `kagglehub`.

## Reference

Arbelaez, P., Maire, M., Fowlkes, C., Malik, J. *Contour Detection and Hierarchical Image Segmentation.* IEEE TPAMI, 2011.
