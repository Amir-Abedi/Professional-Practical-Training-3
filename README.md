# Edge Detection on BSDS500

U-Net decoder with swappable pretrained backbones, comparing **RGB-only** input against **RGB + multi-level Canny** maps.

## Notebooks

| Notebook | Input | Description |
|:--|:--|:--|
| `edge_unet_backbones_canny.ipynb` | RGB + Canny maps (4 levels) | Canny maps are fed as extra input channels and added to the output logits through a learned 1×1 skip, so the network starts from Canny and learns a correction. |
| `edge_unet_backbones_rgb.ipynb` | RGB only | Baseline with the same decoder, loss, schedule and metric. |

Only the input differs between the two notebooks. In each one, only the encoder changes between runs.

## Backbones

Set `BACKBONES` in the setup cell. Any [timm](https://github.com/huggingface/pytorch-image-models) model with 5 feature stages works, for example:

`resnet34` · `resnet50` · `mobilenetv2_100` · `efficientnet_b0` · `vgg16_bn` / `vgg19_bn` (torchvision encoder split at its max-pools)

## Method

- **Model:** pretrained encoder (5 stages, strides 2 to 32) + U-Net decoder with widths (256, 128, 64, 32, 16); the last step also uses the raw input.
- **Loss:** class-balanced BCE + 0.5 × Dice.
- **Optimiser:** AdamW, one-cycle schedule (cosine decay), 150 epochs, batch size 16, 320 px random crops with 90° rotations and flips.
- **Data:** official BSDS500 split: train 200 / val 100 / test 200. Ground truth is the mean of all annotators, thresholded at 0.5.
- **Model selection:** best epoch and threshold are chosen on val; the test set is scored once.
- **Metric:** F-score with 2 px tolerance after thinning (precision, recall and F per image).
- **Protocol:** 5 seeds per backbone, reported as mean ± std.

## Outputs

- Per-seed results and training curves
- Qualitative predictions next to the image, human edges and Canny baseline
- One test F-score table over all backbones (with the Canny baseline), plus a paired comparison against a reference run
- Parameters, GFLOPs and inference time on the test set
- `results_<run>.json` per backbone, so runs from different sessions can be combined in the final table

## Ablations

One-line changes in the setup cell:

```python
CANNY_SIGMAS = ()      # RGB only
CANNY_SKIP = False     # Canny as input only, no skip to the logits
```

## Usage

1. Open a notebook on Kaggle (it downloads the dataset with `kagglehub`: `dorisdan/bsds500`).
2. Set `BACKBONES` and run all cells.
3. The last cells print the comparison and inference-time tables.

Requirements: `torch`, `torchvision`, `timm`, `opencv-python`, `scikit-image`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `kagglehub`.

## Results

| Backbone | Test F (RGB) | Test F (RGB + Canny) |
|:--|:--:|:--:|
| resnet34 | | |
| resnet50 | | |

## Dataset

[BSDS500](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) (Arbelaez et al., *Contour Detection and Hierarchical Image Segmentation*, TPAMI 2011).
