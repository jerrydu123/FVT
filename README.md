# FVT: Feature-to-Video Transition Modeling for Semantic Change Detection

Repository specified in the paper: [jerrydu123/FVT](https://github.com/jerrydu123/FVT).

PyTorch training, evaluation, and inference code for FVT and FVT-Tiny, extracted from the original training implementation. The model combines a ResNet-34 encoder, an enhanced DINOv3 ViT-L/16 branch, bidirectional 3D temporal modeling, and semantic/change decoders.

Both **FVT** (Full) and **FVT-Tiny** are included. Use `--model full` (default) or `--model tiny` for training, evaluation, and prediction. Tiny uses DINOv3 ViT-S/16. The entry points automatically select the verified Tiny internal processing size: **512 × 512 for SECOND**, **384 × 384 for Landsat-SCD**. Both use **10 frames and linear feature interpolation**. Output maps retain the original image size. See [Tiny configuration and weights](docs/TINY.md).

Paper: **FVT: Feature-to-Video Transition Modeling for Semantic Change Detection**, IEEE JSTARS, 2026. DOI: [10.1109/JSTARS.2026.3727661](https://doi.org/10.1109/JSTARS.2026.3727661). Original-code MIT licensing is planned; final copyright attribution and the root LICENSE are deferred. Third-party materials retain their own licenses; see [notices](docs/THIRD_PARTY_NOTICES.md).

## Repository structure

```text
train.py                     Training and evaluation; single GPU / torchrun
predict.py                   Inference without ground-truth labels
models/FVT.py                FVT model and decoder modules
models/FVT_Tiny.py            FVT-Tiny; configurable internal processing size
models/factory.py             Model selection and dataset-specific Tiny settings
models/video.py              Bidirectional 3D temporal branch
models/video_Tiny.py          Tiny temporal branch used by the verified checkpoints
models/video_323.py           Sequence generator implementations
dinov3_my.py                 Enhanced DINOv3 adapters (at repository root)
dinounet/dinov3/             Required local DINOv3 implementation
dataset_process/datasets.py  SECOND / Landsat-SCD datasets
utils/                       Original losses, metrics, augmentation, initialization
SOURCE_MANIFEST.json          Original source file hashes for traceability
docs/                        Extraction notes, checks, third-party notices
```

## Installation

Use a Python environment with a compatible PyTorch and torchvision pair. Install the CUDA build appropriate for your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
python train.py --help
python predict.py --help
```

The requirements list records direct dependencies, not a complete environment snapshot. It is not a locked, fully validated environment; see [validation notes](docs/VALIDATION.md).

## Dataset preparation

Prepare SECOND or Landsat-SCD using the following structure. All four folders must use matching PNG filenames. Images are RGB; labels may be RGB palette images or integer class indices.

```text
datasets/SECOND/
├── train/
│   ├── im1/     # images from the first acquisition
│   ├── im2/     # images from the second acquisition
│   ├── label1/  # semantic change labels for the first acquisition
│   └── label2/  # semantic change labels for the second acquisition
├── val/        # same four folders
└── test/       # optional; same four folders
```

Class 0 means unchanged. Labels must be semantic **change** labels: changed areas contain their semantic class and unchanged areas contain 0. Ordinary full-scene land-cover labels are not interchangeable. The binary training target is derived from `label1 > 0`, as in the original training code.

SECOND classes: unchanged, water, ground, low vegetation, tree, building, sports field. Landsat-SCD classes: unchanged, farmland, desert, building, water. Palette and normalization constants are preserved from the original loaders in `dataset_process/datasets.py`. Confirm they match your published experiments. A batch requires images of equal size; no resizing/cropping was added to the loader.

## Pretrained weights

Place your authorized DINOv3 ViT-L/16 SAT-493M initialization file outside version control, for example:

```text
weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

Training requires an explicit path to the appropriate DINOv3 initialization file (ViT-L/16 for Full or ViT-S/16 for Tiny). An invalid path raises an error rather than silently downloading a different initialization. ResNet-34 uses torchvision ImageNet initialization and may download it on first training run. Evaluation and prediction initialize both branches without downloading weights, then strictly load a complete FVT or FVT-Tiny state dictionary.

Datasets and trained weights are distributed separately from this source repository. Download the trained FVT and FVT-Tiny checkpoints from the [GitHub weights release](https://github.com/jerrydu123/FVT/releases/tag/weights), under **Assets**. The four validated checkpoint filenames, configurations, placement instructions, and checksum verification steps are listed in [Weights](docs/WEIGHTS.md).

## Training

Single GPU (batch size is per GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset second --data-root datasets/SECOND \
  --dinov3-weights weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --output-dir outputs/second
```

Multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --dataset second --data-root datasets/SECOND \
  --dinov3-weights weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --output-dir outputs/second
```

For Landsat-SCD, set `--dataset landsat` and change the data/output paths. Default hyperparameters follow the active script: 50 epochs, per-GPU batch size 3, AdamW, learning rate 0.0003, betas (0.9, 0.99), weight decay 1e-7, polynomial decay power 1.0, seed 42. Changing GPU count changes the global batch size.

The active objective is `0.5 * CE_A + 0.5 * CE_B + weighted_BCE + ChangeSimilarity`. The temporal auxiliary loss is returned by the model but excluded from the objective, matching the active original script. See [extraction notes](docs/EXTRACTION.md) for preserved behavior and differences.

The output directory contains `best.pth` (selected by validation mIoU), `config.json`, `metrics.json`, and TensorBoard logs. Validation mIoU follows the original evaluator: mean IoU of unchanged/changed classes; Fscd and Sek also account for semantic classes. The checkpoint contains model parameters only; optimizer-state resume is not implemented.

## Evaluation

```bash
python train.py --eval-only --dataset second --data-root datasets/SECOND \
  --split test --checkpoint outputs/second/best.pth
```

Outputs Fscd, mIoU, Sek, OA, and semantic validation loss. To evaluate on CPU add `--device cpu`; the full model is large and CPU execution can be slow.

## Prediction

```bash
python predict.py --dataset second \
  --im1-dir datasets/SECOND/test/im1 --im2-dir datasets/SECOND/test/im2 \
  --checkpoint outputs/second/best.pth --output-dir predictions/second
```

Prediction needs no labels. It saves a 0/255 binary change map, two semantic index maps, and their RGB visualizations. Semantic predictions are masked by the binary change probability > 0.5.

## Publication status

Only the contents of this directory should form the GitHub repository. Experimental copies, comparison methods, manuscripts/reviewer responses, datasets, weights, build artifacts, and logs have been excluded.

Full and Tiny GPU evaluation has been completed from this source directory on both locally available datasets; check [validation notes](docs/VALIDATION.md) for scores and dataset limitations. Copyright attribution and the final original-code MIT license are deferred; applicable third-party texts are included as described in [third-party notices](docs/THIRD_PARTY_NOTICES.md). Training from scratch has not been rerun.
