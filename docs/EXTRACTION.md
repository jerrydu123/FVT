# Release scope and source provenance

The Full model follows the active implementation imported by `train.py` at the original workspace root. Tiny is extracted from the final effective class in `models/TBFFNet_Tiny.py` and its actual dependency, `0_Manuscipt_decision/model/video_Tiny.py`. Its dataset-specific internal processing sizes follow the validated configurations: 512 for SECOND and 384 for Landsat-SCD. The original training model files have not been modified. See [Tiny configuration and weights](TINY.md). Trans variants, ablation experiments, and comparison methods are outside the release scope.

| Release file | Original source or purpose |
|---|---|
| `train.py` | Reorganized training workflow preserving the active optimizer, losses, and learning rate formula |
| `models/FVT.py` | Originally `models/TBFFNet.py`, the model imported by the active training entry point; retains the encoder, fusion modules, decoders, and classifiers |
| `models/video.py` | Bidirectional 3D convolutions, frame-difference enhancement, and temporal aggregation |
| `models/FVT_Tiny.py` | Final effective Tiny model from `models/TBFFNet_Tiny.py` and its dependencies, with duplicate definitions and unrelated imports removed |
| `models/video_Tiny.py` | Actual temporal dependency used by Tiny, preserving the computation associated with historical checkpoints |
| `models/factory.py` | Full/Tiny selection and validated dataset-specific Tiny processing sizes |
| `models/video_323.py` | Sequence generators imported by `models/video.py` |
| `dinov3_my.py` | Active `EnhancedSimpleDino` implementation, adapters, and related modules |
| `dinounet/dinov3/` | Backbone, layers, hub, and utilities required by the DINOv3 branch |
| `dataset_process/datasets.py` | Merged directory conventions, classes, color palettes, and normalization constants from `SECOND_process.py` and `Landsat_SCD.py` |
| `utils/loss.py` | Original loss implementations, including the active `CrossEntropyLoss2d`, `weighted_BCE_logits`, and `ChangeSimilarity` |
| `utils/utils.py`, `utils/eval_segm.py` | Original evaluation implementations for OA, Fscd, mIoU, and Sek |
| `utils/transform.py`, `utils/misc.py` | Original data augmentation and weight initialization utilities |
| `predict.py` | Label-free prediction entry point using the binary-mask postprocessing from training |

## Preserved experimental behavior

- Full uses DINOv3 ViT-L/16; Tiny uses ViT-S/16. Supply the corresponding pretrained initialization weights for training.
- The active temporal branch uses **10 frames, linear interpolation, and lightweight 3D blocks**. The separate legacy field `self.video_len=4` does not determine the frame count used by this branch.
- Full resizes an input to 512 × 512 when either spatial dimension is smaller than 512. Tiny uses a fixed internal size of 512 × 512 for SECOND and 384 × 384 for Landsat-SCD. Final outputs are restored to the original size of the first input image.
- Model initialization retains the original fixed seed of 42. The command-line seed controls randomness during subsequent training.
- Semantic cross-entropy ignores class 0. Inputs to `ChangeSimilarity` exclude class 0. Binary targets are derived from the first-time semantic change labels using `label1 > 0`.
- The total objective excludes the temporal auxiliary loss returned by the model. It also excludes the `AdaptiveChangeCriterion`, Dice, and FFT terms that were only instantiated or commented out in the original script.
- AdamW settings, the per-iteration polynomial learning rate schedule, and best-checkpoint selection by validation mIoU follow the original implementation. The original `ReduceLROnPlateau.step()` call was commented out, so no second scheduler is added.
- Automatic mixed precision (AMP) and gradient accumulation are not enabled. Although the original script defined `accumulation_steps=2`, it called `optimizer.step()` after every batch.

## Changes made for release

- Delegate GPU selection to `CUDA_VISIBLE_DEVICES` and the `LOCAL_RANK` provided by `torchrun`, and support single-process training. The original entry point required an initialized distributed environment.
- Run validation, write TensorBoard logs, and save checkpoints only in the main process to avoid concurrent writes. Evaluate the full validation set without a distributed sampler that pads it with duplicate samples.
- Remove hard-coded user paths, process names, and the local source-copy operation performed at the end of training.
- Remove unused model classes, commented-out experiments, and imports of TimeSformer, the earlier DinoUNet, `extra_model`, and `pyramid`. The release therefore does not require the nnUNet, Mamba, causal-conv1d, or TimeSformer training libraries.
- Limit DINOv3 package initialization to modules required by the released models, without automatically importing the nnUNet API or DINOv3 pretraining framework.
- Replace the deprecated torchvision `pretrained` argument with `weights` while preserving the default initialization behavior.
- Merge the data loaders, sort filenames, and load samples on demand instead of preloading all labels. Add checks for matching pairs, image dimensions, and unknown label colors. Support both RGB and class-index labels.
- Replace `change_mask.squeeze()` with `change_mask.squeeze(1)` to preserve explicit dimensions when the batch size is 1.
- Remove unused one-hot intermediate tensors, optimizer and loss objects, and metric plots tied to fixed positions. Save metrics in `metrics.json` instead.
- Supply `labels_bn` during evaluation, matching the original `nb` revalidation workflow, and pass `None` during label-free prediction. These labels affect only the auxiliary loss; a single-sample comparison confirmed that they do not affect current model predictions.

These runtime changes may alter sample order or random-number consumption. This release does not claim bitwise equivalence with the earlier scripts. Reproducing the paper results through training requires verification with the same data splits, GPU count, batch size, and training environment.

## Excluded materials

Datasets, all weights, training results, logs, visualizations, manuscripts and review materials, historical backups, comparison models, installation artifacts (`build/`, `dist/`, and `*.egg-info/`), and caches are excluded.

The original `pred_SCD.py` and earlier README describe an older implementation and were not reused directly. Publication metadata, checkpoint identities, and reported metrics must be supplied or confirmed by the author; they must not be inferred from those earlier files.

## Publication model names

The release exposes `FVT` in `models/FVT.py` and `FVT_Tiny` in `models/FVT_Tiny.py`, with public names **FVT** and **FVT-Tiny**. The naming update changed only filenames, class names, and imports. Internal module names, computation, and `state_dict` keys remain unchanged. Historical source paths above and the `source` fields in `SOURCE_MANIFEST.json` retain their original names for traceability. Select the models with `--model full` or `--model tiny`.
