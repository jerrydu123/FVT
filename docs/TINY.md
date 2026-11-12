# Tiny configuration and weights

Training, evaluation, and label-free prediction support `--model tiny`. The factory in `models/factory.py` selects the configuration according to `--dataset`:

| Dataset | Classes | Internal processing size | Temporal hidden channels | Frames | Feature interpolation |
|---|---:|---:|---:|---:|---|
| SECOND | 7 | 512 × 512 | 64 | 10 | Linear |
| Landsat-SCD | 5 | 384 × 384 | 64 | 10 | Linear |

The internal processing size applies before model encoding. It does not change the source dataset files or final output dimensions. Class definitions, color mappings, and normalization constants remain specific to each dataset.

## Checkpoint files

Weights are distributed separately from source code in the [GitHub weights release](https://github.com/jerrydu123/FVT/releases/tag/weights), under **Assets**. See [Weights](WEIGHTS.md) for the published filenames, extraction instructions, and checksum verification. Place downloaded checkpoints in `weights/`. The original checkpoint names are listed below for traceability:

- SECOND: `ZhangXiaochen_23e_mIoU73.68_Sek24.24_Fscd63.59_OA87.81.pth`
- Landsat-SCD: `ZhangXiaochen_49e_mIoU90.06_Sek63.53_Fscd90.17_OA96.60.pth`

```bash
python train.py --eval-only --model tiny --dataset second \
  --data-root datasets/SECOND --split val --val-batch-size 1 \
  --checkpoint weights/FVT_tiny_SECOND.pth

python train.py --eval-only --model tiny --dataset landsat \
  --data-root datasets/Landsat-SCD --split val --val-batch-size 1 \
  --checkpoint weights/FVT_tiny_LandsatSCD.pth

python predict.py --model tiny --dataset second \
  --im1-dir datasets/SECOND/val/im1 --im2-dir datasets/SECOND/val/im2 \
  --checkpoint weights/FVT_tiny_SECOND.pth \
  --output-dir predictions/tiny_second
```

For training, also select `--model tiny` and either `--dataset second` or `--dataset landsat`, and supply **ViT-S/16** pretrained weights with `--dinov3-weights`. The entry point preserves the active loss and optimizer settings from the original training script. Successful evaluation of existing checkpoints does not establish reproduction through retraining.

## Compatibility notes

- The original Tiny source defines `TBFFNet` three times. This release retains only the final effective class and its dependencies, preserving parameter names.
- The original code uses an absolute `sys.path` entry to import `video_Tiny.py` from the manuscript directory. This release includes that actual dependency as `models/video_Tiny.py` and uses an explicit package import.
- For SECOND, an internal size of 384 yields mIoU = 73.0522%. Restoring the size to 512 with all other settings unchanged reproduces all four paper metrics to two decimal places.
- Decoder residual projection paths remain disabled. The historical temporal computation, in which the output of `backward_layer1` is overwritten on the next line, is preserved to maintain inference behavior with existing weights.
- The original training script always copied the Full `TBFFNet.py` when saving source snapshots. A model snapshot with that name beside a Tiny checkpoint therefore does not reliably identify the Tiny implementation.
- Checkpoints are loaded strictly. Evaluation supplies `labels_bn`; label-free prediction accepts `None`. Evaluation and prediction do not require separate ImageNet or DINOv3 initialization weights.
- The locally available Landsat-SCD validation split contains 476 pairs, compared with 477 described in the paper. The missing pair remains unresolved. See [Validation](VALIDATION.md).

## Independent evaluation from the release directory

| Dataset | Internal processing size | Pairs | mIoU (%) | Sek (%) | Fscd (%) | OA (%) |
|---|---:|---:|---:|---:|---:|---:|
| SECOND | 512 | 593 | 73.678387 | 24.239716 | 63.587918 | 87.806314 |
| Landsat-SCD | 384 | 476 | 90.058361 | 63.526253 | 90.173151 | 96.602454 |
