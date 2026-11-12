# FVT checkpoints

Source code and checkpoint downloads are distributed separately. The source repository contains no `.pth` files.

**Download: [GitHub weights release](https://github.com/jerrydu123/FVT/releases/tag/weights).** Open the release page and download the checkpoint files or weight package from **Assets**. The source-code archives contain no trained weights.

| Model | Dataset | Download filename | Internal processing size | Size |
|---|---|---|---|---|
| Full | SECOND | `FVT_full_SECOND.pth` | Full resize policy; minimum dimension 512 | Approximately 1.25 GiB |
| Full | Landsat-SCD | `FVT_full_LandsatSCD.pth` | Full resize policy; minimum dimension 512 | Approximately 1.25 GiB |
| Tiny | SECOND | `FVT_tiny_SECOND.pth` | 512 × 512 | Approximately 193 MiB |
| Tiny | Landsat-SCD | `FVT_tiny_LandsatSCD.pth` | 384 × 384 | Approximately 193 MiB |

After download, place the files in `weights/`. If downloading the complete bundle, extract it there. Select both `--model` and `--dataset`; dataset selection controls the class mapping, normalization and Tiny processing size.

```bash
python train.py --eval-only --model full --dataset second \
  --data-root datasets/SECOND --split val --val-batch-size 1 \
  --checkpoint weights/FVT_full_SECOND.pth

python train.py --eval-only --model full --dataset landsat \
  --data-root datasets/Landsat-SCD --split val --val-batch-size 1 \
  --checkpoint weights/FVT_full_LandsatSCD.pth

python train.py --eval-only --model tiny --dataset second \
  --data-root datasets/SECOND --split val --val-batch-size 1 \
  --checkpoint weights/FVT_tiny_SECOND.pth

python train.py --eval-only --model tiny --dataset landsat \
  --data-root datasets/Landsat-SCD --split val --val-batch-size 1 \
  --checkpoint weights/FVT_tiny_LandsatSCD.pth
```

The local SECOND evaluation split is named `val` and contains 593 pairs. Match `--split` to your prepared directory name. The available Landsat split has 476 pairs; the paper describes 477.

## Checkpoint export and verification

The Full training snapshots contain a uniform `module.` prefix and 1,210 additional tensors from modules omitted from the minimal inference implementation. The published Full files remove the prefix and include exactly the 1,405 tensors required by the release model, with retained tensor values unchanged. Both Full exports pass strict loading. The original training files are preserved locally.

Tiny files are byte-identical copies of the corresponding original checkpoints, with clearer filenames. Both pass strict loading. The bundle includes `weights_manifest.json` with source checkpoint names, sizes and SHA-256 values, and `SHA256SUMS.txt` for verification:

```bash
cd weights
sha256sum -c SHA256SUMS.txt
```

These full-model checkpoints include DINOv3 parameters. The DINOv3 materials and their derivatives remain subject to the included DINOv3 License Agreement; do not assume a blanket MIT license for the weight files. See [third-party notices](THIRD_PARTY_NOTICES.md). GPU scoring records are in [Validation](VALIDATION.md).

## GPU verification of publication files

| Model | Dataset | Pairs | mIoU (%) | Sek (%) | Fscd (%) | OA (%) |
|---|---|---:|---:|---:|---:|---:|
| Full | SECOND | 593 | 74.601734 | 25.815778 | 65.032570 | 88.376528 |
| Full | Landsat-SCD | 476 | 91.490172 | 68.002818 | 91.696838 | 97.115015 |
| Tiny | SECOND | 593 | 73.678387 | 24.239716 | 63.587918 | 87.806314 |
| Tiny | Landsat-SCD | 476 | 90.058361 | 63.526253 | 90.173151 | 96.602454 |

See [Validation](VALIDATION.md) for the missing Landsat-SCD pair and the Full OA rounding difference.
