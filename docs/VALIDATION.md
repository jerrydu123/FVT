# Release validation

## Final checkpoint exports: Full and Tiny (2026-11-10)

All four publication checkpoint files pass strict loading. The final release `train.py` entry point has completed GPU evaluation for all four model/dataset combinations in the `df` environment. Full exports contain only tensors used by the minimal release model, with values unchanged; Tiny files are byte-identical copies of the original checkpoints.

| Model | Dataset | Pairs | mIoU (%) | Sek (%) | Fscd (%) | OA (%) |
|---|---|---:|---:|---:|---:|---:|
| Full | SECOND | 593 | 74.601734 | 25.815778 | 65.032570 | 88.376528 |
| Full | Landsat-SCD | 476 | 91.490172 | 68.002818 | 91.696838 | 97.115015 |
| Tiny | SECOND | 593 | 73.678387 | 24.239716 | 63.587918 | 87.806314 |
| Tiny | Landsat-SCD | 476 | 90.058361 | 63.526253 | 90.173151 | 96.602454 |

Full SECOND and Tiny SECOND reproduce all four paper metrics to two decimal places. Tiny Landsat also matches to two decimals; Full Landsat OA rounds to 97.12 versus the paper's 97.11, while the other three metrics match. Both Landsat runs use the 476 pairs available locally, compared to 477 described in the paper. These are checkpoint inference validations, not new training runs.

The weight files are packaged separately from the source ZIP. The independent download URL is pending upload to the author's chosen host. See [Weights](WEIGHTS.md).

## Tiny synchronization (2026-11-10)

The release now includes Tiny, selected with `--model tiny` in both entry points. Configuration is dataset-specific: SECOND uses internal size 512; Landsat-SCD uses 384. Both use 10 linear interpolation frames, video hidden channels 64, and disabled decoder residual projections. The temporal implementation is copied from the actual historical import under `0_Manuscipt_decision/model/video_Tiny.py` rather than the other same-named module.

- All 38 Python files parse successfully; both entry points expose model selection in `--help`.
- Thirteen retained non-constructor Tiny methods match the effective original class definitions by AST, after normalizing the documented configurable size and optional labels argument.
- All Tiny temporal class/function definitions are AST-identical to the actual historical dependency.
- Strict checkpoint loading passed for both datasets from this release directory. Evaluation initializes without downloading pretrained weights and supplies `labels_bn`.

Both Tiny checkpoints completed full available-dataset evaluation from this release directory in the `df` environment on RTX 4090 GPUs.

| Dataset | Internal size | Pairs | mIoU (%) | Sek (%) | Fscd (%) | OA (%) |
|---|---:|---:|---:|---:|---:|---:|
| SECOND | 512 | 593 | 73.678387 | 24.239716 | 63.587918 | 87.806314 |
| Landsat-SCD | 384 | 476 | 90.058361 | 63.526253 | 90.173151 | 96.602454 |

SECOND reproduces all four paper metrics to two decimal places. Landsat also matches to two decimal places, but only 476/477 published pairs are available locally. OA differs from the original accumulation by less than 0.000002 percentage points.

The label-free prediction CLI passed on a real SECOND pair and generated all five expected maps at 512 × 512. An evaluation-only guard now assigns zero semantic loss when every label in a target is ignored; its mixed unchanged/changed-batch check passed with finite loss and perfect metrics. The full scoring runs preceded this loss-only guard; prediction and metric computations were unchanged.

Tiny training from scratch and multi-GPU training have not been rerun. See [Tiny configuration and weights](TINY.md).

## Earlier Full extraction checks (2026-11-09)

The following records describe the initial Full-only extraction checks at that time; they are not a statement about the later Tiny runs above.

Checks performed on 2026-11-09:

- Syntax compilation passed for all 35 Python source files.
- `train.py --help` and `predict.py --help` passed from the release directory.
- Importing the model and training module passed without accessing the original project packages.
- AST comparison: the bodies of all retained non-constructor model methods in `models/FVT.py` (originally `models/TBFFNet.py`) and `models/video.py` match their original source. Constructors only received the documented initialization configuration changes.
- Original versus extracted full model: all **1,405 state dictionary keys and tensor shapes match**. This comparison instantiated the full architecture with meta tensors; pretrained loading was disabled for comparison. It does not establish numerical equivalence of training or validate a real checkpoint.
- Full model meta forward passed with batch size 1 and paired 256 × 256 inputs. Output shapes: change `(1,1,256,256)`, semantic A/B `(1,7,256,256)`. Meta execution verifies dimensions, not numerical correctness.
- Temporal branch real CPU forward/backward passed with 10 linear interpolation frames, lightweight 3D blocks, an auxiliary binary loss, and finite input gradients. Label-free inference also passed.
- Synthetic SECOND and Landsat-SCD samples passed RGB/index label decoding, normalization tensor conversion, synchronized augmentation, and invalid-color rejection.
- Batch-size-one evaluation against synthetic perfect predictions returned Fscd, mIoU, Sek, and OA equal to 1 for both datasets.

The interpreter used was Python 3.9.23 from the existing `df` environment, with PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. CUDA was unavailable to this session. The system Python environment lacked several dependencies and was not used for runtime validation.

Full GPU training, multi-GPU execution, pretrained weight ingestion, complete FVT numerical CPU/GPU inference, and reproduction of paper results have **not** been run. The above lightweight checks do not replace those validations. No trained accuracy or reproducibility claim is made for this extracted version.

`requirements.txt` records direct dependencies with version bounds, not a tested matrix of all allowed combinations. Establish and publish an exact environment lock after the full reproduction run.

Additional versions in that validation environment: NumPy 1.24.4, SciPy 1.13.1, scikit-image 0.24.0, Pillow 11.3.0, OpenCV 4.5.5.64, Matplotlib 3.9.4, and tensorboardX 2.6.4. The OpenCV installation predates the requirement bound inherited from the old dependency list; the stated minimum install combination has not been independently tested.

## Publication model names

The release uses `FVT` in `models/FVT.py` and `FVT_Tiny` in `models/FVT_Tiny.py`. AST comparison after normalizing the class names confirms that both model implementations are otherwise unchanged. All release Python files compile, and the renamed factory models strictly load all four published checkpoints (Full/Tiny × SECOND/Landsat-SCD). This naming check does not rerun dataset scoring. Historical source paths retain their original names for provenance.
