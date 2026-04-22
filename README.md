# YOLO Image Segmentation Training Lab

[阅读中文文档 (README_zh.md)](README_zh.md)

A small project built on Ultralytics YOLO for image segmentation training. The repository separates training flow, configuration management, logging, and validation to make experiments reproducible and easy to compare.

---

## Quick overview

- Goal: provide a reproducible and manageable segmentation training workflow (supports new training, resume from interruption, and fine-tuning from a historical best.pt).
- Language: Python 3.8+
- Dependencies: ultralytics, pyyaml (installation shown below)

---

## Key features

- Three training modes: new training / resume last run / continue from historical best.pt
- Pre-train confirmation that prints key parameters to avoid mistakes
- Toggleable data augmentation with centralized configuration
- Automatic validation and CSV logging (overall and per-class)
- Experiment isolation: each run creates its own result folder and logs for easy comparison

---

## Project structure (simplified)

```text
CODE/
├── dataset_tools/         # data splitting and label utilities
├── pretrained_models/     # common pretrained weights
├── result/                # per-experiment result folders
├── scripts/               # training, config and logging scripts
│   ├── config.py
│   ├── paths.py
│   ├── train_logger.py
│   └── train_segment.py
├── train_logs/            # CSV logs: train_log / result_summary / result_per_class
└── data.yaml              # dataset config (classes, train/val paths)
```

---

## Setup

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics pyyaml
```

For GPU support, ensure the correct CUDA drivers and a matching PyTorch build are installed; Ultralytics uses the system PyTorch.

---

## Configuration

All main settings are in `scripts/config.py` (managed via `TrainConfig`):

- Paths: `data_yaml`, `model_file`, `results_dir`, `log_dir`
- Training hyperparameters: `epochs`, `imgsz`, `batch`, `device`
- Experiment: `experiment_name` (used to generate `save_dir`)
- Augmentation: `use_augment` and parameters such as `hsv_h`, `hsv_s`, `hsv_v`, `translate`, `scale`, `mosaic`, `mixup`, `copy_paste`
- Auto properties: `save_dir`, `last_pt`, `best_pt` (computed via properties)

After editing config, double-check `experiment_name` and `results_dir` to avoid overwriting existing experiments.

---

## Quick start

1. Edit `scripts/config.py`: set `data_yaml`, `model_file`, `experiment_name`, and training hyperparameters.
2. Start training:

```bash
python scripts/train_segment.py
```

The script will prompt for a training mode:

- Enter `1`: start a new training (uses `config.model_file` as the starting weights)
- Enter `2`: resume the last interrupted run (requires `last.pt` in the current experiment folder)
- Enter `3`: continue from a historical `best.pt` (scans `results_dir` and lets you pick an experiment)

Before training starts the script prints key parameters and asks whether to enable augmentation if not fixed in the config.

---

## Logs and validation

The project writes three CSV logs to `train_logs/`:

- `train_log.csv`: training process records (time, mode, status, paths, hyperparams, save locations, etc.)
- `result_summary_log.csv`: overall validation metrics (images/instances, box/mask mAP, precision/recall)
- `result_per_class_log.csv`: per-class metrics and sample distribution (useful to locate weak classes)

After training, validation runs automatically and results are appended to logs. The logs also include the number of images and instances per class found in the validation set.

---

## Recommended workflow

1. Prepare and check your dataset using `dataset_tools/`.
2. Set a new `experiment_name` in `scripts/config.py` to avoid collisions.
3. Run `python scripts/train_segment.py` and choose the appropriate mode.
4. After training, inspect `result/` for `best.pt` and `last.pt`, and review corresponding entries in `train_logs/`.

---

## FAQ & tips

- To fine-tune from a specific historical model, use mode 3 and choose that experiment's `best.pt`.
- If `last.pt` is missing when selecting mode 2, the script will notify you and offer to start a new training instead.
- If GPU memory is insufficient, reduce `batch` or `imgsz`, or switch to CPU (`device='cpu'`) for reproducibility (slower).
- Consider including key hyperparameters (epochs/imgsz/batch) in `experiment_name` or saving them to the experiment folder for easier reproduction.

---

## dataset_tools

`dataset_tools/` contains utilities for preparing and splitting datasets so images and labels are organized for train/val/test workflows.

Main scripts and purpose:

- `dataset_tools/create_empty_labels.py` — generate empty YOLO-style label files for images without annotations (useful for placeholders or pseudo-labeling).
- `dataset_tools/split_images_only/` — split images only:
  - `split_every_5th_images_only.py`: sample every N-th image into val/test (periodic sampling).
  - `split_random_images_only.py`: randomly sample a proportion of images into val/test.
- `dataset_tools/split_train_val/` — split images and corresponding labels into train/val:
  - `split_every_5th_with_labels.py`: interval-based split and move matching label files.
  - `split_random_with_labels.py`: random split while keeping image/label pairs intact.
- `dataset_tools/split_train_val_test/` — support three-way splits (train/val/test), e.g. `split_random_with_labels.py` for independent test sets.

Usage tips:

- Back up original data or test the scripts on a copy before modifying your main dataset.
- Scripts typically accept source dir, target dir, and ratio/interval parameters — check top-of-file comments for usage.
- After splitting, verify that `data.yaml` `train`/`val`/`test` paths point to the correct locations.
- If your label format differs from standard YOLO (class x_center y_center w h per line), convert labels first.

These utilities speed up dataset preparation and reduce manual errors. Contributions for extra features (class-balanced sampling, resolution filters, etc.) are welcome.

---

## Future improvements

- Add a unified inference script and optional export (ONNX / TensorRT)
- Add visualization tools (training curves, confusion matrix, per-class comparisons)
- Auto-save training parameters as JSON/TXT into the experiment folder
- Add test-set evaluation and CI checks

---

## Contributing and maintenance

PRs and issues are welcome:
- improve validation scripts
- add compatibility notes for different Ultralytics versions

---

Final note: after large changes, update `experiment_name` and keep previous experiments for comparison and traceability.
