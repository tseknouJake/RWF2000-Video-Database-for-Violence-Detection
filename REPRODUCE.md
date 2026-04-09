# RWF-2000 Reproduction Guide

This repository originally provided notebook-only code. The commands below turn it into a repeatable preprocessing, training, and evaluation workflow so you can fill the `cctv_violence_reproduction_tracker.xlsx` row for the official RWF-2000 baseline.

## 1. Environment

Use the locally available Python 3.11 interpreter instead of Python 3.13, then install the project dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset Layout

The scripts expect the raw dataset to preserve the original split/class structure:

```text
data/raw/RWF-2000/
  train/
    Fight/
    NonFight/
  val/
    Fight/
    NonFight/
```

If your dataset root already follows that pattern, keep it as-is. The preprocessing step preserves the relative directory structure when it writes `.npy` files.

## 3. Preprocess The Videos

Convert every raw video into the 5-channel tensor format used by the notebooks:

```bash
python scripts/preprocess_rwf2000.py \
  --source-dir data/raw/RWF-2000 \
  --target-dir data/processed/RWF2000-opt
```

That produces:

```text
data/processed/RWF2000-opt/
  train/
    Fight/*.npy
    NonFight/*.npy
  val/
    Fight/*.npy
    NonFight/*.npy
```

## 4. Train A Baseline

The original repo contains three notebook baselines. The most important one for reproduction is the flow-gated model from the paper:

```bash
python scripts/train_rwf2000.py \
  --architecture flow-gated \
  --train-dir data/processed/RWF2000-opt/train \
  --val-dir data/processed/RWF2000-opt/val \
  --output-dir outputs/flow-gated
```

Other notebook variants:

```bash
python scripts/train_rwf2000.py --architecture rgb --train-dir data/processed/RWF2000-opt/train --val-dir data/processed/RWF2000-opt/val --output-dir outputs/rgb
python scripts/train_rwf2000.py --architecture opt --train-dir data/processed/RWF2000-opt/train --val-dir data/processed/RWF2000-opt/val --output-dir outputs/opt
```

## 5. Evaluate A Model

Evaluate the best checkpoint from your run:

```bash
python scripts/evaluate_rwf2000.py \
  --model-path outputs/flow-gated/best_model.keras \
  --data-dir data/processed/RWF2000-opt/val \
  --architecture flow-gated \
  --metrics-out outputs/flow-gated/metrics.json
```

You can also try the shipped pretrained model:

```bash
python scripts/evaluate_rwf2000.py \
  --model-path Models/keras_model.h5 \
  --data-dir data/processed/RWF2000-opt/val \
  --architecture flow-gated
```

## 6. What To Put In The Tracker

For the `mchengny/RWF2000-Video-Database-for-Violence-Detection` row in `cctv_violence_reproduction_tracker.xlsx`:

- `Reported metric`: Flow Gated Network accuracy on RWF-2000 test/validation split as reported by the paper.
- `Metric I reproduced`: Copy the `accuracy` value from `outputs/flow-gated/metrics.json`.
- `GPU/CPU used`: Record the machine or GPU model you actually used.
- `Problems encountered`: Note environment issues, TensorFlow version, dataset path fixes, or any model-loading issue with `keras_model.h5`.
- `Notes`: Record exact commands, batch size changes, or any deviation from the notebook defaults.

## 7. Known Reproducibility Gaps In The Original Repo

- The repository does not include the dataset itself.
- The notebooks hard-code multi-GPU training and old standalone Keras APIs.
- `Flow Gated Network.ipynb` points at `ViolentFlow-opt` instead of `RWF2000-opt`; this guide uses the RWF-2000 path consistently.
- The official README does not contain a full end-to-end command sequence.

