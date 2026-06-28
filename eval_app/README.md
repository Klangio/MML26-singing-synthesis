# MergeKit Dataset 8 evaluation app

This directory reproduces the Dataset 8 model merge and builds a small Gradio app for
listening to its predictions on the 35-song Klangio validation set. It keeps the three
stages separate:

1. Merge two compatible Basic Pitch checkpoints with MergeKit.
2. Run the merged Lightning checkpoint over `klangiodataset`.
3. Turn the predicted JSON/MIDI files into static audio, metrics, and piano-roll assets.

The published Dataset 8 checkpoint is a normalized linear blend of 99% Dataset 7
(audio augmentations) and 1% Dataset 2 (SoulX). The selected BF16 validation scores are
`COnPOff_f1=0.2409`, `COnP_f1=0.4199`, and `COn_f1=0.6615`. Klangio is used for model
selection here, so these numbers are validation results rather than held-out test results.

## Why a checkpoint bridge is needed

The challenge models are PyTorch Lightning `.ckpt` files. MergeKit's raw PyTorch mode
expects a flat mapping of tensor names to tensors. `tools/build_mergekit_dataset8.py`
extracts the floating-point `state_dict` tensors, calls `mergekit-pytorch`, then packages
the merged tensors back into the original Lightning envelope. Integer BatchNorm counters
are copied from Dataset 7 because MergeKit's linear arithmetic is intended for floating
point tensors.

The two parents were trained independently, so broad 50/50 model soups perform poorly:
hidden channels are not guaranteed to remain permutation-aligned. The 99/1 blend was the
best validation result among the tested MergeKit candidates while retaining a measurable
SoulX contribution.

## Setup

From the repository root, install the normal project dependencies plus MergeKit:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r eval_app/requirements.txt
```

Download the two parent checkpoints:

```bash
hf download krishnakalyan3/mml26-singing-synthesis-checkpoints \
  checkpoints/best-epoch75-step003116.ckpt \
  checkpoints/syntheticdataset-soulx-best-epoch22-COnPOff_f1-0.2498.ckpt \
  --local-dir hf_checkpoints
```

## 1. Build Dataset 8 with MergeKit

```bash
python eval_app/tools/build_mergekit_dataset8.py \
  --augmented-checkpoint \
    hf_checkpoints/checkpoints/best-epoch75-step003116.ckpt \
  --soulx-checkpoint \
    hf_checkpoints/checkpoints/syntheticdataset-soulx-best-epoch22-COnPOff_f1-0.2498.ckpt \
  --output-checkpoint \
    comparison_runs/checkpoints/mergekit-dataset8-aug75-soulx-99-01.ckpt
```

The script also writes the resolved MergeKit YAML and intermediate raw tensor files under
`comparison_runs/mergekit_dataset8/`. The conceptual configuration is checked in at
`mergekit/dataset8-merge.yml`.

## 2. Run Klangio inference

```bash
python -m src.inference \
  --checkpoint-path \
    comparison_runs/checkpoints/mergekit-dataset8-aug75-soulx-99-01.ckpt \
  --input-path klangiodataset \
  --output-dir comparison_runs/inference/mergekit_dataset8 \
  --output-format all \
  --accelerator gpu
```

This creates one prediction JSON and MIDI file per Klangio song. The checkpoint contains
the selected onset/frame thresholds (`0.70`/`0.01`), so the inference command does not need
additional threshold arguments.

## 3. Build and launch the evaluator

```bash
python eval_app/tools/prepare_assets.py . eval_app
python eval_app/app.py
```

Open the local Gradio URL printed by the second command. The app provides:

- original Klangio audio and a sonification of predicted notes;
- aggregate and per-song COn, COnP, COnOff, and COnPOff metrics;
- aligned ground-truth/prediction piano rolls;
- downloadable reference TSV, prediction JSON, and prediction MIDI files.

Generated `assets/` and `data/` are deliberately ignored by Git. Build them before a local
launch or include them when deploying `eval_app/` as a Hugging Face Gradio Space.

## Published artifacts

- [Dataset 8 checkpoint](https://huggingface.co/krishnakalyan3/mml26-singing-synthesis-checkpoints/blob/main/checkpoints/mergekit-dataset8-aug75-soulx-99-01-COnPOff_f1-0.2409.ckpt)
- [Merge configuration](https://huggingface.co/krishnakalyan3/mml26-singing-synthesis-checkpoints/blob/main/mergekit/dataset8-merge.yml)
- [Validation metadata](https://huggingface.co/krishnakalyan3/mml26-singing-synthesis-checkpoints/blob/main/mergekit/dataset8-validation.json)
- [Live evaluation Space](https://huggingface.co/spaces/krishnakalyan3/mml26-singing-synthesis-eval)
