"""Build Dataset 8 with MergeKit and package it as a Lightning checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as load_safetensors


MERGEKIT_VERSION = "0.1.4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_mergeable(checkpoint_path: Path, output_path: Path) -> list[str]:
    state_dict = load_checkpoint(checkpoint_path)["state_dict"]
    mergeable = {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mergeable, output_path)
    return sorted(mergeable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmented-checkpoint", type=Path, required=True)
    parser.add_argument("--soulx-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("comparison_runs/mergekit_dataset8"),
    )
    parser.add_argument("--mergekit-command", default="mergekit-pytorch")
    args = parser.parse_args()

    for checkpoint_path in (args.augmented_checkpoint, args.soulx_checkpoint):
        if not checkpoint_path.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint_path}")

    raw_dir = args.work_dir / "raw"
    augmented_raw = raw_dir / "best-epoch75-step003116.pt"
    soulx_raw = raw_dir / "syntheticdataset-soulx-best-epoch22-COnPOff_f1-0.2498.pt"
    augmented_keys = extract_mergeable(args.augmented_checkpoint, augmented_raw)
    soulx_keys = extract_mergeable(args.soulx_checkpoint, soulx_raw)
    if augmented_keys != soulx_keys:
        raise ValueError("Parent checkpoints do not expose the same floating-point tensors")

    config = {
        "merge_method": "linear",
        "dtype": "float32",
        "parameters": {"normalize": True},
        "models": [
            {"model": str(augmented_raw.resolve()), "parameters": {"weight": 0.99}},
            {"model": str(soulx_raw.resolve()), "parameters": {"weight": 0.01}},
        ],
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.work_dir / "resolved-merge.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    merged_dir = args.work_dir / "merged"
    subprocess.run(
        [
            args.mergekit_command,
            str(config_path),
            str(merged_dir),
            "--no-write-model-card",
        ],
        check=True,
    )
    merged_files = sorted(merged_dir.glob("*.safetensors"))
    if len(merged_files) != 1:
        raise RuntimeError(f"Expected one merged safetensors file, found {merged_files}")
    merged = load_safetensors(str(merged_files[0]), device="cpu")
    if set(merged) != set(augmented_keys):
        raise ValueError("MergeKit output does not match the parent tensor schema")

    template = load_checkpoint(args.augmented_checkpoint)
    output = copy.deepcopy(template)
    for key, value in merged.items():
        output["state_dict"][key] = value
    output["epoch"] = -1
    output["global_step"] = 0
    output["optimizer_states"] = []
    output["lr_schedulers"] = []
    output["onset_threshold"] = 0.70
    output["frame_threshold"] = 0.01
    output["mergekit_provenance"] = {
        "mergekit_version": MERGEKIT_VERSION,
        "method": "linear",
        "parents": [
            {
                "checkpoint": str(args.augmented_checkpoint),
                "sha256": sha256(args.augmented_checkpoint),
                "weight": 0.99,
            },
            {
                "checkpoint": str(args.soulx_checkpoint),
                "sha256": sha256(args.soulx_checkpoint),
                "weight": 0.01,
            },
        ],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "validation": {
            "dataset": "klangiodataset",
            "songs": 35,
            "precision": "bf16-mixed",
            "selection_metric": "COnPOff_f1",
            "onset_threshold": 0.70,
            "frame_threshold": 0.01,
            "metrics": {
                "COnPOff_f1": 0.24085187911987305,
                "COnP_f1": 0.419907808303833,
                "COn_f1": 0.6614999771118164,
            },
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_checkpoint)

    manifest = {
        "output_checkpoint": str(args.output_checkpoint),
        "sha256": sha256(args.output_checkpoint),
        "bytes": args.output_checkpoint.stat().st_size,
        "mergekit_provenance": output["mergekit_provenance"],
    }
    manifest_path = args.work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_checkpoint}")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
