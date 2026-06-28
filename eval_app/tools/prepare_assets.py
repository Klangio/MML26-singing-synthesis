"""Build static Dataset 8 audio, note, and metric assets for the Gradio app."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


RUN_KEY = "mergekit_dataset8"
RUN_SOURCE = Path("comparison_runs/inference/mergekit_dataset8")
RUN_METADATA = {
    "tab_title": "Dataset 8 · MergeKit 99/1",
    "title": "Dataset 8 — MergeKit (99% Dataset 7 + 1% Dataset 2)",
    "description": (
        "Normalized linear MergeKit blend of 99% audio-augmentation checkpoint "
        "and 1% SoulX checkpoint; no retraining."
    ),
    "dataset_name": "MergeKit: 99% Dataset 7 + 1% Dataset 2",
    "training_hours": 0.0,
    "training_summary": "Model merge (no retraining)",
    "github_url": "https://github.com/arcee-ai/mergekit",
    "source_url": (
        "https://huggingface.co/krishnakalyan3/"
        "mml26-singing-synthesis-checkpoints/blob/main/mergekit/dataset8-merge.yml"
    ),
    "checkpoint": "mergekit-dataset8-aug75-soulx-99-01-COnPOff_f1-0.2409.ckpt",
    "best_epoch": "MergeKit",
    "selection_context": "MergeKit validation sweep",
    "selection_COnPOff_f1": 0.24085187911987305,
    "aggregate_scope": (
        "FP32 inference on all 35 Klangio validation songs using saved "
        "onset/frame thresholds 0.70/0.01"
    ),
}


def evaluate_prediction_set(source: Path, prediction_dir: Path) -> tuple[dict, dict]:
    sys.path.insert(0, str(source))
    from src.transcription_utils.evaluation import get_metrics_dict

    rows = {}
    for wav_path in sorted((source / "klangiodataset").glob("*.wav")):
        song_id = wav_path.stem
        prediction_path = prediction_dir / f"{song_id}_notes.json"
        midi_path = prediction_dir / f"{song_id}.mid"
        if not prediction_path.is_file() or not midi_path.is_file():
            raise FileNotFoundError(
                f"Missing Dataset 8 inference output for {song_id}; run the README inference step"
            )
        reference = np.loadtxt(wav_path.with_suffix(".tsv"), delimiter="\t", comments="#")
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        estimated = np.asarray(
            [[note["onset"], note["offset"], note["pitch"]] for note in prediction],
            dtype=np.float64,
        ).reshape(-1, 3)
        metrics = get_metrics_dict(reference, estimated)
        rows[song_id] = {
            "song_id": song_id,
            "reference_notes": len(reference),
            "predicted_notes": len(estimated),
            **metrics,
        }

    metric_names = ("COn_f1", "COnP_f1", "COnOff_f1", "COnPOff_f1", "pitch_mse")
    aggregate = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows.values()], dtype=np.float64)
        aggregate[metric_name] = float(values[np.isfinite(values)].mean())
    return rows, aggregate


def synthesize_notes(notes: list[dict], duration: float, sample_rate: int = 16000) -> np.ndarray:
    audio = np.zeros(int(np.ceil(duration * sample_rate)), dtype=np.float32)
    for note in notes:
        start = max(0, int(round(float(note["onset"]) * sample_rate)))
        end = min(len(audio), int(round(float(note["offset"]) * sample_rate)))
        if end <= start:
            continue
        count = end - start
        time = np.arange(count, dtype=np.float32) / sample_rate
        frequency = 440.0 * (2.0 ** ((int(note["pitch"]) - 69) / 12.0))
        tone = (
            0.68 * np.sin(2 * np.pi * frequency * time)
            + 0.22 * np.sin(2 * np.pi * frequency * 2 * time)
            + 0.10 * np.sin(2 * np.pi * frequency * 3 * time)
        )
        attack = min(count, int(0.018 * sample_rate))
        release = min(count, int(0.055 * sample_rate))
        envelope = np.ones(count, dtype=np.float32)
        if attack:
            envelope[:attack] = np.linspace(0, 1, attack, endpoint=False)
        if release:
            envelope[-release:] *= np.linspace(1, 0, release, endpoint=True)
        audio[start:end] += 0.24 * tone.astype(np.float32) * envelope
    return np.tanh(audio * 1.15).astype(np.float32)


def write_mp3(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        audio,
        sample_rate,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=0.72,
        bitrate_mode="VARIABLE",
    )


def build(source: Path, destination: Path) -> None:
    prediction_dir = source / RUN_SOURCE
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset 8 inference directory does not exist: {prediction_dir}"
        )

    assets = destination / "assets"
    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    song_ids = sorted(path.stem for path in (source / "klangiodataset").glob("*.wav"))
    if not song_ids:
        raise FileNotFoundError(f"No Klangio WAV files found under {source / 'klangiodataset'}")

    lookup, aggregate = evaluate_prediction_set(source, prediction_dir)
    manifest = {
        "sample_ids": song_ids,
        "run_order": [RUN_KEY],
        "runs": {},
    }

    for song_id in song_ids:
        gt_wav = source / "klangiodataset" / f"{song_id}.wav"
        gt_tsv = source / "klangiodataset" / f"{song_id}.tsv"
        gt_audio, gt_rate = sf.read(gt_wav, dtype="float32")
        if gt_rate != 16000:
            gt_audio = resample_poly(gt_audio, 16000, gt_rate).astype(np.float32)
        write_mp3(assets / "ground_truth" / f"{song_id}.mp3", gt_audio)
        gt_out = assets / "ground_truth" / f"{song_id}.tsv"
        gt_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_tsv, gt_out)

    run = dict(RUN_METADATA)
    run["aggregate"] = {key: float(value) for key, value in aggregate.items()}
    run["samples"] = {}
    for song_id in song_ids:
        prediction_json = prediction_dir / f"{song_id}_notes.json"
        prediction_midi = prediction_dir / f"{song_id}.mid"
        notes = json.loads(prediction_json.read_text(encoding="utf-8"))
        duration = sf.info(source / "klangiodataset" / f"{song_id}.wav").duration
        run_asset_dir = assets / RUN_KEY
        write_mp3(run_asset_dir / f"{song_id}.mp3", synthesize_notes(notes, duration))
        run_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prediction_json, run_asset_dir / f"{song_id}_notes.json")
        shutil.copy2(prediction_midi, run_asset_dir / f"{song_id}.mid")

        row = lookup[song_id]
        run["samples"][song_id] = {
            "song_id": song_id,
            "reference_notes": int(row["reference_notes"]),
            "predicted_notes": int(row["predicted_notes"]),
            "metrics": {
                key: float(row[key])
                for key in ("COn_f1", "COnP_f1", "COnOff_f1", "COnPOff_f1", "pitch_mse")
            },
            "gt_audio": f"assets/ground_truth/{song_id}.mp3",
            "gt_notes": f"assets/ground_truth/{song_id}.tsv",
            "prediction_audio": f"assets/{RUN_KEY}/{song_id}.mp3",
            "prediction_notes": f"assets/{RUN_KEY}/{song_id}_notes.json",
            "prediction_midi": f"assets/{RUN_KEY}/{song_id}.mid",
        }

    manifest["runs"][RUN_KEY] = run
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built {len(song_ids)} Dataset 8 samples in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MML26-singing-synthesis repository")
    parser.add_argument("destination", type=Path, help="Evaluation app directory")
    args = parser.parse_args()
    build(args.source.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
