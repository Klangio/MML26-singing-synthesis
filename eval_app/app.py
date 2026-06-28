from __future__ import annotations

import html
import json
from functools import lru_cache, partial
from pathlib import Path

import gradio as gr


BASE_DIR = Path(__file__).resolve().parent
DATA = json.loads((BASE_DIR / "data" / "manifest.json").read_text())
SAMPLE_IDS = DATA["sample_ids"]
DEFAULT_SAMPLE_ID = "c4dd884fa1a7b263"

if DEFAULT_SAMPLE_ID not in SAMPLE_IDS:
    raise ValueError(f"Default sample {DEFAULT_SAMPLE_ID!r} is not in the manifest")


CSS = """
.gradio-container { max-width: 1240px !important; }
.hero { text-align: center; padding: 1rem 0 .25rem; }
.hero h1 { font-size: 2.25rem; margin-bottom: .35rem; }
.hero p { color: var(--body-text-color-subdued); font-size: 1.05rem; }
.run-summary { border-left: 4px solid #7c3aed; padding: .55rem 1rem; margin: .5rem 0 1rem; }
.dataset-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .6rem; margin: .8rem 0; }
.dataset-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .7rem; background: var(--background-fill-primary); }
.dataset-card .label { color: var(--body-text-color-subdued); font-size: .78rem; }
.dataset-card .value { font-weight: 650; margin-top: .18rem; }
.metric-grid { display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: .6rem; margin: .45rem 0; }
.metric-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .7rem; background: var(--background-fill-secondary); }
.metric-card .label { color: var(--body-text-color-subdued); font-size: .78rem; }
.metric-card .value { font-size: 1.35rem; font-weight: 700; margin-top: .15rem; }
.sample-meta { margin: .45rem 0 .15rem; color: var(--body-text-color-subdued); }
.piano-roll { overflow-x: auto; border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .35rem; background: #111827; }
.piano-roll svg { display: block; min-width: 760px; width: 100%; height: auto; }
.legend { display: flex; gap: 1.25rem; align-items: center; margin: .2rem 0 .5rem; font-size: .86rem; }
.dot { display: inline-block; width: .75rem; height: .75rem; border-radius: 3px; margin-right: .3rem; }
@media (max-width: 760px) { .metric-grid, .dataset-grid { grid-template-columns: repeat(2, 1fr); } }
"""


def absolute(relative_path: str) -> str:
    return str(BASE_DIR / relative_path)


@lru_cache(maxsize=256)
def load_notes(relative_path: str) -> list[dict]:
    path = BASE_DIR / relative_path
    if path.suffix == ".json":
        return json.loads(path.read_text())

    notes = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        onset, offset, pitch = line.split("\t")[:3]
        notes.append(
            {"onset": float(onset), "offset": float(offset), "pitch": int(float(pitch))}
        )
    return notes


def metric_cards(sample: dict) -> str:
    metrics = sample["metrics"]
    cards = [
        ("COnPOff_f1", metrics["COnPOff_f1"]),
        ("COnP_f1", metrics["COnP_f1"]),
        ("COn_f1", metrics["COn_f1"]),
    ]
    rendered = "".join(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value:.3f}</div></div>'
        for label, value in cards
    )
    split = sample.get("split")
    split_text = f" · split: <strong>{html.escape(split)}</strong>" if split else ""
    return (
        f'<div class="sample-meta"><strong>{html.escape(sample["song_id"])}</strong>'
        f' · {sample["reference_notes"]} reference notes · '
        f'{sample["predicted_notes"]} predicted notes{split_text}</div>'
        f'<div class="metric-grid">{rendered}</div>'
    )


def piano_roll(gt_path: str, prediction_path: str) -> str:
    gt = load_notes(gt_path)
    pred = load_notes(prediction_path)
    all_notes = gt + pred
    if not all_notes:
        return "<p>No notes available.</p>"

    min_pitch = min(n["pitch"] for n in all_notes) - 1
    max_pitch = max(n["pitch"] for n in all_notes) + 1
    duration = max(n["offset"] for n in all_notes)
    duration = max(duration, 1.0)

    width, height = 1000, 330
    left, right, top, bottom = 52, 15, 20, 36
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(value: float) -> float:
        return left + (value / duration) * plot_w

    def y(pitch: int, lane_offset: float) -> float:
        span = max(max_pitch - min_pitch, 1)
        return top + ((max_pitch - pitch) / span) * plot_h + lane_offset

    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Ground-truth and predicted note piano roll">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#111827" rx="9"/>',
    ]

    tick_step = 5 if max_pitch - min_pitch > 22 else 2
    first_tick = ((min_pitch + tick_step - 1) // tick_step) * tick_step
    for pitch in range(first_tick, max_pitch + 1, tick_step):
        yy = y(pitch, 0)
        pieces.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
            'stroke="#374151" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" fill="#9ca3af" '
            f'font-size="11">{pitch}</text>'
        )

    seconds_step = 5 if duration > 20 else 2
    second = 0
    while second <= duration:
        xx = x(second)
        pieces.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" '
            'stroke="#374151" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{xx:.1f}" y="{height-12}" text-anchor="middle" fill="#9ca3af" '
            f'font-size="11">{second}s</text>'
        )
        second += seconds_step

    def add_notes(notes: list[dict], color: str, lane_offset: float, opacity: float) -> None:
        span = max(max_pitch - min_pitch, 1)
        note_h = max(3.5, min(9.0, plot_h / span * 0.42))
        for note in notes:
            xx = x(note["onset"])
            ww = max(2.0, x(note["offset"]) - xx)
            yy = y(note["pitch"], lane_offset)
            pieces.append(
                f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{ww:.1f}" height="{note_h:.1f}" '
                f'fill="{color}" fill-opacity="{opacity}" rx="2"/>'
            )

    add_notes(gt, "#38bdf8", -4.5, 0.9)
    add_notes(pred, "#fb7185", 2.0, 0.78)
    pieces.append("</svg>")

    return (
        '<div class="legend">'
        '<span><span class="dot" style="background:#38bdf8"></span>Ground truth</span>'
        '<span><span class="dot" style="background:#fb7185"></span>Prediction</span>'
        '</div><div class="piano-roll">' + "".join(pieces) + "</div>"
    )


def sample_payload(run_key: str, song_id: str):
    sample = DATA["runs"][run_key]["samples"][song_id]
    return (
        absolute(sample["gt_audio"]),
        absolute(sample["prediction_audio"]),
        metric_cards(sample),
        piano_roll(sample["gt_notes"], sample["prediction_notes"]),
        absolute(sample["gt_notes"]),
        absolute(sample["prediction_notes"]),
        absolute(sample["prediction_midi"]),
    )


def run_summary(run: dict) -> str:
    aggregate = run["aggregate"]
    cards = [
        ("COnPOff_f1", aggregate["COnPOff_f1"]),
        ("COnP_f1", aggregate["COnP_f1"]),
        ("COn_f1", aggregate["COn_f1"]),
    ]
    metric_html = "".join(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value:.3f}</div></div>'
        for label, value in cards
    )
    github_url = html.escape(run["github_url"], quote=True)
    source_link = ""
    if run.get("source_url"):
        source_url = html.escape(run["source_url"], quote=True)
        source_link = f' · <a href="{source_url}" target="_blank" rel="noopener">Dataset source</a>'
    wandb_card = ""
    if run.get("wandb_url"):
        wandb_url = html.escape(run["wandb_url"], quote=True)
        wandb_card = (
            '<div class="dataset-card"><div class="label">Experiment tracking</div>'
            f'<div class="value"><a href="{wandb_url}" target="_blank" '
            'rel="noopener">Weights & Biases ↗</a></div></div>'
        )
    training_summary = run.get("training_summary")
    if training_summary is None:
        training_summary = f'{run["training_hours"]:.2f} hours'
    selection_context = run.get("selection_context", f'epoch {run["best_epoch"]}')
    dataset_html = (
        '<div class="dataset-grid">'
        '<div class="dataset-card"><div class="label">Dataset name</div>'
        f'<div class="value">{html.escape(run["dataset_name"])}</div></div>'
        '<div class="dataset-card"><div class="label">Training audio</div>'
        f'<div class="value">{html.escape(training_summary)}</div></div>'
        '<div class="dataset-card"><div class="label">Code / provenance</div>'
        f'<div class="value"><a href="{github_url}" target="_blank" rel="noopener">GitHub ↗</a>'
        f'{source_link}</div></div>{wandb_card}</div>'
    )
    return (
        '<div class="run-summary">'
        f'<h3>{html.escape(run["title"])}</h3>'
        f'<p>{html.escape(run["description"])}</p>'
        f'{dataset_html}'
        f'<p><strong>Best checkpoint:</strong> {html.escape(run["checkpoint"])}<br>'
        f'<strong>Selection score:</strong> COnPOff F1 = {run["selection_COnPOff_f1"]:.4f} '
        f'({html.escape(selection_context)})<br>'
        f'<strong>Displayed aggregate:</strong> {html.escape(run["aggregate_scope"])}</p>'
        f'<div class="metric-grid">{metric_html}</div></div>'
    )


with gr.Blocks(title="MML26 Singing Transcription Evaluation") as demo:
    gr.HTML(
        '<div class="hero"><h1>🎙️ MML26 Singing Transcription Evaluation</h1>'
        '<p>Listen to Klangio ground truth beside predictions from the Dataset 8 MergeKit checkpoint.</p></div>'
    )
    gr.Markdown(
        "Select a validation song in any dataset tab. **Ground truth** is the original singing; "
        "**prediction** is a simple synthesized rendering of the predicted MIDI notes, so timing "
        "and pitch errors are directly audible."
    )

    with gr.Tabs():
        for run_key in DATA["run_order"]:
            run = DATA["runs"][run_key]
            with gr.Tab(run["tab_title"]):
                gr.HTML(run_summary(run))
                selector = gr.Dropdown(
                    choices=SAMPLE_IDS,
                    value=DEFAULT_SAMPLE_ID,
                    label="Klangio validation song",
                    filterable=True,
                )
                initial = sample_payload(run_key, DEFAULT_SAMPLE_ID)
                with gr.Row():
                    gt_audio = gr.Audio(
                        value=initial[0], label="Ground truth singing", interactive=False
                    )
                    prediction_audio = gr.Audio(
                        value=initial[1], label="Predicted notes (sonified)", interactive=False
                    )
                sample_metrics = gr.HTML(initial[2])
                roll = gr.HTML(initial[3])
                with gr.Row():
                    gt_file = gr.File(value=initial[4], label="Ground-truth TSV")
                    prediction_file = gr.File(value=initial[5], label="Prediction JSON")
                    midi_file = gr.File(value=initial[6], label="Prediction MIDI")

                selector.change(
                    fn=partial(sample_payload, run_key),
                    inputs=selector,
                    outputs=[
                        gt_audio,
                        prediction_audio,
                        sample_metrics,
                        roll,
                        gt_file,
                        prediction_file,
                        midi_file,
                    ],
                    show_progress="hidden",
                )

    gr.Markdown(
        "Metrics follow the challenge evaluator. **COn** = onset, **P** = pitch, "
        "**Off** = offset. Higher F1 is better; lower pitch MSE is better. "
        "Klangio is the validation set used to select this merge and its decoding thresholds."
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
