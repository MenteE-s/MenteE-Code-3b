"""Generate real charts from real run artifacts — no invented numbers.

    python scripts/make_charts.py \
      --trainer-state outputs/full_run/trainer_state.json \
      --humaneval-tuned outputs/humaneval_results.json \
      --humaneval-base outputs/humaneval_results_base.json \
      --gpu-log outputs/gpu_log.csv \
      --out outputs/charts

Every chart skips itself (with a note) if its input file is missing, so you can
run this with whatever artifacts you have. Requires: pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt

MENTEE = "#7c3aed"  # MenteE purple
BASE = "#64748b"


def chart_loss(log_history: list[dict], out: Path):
    steps = [h["step"] for h in log_history if "loss" in h]
    loss = [h["loss"] for h in log_history if "loss" in h]
    if not steps:
        print("  - loss chart: no loss entries found, skipped")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, loss, color=MENTEE, linewidth=1.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss")
    ax.set_title("MenteE Code 3B — training loss (measured)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  - loss_curve.png ({len(steps)} points)")


def chart_accuracy(log_history: list[dict], out: Path):
    pts = [(h["step"], h["mean_token_accuracy"]) for h in log_history if "mean_token_accuracy" in h]
    if not pts:
        print("  - accuracy chart: field not in logs, skipped")
        return
    steps, acc = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, [100 * a for a in acc], color=MENTEE, linewidth=1.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("next-token accuracy (%)")
    ax.set_title("MenteE Code 3B — token prediction accuracy (measured)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_curve.png", dpi=150)
    plt.close(fig)
    print(f"  - accuracy_curve.png ({len(steps)} points)")


def chart_lr(log_history: list[dict], out: Path):
    pts = [(h["step"], h["learning_rate"]) for h in log_history if "learning_rate" in h]
    if not pts:
        print("  - lr chart: skipped")
        return
    steps, lr = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, lr, color=BASE, linewidth=1.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("learning rate")
    ax.set_title("MenteE Code 3B — learning-rate schedule")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lr_schedule.png", dpi=150)
    plt.close(fig)
    print(f"  - lr_schedule.png")


def chart_gpu(gpu_log: Path, out: Path):
    """nvidia-smi CSV: timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu"""
    if not gpu_log.exists():
        print("  - gpu chart: no gpu_log.csv, skipped")
        return
    rows = list(csv.DictReader(gpu_log.open()))
    if len(rows) < 3:
        print("  - gpu chart: too few samples, skipped")
        return

    def num(val: str) -> float:
        m = re.search(r"-?\d+(?:\.\d+)?", str(val))
        return float(m.group()) if m else 0.0

    def norm(r: dict) -> dict:
        # headers look like "utilization.gpu [%]", "power.draw [W]" -> normalize keys
        return {k.split(" [")[0].strip(): v for k, v in r.items()}

    def parse_ts(s: str) -> datetime:
        # nvidia-smi timestamps may include fractional seconds: 2026/10/25 08:20:15.802
        return datetime.strptime(s.split(".")[0].strip(), "%Y/%m/%d %H:%M:%S")

    rows = [norm(r) for r in rows]
    t0 = parse_ts(rows[0]["timestamp"])
    minutes = [(parse_ts(r["timestamp"]) - t0).total_seconds() / 60 for r in rows]
    util = [num(r["utilization.gpu"]) for r in rows]
    vram = [num(r["memory.used"]) / 1024 for r in rows]
    power = [num(r["power.draw"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(minutes, util, color=MENTEE)
    axes[0].set_ylabel("GPU util (%)")
    axes[0].set_ylim(0, 105)
    axes[1].plot(minutes, vram, color="#0ea5e9")
    axes[1].set_ylabel("VRAM (GB)")
    axes[2].plot(minutes, power, color="#f59e0b")
    axes[2].set_ylabel("power (W)")
    axes[2].set_xlabel("minutes since training start")
    fig.suptitle("MenteE Code 3B — GPU telemetry during training (measured)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gpu_telemetry.png", dpi=150)
    plt.close(fig)
    print(f"  - gpu_telemetry.png ({len(rows)} samples)")


def chart_humaneval(tuned: Path | None, base: Path | None, out: Path):
    bars, labels, colors = [], [], []
    if base and base.exists():
        d = json.loads(base.read_text())
        bars.append(d["pass_at_1"])
        labels.append(f"Qwen2.5-Coder-3B\n(base, measured)")
        colors.append(BASE)
    if tuned and tuned.exists():
        d = json.loads(tuned.read_text())
        bars.append(d["pass_at_1"])
        labels.append(f"MenteE Code 3B\n(measured)")
        colors.append(MENTEE)
    if not bars:
        print("  - humaneval chart: no result files, skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, bars, color=colors, width=0.5)
    for i, v in enumerate(bars):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("HumanEval pass@1 (%)")
    ax.set_title("HumanEval — measured on identical hardware")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "humaneval_compare.png", dpi=150)
    plt.close(fig)
    print("  - humaneval_compare.png")


# vendor-reported approximations, clearly labeled as such on the chart
LANDSCAPE = [
    ("StarCoder2-3B", 3, 31),
    ("Phi-3-mini (3.8B)", 3.8, 62),
    ("Qwen2.5-Coder-3B", 3.1, 84),
    ("GPT-4o", None, 91),  # closed: no param count -> plotted on the right edge
    ("Claude 3.5 Sonnet", None, 92),
]


def chart_landscape(tuned: Path | None, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, params, score in LANDSCAPE:
        if params:
            ax.scatter(params, score, color=BASE, zorder=3)
            ax.annotate(f"{name}\n({score}%)", (params, score), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color="#334155")
    # closed models: horizontal reference lines (honest — no param count to plot)
    ax.axhline(91, color=BASE, linestyle=":", alpha=0.5)
    ax.text(0.35, 91.5, "GPT-4o ~91 (vendor)", fontsize=8, color=BASE)
    ax.axhline(92, color=BASE, linestyle=":", alpha=0.5)
    ax.text(0.35, 92.5, "Claude 3.5 Sonnet ~92 (vendor)", fontsize=8, color=BASE)

    if tuned and tuned.exists():
        d = json.loads(tuned.read_text())
        ax.scatter(3.1, d["pass_at_1"], s=180, color=MENTEE, zorder=4, marker="*")
        ax.annotate(f"MenteE Code 3B\n({d['pass_at_1']}%, measured)",
                    (3.1, d["pass_at_1"]), textcoords="offset points",
                    xytext=(10, -10), fontsize=9, fontweight="bold", color=MENTEE)

    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_ylabel("HumanEval pass@1 (%)")
    ax.set_title("Code-model landscape — our point is measured, others vendor-reported")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "model_landscape.png", dpi=150)
    plt.close(fig)
    print("  - model_landscape.png")


def find_trainer_state(primary: str) -> Path | None:
    """Accept the file directly, or find the newest checkpoint's copy."""
    p = Path(primary)
    if p.exists():
        return p
    base = p.parent
    if base.exists():
        cands = sorted(base.glob("checkpoint-*/trainer_state.json"))
        if cands:
            return cands[-1]  # highest checkpoint-N
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-state", default="outputs/full_run/trainer_state.json")
    ap.add_argument("--humaneval-tuned", default="outputs/humaneval_results_tuned.json")
    ap.add_argument("--humaneval-base", default="outputs/humaneval_results_base.json")
    ap.add_argument("--gpu-log", default="outputs/gpu_log.csv")
    ap.add_argument("--out", default="outputs/charts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log_history = []
    state_path = find_trainer_state(args.trainer_state)
    if state_path:
        print(f"  trainer state: {state_path}")
        state = json.loads(state_path.read_text())
        log_history = state.get("log_history", [])
    else:
        print(f"  ! no trainer_state found at {args.trainer_state} (or checkpoint-*/)" +
              " — training charts skipped")

    if log_history:
        chart_loss(log_history, out)
        chart_accuracy(log_history, out)
        chart_lr(log_history, out)
    chart_gpu(Path(args.gpu_log), out)
    chart_humaneval(Path(args.humaneval_tuned), Path(args.humaneval_base), out)
    chart_landscape(Path(args.humaneval_tuned), out)

    print(f"\n  charts -> {out}/")


if __name__ == "__main__":
    raise SystemExit(main())
