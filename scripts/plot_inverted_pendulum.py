"""Plot recorded experiments; no simulator runs or synthetic measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    runs = [(directory, label, color) for directory, label, color in [
        ("matched-h64", "64-step gradients", "#727b84"),
        ("matched-h128", "128-step gradients", "#2166ac"),
        ("matched-h256", "256-step gradients", "#d55e00"),
        ("matched-h1000", "1,000-step gradients", "#398145"),
    ] if (args.root / directory / "evaluation.json").exists()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for directory, label, color in runs:
        rows = [json.loads(line) for line in (args.root / directory / "metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if "validation" in r]
        axes[0].plot([r["environment_steps"] / 1e6 for r in rows],
                     [r["validation"]["mean_return"] for r in rows],
                     label=label, color=color, linewidth=1.4, marker="o", markersize=4)
    axes[0].set(title="Genesis validation: 64 fixed reset states", xlabel="Training transitions (millions)",
                ylabel="Mean survival return / 1,000", ylim=(0, 1100))
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)
    baseline = json.loads((args.root / "matched-h64" / "baseline_gym.json").read_text())
    scores = [[r["return"] for r in baseline["records"]]]
    labels = ["Zero output"]
    for directory, label, _ in runs:
        result = json.loads((args.root / directory / "evaluation.json").read_text())
        scores.append([r["return"] for r in result["gymnasium_mujoco"]["records"]])
        labels.append(label.replace(" gradients", "\ntrained"))
    independent = args.root / "independent-evaluation" / "evaluation.json"
    if independent.exists():
        result = json.loads(independent.read_text())
        scores.append([r["return"] for r in result["records"]])
        labels.append("Final policy\nnew seeds")
    axes[1].boxplot(scores, tick_labels=labels,
                    showmeans=True, whis=(0, 100))
    for i, values in enumerate(scores, start=1):
        axes[1].text(i, max(values) + 30, f"mean {np.mean(values):.1f}", ha="center", fontsize=8)
    axes[1].axhline(1000, color="#398145", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].set(title="Unmodified MuJoCo benchmark: 100 episodes each",
                ylabel="Undiscounted survival return / 1,000", ylim=(0, 1100))
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Public Genesis 1.4.5 · end-to-end physics gradients · RTX 4090 GPU 1", fontsize=11)
    for extension in ("png", "svg"):
        fig.savefig(args.root / f"learning.{extension}", dpi=180)


if __name__ == "__main__":
    main()
