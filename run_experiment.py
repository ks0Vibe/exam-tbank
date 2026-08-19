"""Run controlled comparison of memory mechanisms."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from src.environment import TextRuleEnvironment
from src.memory import DecisionAwareMemory, RecentMemory, SemanticMemory
from src.model import train_decision_encoder


METHODS = ("recent", "semantic", "decision_aware")


def build_memory(name: str, budget: int, encoder):
    if name == "recent":
        return RecentMemory(budget)
    if name == "semantic":
        return SemanticMemory(budget)
    return DecisionAwareMemory(budget, encoder)


def evaluate(regime: str, budget: int, seed: int, episodes: int, encoder) -> dict[str, float]:
    scores = {name: [] for name in METHODS}
    env = TextRuleEnvironment(regime, seed)
    for _ in range(episodes):
        calibration, queries = env.episode()
        memories = {name: build_memory(name, budget, encoder) for name in METHODS}
        for item in calibration:
            for memory in memories.values():
                memory.add(item)
        for query in queries:
            for name, memory in memories.items():
                prediction = memory.predict(query)
                scores[name].append(float(prediction == query.action))
    return {name: mean(values) for name, values in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.episodes, args.budgets = min(args.episodes, 40), [2, 6]
    encoder, encoder_accuracy = train_decision_encoder()
    if encoder_accuracy < 0.99:
        raise RuntimeError(f"Decision encoder did not fit cue task: {encoder_accuracy:.3f}")

    rows = []
    for regime in ("aligned", "confounded"):
        for budget in args.budgets:
            per_seed = [evaluate(regime, budget, seed, args.episodes, encoder) for seed in args.seeds]
            for method in METHODS:
                values = [out[method] for out in per_seed]
                rows.append({"regime": regime, "budget": budget, "method": method,
                             "accuracy_mean": mean(values), "accuracy_std": float(np.std(values)),
                             "n_seeds": len(values), "episodes_per_seed": args.episodes,
                             "encoder_cue_accuracy": encoder_accuracy})

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['regime']:10s} budget={row['budget']} {row['method']:14s} "
              f"accuracy={row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, regime in zip(axes, ("aligned", "confounded")):
        for method in METHODS:
            subset = [r for r in rows if r["regime"] == regime and r["method"] == method]
            ax.plot([r["budget"] for r in subset], [r["accuracy_mean"] for r in subset], marker="o", label=method)
        ax.set_title(regime); ax.set_xlabel("memory budget (experiences)"); ax.set_ylim(0, 1.05); ax.grid(alpha=.25)
    axes[0].set_ylabel("query action accuracy"); axes[1].legend(frameon=False)
    fig.suptitle("Decision-aware memory only helps when similarity is confounded")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_budget.png", dpi=160)


if __name__ == "__main__":
    main()
