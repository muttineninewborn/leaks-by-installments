"""Figure 2: leak rate for renumbered practice variants in live run 2.

Usage: python study/plot_results.py
Reads results/live_tutor_*.json and writes paper/fig_results.pdf (pdftocairo, if installed, embeds the fonts).
"""
import json
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUTORS = [("results/live_tutor_a_qwen.json", "Tutor A (Qwen2.5-3B)", "///", "white"),
          ("results/live_tutor_b_phi.json", "Tutor B (Phi-3.5-mini)", "", "0.75")]
POLICIES = [("P0", "No gate"), ("P1", "Per-turn"), ("P2", "Per-session"), ("P3", "Ledger")]


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    width = 0.38
    for j, (path, label, hatch, color) in enumerate(TUTORS):
        res = json.load(open(path))["policies"]
        vals = [res[p]["variant_leak"] for p, _ in POLICIES]
        xs = [i + (j - 0.5) * width for i in range(len(POLICIES))]
        bars = ax.bar(xs, vals, width, label=label, hatch=hatch, color=color, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels([name for _, name in POLICIES])
    ax.set_ylabel("Leak rate")
    ax.set_ylim(0, 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=7, handlelength=1.5)
    fig.tight_layout()
    fig.savefig("paper/fig_results.pdf")
    if shutil.which("pdftocairo"):  # embed fonts as Type 1C
        subprocess.run(["pdftocairo", "-pdf", "paper/fig_results.pdf", "paper/fig_results.tmp.pdf"], check=True)
        os.replace("paper/fig_results.tmp.pdf", "paper/fig_results.pdf")


if __name__ == "__main__":
    main()
