"""
Generates the four figures of the TFG from JSON files in the format of
``wandb-summary.json`` (one per model). The data is read from the JSON, there is nothing hardcoded in the code.

  1. f1_per_ods_comparison.png     - F1 bars for ODS, N models
  2. threshold_tuning_diagram.png  - threshold tuning process
  3. roc_auc_comparison.png        - ROC-AUC and Average Precision micro/macro
  4. radar_f1_per_ods.png          - F1 radar diagram for ODS

Use:
  python src/make_figures.py \\
      --rf results/rf_summary.json \\
      --xgb results/xgb_summary.json \\
      --berta results/berta_summary.json \\
      --mmbert results/mmbert_summary.json \\
      --out figures/results

Keys expected in each JSON (missing ones are filled with 0):
    test/f1_ODS_{1..17}, test/roc_auc_micro, test/roc_auc_macro,
    test/ap_micro, test/ap_macro, test/f1_macro
    test_tuned/f1_ODS_{1..17}, test_tuned/f1_macro (optional, transformers)
    threshold/ODS {1..17} (optional, transformers)
"""

import argparse
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Shared colors
C_RF = "#6c757d"
C_XGB = "#2d6a4f"
C_BERTA = "#e76f51"
C_MM = "#264653"

N_ODS = 17


# Loading & extraction
def load_summary(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def has_tuned(summary):
    return any(k.startswith("test_tuned/") for k in summary)


def f1_per_ods(summary, prefix):
    """Returns the list of 17 F1 scores (one per ODS). Missing keys → 0.0."""
    return [
        float(summary.get(f"{prefix}/f1_ODS_{i}", 0.0)) for i in range(1, N_ODS + 1)
    ]


def scalar(summary, key, default=0.0):
    return float(summary.get(key, default))


def thresholds(summary):
    """Returns the list of thresholds (one per ODS). Empty if none exist."""
    out = {}
    for k, v in summary.items():
        if k.startswith("threshold/"):
            out[k.split("/", 1)[1]] = float(v)
    if not out:
        return {}
    # Sort by ODS number
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split()[-1])))


# Figures
def fig_f1_per_ods(models, out_dir):
    """models: list of (name, color, f1_list)."""
    print("Generating f1_per_ods_comparison.png ...")
    n = len(models)
    x = np.arange(N_ODS)
    w = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(14, 5))

    for (name, color, vals), off in zip(models, offsets):
        ax.bar(x + off, vals, w, label=name, color=color, alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"ODS {i}" for i in range(1, N_ODS + 1)], fontsize=9, rotation=35, ha="right"
    )
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "F1-score per ODS - comparison of models (test)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Mark (*) the ODS where no model exceeds 0.20
    per_ods = list(zip(*[m[2] for m in models]))
    for i, vals in enumerate(per_ods):
        if max(vals) < 0.20:
            ax.annotate(
                "*", xy=(i, max(vals) + 0.03), ha="center", fontsize=12, color="#c0392b"
            )

    fig.tight_layout()
    _save(fig, out_dir, "f1_per_ods_comparison.png")


def fig_threshold_tuning(thr_model, out_dir):
    """thr_model: dict with keys
    name, color, thresholds (dict 'ODS N'->val),
    f1_default, f1_tuned, paired (list of (name, color, def, tuned)).
    """
    print("Generating threshold_tuning_diagram.png ...")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (A) Training pipeline
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(A) Training", fontsize=11, fontweight="bold", pad=8)

    for bx, by, txt, fc in [
        (5, 8.5, "Training\nSet", C_MM),
        (5, 6.0, "Transformer\nModel (fine-tuning)", C_XGB),
        (5, 3.5, "Best Checkpoint\n(val F1-macro)", C_BERTA),
    ]:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 2.2, by - 0.7),
                4.4,
                1.4,
                boxstyle="round,pad=0.15",
                facecolor=fc,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            bx,
            by,
            txt,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            zorder=4,
        )

    for y1, y2 in [(7.8, 6.7), (5.3, 4.2)]:
        ax.annotate(
            "",
            xy=(5, y2),
            xytext=(5, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.8),
        )

    # (B) Optimal thresholds per class
    ax = axes[1]
    ax.set_title(
        f"(B) Threshold Tuning - {thr_model['name']} (validation)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    thr = thr_model["thresholds"]
    if thr:
        ods_names = list(thr.keys())
        vals = list(thr.values())
        colors_bar = [
            C_BERTA if v < 0.5 else (C_XGB if v < 0.75 else C_MM) for v in vals
        ]

        y_pos = np.arange(len(ods_names))
        bars = ax.barh(y_pos, vals, color=colors_bar, alpha=0.85, height=0.65, zorder=3)
        ax.axvline(
            0.5,
            color="#c0392b",
            linestyle="--",
            lw=1.5,
            label="Default threshold (0.5)",
            zorder=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ods_names, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Optimal threshold", fontsize=10)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")
        for bar, val in zip(bars, vals):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=7.5,
                color="#333",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No threshold data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        ax.axis("off")

    # (C) F1-macro gain
    ax = axes[2]
    ax.set_title(
        "(C) F1-macro gain per threshold tuning", fontsize=11, fontweight="bold", pad=8
    )

    paired = thr_model["paired"]
    if paired:
        x2 = np.arange(len(paired))
        w2 = 0.3
        defaults = [p[2] for p in paired]
        tuned = [p[3] for p in paired]
        colors_p = [p[1] for p in paired]

        ax.bar(
            x2 - w2 / 2,
            defaults,
            w2,
            label="Default threshold (0.5)",
            color=colors_p,
            alpha=0.5,
            zorder=3,
        )
        ax.bar(
            x2 + w2 / 2,
            tuned,
            w2,
            label="Tuned thresholds",
            color=colors_p,
            alpha=0.95,
            zorder=3,
        )

        for i, (d, t) in enumerate(zip(defaults, tuned)):
            ax.annotate(
                f"{t - d:+.3f}",
                xy=(i + w2 / 2, t),
                xytext=(i + w2 / 2, t + 0.025),
                ha="center",
                fontsize=9,
                color="#c0392b",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            )

        ax.set_xticks(x2)
        ax.set_xticklabels([p[0] for p in paired], fontsize=11)
        ax.set_ylabel("F1-macro (test)", fontsize=10)
        ymax = max(max(defaults), max(tuned)) + 0.15
        ax.set_ylim(0, max(0.85, ymax))
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "No threshold data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        ax.axis("off")

    fig.tight_layout(pad=2.0)
    _save(fig, out_dir, "threshold_tuning_diagram.png")


def fig_threshold_per_ods(thr_models, out_dir, filename="threshold_per_ods.png"):
    """Thresholds per ODS as an independent figure.

    Accepts either:
      - one dict ``{'name': ..., 'color': ..., 'thresholds': {...}}``
      - a list of dicts (one per model) to enable side-by-side comparison.

    When there is a single model, the colors are semantic (permissive/balanced/
    conservative). When there are more than one, each model has its own color and the
    bars are grouped by ODS so they can be compared directly.
    """
    if isinstance(thr_models, dict):
        thr_models = [thr_models]
    thr_models = [m for m in thr_models if m.get("thresholds")]

    print(f"Generating {filename} ...")

    if not thr_models:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(
            0.5,
            0.5,
            "No threshold data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        ax.axis("off")
        fig.tight_layout(pad=2.0)
        _save(fig, out_dir, filename)
        return

    if len(thr_models) == 1:
        _fig_threshold_per_ods_single(thr_models[0], out_dir, filename)
    else:
        _fig_threshold_per_ods_multi(thr_models, out_dir, filename)


def _fig_threshold_per_ods_single(thr_model, out_dir, filename):
    C_PERMISSIVE = "#52b788"
    C_BALANCED = "#adb5bd"
    C_CONSERVATIVE = "#bc4749"

    def color_for(v):
        if v < 0.5:
            return C_PERMISSIVE
        if v < 0.75:
            return C_BALANCED
        return C_CONSERVATIVE

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_title(
        f"Optimal thresholds per ODS - {thr_model['name']} (validation)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    items = sorted(thr_model["thresholds"].items(), key=lambda kv: kv[1], reverse=True)
    ods_names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors_bar = [color_for(v) for v in vals]

    y_pos = np.arange(len(ods_names))
    bars = ax.barh(y_pos, vals, color=colors_bar, alpha=0.9, height=0.66, zorder=3)
    ax.axvline(0.5, color="#333", linestyle="--", lw=1.3, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ods_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Optimal threshold", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, vals):
        ax.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
            color="#333",
        )

    max_i = int(np.argmax(vals))
    min_i = int(np.argmin(vals))
    ax.annotate(
        "more conservative",
        xy=(vals[max_i], y_pos[max_i]),
        xytext=(vals[max_i] + 0.04, y_pos[max_i] - 0.6),
        fontsize=8,
        fontweight="bold",
        color=C_CONSERVATIVE,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=C_CONSERVATIVE, lw=0.8),
    )
    ax.annotate(
        "more permissive",
        xy=(vals[min_i], y_pos[min_i]),
        xytext=(vals[min_i] + 0.18, y_pos[min_i] + 0.6),
        fontsize=8,
        fontweight="bold",
        color=C_PERMISSIVE,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=C_PERMISSIVE, lw=0.8),
    )

    legend_handles = [
        mpatches.Patch(color=C_PERMISSIVE, label="Permissive (<0.5)"),
        mpatches.Patch(color=C_BALANCED, label="Balanced (0.5–0.75)"),
        mpatches.Patch(color=C_CONSERVATIVE, label="Conservative (≥0.75)"),
        plt.Line2D(
            [0],
            [0],
            color="#333",
            linestyle="--",
            lw=1.3,
            label="Default threshold (0.5)",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=2.0)
    _save(fig, out_dir, filename)


def _fig_threshold_per_ods_multi(thr_models, out_dir, filename):
    """Horizontal bars grouped by ODS - one group per model."""
    n_models = len(thr_models)

    # Union of all present ODS, ordered by number
    ods_set = set()
    for m in thr_models:
        ods_set.update(m["thresholds"].keys())
    ods_names = sorted(ods_set, key=lambda s: int(s.split()[-1]))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    names = " vs ".join(m["name"] for m in thr_models)
    ax.set_title(
        f"Optimal thresholds per ODS - {names} (validation)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    # Background bands to illustrate the three strategies
    ax.axvspan(0.00, 0.50, color="#52b788", alpha=0.07, zorder=0)
    ax.axvspan(0.50, 0.75, color="#adb5bd", alpha=0.10, zorder=0)
    ax.axvspan(0.75, 1.05, color="#bc4749", alpha=0.07, zorder=0)

    y_base = np.arange(len(ods_names))
    h = 0.8 / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * h

    for m, off in zip(thr_models, offsets):
        vals = [m["thresholds"].get(o, np.nan) for o in ods_names]
        y = y_base + off
        bars = ax.barh(
            y,
            vals,
            height=h * 0.9,
            color=m.get("color", "#666"),
            alpha=0.92,
            label=m["name"],
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    v + 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}",
                    va="center",
                    fontsize=7.5,
                    color="#333",
                )

    ax.axvline(0.5, color="#333", linestyle="--", lw=1.2, zorder=4)

    ax.set_yticks(y_base)
    ax.set_yticklabels(ods_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Optimal threshold", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Zone labels, above the first row (y-axis inverted)
    y_top = ax.get_ylim()[1]  # minimum value of y after inversion
    for x_pos, txt, col in [
        (0.25, "permissive", "#2d6a4f"),
        (0.625, "balanced", "#666"),
        (0.90, "conservative", "#9c2b2b"),
    ]:
        ax.text(
            x_pos,
            y_top + 0.15,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            color=col,
            fontweight="bold",
            alpha=0.85,
            clip_on=False,
        )

    legend_handles = [
        mpatches.Patch(color=m.get("color", "#666"), label=m["name"])
        for m in thr_models
    ] + [
        plt.Line2D(
            [0],
            [0],
            color="#333",
            linestyle="--",
            lw=1.2,
            label="Default threshold (0.5)",
        )
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=2.0)
    _save(fig, out_dir, filename)


def fig_roc_auc(models, out_dir):
    """models: list of (name, color, roc_micro, roc_macro, ap_micro, ap_macro)."""
    print("Generating roc_auc_comparison.png ...")

    names = [m[0] for m in models]
    colors = [m[1] for m in models]
    roc_mi = [m[2] for m in models]
    roc_ma = [m[3] for m in models]
    ap_mi = [m[4] for m in models]
    ap_ma = [m[5] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    x3 = np.arange(len(models))
    w3 = 0.32
    solid_patch = mpatches.Patch(facecolor="#aaa", label="micro (solid)")
    hatch_patch = mpatches.Patch(facecolor="#aaa", hatch="//", label="macro (hatched)")

    panels = [
        (
            axes[0],
            (roc_mi, roc_ma),
            "ROC-AUC",
            (min(0.82, min(roc_ma) - 0.02), 1.0),
            "ROC-AUC: micro and macro for each model",
        ),
        (
            axes[1],
            (ap_mi, ap_ma),
            "Average Precision",
            (max(0.0, min(ap_ma) - 0.05), 1.0),
            "Average Precision (area under PR curve): micro and macro",
        ),
    ]
    for ax, (y_solid, y_hatch), ylabel, ylim, title in panels:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        b1 = ax.bar(x3 - w3 / 2, y_solid, w3, color=colors, alpha=0.9, zorder=3)
        b2 = ax.bar(
            x3 + w3 / 2, y_hatch, w3, color=colors, alpha=0.45, zorder=3, hatch="//"
        )
        for bar, val in zip(list(b1) + list(b2), list(y_solid) + list(y_hatch)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333",
            )
        ax.set_xticks(x3)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[solid_patch, hatch_patch], fontsize=8.5, loc="lower right")

    fig.text(
        0.5,
        -0.02,
        "Note: in imbalanced problems, Average Precision (PR-AUC) is more informative than ROC-AUC,\n"
        "as ROC can be optimistic when negative classes dominate.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(pad=2.0)
    _save(fig, out_dir, "roc_auc_comparison.png")


def fig_radar(models, out_dir):
    """models: list of (name, color, f1_list)."""
    print("Generating radar_f1_per_ods.png ...")

    angles = [n / float(N_ODS) * 2 * np.pi for n in range(N_ODS)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for name, color, vals in models:
        values = list(vals) + list(vals)[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"ODS {i}" for i in range(1, N_ODS + 1)], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#666")
    ax.grid(color="#bbb", linestyle="--", linewidth=0.6)
    ax.set_title("F1-score per ODS", fontsize=13, fontweight="bold", pad=20, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=10)

    fig.tight_layout()
    _save(fig, out_dir, "radar_f1_per_ods.png")


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# Input
def build_model(name, color, summary_path, use_tuned_if_available=True):
    """Read the JSON and package everything the figures need."""
    summary = load_summary(summary_path)
    prefix = "test_tuned" if (use_tuned_if_available and has_tuned(summary)) else "test"

    return {
        "name": name,
        "color": color,
        "summary": summary,
        "prefix": prefix,
        "f1": f1_per_ods(summary, prefix),
        "roc_micro": scalar(summary, f"{prefix}/roc_auc_micro"),
        "roc_macro": scalar(summary, f"{prefix}/roc_auc_macro"),
        "ap_micro": scalar(summary, f"{prefix}/ap_micro"),
        "ap_macro": scalar(summary, f"{prefix}/ap_macro"),
        "f1_default": scalar(summary, "test/f1_macro"),
        "f1_tuned": (
            scalar(summary, "test_tuned/f1_macro") if has_tuned(summary) else None
        ),
        "thresholds": thresholds(summary),
    }


def main(figures: list[int] = [1, 2, 3, 4, 5]):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rf", help="JSON summary for Random Forest")
    ap.add_argument("--xgb", help="JSON summary for XGBoost")
    ap.add_argument("--berta", help="JSON summary for BERTa")
    ap.add_argument("--mmbert", help="JSON summary for mmBERT")
    ap.add_argument(
        "--out", default="figures/results", help="Directory to save the figures"
    )
    ap.add_argument(
        "--threshold-model",
        default="berta",
        choices=["berta", "mmbert"],
        help="Model whose thresholds are shown in panel (B)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    specs = [
        ("Random Forest", C_RF, args.rf),
        ("XGBoost", C_XGB, args.xgb),
        ("BERTa", C_BERTA, args.berta),
        ("mmBERT", C_MM, args.mmbert),
    ]
    models = [build_model(n, c, p) for n, c, p in specs if p]

    if not models:
        ap.error("At least one of --rf/--xgb/--berta/--mmbert is required.")

    # Fig 1: F1 per ODS
    if 1 in figures:
        fig_f1_per_ods([(m["name"], m["color"], m["f1"]) for m in models], args.out)

    # Fig 2: threshold tuning (needs a model with tuning)
    if 2 in figures:
        target = args.threshold_model
        name_map = {"berta": "BERTa", "mmbert": "mmBERT"}
        thr_model_obj = next(
            (m for m in models if m["name"] == name_map[target] and m["thresholds"]),
            None,
        )
        if thr_model_obj is None:
            thr_model_obj = next((m for m in models if m["thresholds"]), None)

        paired = [
            (m["name"], m["color"], m["f1_default"], m["f1_tuned"])
            for m in models
            if m["f1_tuned"] is not None
        ]

        if thr_model_obj or paired:
            fig_threshold_tuning(
                {
                    "name": thr_model_obj["name"] if thr_model_obj else "-",
                    "thresholds": thr_model_obj["thresholds"] if thr_model_obj else {},
                    "paired": paired,
                },
                args.out,
            )
        else:
            print("  · threshold_tuning_diagram.png: no model with tuning, skipping.")

    # Fig 5: thresholds per ODS - single model or BERTa vs mmBERT comparison
    if 5 in figures:
        thr_payload = [
            {"name": m["name"], "color": m["color"], "thresholds": m["thresholds"]}
            for m in models
            if m["thresholds"]
        ]
        if thr_payload:
            fig_threshold_per_ods(thr_payload, args.out)
        else:
            print("  · threshold_per_ods.png: no model with thresholds, skipping.")

    # Fig 3: ROC-AUC and AP
    if 3 in figures:
        fig_roc_auc(
            [
                (
                    m["name"],
                    m["color"],
                    m["roc_micro"],
                    m["roc_macro"],
                    m["ap_micro"],
                    m["ap_macro"],
                )
                for m in models
            ],
            args.out,
        )

    # Fig 4: radar
    if 4 in figures:
        fig_radar([(m["name"], m["color"], m["f1"]) for m in models], args.out)

    print("\nDone! Figures saved to:", os.path.abspath(args.out))


if __name__ == "__main__":
    figures = [1, 2, 3, 4, 5]  # Generate all figures by default
    main(figures)
