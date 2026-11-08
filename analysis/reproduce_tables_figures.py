"""Standalone analysis script extracted from the supplied research notebook.
Run from the repository root after generating the zero-shot and fine-tuned result CSVs.
"""


# ===== Original notebook code cell 1 =====
from pathlib import Path
import itertools
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", message="Pandas requires version")

# Locate the results directory whether the notebook is launched here or nearby.
candidates = [Path.cwd(), Path.cwd() / "results", Path.cwd().parent]
ROOT = next((p.resolve() for p in candidates if (p / "suite_summary__test.csv").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Could not locate suite_summary__test.csv. Run this notebook from the results directory.")

OUT = ROOT / "journal_outputs"
OUT.mkdir(parents=True, exist_ok=True)

OVERLAPS = ["overlap_15_35", "overlap_35_65", "overlap_65_85", "overlap_85_100"]
OVERLAP_LABELS = ["15–35%", "35–65%", "65–85%", "85–100%"]
LABEL_MAP = dict(zip(OVERLAPS, OVERLAP_LABELS))

FT_DIRS = {
    "Conv-TasNet": "convtasnet__30_epoch_finetune",
    "DPRNN": "dprnn__30_epoch_finetune",
    "SepFormer": "sepformer__30_epoch_finetune_cachefix",
    "MossFormer2": "mossformer2__30_epoch_finetune",
}
ZS_DIRS = {
    "Conv-TasNet": "results_zero_shot/convtasnet",
    "DPRNN": "results_zero_shot/dprnn",
    "SepFormer": "results_zero_shot/sepformer",
    "MossFormer2": "results_zero_shot/mossformer2",
}
ABLATION_DIR = "mossformer2__mossformer2_partial_9epoch"
MODELS = list(FT_DIRS)

# Okabe–Ito-inspired, colorblind-safe palette.
COLORS = {
    "Conv-TasNet": "#0072B2",
    "DPRNN": "#E69F00",
    "SepFormer": "#009E73",
    "MossFormer2": "#CC79A7",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

METRICS = ["SI-SDRi", "SDRi", "PESQi", "STOIi"]
PAIR_RE = re.compile(r"speaker\d+")
SOURCE_RE = re.compile(r"(speaker\d+)_u(\d+)")

def speaker_pair(mixture_id):
    speakers = sorted(PAIR_RE.findall(str(mixture_id))[:2])
    return "|".join(speakers)

def load_experiment(relative_dir):
    paths = [ROOT / relative_dir / "2spk" / f"test__{variation}.csv" for variation in OVERLAPS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing expected files:\n" + "\n".join(missing))
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    frame["speaker_pair"] = frame["id"].map(speaker_pair)
    return frame

def cluster_bootstrap_ci(values, clusters, n_boot=5000, seed=20260811):
    # Percentile CI for a mean, resampling speaker-pair clusters.
    work = pd.DataFrame({"value": np.asarray(values, float), "cluster": np.asarray(clusters)})
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    grouped = work.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    n_clusters = len(grouped)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, float)
    chunk = 500
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        boot[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))

def export_table(df, stem, index=False, float_format="%.4f"):
    csv_path = OUT / f"{stem}.csv"
    tex_path = OUT / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    # Lightweight Booktabs exporter avoids pandas Styler/Jinja version coupling.
    table = df.reset_index() if index else df.copy()
    def tex_escape(value):
        text = str(value)
        for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"),
                         ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
            text = text.replace(old, new)
        return text
    def tex_value(value):
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            return float_format % value
        return tex_escape(value)
    align = "l" + "r" * (len(table.columns) - 1)
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in table.columns) + r" \\", r"\midrule"]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(tex_value(v) for v in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {csv_path.name} and {tex_path.name}")

def export_figure(fig, stem):
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    print(f"Saved {stem}.pdf, {stem}.svg, and {stem}.png")

print("Results root:", ROOT)
print("Output directory:", OUT)


# ===== Original notebook code cell 3 =====
fine_tuned = {name: load_experiment(path) for name, path in FT_DIRS.items()}
zero_shot = {name: load_experiment(path) for name, path in ZS_DIRS.items()}

all_csvs = sorted(p for p in ROOT.rglob("*.csv") if OUT not in p.parents)
all_utterance_csvs = [p for p in all_csvs if p.name.startswith("test__overlap_")]
excluded_fragments = {
    "mossformer2__mossformer2_partial_9epoch",
    "sepformer__30_epoch_finetune",
}
utterance_csvs = [
    p for p in all_utterance_csvs
    if not any(fragment in p.parts for fragment in excluded_fragments)
]
summary_csvs = [p for p in all_csvs if p.name == "summary__test.csv"]
suite_csvs = [p for p in all_csvs if p.name == "suite_summary__test.csv"]

structure = pd.DataFrame({
    "CSV category": [
        "All source CSV files discovered",
        "Primary utterance-level files analyzed",
        "Excluded utterance files (old SepFormer + partial 9 epoch)",
        "Per-experiment summary files",
        "Suite summary files",
    ],
    "Files": [len(all_csvs), len(utterance_csvs),
              len(all_utterance_csvs) - len(utterance_csvs),
              len(summary_csvs), len(suite_csvs)],
    "Rows represented": [np.nan,
                         sum(sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in utterance_csvs),
                         sum(sum(1 for _ in open(p, encoding="utf-8")) - 1
                             for p in all_utterance_csvs if p not in utterance_csvs),
                         len(summary_csvs) * 4, 24 + 16],
})
display(structure)


# ===== Original notebook code cell 5 =====
audit_rows = []
identity_specs = [
    ("SI-SDRi", "SI-SDR", "SI-SDR_mixture"),
    ("SDRi", "SDR", "SDR_mixture"),
    ("PESQi", "PESQ", "PESQ_mixture"),
    ("STOIi", "STOI", "STOI_mixture"),
]

for path in utterance_csvs:
    df = pd.read_csv(path)
    max_identity_error = max(
        float(np.nanmax(np.abs(df[out] - (df[enh] - df[base]))))
        for out, enh, base in identity_specs
    )
    audit_rows.append({
        "file": str(path.relative_to(ROOT)),
        "rows": len(df),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "missing_metric_values": int(df[METRICS].isna().sum().sum()),
        "max_improvement_identity_error": max_identity_error,
    })

audit = pd.DataFrame(audit_rows)
audit_summary = pd.DataFrame({
    "Check": ["Utterance files", "Files with row count != 2,000", "Duplicate IDs",
              "Missing metric values", "Largest metric identity error"],
    "Result": [len(audit), int((audit.rows != 2000).sum()), int(audit.duplicate_ids.sum()),
               int(audit.missing_metric_values.sum()), audit.max_improvement_identity_error.max()],
})
display(audit_summary)

# Verify that every main model sees exactly the same IDs within each overlap bin.
alignment = []
for variation in OVERLAPS:
    sets = [set(fine_tuned[m].loc[fine_tuned[m].variation == variation, "id"]) for m in MODELS]
    sets += [set(zero_shot[m].loc[zero_shot[m].variation == variation, "id"]) for m in MODELS]
    alignment.append({
        "Overlap": LABEL_MAP[variation],
        "IDs per condition": len(sets[0]),
        "All IDs aligned": all(s == sets[0] for s in sets[1:]),
    })
display(pd.DataFrame(alignment))
export_table(audit, "table_supplement_file_audit")


# ===== Original notebook code cell 7 =====
reference = fine_tuned["SepFormer"].copy()
dataset_rows = []
for variation in OVERLAPS:
    group = reference[reference.variation == variation]
    speakers, sources = set(), set()
    for mixture_id in group.id:
        for speaker, utt in SOURCE_RE.findall(mixture_id):
            speakers.add(speaker)
            sources.add(f"{speaker}_u{utt}")
    dataset_rows.append({
        "Overlap": LABEL_MAP[variation],
        "Mixtures": len(group),
        "Duration_h": group.duration_sec.sum() / 3600,
        "Mean_duration_s": group.duration_sec.mean(),
        "SD_duration_s": group.duration_sec.std(ddof=1),
        "Speakers": len(speakers),
        "Speaker_pairs": group.speaker_pair.nunique(),
        "Unique_source_utterances": len(sources),
        "Mixture_PESQ": group.PESQ_mixture.mean(),
        "Mixture_STOI": group.STOI_mixture.mean(),
    })

dataset_table = pd.DataFrame(dataset_rows)
all_sources = {
    f"{speaker}_u{utt}"
    for mixture_id in reference.id
    for speaker, utt in SOURCE_RE.findall(mixture_id)
}
total_row = {
    "Overlap": "Total / overall", "Mixtures": len(reference),
    "Duration_h": reference.duration_sec.sum() / 3600,
    "Mean_duration_s": reference.duration_sec.mean(),
    "SD_duration_s": reference.duration_sec.std(ddof=1),
    "Speakers": len(set(PAIR_RE.findall(" ".join(reference.id.astype(str))))),
    "Speaker_pairs": reference.speaker_pair.nunique(),
    "Unique_source_utterances": len(all_sources),
    "Mixture_PESQ": reference.PESQ_mixture.mean(),
    "Mixture_STOI": reference.STOI_mixture.mean(),
}
dataset_table = pd.concat([dataset_table, pd.DataFrame([total_row])], ignore_index=True)
display(dataset_table.round(4))
export_table(dataset_table, "table_01_dataset_composition")


# ===== Original notebook code cell 9 =====
main_rows = []
for model_idx, model in enumerate(MODELS):
    df = fine_tuned[model]
    row = {"Model": model, "N": len(df)}
    for variation in OVERLAPS:
        row[f"SI-SDRi_{LABEL_MAP[variation]}"] = df.loc[df.variation == variation, "SI-SDRi"].mean()
    low, high = cluster_bootstrap_ci(df["SI-SDRi"], df.speaker_pair, seed=1000 + model_idx)
    row.update({
        "SI-SDRi_overall": df["SI-SDRi"].mean(),
        "SI-SDRi_CI_low": low,
        "SI-SDRi_CI_high": high,
        "SDRi_overall": df.SDRi.mean(),
        "PESQi_overall": df.PESQi.mean(),
        "STOIi_overall": df.STOIi.mean(),
        "Median_SI-SDRi": df["SI-SDRi"].median(),
        "P10_SI-SDRi": df["SI-SDRi"].quantile(0.10),
        "Failure_<0dB_pct": 100 * (df["SI-SDRi"] < 0).mean(),
        "Failure_<5dB_pct": 100 * (df["SI-SDRi"] < 5).mean(),
        "Inference_RTF_observed": df.inference_time_sec.sum() / df.duration_sec.sum(),
    })
    main_rows.append(row)

main_results = pd.DataFrame(main_rows)
display(main_results.round(4))
export_table(main_results, "table_02_main_finetuned_results")


# ===== Original notebook code cell 11 =====
fig, ax = plt.subplots(figsize=(7.1, 4.2))
x = np.arange(len(OVERLAPS))
for model_idx, model in enumerate(MODELS):
    df = fine_tuned[model]
    means, lows, highs = [], [], []
    for overlap_idx, variation in enumerate(OVERLAPS):
        group = df[df.variation == variation]
        mean = group["SI-SDRi"].mean()
        low, high = cluster_bootstrap_ci(
            group["SI-SDRi"], group.speaker_pair,
            seed=2000 + 10 * model_idx + overlap_idx,
        )
        means.append(mean); lows.append(low); highs.append(high)
    means, lows, highs = map(np.asarray, (means, lows, highs))
    ax.plot(x, means, marker="o", color=COLORS[model], label=model)
    ax.fill_between(x, lows, highs, color=COLORS[model], alpha=0.14, linewidth=0)

ax.set_xticks(x, OVERLAP_LABELS)
ax.set_xlabel("Nominal speech-overlap range")
ax.set_ylabel("SI-SDR improvement (dB)")
ax.set_title("Fine-tuned separation performance across overlap conditions")
ax.legend(frameon=False, ncol=2)
ax.margins(x=0.03)
fig.tight_layout()
export_figure(fig, "figure_01_sisdr_improvement_vs_overlap")
plt.show()


# ===== Original notebook code cell 14 =====
adaptation_rows = []
gain_by_overlap = pd.DataFrame(index=MODELS, columns=OVERLAP_LABELS, dtype=float)

for model_idx, model in enumerate(MODELS):
    ft = fine_tuned[model].set_index(["variation", "id"]).sort_index()
    zs = zero_shot[model].set_index(["variation", "id"]).sort_index()
    common = ft.index.intersection(zs.index)
    ft, zs = ft.loc[common], zs.loc[common]
    delta = ft["SI-SDRi"] - zs["SI-SDRi"]
    clusters = [speaker_pair(idx[1]) for idx in common]
    low, high = cluster_bootstrap_ci(delta, clusters, seed=3000 + model_idx)
    row = {
        "Model": model,
        "N_paired": len(common),
        "Zero_shot_SI-SDRi": zs["SI-SDRi"].mean(),
        "Fine_tuned_SI-SDRi": ft["SI-SDRi"].mean(),
        "SI-SDRi_gain": delta.mean(),
        "Gain_CI_low": low,
        "Gain_CI_high": high,
        "SI-SDRi_win_pct": 100 * (delta > 0).mean(),
        "Zero_shot_failure_<0dB_pct": 100 * (zs["SI-SDRi"] < 0).mean(),
        "Fine_tuned_failure_<0dB_pct": 100 * (ft["SI-SDRi"] < 0).mean(),
    }
    for metric in ["SDRi", "PESQi", "STOIi"]:
        d = ft[metric] - zs[metric]
        row[f"{metric}_gain"] = d.mean()
        row[f"{metric}_win_pct"] = 100 * (d > 0).mean()
    adaptation_rows.append(row)

    for variation, label in zip(OVERLAPS, OVERLAP_LABELS):
        mask = ft.index.get_level_values("variation") == variation
        gain_by_overlap.loc[model, label] = (ft.loc[mask, "SI-SDRi"] - zs.loc[mask, "SI-SDRi"]).mean()

adaptation = pd.DataFrame(adaptation_rows)
display(adaptation.round(4))
export_table(adaptation, "table_03_zero_shot_to_finetuned_gain")
export_table(gain_by_overlap.reset_index(names="Model"), "table_03b_gain_by_overlap")


# ===== Original notebook code cell 16 =====
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), gridspec_kw={"width_ratios": [1, 1.2]})

# Paired condition means.
ax = axes[0]
for y, model in enumerate(MODELS):
    row = adaptation.loc[adaptation.Model == model].iloc[0]
    ax.plot([row["Zero_shot_SI-SDRi"], row["Fine_tuned_SI-SDRi"]], [y, y], color="#999999", lw=1.4)
    ax.scatter(row["Zero_shot_SI-SDRi"], y, facecolors="white", edgecolors=COLORS[model], s=42, zorder=3)
    ax.scatter(row["Fine_tuned_SI-SDRi"], y, color=COLORS[model], s=42, zorder=3)
    ax.text(row["Fine_tuned_SI-SDRi"] + 0.35, y, f"+{row['SI-SDRi_gain']:.2f}", va="center", fontsize=8)
ax.set_yticks(range(len(MODELS)), MODELS)
ax.invert_yaxis()
ax.set_xlabel("Overall SI-SDR improvement (dB)")
ax.set_title("Zero-shot → fine-tuned\n(open → filled markers)")

# Heatmap of fine-tuning gains by overlap.
ax = axes[1]
matrix = gain_by_overlap.loc[MODELS, OVERLAP_LABELS].to_numpy(float)
im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=np.ceil(matrix.max()))
ax.set_xticks(range(len(OVERLAP_LABELS)), OVERLAP_LABELS, rotation=30, ha="right")
ax.set_yticks(range(len(MODELS)), MODELS)
ax.set_title("SI-SDRi gain by overlap (dB)")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        color = "white" if matrix[i, j] > 0.58 * matrix.max() else "black"
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Fine-tuning gain (dB)")
fig.tight_layout()
export_figure(fig, "figure_02_zero_shot_to_finetuned_gain")
plt.show()


# ===== Original notebook code cell 19 =====
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharey=True)
for ax, condition, data in zip(axes, ["Zero-shot", "Fine-tuned"], [zero_shot, fine_tuned]):
    for model in MODELS:
        values = np.sort(data[model]["SI-SDRi"].dropna().to_numpy())
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, ecdf, color=COLORS[model], label=model)
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.axvline(5, color="#777777", ls=":", lw=0.9)
    ax.set_title(condition)
    ax.set_xlabel("SI-SDR improvement (dB)")
    ax.set_xlim(-15, 35)
axes[0].set_ylabel("Empirical cumulative probability")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Distribution and failure behavior across 8,000 mixtures per system", y=1.02)
fig.tight_layout()
export_figure(fig, "figure_03_sisdr_ecdf_reliability")
plt.show()


# ===== Original notebook code cell 22 =====
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))
x = np.arange(len(OVERLAPS))
panel_specs = [("PESQ", "PESQ_mixture", "PESQ"), ("STOI", "STOI_mixture", "STOI")]

for panel_idx, (metric, baseline_metric, ylabel) in enumerate(panel_specs):
    ax = axes[panel_idx]
    for model_idx, model in enumerate(MODELS):
        df = fine_tuned[model]
        means, lows, highs = [], [], []
        for overlap_idx, variation in enumerate(OVERLAPS):
            group = df[df.variation == variation]
            mean = group[metric].mean()
            low, high = cluster_bootstrap_ci(
                group[metric], group.speaker_pair,
                seed=4000 + panel_idx * 100 + model_idx * 10 + overlap_idx,
            )
            means.append(mean); lows.append(low); highs.append(high)
        ax.plot(x, means, marker="o", color=COLORS[model], label=model)
        ax.fill_between(x, lows, highs, color=COLORS[model], alpha=0.10, linewidth=0)

    baseline = [reference.loc[reference.variation == v, baseline_metric].mean() for v in OVERLAPS]
    ax.plot(x, baseline, marker="s", color="#555555", ls="--", label="Unprocessed mixture")
    ax.set_xticks(x, OVERLAP_LABELS, rotation=20)
    ax.set_xlabel("Nominal speech-overlap range")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Absolute {metric}")

axes[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
export_figure(fig, "figure_04_absolute_pesq_stoi")
plt.show()


# ===== Original notebook code cell 25 =====
efficiency_rows = []
for model in MODELS:
    df = fine_tuned[model]
    rtf = df.inference_time_sec.sum() / df.duration_sec.sum()
    efficiency_rows.append({
        "Model": model,
        "SI-SDRi_overall": df["SI-SDRi"].mean(),
        "Inference_RTF_observed": rtf,
        "Times_real_time": 1 / rtf,
        "Batch_mode": str(df.batch_mode.drop_duplicates().tolist()),
        "Model_output_sample_rate_Hz": str(df.model_output_sample_rate.drop_duplicates().tolist()),
        "Metric_sample_rate_Hz": str(df.metric_sample_rate.drop_duplicates().tolist()),
    })
efficiency = pd.DataFrame(efficiency_rows)
display(efficiency.round(4))
export_table(efficiency, "table_04_observed_efficiency")

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for _, row in efficiency.iterrows():
    ax.scatter(row.Inference_RTF_observed, row["SI-SDRi_overall"], s=62,
               color=COLORS[row.Model], edgecolor="white", linewidth=0.7, zorder=3)
    ax.annotate(row.Model, (row.Inference_RTF_observed, row["SI-SDRi_overall"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Observed inference real-time factor (lower is faster)")
ax.set_ylabel("Overall SI-SDR improvement (dB)")
ax.set_title("Observed quality–throughput trade-off")
fig.tight_layout()
export_figure(fig, "figure_05_observed_quality_throughput")
plt.show()


# ===== Original notebook code cell 28 =====
pairwise_rows = []
for comparison_idx, (model_a, model_b) in enumerate(itertools.combinations(MODELS, 2)):
    a = fine_tuned[model_a].set_index(["variation", "id"]).sort_index()
    b = fine_tuned[model_b].set_index(["variation", "id"]).sort_index()
    common = a.index.intersection(b.index)
    delta = a.loc[common, "SI-SDRi"] - b.loc[common, "SI-SDRi"]
    clusters = [speaker_pair(idx[1]) for idx in common]
    low, high = cluster_bootstrap_ci(delta, clusters, seed=6000 + comparison_idx)
    pairwise_rows.append({
        "Model_A": model_a, "Model_B": model_b,
        "Mean_A_minus_B_dB": delta.mean(),
        "CI_low": low, "CI_high": high,
        "A_win_pct": 100 * (delta > 0).mean(),
        "N_paired": len(delta),
    })
pairwise = pd.DataFrame(pairwise_rows)
display(pairwise.round(4))
export_table(pairwise, "table_supplement_pairwise_effect_sizes")


# ===== Original notebook code cell 30 =====
correlation_rows = []
fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.1), sharex=True, sharey=True)
image_handle = None

for ax, model in zip(axes.flat, MODELS):
    df = fine_tuned[model]
    centered = df[METRICS] - df.groupby("variation")[METRICS].transform("mean")
    corr = centered.corr(method="spearman")

    for metric_a in METRICS:
        for metric_b in METRICS:
            correlation_rows.append({
                "Model": model,
                "Metric_A": metric_a,
                "Metric_B": metric_b,
                "Spearman_rho_within_overlap": corr.loc[metric_a, metric_b],
            })

    image_handle = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(METRICS)), METRICS, rotation=35, ha="right")
    ax.set_yticks(range(len(METRICS)), METRICS)
    ax.set_title(model)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) >= 0.65 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

cbar = fig.colorbar(image_handle, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
cbar.set_label("Within-overlap Spearman correlation")
fig.suptitle("Agreement among utterance-level improvement metrics", y=0.99)
fig.subplots_adjust(left=0.11, right=0.88, bottom=0.10, top=0.92, wspace=0.22, hspace=0.30)
export_figure(fig, "figure_06_metric_correlation_matrix")
plt.show()

metric_correlations = pd.DataFrame(correlation_rows)
display(metric_correlations.round(4))
export_table(metric_correlations, "table_05_metric_correlation_matrix")


# ===== Original notebook code cell 33 =====
generated = sorted(p.name for p in OUT.iterdir() if p.is_file())
print(f"Generated {len(generated)} journal assets in {OUT}:\n")
print("\n".join(generated))
