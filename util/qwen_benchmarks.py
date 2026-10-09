"""Published Qwen3.5 benchmarks versus the free energy used in llm_ht.ipynb.

Run `python qwen_benchmarks.py --refresh` to retrieve official model-card scores;
subsequent runs use the saved CSV and do not require network access.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qwen3_5"
FIGURES = ROOT / "figures" / "paper"
MODELS = {
    "0.8b": "0.8B", "2b": "2B", "4b": "4B", "9b": "9B",
    "35b": "35B-A3B", "122b": "122B-A10B", "397b": "397B-A17B",
}
BENCHMARKS = ["MMLU-Pro", "MMLU-Redux", "C-Eval", "SuperGPQA"]


def fetch_benchmarks():
    """Extract each model's own column at a recorded Hugging Face revision."""
    import requests

    records, provenance = [], []
    for size, suffix in MODELS.items():
        repo = f"Qwen/Qwen3.5-{suffix}"
        response = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
        url = f"https://huggingface.co/{repo}/raw/{revision}/README.md"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        source = response.text
        table = re.findall(r"<table\b[^>]*>(.*?)</table>", source, re.S)[0]
        rows = []
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S):
            rows.append([
                html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
                for cell in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row, re.S)
            ])
        column = rows[0].index(f"Qwen3.5-{suffix}")
        section, scores, sections = "", {}, {}
        for row in rows[1:]:
            if len(row) == 1:
                section = row[0]
                continue
            if not row or row[0] not in BENCHMARKS:
                continue
            # Small models publish two modes: explicitly select Thinking.
            if size in ("0.8b", "2b") and section != "Knowledge & STEM (Thinking)":
                continue
            if row[0] in scores:
                raise ValueError(f"Ambiguous benchmark row: {repo}, {row[0]}")
            scores[row[0]] = float(row[column])
            sections[row[0]] = section
        if set(scores) != set(BENCHMARKS):
            raise ValueError(f"Missing benchmarks in {repo}: {scores}")
        if not all(0 <= score <= 100 for score in scores.values()):
            raise ValueError(f"Invalid scores in {repo}")
        records.append({
            "model_size": size, "model": repo,
            "local_weights": "FP8 folder" if size in ("35b", "122b", "397b") else "standard folder",
            **scores, "source_url": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
        })
        provenance.append({
            "model": repo, "revision": revision, "raw_url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "readme_sha256": hashlib.sha256(response.content).hexdigest(),
            "source_sections": sections,
            "mode_basis": "explicit Thinking section" if size in ("0.8b", "2b") else "main language table; model card states thinking mode is default",
        })
    df = pd.DataFrame(records)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "published_benchmarks.csv", index=False)
    (DATA / "published_benchmarks_sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return df


def compute_free_energy(e):
    """Use the current per-eigenvalue free energy (n=len(e), jitter=1e-20).

    Preserve tensor dtype, torch reductions, and lambda = 1 / mean(e).
    """
    if e.ndim != 1 or not torch.isfinite(e).all() or not torch.all(e + 1e-20 > 0):
        raise ValueError("Expected a finite, positive saved spectrum")
    n = len(e)
    fro_norm = torch.sum(e + 1e-20).item() / n
    lam = 1 / e.mean().item()
    logdet_term = torch.sum(torch.log((e + 1e-20))).item() / n
    value = lam / 2 * fro_norm - 0.5 * logdet_term + 1/2 * np.log(2 * np.pi / lam)
    return float(lam), float(value)

def build_comparison():
    df = pd.read_csv(DATA / "published_benchmarks.csv").set_index("model_size")
    if set(df.index) != set(MODELS) or not df.index.is_unique:
        raise ValueError("Expected exactly the seven notebook models")
    df = df.loc[list(MODELS)].copy()
    for size in MODELS:
        path = DATA / f"{size}_lm_head_eigs.pt"
        spectrum = torch.load(path, map_location="cpu", weights_only=False)
        e = torch.from_numpy(spectrum["eigvals"])
        lam, energy = compute_free_energy(e)
        df.loc[size, "optimal_lambda"] = lam
        df.loc[size, "free_energy"] = energy
        df.loc[size, "spectrum_length"] = len(e)
        df.loc[size, "spectrum_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    df = df.reset_index()
    df.to_csv(DATA / "free_energy_vs_benchmarks.csv", index=False)
    return df


def plot_comparison(df, benchmark="MMLU-Pro", *, min_size_b=None,
                    model_sizes=None, figsize=(0.48 * 5.5, 2.0)):
    """Return one benchmark figure and its correlations; save a PDF like Pythia.

    Call set_iclr_style() first to use the notebook's formatting.
    min_size_b filters by total parameters in billions (including MoE models).
    model_sizes optionally selects explicit keys, e.g. ["2b", "4b", "9b"].
    When both filters are supplied, their intersection is plotted. Colours
    remain fixed across subsets. Each call overwrites the same Qwen PDF.
    """
    from matplotlib.ticker import FuncFormatter
    from scipy import stats

    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}")
    selected = df.copy()
    if model_sizes is not None:
        if isinstance(model_sizes, str):
            model_sizes = [model_sizes]
        unknown = set(model_sizes) - set(MODELS)
        if unknown:
            raise ValueError(f"Unknown model sizes: {sorted(unknown)}")
        selected = selected[selected["model_size"].isin(model_sizes)]
    if min_size_b is not None:
        sizes_b = selected["model_size"].map(lambda size: float(size[:-1]))
        selected = selected[sizes_b >= min_size_b]
    if len(selected) < 2:
        raise ValueError("Select at least two models for the fit and correlations")
    x = selected["free_energy"].to_numpy()
    y = selected[benchmark].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Free energy and benchmark scores must be finite")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Free energy and benchmark scores must vary across models")

    r = float(np.corrcoef(x, y)[0, 1])
    rho = float(stats.spearmanr(x, y).statistic)
    kendall = stats.kendalltau(x, y)
    correlations = pd.DataFrame([{
        "benchmark": benchmark, "pearson_r": r, "spearman_rho": rho,
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue), "n": len(selected),
    }])
    print(f"Spearman correlation ({benchmark}): {rho:.2f}")
    print(f"Pearson correlation ({benchmark}): {r:.2f}")
    print(f"Kendall correlation ({benchmark}): {kendall.statistic:.2f}, "
          f"p-value: {kendall.pvalue:.2e}")

    model_colors = {size: plt.get_cmap("tab10")(i % 10)
                    for i, size in enumerate(MODELS)}
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for _, row in selected.iterrows():
        size = row["model_size"]
        ax.scatter(row["free_energy"], row[benchmark], marker="o",
                   color=model_colors[size], label=size.upper())
    m, b = np.polyfit(x, y, 1)
    x_fit = np.sort(x)
    ax.plot(x_fit, m * x_fit + b, linestyle="--", color="green")
    ax.set_title("Qwen3.5")
    ax.set_xlabel("Free Energy")
    ax.set_ylabel(f"Test Acc ({benchmark})")
    ax.set_xticks(np.linspace(x.min(), x.max(), 3))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, pos: f"{value:.3f}".rstrip("0").rstrip(".")
    ))
    ax.legend(loc="best", frameon=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "free_energy_vs_acc_qwen3_5.pdf",
                bbox_inches="tight", format="pdf")
    return fig, correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch scores from official model cards")
    args = parser.parse_args()
    if args.refresh:
        fetch_benchmarks()
    comparison = build_comparison()
    figure, correlations = plot_comparison(comparison)
    print(comparison[["model_size", "free_energy", *BENCHMARKS]].to_string(index=False))
    print(correlations.to_string(index=False))
    plt.close(figure)
