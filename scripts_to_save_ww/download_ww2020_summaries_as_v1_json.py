from pathlib import Path
from urllib.request import urlretrieve
import ast
import json
import time

import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RAW_BASE = "https://raw.githubusercontent.com/CalculatedContent/ww-trends-2020/master/data"

OUT_DIR = Path("ww_trends_2020_repo_v1_summaries")
SUMMARIES_DIR = OUT_DIR / "summaries"
DETAILS_DIR = OUT_DIR / "details"
RAW_DIR = OUT_DIR / "raw_repo_files"

OUT_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR.mkdir(exist_ok=True)
DETAILS_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUT_DIR / "summary_metrics_repo_v1.csv"


# ---------------------------------------------------------------------
# Models used in the WW trends notebooks
# ---------------------------------------------------------------------

MODELS = [
    # ResNet
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",

    # VGG
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",

    # DenseNet
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
]


# Reported PyTorch ImageNet top-1/top-5 errors used in the notebooks.
# Accuracy = 100 - error.
TOP1_ERROR = {
    # ResNet
    "resnet18": 30.24,
    "resnet34": 26.70,
    "resnet50": 23.85,
    "resnet101": 22.63,
    "resnet152": 21.69,

    # VGG
    "vgg11": 30.98,
    "vgg13": 30.07,
    "vgg16": 28.41,
    "vgg19": 27.62,
    "vgg11_bn": 29.62,
    "vgg13_bn": 28.45,
    "vgg16_bn": 26.63,
    "vgg19_bn": 25.76,

    # DenseNet
    "densenet121": 25.35,
    "densenet169": 24.00,
    "densenet201": 22.80,
    "densenet161": 22.35,
}

TOP5_ERROR = {
    # ResNet
    "resnet18": 10.92,
    "resnet34": 8.58,
    "resnet50": 7.13,
    "resnet101": 6.44,
    "resnet152": 5.94,

    # VGG
    "vgg11": 11.37,
    "vgg13": 10.75,
    "vgg16": 9.62,
    "vgg19": 9.12,
    "vgg11_bn": 10.19,
    "vgg13_bn": 9.63,
    "vgg16_bn": 8.50,
    "vgg19_bn": 8.15,

    # DenseNet
    "densenet121": 7.83,
    "densenet169": 7.00,
    "densenet201": 6.43,
    "densenet161": 6.20,
}


# Old torchvision checkpoint URLs.
# These are useful metadata only; this script does not download model weights.
WEIGHTS_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",

    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",

    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet161": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
}


def download_file(model_name, suffix):
    """
    Download repo data/{model_name}.{suffix}.
    suffix should be 'txt' or 'csv'.
    """
    url = f"{RAW_BASE}/{model_name}.{suffix}"
    local_path = RAW_DIR / f"{model_name}.{suffix}"

    if not local_path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, local_path)

    return local_path


def load_repo_summary(model_name):
    """
    Load the saved WeightWatcher summary dict from data/{model_name}.txt.
    """
    path = download_file(model_name, "txt")
    text = path.read_text()

    # The repo saved summaries using str(dict), so literal_eval is appropriate.
    summary = ast.literal_eval(text)

    if not isinstance(summary, dict):
        raise ValueError(f"{path} did not parse as a dict")

    return summary


def maybe_download_details(model_name, model_alias):
    """
    Download the repo's layer-level CSV if available, and save it under the
    same style as the previous script's details directory.
    """
    try:
        path = download_file(model_name, "csv")
    except Exception as e:
        print(f"[WARN] Could not download details CSV for {model_name}: {e}")
        return None

    out_path = DETAILS_DIR / f"{model_alias}_details.csv"

    try:
        df = pd.read_csv(path)
        df.to_csv(out_path, index=False)
    except Exception:
        # If pandas struggles with old formatting, still preserve the raw file.
        out_path.write_bytes(path.read_bytes())

    return out_path


def sanitize_value(x):
    """
    Make values JSON/CSV friendly.
    """
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    try:
        return float(x)
    except Exception:
        return str(x)


def make_row(model_name, summary):
    """
    Convert repo summary to the same row-style format used by the previous script.
    """
    model_alias = f"{model_name}_v1"

    row = {
        "model": model_alias,
        "source_model": model_name,
        "status": "ok",
        "elapsed_sec": None,

        # Metadata fields matching the previous script style
        "weights_name": f"{model_name}.IMAGENET1K_V1",
        "weights_url": WEIGHTS_URLS.get(model_name),
        "top1_acc": 100.0 - TOP1_ERROR[model_name],
        "top5_acc": 100.0 - TOP5_ERROR[model_name],
        "metric_dataset": "ImageNet-1K",
        "num_params": None,
        "recipe": "torchvision legacy pretrained=True / ImageNet1K_V1",

        # Marks that these are downloaded repo summaries, not recomputed locally
        "ww_source": "CalculatedContent/ww-trends-2020/data",
        "ww_recomputed": False,
    }

    # Preserve all saved WW summary metrics, including spectral norm.
    for k, v in summary.items():
        row[k] = sanitize_value(v)

    # Add explicit modern aliases if only old names exist.
    if "spectral_norm" not in row and "spectralnorm" in row:
        row["spectral_norm"] = row["spectralnorm"]

    if "log_spectral_norm" not in row and "logspectralnorm" in row:
        row["log_spectral_norm"] = row["logspectralnorm"]

    if "log_norm" not in row and "lognorm" in row:
        row["log_norm"] = row["lognorm"]

    if "stable_rank" not in row and "softrank" in row:
        row["stable_rank"] = row["softrank"]

    if "log_stable_rank" not in row and "softranklog" in row:
        row["log_stable_rank"] = row["softranklog"]

    if "mp_softrank" not in row and "softrank_mp" in row:
        row["mp_softrank"] = row["softrank_mp"]

    if "log_alpha_norm" not in row and "logpnorm" in row:
        row["log_alpha_norm"] = row["logpnorm"]

    return row


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    rows = []

    for i, model_name in enumerate(MODELS, start=1):
        print(f"\n===== {i}/{len(MODELS)}: {model_name} =====")

        t0 = time.time()

        summary = load_repo_summary(model_name)
        row = make_row(model_name, summary)

        # Keep elapsed time as download/parse time, not WW runtime.
        row["elapsed_sec"] = time.time() - t0

        model_alias = row["model"]

        json_path = SUMMARIES_DIR / f"{model_alias}_summary.json"
        save_json(json_path, row)

        maybe_download_details(model_name, model_alias)

        rows.append(row)

        print(f"Saved {json_path}")
        print(
            "spectral_norm =",
            row.get("spectral_norm"),
            "| log_spectral_norm =",
            row.get("log_spectral_norm"),
        )

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved combined CSV:")
    print(SUMMARY_CSV)

    print("\nColumns containing spectral:")
    for c in df.columns:
        if "spectral" in c.lower():
            print(" ", c)

    preview_cols = [
        "model",
        "top1_acc",
        "top5_acc",
        "alpha",
        "alpha_weighted",
        "log_norm",
        "spectral_norm",
        "log_spectral_norm",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nPreview:")
    print(df[preview_cols])


if __name__ == "__main__":
    main()