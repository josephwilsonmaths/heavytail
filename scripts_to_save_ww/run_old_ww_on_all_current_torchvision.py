import os
import sys
import gc
import json
import time
import shutil
import types
import traceback
from pathlib import Path
from importlib.machinery import ModuleSpec

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
    module=r"weightwatcher\.weightwatcher",
)

import numpy as np

# ---------------------------------------------------------------------
# NumPy compatibility for old WeightWatcher 0.2.7
# ---------------------------------------------------------------------

if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf


# ---------------------------------------------------------------------
# TensorFlow/Keras stubs for old WeightWatcher 0.2.7
# ---------------------------------------------------------------------

def _add_spec(module, name, is_package=False):
    """
    Give fake modules a valid importlib spec.
    This avoids torch._dynamo / importlib.find_spec failures.
    """
    module.__spec__ = ModuleSpec(name=name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []
    return module


def install_tf_keras_stubs():
    """
    weightwatcher==0.2.7 imports/checks tensorflow/keras even for PyTorch models.
    This provides enough dummy Keras/TensorFlow structure for PyTorch-only analysis.
    """

    class DummyKerasModel:
        pass

    class DummyKerasLayer:
        pass

    def dummy_load_model(*args, **kwargs):
        raise RuntimeError(
            "Keras load_model was called, but this script is intended for "
            "PyTorch torchvision models only."
        )

    keras_models_stub = _add_spec(
        types.ModuleType("keras.models"),
        "keras.models",
        is_package=False,
    )
    keras_models_stub.Model = DummyKerasModel
    keras_models_stub.Sequential = DummyKerasModel
    keras_models_stub.load_model = dummy_load_model

    keras_core_stub = _add_spec(
        types.ModuleType("keras.layers.core"),
        "keras.layers.core",
        is_package=False,
    )
    keras_core_stub.Dense = DummyKerasLayer
    keras_core_stub.Flatten = DummyKerasLayer
    keras_core_stub.Dropout = DummyKerasLayer
    keras_core_stub.Activation = DummyKerasLayer

    keras_conv_stub = _add_spec(
        types.ModuleType("keras.layers.convolutional"),
        "keras.layers.convolutional",
        is_package=False,
    )
    keras_conv_stub.Conv1D = DummyKerasLayer
    keras_conv_stub.Conv2D = DummyKerasLayer
    keras_conv_stub.Conv3D = DummyKerasLayer

    keras_norm_stub = _add_spec(
        types.ModuleType("keras.layers.normalization"),
        "keras.layers.normalization",
        is_package=False,
    )
    keras_norm_stub.BatchNormalization = DummyKerasLayer

    keras_pool_stub = _add_spec(
        types.ModuleType("keras.layers.pooling"),
        "keras.layers.pooling",
        is_package=False,
    )
    keras_pool_stub.MaxPooling1D = DummyKerasLayer
    keras_pool_stub.MaxPooling2D = DummyKerasLayer
    keras_pool_stub.AveragePooling1D = DummyKerasLayer
    keras_pool_stub.AveragePooling2D = DummyKerasLayer
    keras_pool_stub.GlobalAveragePooling2D = DummyKerasLayer

    keras_backend_stub = _add_spec(
        types.ModuleType("keras.backend"),
        "keras.backend",
        is_package=False,
    )

    keras_layers_stub = _add_spec(
        types.ModuleType("keras.layers"),
        "keras.layers",
        is_package=True,
    )
    keras_layers_stub.Layer = DummyKerasLayer
    keras_layers_stub.Dense = DummyKerasLayer
    keras_layers_stub.Flatten = DummyKerasLayer
    keras_layers_stub.Dropout = DummyKerasLayer
    keras_layers_stub.Activation = DummyKerasLayer
    keras_layers_stub.Conv1D = DummyKerasLayer
    keras_layers_stub.Conv2D = DummyKerasLayer
    keras_layers_stub.Conv3D = DummyKerasLayer
    keras_layers_stub.BatchNormalization = DummyKerasLayer

    # Old Keras nested-module style
    keras_layers_stub.core = keras_core_stub
    keras_layers_stub.convolutional = keras_conv_stub
    keras_layers_stub.normalization = keras_norm_stub
    keras_layers_stub.pooling = keras_pool_stub

    keras_stub = _add_spec(
        types.ModuleType("keras"),
        "keras",
        is_package=True,
    )
    keras_stub.__version__ = "2.3.1"
    keras_stub.models = keras_models_stub
    keras_stub.layers = keras_layers_stub
    keras_stub.backend = keras_backend_stub

    tf_compat_v1_stub = _add_spec(
        types.ModuleType("tensorflow.compat.v1"),
        "tensorflow.compat.v1",
        is_package=False,
    )

    tf_compat_stub = _add_spec(
        types.ModuleType("tensorflow.compat"),
        "tensorflow.compat",
        is_package=True,
    )
    tf_compat_stub.v1 = tf_compat_v1_stub

    tf_stub = _add_spec(
        types.ModuleType("tensorflow"),
        "tensorflow",
        is_package=True,
    )
    tf_stub.__version__ = "2.0.0"
    tf_stub.keras = keras_stub
    tf_stub.compat = tf_compat_stub

    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.compat"] = tf_compat_stub
    sys.modules["tensorflow.compat.v1"] = tf_compat_v1_stub

    sys.modules["keras"] = keras_stub
    sys.modules["keras.models"] = keras_models_stub
    sys.modules["keras.layers"] = keras_layers_stub
    sys.modules["keras.layers.core"] = keras_core_stub
    sys.modules["keras.layers.convolutional"] = keras_conv_stub
    sys.modules["keras.layers.normalization"] = keras_norm_stub
    sys.modules["keras.layers.pooling"] = keras_pool_stub
    sys.modules["keras.backend"] = keras_backend_stub


# ---------------------------------------------------------------------
# Import modern torch / torchvision FIRST
# ---------------------------------------------------------------------

import torch
import pandas as pd
import torchvision
import torchvision.models as tv_models

# Now install TensorFlow/Keras stubs before importing old WeightWatcher
install_tf_keras_stubs()

import weightwatcher as ww


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUT_DIR = Path("old_ww_current_torchvision_all_default_spectralnorms")
DETAILS_DIR = OUT_DIR / "details"
SUMMARIES_DIR = OUT_DIR / "summaries"
CACHE_ROOT = OUT_DIR / "_temporary_torch_cache"

OUT_DIR.mkdir(exist_ok=True)
DETAILS_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR.mkdir(exist_ok=True)
CACHE_ROOT.mkdir(exist_ok=True)

SUMMARY_CSV = OUT_DIR / "summary_metrics.csv"
FAILURES_CSV = OUT_DIR / "failures.csv"
DISCOVERED_JSON = OUT_DIR / "discovered_models.json"

# Set to e.g. 3 for testing. Leave None for the full run.
MAX_MODELS = None

# Delete downloaded checkpoint cache after each model.
DELETE_CACHE_AFTER_EACH_MODEL = True

# Skip models whose summary and details already exist.
SKIP_COMPLETED = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_default_weight(model_name):
    """
    Return torchvision's DEFAULT weights enum for a model name.
    """
    try:
        weights_enum = tv_models.get_model_weights(model_name)
    except Exception:
        return None

    try:
        return weights_enum.DEFAULT
    except Exception:
        return None


def extract_metrics_from_weight_meta(weights):
    """
    Extract top-1/top-5 accuracy from torchvision weight metadata.
    """
    out = {
        "weights_name": str(weights),
        "weights_url": getattr(weights, "url", None),
        "top1_acc": None,
        "top5_acc": None,
        "metric_dataset": None,
        "num_params": None,
        "recipe": None,
    }

    meta = getattr(weights, "meta", {}) or {}

    out["num_params"] = meta.get("num_params", None)
    out["recipe"] = meta.get("recipe", None)

    metrics = meta.get("_metrics", {}) or {}

    if isinstance(metrics, dict) and len(metrics) > 0:
        dataset_key = "ImageNet-1K" if "ImageNet-1K" in metrics else next(iter(metrics.keys()))
        dataset_metrics = metrics.get(dataset_key, {}) or {}

        out["metric_dataset"] = dataset_key
        out["top1_acc"] = dataset_metrics.get("acc@1", None)
        out["top5_acc"] = dataset_metrics.get("acc@5", None)

    return out


def safe_scalar(x):
    """
    Convert numpy scalars / tensors / unusual objects into JSON/CSV-safe values.
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return str(x.detach().cpu().numpy())

    if isinstance(x, np.generic):
        return x.item()

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (str, int, float, bool)):
        return x

    return str(x)


def flatten_summary(summary):
    """
    Convert WeightWatcher summary into a flat dict.
    """
    if summary is None:
        return {}

    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()

    if not isinstance(summary, dict):
        return {"ww_summary_raw": str(summary)}

    return {str(k): safe_scalar(v) for k, v in summary.items()}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def clear_model_and_cache(model, cache_dir):
    """
    Delete model from RAM and delete this model's temporary checkpoint cache.
    """
    try:
        del model
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if DELETE_CACHE_AFTER_EACH_MODEL and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def discover_default_weight_models():
    """
    Discover top-level torchvision.models classification models with DEFAULT weights.
    Keeps models with ImageNet-style top1/top5 accuracy metadata.
    """
    names = tv_models.list_models()
    rows = []

    for name in names:
        weights = get_default_weight(name)
        if weights is None:
            continue

        meta = extract_metrics_from_weight_meta(weights)

        # Classification weights should have acc@1 / acc@5 metadata.
        if meta["top1_acc"] is None and meta["top5_acc"] is None:
            continue

        rows.append((name, weights, meta))

    rows = sorted(rows, key=lambda x: x[0])

    if MAX_MODELS is not None:
        rows = rows[:MAX_MODELS]

    return rows


def run_one_model(model_name, weights, weight_meta):
    details_path = DETAILS_DIR / f"{model_name}_details.csv"
    summary_path = SUMMARIES_DIR / f"{model_name}_summary.json"

    if SKIP_COMPLETED and details_path.exists() and summary_path.exists():
        print(f"[SKIP] {model_name}: already completed")
        return json.loads(summary_path.read_text())

    model_cache_dir = CACHE_ROOT / model_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Redirect torch hub cache so each model's checkpoint can be deleted after use.
    torch.hub.set_dir(str(model_cache_dir))

    print(f"\n[LOAD] {model_name}")
    print(f"       weights = {weights}")
    print(f"       acc@1   = {weight_meta.get('top1_acc')}")
    print(f"       acc@5   = {weight_meta.get('top5_acc')}")

    t0 = time.time()
    model = None

    try:
        model = tv_models.get_model(model_name, weights=weights)
        model.eval()

        print(f"[WW]   Running WeightWatcher 0.2.7 on {model_name}")
        watcher = ww.WeightWatcher(model=model)

        # Old repo-era call.
        details = watcher.analyze(alphas=True, spectralnorms=True)

        try:
            details = watcher.get_details()
        except Exception:
            pass

        summary = watcher.get_summary()
        summary_flat = flatten_summary(summary)

        details_df = pd.DataFrame(details)
        details_df.to_csv(details_path, index=False)

        elapsed_sec = time.time() - t0

        row = {
            "model": model_name,
            "status": "ok",
            "elapsed_sec": elapsed_sec,
            **weight_meta,
            **summary_flat,
        }

        save_json(summary_path, row)

        print(f"[DONE] {model_name}: {elapsed_sec:.1f} sec")
        return row

    finally:
        clear_model_and_cache(model, model_cache_dir)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Python:", sys.version)
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("weightwatcher:", ww.__version__)
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)

    model_specs = discover_default_weight_models()

    print(f"\nFound {len(model_specs)} torchvision classification models with DEFAULT weights.")

    save_json(
        DISCOVERED_JSON,
        [
            {
                "model": name,
                **meta,
            }
            for name, weights, meta in model_specs
        ],
    )

    rows = []
    failures = []

    # Load existing partial results if present
    if SUMMARY_CSV.exists():
        try:
            rows = pd.read_csv(SUMMARY_CSV).to_dict(orient="records")
        except Exception:
            rows = []

    if FAILURES_CSV.exists():
        try:
            failures = pd.read_csv(FAILURES_CSV).to_dict(orient="records")
        except Exception:
            failures = []

    for i, (model_name, weights, weight_meta) in enumerate(model_specs, start=1):
        print(f"\n===== {i}/{len(model_specs)}: {model_name} =====")

        try:
            row = run_one_model(model_name, weights, weight_meta)

            # Replace old row for this model if rerun
            rows = [r for r in rows if r.get("model") != model_name]
            rows.append(row)

            pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results.")
            break

        except Exception as e:
            print(f"[FAIL] {model_name}: {repr(e)}")

            failure = {
                "model": model_name,
                "weights": str(weights),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

            failures = [f for f in failures if f.get("model") != model_name]
            failures.append(failure)

            pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)

            shutil.rmtree(CACHE_ROOT / model_name, ignore_errors=True)

    if rows:
        pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
        print(f"\nSaved summary metrics to: {SUMMARY_CSV}")

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)
        print(f"Saved failures to: {FAILURES_CSV}")

    if DELETE_CACHE_AFTER_EACH_MODEL:
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)

    print("\nFinished.")


if __name__ == "__main__":
    main()