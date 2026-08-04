from matplotlib import transforms
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import shutil
import mpmath
import numpy as np
from safetensors import safe_open
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.func import functional_call, vmap, jacrev
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import math

from scipy import stats
from scipy.stats import rv_continuous
from scipy.special import hyp1f1, loggamma
from scipy.optimize import fmin
import numpy as np
import mpmath as mpm
from scipy.stats._distn_infrastructure import _ShapeInfo
from scipy.special import gammaln


def _normalize_pythia_model_size(model_size):
    model_size = str(model_size).strip().lower()
    model_size = model_size.removeprefix("eleutherai/")
    model_size = model_size.removeprefix("pythia-")

    aliases = {
        "19m": "70m",
        "70m": "70m",
        "125m": "160m",
        "160m": "160m",
        "350m": "410m",
        "410m": "410m",
        "800m": "1b",
        "1b": "1b",
        "1.3b": "1.4b",
        "1.4b": "1.4b",
        "2.7b": "2.8b",
        "2.8b": "2.8b",
        "6.7b": "6.9b",
        "6.9b": "6.9b",
        "13b": "12b",
        "12b": "12b",
        "14m": "14m",
        "31m": "31m",
    }
    return aliases.get(model_size, model_size)


def _extract_pythia_eval_step(payload, file_path):
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_args = str(config.get("model_args", ""))
    match = re.search(r"step(\d+)", model_args)
    if match:
        return int(match.group(1))

    for candidate in [str(file_path), file_path.stem]:
        match = re.search(r"step(\d+)", candidate)
        if match:
            return int(match.group(1))

    number_matches = re.findall(r"(?<!\d)(\d{3,6})(?!\d)", file_path.stem)
    if number_matches:
        return int(max(number_matches, key=int))

    raise ValueError(f"Could not infer step from {file_path}")


def _resolve_pythia_metric(task_metrics, metric):
    if metric in task_metrics:
        return metric

    normalized_requested = metric.replace(",none", "")
    normalized_lookup = {
        key.replace(",none", ""): key
        for key in task_metrics
    }
    if normalized_requested in normalized_lookup:
        return normalized_lookup[normalized_requested]

    available_metrics = ", ".join(sorted(task_metrics))
    raise KeyError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")


def _get_pythia_eval_model_dir(
    model_size,
    *,
    evals_root,
    suite="pythia-v1",
    deduped=True,
    shot="zero-shot",
):
    canonical_size = _normalize_pythia_model_size(model_size)
    model_dir_name = f"pythia-{canonical_size}"
    if deduped:
        model_dir_name += "-deduped"

    model_dir = Path(evals_root) / suite / model_dir_name / shot
    if not model_dir.exists():
        raise FileNotFoundError(f"Could not find model eval directory: {model_dir}")

    return model_dir


def _normalize_pythia_model_name(model_name):
    model_name = str(model_name).strip()
    model_name = model_name.removeprefix("EleutherAI/")
    if not model_name.startswith("pythia-"):
        model_name = f"pythia-{model_name}"
    return model_name


def _get_pythia_eval_identifiers(model_name):
    short_model_name = _normalize_pythia_model_name(model_name)
    model_stub = short_model_name.removeprefix("pythia-")
    eval_model_dir = short_model_name
    eval_file_prefix = model_stub
    return short_model_name, eval_model_dir, eval_file_prefix


def _load_remote_pythia_eval_json(model_name, step, *, shot="zero-shot", evals_root="evals", suite="pythia-v1"):
    short_model_name, eval_model_dir, eval_file_prefix = _get_pythia_eval_identifiers(model_name)
    local_eval_path = Path(evals_root) / suite / eval_model_dir / shot / f"{eval_file_prefix}_step{int(step)}.json"
    if not local_eval_path.exists():
        local_eval_path.parent.mkdir(parents=True, exist_ok=True)
        raw_url = (
            "https://raw.githubusercontent.com/EleutherAI/pythia/main/"
            f"evals/{suite}/{eval_model_dir}/{shot}/{eval_file_prefix}_step{int(step)}.json"
        )
        with urlopen(raw_url) as response:
            payload = response.read().decode("utf-8")
        local_eval_path.write_text(payload, encoding="utf-8")

    with local_eval_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_pythia_eval_steps(
    model_name,
    *,
    evals_root="evals",
    suite="pythia-v1",
    shot="zero-shot",
):
    """Return all checkpoint steps with available evaluation results for a Pythia model.

    This first looks for cached/local eval JSONs under evals_root. If none are present,
    it queries the GitHub contents API for the upstream Pythia eval directory.
    """

    short_model_name, eval_model_dir, _ = _get_pythia_eval_identifiers(model_name)
    local_dir = Path(evals_root) / suite / eval_model_dir / shot
    steps = set()

    if local_dir.exists():
        for file_path in local_dir.glob("*.json"):
            match = re.search(r"_step(\d+)\.json$", file_path.name)
            if match:
                steps.add(int(match.group(1)))

    if not steps:
        api_url = (
            "https://api.github.com/repos/EleutherAI/pythia/contents/"
            f"evals/{suite}/{eval_model_dir}/{shot}"
        )
        request = Request(api_url, headers={"User-Agent": "GitHub-Copilot"})
        with urlopen(request) as response:
            payload = json.load(response)

        for entry in payload:
            if entry.get("type") != "file":
                continue
            match = re.search(r"_step(\d+)\.json$", entry.get("name", ""))
            if match:
                steps.add(int(match.group(1)))

    if not steps:
        raise ValueError(
            f"No evaluation steps found for {short_model_name} in {suite}/{eval_model_dir}/{shot}"
        )

    return sorted(steps)


def save_pythia_checkpoint_summary(
    model_name,
    step,
    *,
    evals_root="evals",
    model_cache_root=".",
    results_dir="results/htmp/epoch",
    print_possible_steps=False,
    delete_model_cache=False,
    access_token=None,
):
    """Download a Pythia checkpoint, compute spectral stats, and save an epoch-style result file.

    The saved dictionary contains:
        - QueryKey: per-layer eigenvalues/aspect ratio for W=query_key_value
        - Dense: per-layer eigenvalues/aspect ratio for W=dense_h_to_4h
        - EmbedOut: eigenvalues/aspect ratio for W=embed_out

    The output filename follows epoch.py conventions:
        <model>_pile_b2m_<step>_0.pt
    """

    from transformers import GPTNeoXForCausalLM

    if print_possible_steps:
        steps = np.concatenate([np.array([0]), np.logspace(0, 9, 10, base=2), np.linspace(1000, 143000, 143)])
        print("Possible steps for Pythia models:")
        print(steps)

    short_model_name = _normalize_pythia_model_name(model_name)
    repo_id = model_name if str(model_name).startswith("EleutherAI/") else f"EleutherAI/{short_model_name}"
    step = int(step)

    model_cache_dir = Path(model_cache_root) / short_model_name / f"step{step}"
    model = GPTNeoXForCausalLM.from_pretrained(
        repo_id,
        revision=f"step{step}",
        cache_dir=str(model_cache_dir),
        token=access_token
    )
    model.eval()

    querykey = {}
    dense = {}
    for idx, layer in enumerate(model.gpt_neox.layers):
        qk_w = layer.attention.query_key_value.weight.detach().to(dtype=torch.float32)
        qk_eigvals = torch.linalg.eigvalsh(qk_w.T @ qk_w)
        querykey[idx] = {
            "eigvals": qk_eigvals.cpu().numpy(),
            "aspect_ratio": qk_w.shape[1] / qk_w.shape[0],
        }

        dense_w = layer.mlp.dense_h_to_4h.weight.detach().to(dtype=torch.float32)
        dense_eigvals = torch.linalg.eigvalsh(dense_w.T @ dense_w)
        dense[idx] = {
            "eigvals": dense_eigvals.cpu().numpy(),
            "aspect_ratio": dense_w.shape[1] / dense_w.shape[0],
        }

    embed_out_w = model.embed_out.weight.detach().to(dtype=torch.float32)
    embed_out_eigvals = torch.linalg.eigvalsh(embed_out_w.T @ embed_out_w)
    embed_out = {
        "eigvals": embed_out_eigvals.cpu().numpy(),
        "aspect_ratio": embed_out_w.shape[1] / embed_out_w.shape[0],
    }

    res_dict = {
        "QueryKey": querykey,
        "Dense": dense,
        "EmbedOut": embed_out,
        "model": short_model_name,
        "dataset": "pile",
        "epoch": step,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{short_model_name}_pile_b2m_{step}_0.pt"
    torch.save(res_dict, save_path)

    del model
    if delete_model_cache and model_cache_dir.exists():
        shutil.rmtree(model_cache_dir, ignore_errors=True)

    return res_dict, save_path


def list_pythia_eval_tasks(
    model_size,
    *,
    evals_root,
    suite="pythia-v1",
    deduped=True,
    shot="zero-shot",
):
    """Return all evaluation task names available for a given Pythia model.

    Args:
        model_size: Size label such as "70m", "160m", "1b", or legacy aliases like "19m".
        evals_root: Path to the repository's evals directory.
        suite: Usually "pythia-v1".
        deduped: Whether to look under the deduped model directory.
        shot: Subdirectory under the model directory, usually "zero-shot" or "five-shot".

    Returns:
        Sorted list of unique task names found across the model's evaluation JSON files.
    """

    model_dir = _get_pythia_eval_model_dir(
        model_size=model_size,
        evals_root=evals_root,
        suite=suite,
        deduped=deduped,
        shot=shot,
    )

    tasks = set()
    json_files = sorted(model_dir.rglob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON eval files found in {model_dir}")

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tasks.update(payload.get("results", {}).keys())

    if not tasks:
        raise ValueError(f"No evaluation tasks found in JSON files under {model_dir}")

    return sorted(tasks)


def load_pythia_eval_metric(
    model_size,
    task,
    metric,
    *,
    evals_root,
    suite="pythia-v1",
    deduped=True,
    shot="zero-shot",
):
    """Load a metric over training steps from EleutherAI Pythia eval JSON files.

    Args:
        model_size: Size label such as "70m", "160m", "1b", or legacy aliases like "19m".
        task: Evaluation task name in the JSON results, for example "arc_easy".
        metric: Metric key to extract, for example "acc" or "acc,none".
        evals_root: Path to the repository's evals directory.
        suite: Usually "pythia-v1".
        deduped: Whether to look under the deduped model directory.
        shot: Subdirectory under the model directory, usually "zero-shot" or "five-shot".

    Returns:
        pandas.DataFrame with columns step, value, metric, task, shot, and file.
    """

    model_dir = _get_pythia_eval_model_dir(
        model_size=model_size,
        evals_root=evals_root,
        suite=suite,
        deduped=deduped,
        shot=shot,
    )

    rows = []
    skipped_files = []
    for file_path in sorted(model_dir.rglob("*.json")):
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        task_results = payload.get("results", {}).get(task)
        if task_results is None:
            skipped_files.append(str(file_path))
            continue

        metric_key = _resolve_pythia_metric(task_results, metric)
        step = _extract_pythia_eval_step(payload, file_path)
        stderr_key = f"{metric_key}_stderr"

        rows.append(
            {
                "step": step,
                "value": float(task_results[metric_key]),
                "metric": metric_key,
                "task": task,
                "shot": shot,
                "file": str(file_path),
                "stderr": float(task_results[stderr_key]) if stderr_key in task_results else np.nan,
            }
        )

    if not rows:
        raise ValueError(
            f"No evals found for task '{task}' in {model_dir}. "
            f"Checked {len(skipped_files)} JSON files without a matching task."
        )

    df = pd.DataFrame(rows).sort_values("step").drop_duplicates(subset=["step"], keep="last")
    return df.reset_index(drop=True)


def plot_pythia_eval_metric(
    model_size,
    task,
    metric,
    *,
    evals_root,
    suite="pythia-v1",
    deduped=True,
    shot="zero-shot",
    ax=None,
    label=None,
    show_stderr=False,
    log_x=False,
):
    """Plot a Pythia evaluation metric against training step.

    Example:
        plot_pythia_eval_metric(
            "70m",
            task="arc_easy",
            metric="acc",
            evals_root="path/to/pythia/evals",
            shot="zero-shot",
        )
    """

    df = load_pythia_eval_metric(
        model_size=model_size,
        task=task,
        metric=metric,
        evals_root=evals_root,
        suite=suite,
        deduped=deduped,
        shot=shot,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    line_label = label or f"pythia-{_normalize_pythia_model_size(model_size)}"
    ax.plot(df["step"], df["value"], marker="o", linewidth=1.8, markersize=4, label=line_label)

    if show_stderr and df["stderr"].notna().any():
        lower = df["value"] - df["stderr"].fillna(0.0)
        upper = df["value"] + df["stderr"].fillna(0.0)
        ax.fill_between(df["step"], lower, upper, alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace(",none", ""))
    ax.set_title(f"{task} on pythia-{_normalize_pythia_model_size(model_size)} ({shot})")
    ax.grid(True, alpha=0.3)

    if log_x:
        ax.set_xscale("log")

    if label is not None:
        ax.legend()

    return df, ax


def list_moonlight_layers(
    optim_name,
    *,
    data_root="data/moonlight",
):
    """Return all available saved Moonlight layer ids for an optimizer."""

    optim_name = str(optim_name).strip().lower()
    optimizer_root = Path(data_root) / optim_name
    if not optimizer_root.exists():
        raise FileNotFoundError(f"Could not find Moonlight optimizer directory: {optimizer_root}")

    layers = []
    for path in sorted(optimizer_root.glob("layer_*")):
        match = re.fullmatch(r"layer_(\d+)", path.name)
        if match and path.is_dir():
            layers.append(int(match.group(1)))

    if not layers:
        raise ValueError(f"No Moonlight layer directories found in {optimizer_root}")

    return sorted(layers)


def _resolve_moonlight_layer_dir(optim_name, *, data_root="data/moonlight", layer=1):
    optim_name = str(optim_name).strip().lower()
    layer = int(layer)
    optimizer_root = Path(data_root) / optim_name
    layer_dir = optimizer_root / f"layer_{layer}"
    if not layer_dir.exists():
        available_layers = list_moonlight_layers(optim_name, data_root=data_root)
        raise FileNotFoundError(
            f"Could not find Moonlight layer directory {layer_dir}. "
            f"Available layers: {available_layers}"
        )

    tensor_layer_index = layer - 1
    if tensor_layer_index < 0:
        raise ValueError(f"Moonlight layer must be positive, got layer={layer}.")

    return optim_name, layer, tensor_layer_index, layer_dir


def list_moonlight_checkpoint_steps(
    optim_name,
    *,
    data_root="data/moonlight",
    layer=1,
):
    """Return all available Moonlight checkpoint steps for an optimizer."""

    _, _, _, layer_dir = _resolve_moonlight_layer_dir(
        optim_name,
        data_root=data_root,
        layer=layer,
    )

    steps = []
    for file_path in sorted(layer_dir.glob("step_*.safetensors")):
        match = re.fullmatch(r"step_(\d+)\.safetensors", file_path.name)
        if match:
            steps.append(int(match.group(1)))

    if not steps:
        raise ValueError(f"No Moonlight safetensor checkpoints found in {layer_dir}")

    return sorted(steps)


def save_all_moonlight_checkpoint_summaries(
    optim_names=("adam", "muon"),
    *,
    steps=None,
    layer=None,
    data_root="data/moonlight",
    results_dir="results/htmp/epoch",
    verbose=False,
):
    """Convert available Moonlight safetensor checkpoints into saved spectral summaries.

    Args:
        optim_names: Iterable of optimizer names to convert, for example ("adam", "muon").
        steps: Optional iterable of step values to restrict conversion. If omitted, converts all available steps.
        layer: Optional saved layer id to summarize. If omitted, converts all available layers.
        data_root: Root directory containing Moonlight safetensor checkpoints.
        results_dir: Output directory for saved summary .pt files.
        verbose: If True, print each converted checkpoint.

    Returns:
        List of saved output paths.
    """

    if isinstance(optim_names, str):
        optim_names = [optim_names]

    requested_steps = None if steps is None else {int(step) for step in steps}
    saved_paths = []

    for optim_name in optim_names:
        layers_to_save = [int(layer)] if layer is not None else list_moonlight_layers(
            optim_name,
            data_root=data_root,
        )

        for layer_value in layers_to_save:
            available_steps = list_moonlight_checkpoint_steps(
                optim_name,
                data_root=data_root,
                layer=layer_value,
            )
            steps_to_save = available_steps if requested_steps is None else [
                step for step in available_steps if step in requested_steps
            ]

            if requested_steps is not None:
                missing_steps = sorted(requested_steps.difference(steps_to_save))
                if missing_steps:
                    raise ValueError(
                        f"Requested Moonlight steps {missing_steps} are not available for optimizer {optim_name} layer {layer_value}."
                    )

            for step in steps_to_save:
                _, save_path = save_moonlight_checkpoint_summary(
                    optim_name,
                    step,
                    layer=layer_value,
                    data_root=data_root,
                    results_dir=results_dir,
                )
                saved_paths.append(save_path)
                if verbose:
                    print(f"Saved Moonlight summary: {save_path}")

    return saved_paths


def save_moonlight_checkpoint_summary(
    optim_name,
    step,
    *,
    layer=1,
    data_root="data/moonlight",
    results_dir="results/htmp/epoch",
):
    """Load a Moonlight checkpoint, compute spectral stats, and save an epoch-style result file.

    The saved dictionary contains:
        - Dense: per-layer eigenvalues/aspect ratio for W=mlp.gate_proj.weight
        - KeyValue: per-layer eigenvalues/aspect ratio for W=self_attn.kv_b_proj.weight
        - Query: per-layer eigenvalues/aspect ratio for W=self_attn.q_proj.weight

    The output filename follows epoch.py conventions, with a layer suffix:
        moonlight_<optim>_openwebtext_b2m_<step>_0_layer<layer>.pt
    """

    optim_name, layer, tensor_layer_index, model_dir = _resolve_moonlight_layer_dir(
        optim_name,
        data_root=data_root,
        layer=layer,
    )
    short_model_name = f"moonlight_{optim_name}"
    step = int(step)

    model_path = model_dir / f"step_{step}.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find Moonlight checkpoint: {model_path}")

    dense_tensor_name = (
        f"model.layers.{tensor_layer_index}.mlp.gate_proj.weight"
        if layer == 1
        else f"model.layers.{tensor_layer_index}.mlp.shared_experts.gate_proj.weight"
    )
    tensor_names = {
        "Dense": dense_tensor_name,
        "KeyValue": f"model.layers.{tensor_layer_index}.self_attn.kv_b_proj.weight",
        "Query": f"model.layers.{tensor_layer_index}.self_attn.q_proj.weight",
    }

    def _tensor_summary(weight):
        weight = weight.detach().to(dtype=torch.float32)
        eigvals = torch.linalg.eigvalsh(weight.T @ weight)
        return {
            "eigvals": eigvals.cpu().numpy(),
            "aspect_ratio": weight.shape[1] / weight.shape[0],
        }

    res_dict = {
        "Dense": {},
        "KeyValue": {},
        "Query": {},
        "model": short_model_name,
        "dataset": "openwebtext",
        "epoch": step,
    }

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        available_names = set(handle.keys())
        available_weight_tensors = {
            weight_type: tensor_name
            for weight_type, tensor_name in tensor_names.items()
            if tensor_name in available_names
        }
        if not available_weight_tensors:
            raise KeyError(
                f"None of the supported Moonlight tensors were found in {model_path}. "
                f"Checked: {', '.join(tensor_names.values())}"
            )

        for weight_type, tensor_name in available_weight_tensors.items():
            res_dict[weight_type][layer] = _tensor_summary(handle.get_tensor(tensor_name))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{short_model_name}_openwebtext_b2m_{step}_0_layer{layer}.pt"
    torch.save(res_dict, save_path)

    return res_dict, save_path


def _split_optimizer_resnet_model_name(model_name):
    normalized_name = str(model_name).strip().lower()
    for base_model in ("resnet9", "resnet18", "resnet34", "resnet50"):
        prefix = f"{base_model}_"
        if normalized_name.startswith(prefix):
            optimizer_name = normalized_name[len(prefix):]
            if optimizer_name:
                return base_model, optimizer_name

    raise ValueError(
        "Optimizer-scoped ResNet models must be named like 'resnet9_adam' or 'resnet50_sgd'."
    )


def _get_optimizer_resnet_checkpoint_dir(model_name, dataset, *, data_root="data/htmp"):
    base_model, optimizer_name = _split_optimizer_resnet_model_name(model_name)
    checkpoint_dir = Path(data_root) / f"{base_model}_{dataset}_{optimizer_name}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Could not find optimizer-scoped ResNet checkpoint directory: {checkpoint_dir}")
    return base_model, optimizer_name, checkpoint_dir


def _get_modelzoo_checkpoint_dir(model_name, dataset, *, data_root="data/modelzoo", seed=0):
    short_model_name = str(model_name).strip().lower()
    short_dataset = str(dataset).strip().lower()
    checkpoint_dir = Path(data_root) / f"{short_model_name}_{short_dataset}" / f"seed_{int(seed)}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Could not find ModelZoo checkpoint directory: {checkpoint_dir}")
    return short_model_name, short_dataset, checkpoint_dir


def _modelzoo_checkpoint_sort_key(file_path):
    match = re.fullmatch(r"checkpoint_(\d+)", file_path.name)
    if match is None:
        raise ValueError(f"Unexpected ModelZoo checkpoint directory name: {file_path.name}")
    return int(match.group(1))


def list_modelzoo_seeds(model_name, dataset, *, data_root="data/modelzoo"):
    short_model_name = str(model_name).strip().lower()
    short_dataset = str(dataset).strip().lower()
    model_dir = Path(data_root) / f"{short_model_name}_{short_dataset}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Could not find ModelZoo model directory: {model_dir}")

    seeds = []
    for file_path in model_dir.glob("seed_*"):
        match = re.fullmatch(r"seed_(\d+)", file_path.name)
        if match:
            seeds.append(int(match.group(1)))

    if not seeds:
        raise ValueError(f"No ModelZoo seed directories found in {model_dir}")

    return sorted(seeds)


def list_modelzoo_checkpoint_steps(model_name, dataset, *, data_root="data/modelzoo", seed=0):
    """Return all available checkpoint steps for one ModelZoo seed directory."""

    _, _, checkpoint_dir = _get_modelzoo_checkpoint_dir(
        model_name,
        dataset,
        data_root=data_root,
        seed=seed,
    )

    steps = []
    for file_path in sorted(checkpoint_dir.glob("checkpoint_*"), key=_modelzoo_checkpoint_sort_key):
        steps.append(_modelzoo_checkpoint_sort_key(file_path))

    if not steps:
        raise ValueError(f"No ModelZoo checkpoints found in {checkpoint_dir}")

    return steps


def _get_modelzoo_progress_row(progress_df, step):
    if "training_iteration" in progress_df.columns:
        step_rows = progress_df.loc[progress_df["training_iteration"] == int(step)]
        if not step_rows.empty:
            return step_rows.iloc[-1]

    if int(step) < 0 or int(step) >= len(progress_df):
        raise IndexError(f"ModelZoo checkpoint step {step} is out of bounds for progress.csv with {len(progress_df)} rows.")

    return progress_df.iloc[int(step)]


def _load_modelzoo_checkpoint_payload(model_name, dataset, step, seed, *, data_root="data/modelzoo"):
    short_model_name, short_dataset, checkpoint_dir = _get_modelzoo_checkpoint_dir(
        model_name,
        dataset,
        data_root=data_root,
        seed=seed,
    )
    step = int(step)
    seed = int(seed)

    checkpoint_path = checkpoint_dir / f"checkpoint_{step:06d}" / "checkpoints"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find ModelZoo checkpoint file: {checkpoint_path}")

    progress_path = checkpoint_dir / "progress.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"Could not find ModelZoo progress file: {progress_path}")

    params_path = checkpoint_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find ModelZoo params file: {params_path}")

    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    progress_df = pd.read_csv(progress_path)
    progress_row = _get_modelzoo_progress_row(progress_df, step)

    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    return {
        "short_model_name": short_model_name,
        "short_dataset": short_dataset,
        "step": step,
        "seed": seed,
        "state_dict": state_dict,
        "progress_row": progress_row,
        "params": params,
    }


def _modelzoo_tensor_summary(weight, *, flatten=False):
    weight = weight.detach().to(dtype=torch.float64)
    if flatten:
        weight = weight.flatten(1, -1)
    eigvals = torch.linalg.eigvalsh(weight @ weight.T)
    return {
        "eigvals": eigvals.cpu().numpy(),
        "aspect_ratio": weight.shape[0] / weight.shape[1],
    }


def _build_modelzoo_summary_dict(payload):
    state_dict = payload["state_dict"]
    progress_row = payload["progress_row"]
    params = payload["params"]

    res_dict = {
        "model": payload["short_model_name"],
        "dataset": payload["short_dataset"],
        "epoch": payload["step"],
        "repeat": payload["seed"],
        "batch_size": int(params["training::batchsize"]),
        "subsample": False,
        "seed": payload["seed"],
        "train_loss": float(progress_row["train_loss"]),
        "train_acc": float(progress_row["train_acc"]),
        "test_loss": float(progress_row["test_loss"]),
        "test_acc": float(progress_row["test_acc"]),
        "FC": {},
        "Conv2": {},
    }

    if "fc.weight" in state_dict:
        res_dict["FC"] = _modelzoo_tensor_summary(state_dict["fc.weight"], flatten=True)

    conv2_tensor_names = sorted(
        [name for name in state_dict.keys() if re.fullmatch(r"layer\d+\.\d+\.conv2\.weight", name)],
        key=_optimizer_resnet_conv2_sort_key,
    )
    for layer_idx, tensor_name in enumerate(conv2_tensor_names):
        res_dict["Conv2"][layer_idx] = {
            **_modelzoo_tensor_summary(state_dict[tensor_name], flatten=True),
            "tensor_name": tensor_name,
        }

    if not res_dict["FC"] and not res_dict["Conv2"]:
        raise KeyError(
            "No supported ModelZoo tensors found. Expected 'fc.weight' and/or 'layer*.conv2.weight'."
        )

    return res_dict


def _save_modelzoo_summary_dict(res_dict, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / (
        f"{res_dict['model']}_{res_dict['dataset']}_b{int(res_dict['batch_size'])}_{int(res_dict['epoch'])}_{int(res_dict['repeat'])}.pt"
    )
    torch.save(res_dict, save_path)
    return save_path


def save_modelzoo_checkpoint_summary(
    model_name,
    dataset,
    step,
    seed,
    *,
    data_root="data/modelzoo",
    results_dir="results/htmp/epoch",
):
    """Convert one ModelZoo checkpoint into an epoch-style summary.

    The saved dictionary contains:
        - FC: eigenvalues/aspect ratio for W=fc.weight
        - Conv2: per-layer eigenvalues/aspect ratio for W=layer*.conv2.weight.flatten(1)
        - train/test metrics from progress.csv
    """

    payload = _load_modelzoo_checkpoint_payload(
        model_name,
        dataset,
        step,
        seed,
        data_root=data_root,
    )
    res_dict = _build_modelzoo_summary_dict(payload)
    save_path = _save_modelzoo_summary_dict(res_dict, results_dir)
    return res_dict, save_path


def save_all_modelzoo_checkpoint_summaries(
    model_name,
    dataset,
    *,
    steps=None,
    seeds=None,
    data_root="data/modelzoo",
    results_dir="results/htmp/epoch",
    verbose=False,
):
    """Convert ModelZoo checkpoints into ESDFit-compatible epoch summaries."""

    seeds_to_save = list_modelzoo_seeds(model_name, dataset, data_root=data_root) if seeds is None else [int(seed) for seed in seeds]
    requested_steps = None if steps is None else {int(step) for step in steps}
    saved_paths = []

    available_steps_per_seed = {}
    for seed in seeds_to_save:
        available_steps = list_modelzoo_checkpoint_steps(
            model_name,
            dataset,
            data_root=data_root,
            seed=seed,
        )
        available_steps_per_seed[seed] = available_steps
        if requested_steps is not None:
            missing_steps = sorted(requested_steps.difference(available_steps))
            if missing_steps:
                raise ValueError(
                    f"Requested ModelZoo steps {missing_steps} are not available for seed {seed}."
                )

    if requested_steps is None:
        shared_steps = sorted(set.intersection(*[set(steps) for steps in available_steps_per_seed.values()]))
        if not shared_steps:
            raise ValueError("No shared ModelZoo checkpoint steps were found across the selected seeds.")
    else:
        shared_steps = sorted(requested_steps)

    for seed in seeds_to_save:
        for step in shared_steps:
            _, save_path = save_modelzoo_checkpoint_summary(
                model_name,
                dataset,
                step,
                seed,
                data_root=data_root,
                results_dir=results_dir,
            )
            saved_paths.append(save_path)
            if verbose:
                print(f"Saved ModelZoo summary: {save_path}")

    return saved_paths


def _optimizer_resnet_checkpoint_sort_key(file_path):
    match = re.fullmatch(r"step_(\d+)_repeat_(\d+)\.safetensors", file_path.name)
    if match is None:
        raise ValueError(f"Unexpected optimizer-scoped ResNet checkpoint filename: {file_path.name}")
    return int(match.group(1)), int(match.group(2))


def _optimizer_resnet_conv2_sort_key(tensor_name):
    match = re.fullmatch(r"layer(\d+)\.(\d+)\.conv2\.weight", tensor_name)
    if match is None:
        raise ValueError(f"Unexpected ResNet conv2 tensor name: {tensor_name}")
    return int(match.group(1)), int(match.group(2))


def _coerce_safetensor_metadata_value(value):
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    lower = normalized.lower()
    if lower == "none":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", normalized):
        return int(normalized)

    try:
        return float(normalized)
    except ValueError:
        return normalized


def list_optimizer_resnet_checkpoint_steps(
    model_name,
    dataset,
    *,
    data_root="data/htmp",
):
    """Return all available saved steps for an optimizer-scoped ResNet model."""

    _, _, checkpoint_dir = _get_optimizer_resnet_checkpoint_dir(
        model_name,
        dataset,
        data_root=data_root,
    )

    steps = set()
    for file_path in checkpoint_dir.glob("step_*_repeat_*.safetensors"):
        step, _ = _optimizer_resnet_checkpoint_sort_key(file_path)
        steps.add(step)

    if not steps:
        raise ValueError(f"No optimizer-scoped ResNet checkpoints found in {checkpoint_dir}")

    return sorted(steps)


def save_resnet_optimizer_checkpoint_summary(
    model_name,
    dataset,
    step,
    repeat,
    *,
    data_root="data/htmp",
    results_dir="results/htmp/epoch",
):
    """Convert one optimizer-scoped ResNet safetensor checkpoint into an epoch-style summary.

    The saved dictionary contains:
        - Conv2: per-layer eigenvalues/aspect ratio for W=conv2.weight.flatten(1, -1)
        - train_loss/train_acc/test_loss/test_acc parsed from checkpoint metadata

    The output filename follows epoch.py conventions with the optimizer included in the model name:
        <resnet_model>_<optimizer>_<dataset>[_subsampled]_b<batch>_<step>_<repeat>.pt
    """

    _, _, checkpoint_dir = _get_optimizer_resnet_checkpoint_dir(
        model_name,
        dataset,
        data_root=data_root,
    )
    step = int(step)
    repeat = int(repeat)
    checkpoint_path = checkpoint_dir / f"step_{step}_repeat_{repeat}.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find optimizer-scoped ResNet checkpoint: {checkpoint_path}")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = {
            key: _coerce_safetensor_metadata_value(value)
            for key, value in (handle.metadata() or {}).items()
        }
        conv2_tensor_names = sorted(
            [name for name in handle.keys() if re.fullmatch(r"layer\d+\.\d+\.conv2\.weight", name)],
            key=_optimizer_resnet_conv2_sort_key,
        )
        if not conv2_tensor_names:
            raise KeyError(
                f"No conv2.weight tensors were found in optimizer-scoped ResNet checkpoint {checkpoint_path}."
            )

        conv2 = {}
        for layer_idx, tensor_name in enumerate(conv2_tensor_names):
            weight = handle.get_tensor(tensor_name).detach().to(dtype=torch.float64).flatten(1, -1)
            eigvals = torch.linalg.eigvalsh(weight @ weight.T)
            conv2[layer_idx] = {
                "eigvals": eigvals.cpu().numpy(),
                "aspect_ratio": weight.shape[0] / weight.shape[1],
                "tensor_name": tensor_name,
            }

    short_model_name = str(model_name).strip().lower()
    save_dataset = str(metadata.get("dataset", dataset))
    save_step = int(metadata.get("epoch", step))
    save_repeat = int(metadata.get("repeat", repeat))
    batch_size = metadata.get("batch_size")
    if batch_size is None:
        raise KeyError(f"Missing batch_size metadata in optimizer-scoped ResNet checkpoint {checkpoint_path}")
    subsample = bool(metadata.get("subsample", False))

    res_dict = {
        "Conv2": conv2,
        "model": short_model_name,
        "dataset": save_dataset,
        "epoch": save_step,
        "repeat": save_repeat,
        "batch_size": int(batch_size),
        "subsample": subsample,
        "optimizer": metadata.get("optimizer"),
        "scheduler": metadata.get("scheduler"),
        "train_loss": float(metadata["train_loss"]) if metadata.get("train_loss") is not None else np.nan,
        "train_acc": float(metadata["train_acc"]) if metadata.get("train_acc") is not None else np.nan,
        "test_loss": float(metadata["test_loss"]) if metadata.get("test_loss") is not None else np.nan,
        "test_acc": float(metadata["test_acc"]) if metadata.get("test_acc") is not None else np.nan,
        "metadata": metadata,
    }

    save_name = f"{short_model_name}_{save_dataset}"
    if subsample:
        save_name += "_subsampled"
    save_name += f"_b{int(batch_size)}_{save_step}_{save_repeat}.pt"

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / save_name
    torch.save(res_dict, save_path)

    return res_dict, save_path


def save_all_resnet_optimizer_checkpoint_summaries(
    model_names,
    dataset,
    *,
    steps=None,
    repeats=None,
    data_root="data/htmp",
    results_dir="results/htmp/epoch",
    verbose=False,
):
    """Convert available optimizer-scoped ResNet safetensor checkpoints into saved summaries."""

    if isinstance(model_names, str):
        model_names = [model_names]

    requested_steps = None if steps is None else {int(step) for step in steps}
    requested_repeats = None if repeats is None else {int(repeat) for repeat in repeats}
    saved_paths = []

    for model_name in model_names:
        _, _, checkpoint_dir = _get_optimizer_resnet_checkpoint_dir(
            model_name,
            dataset,
            data_root=data_root,
        )
        checkpoint_files = sorted(
            checkpoint_dir.glob("step_*_repeat_*.safetensors"),
            key=_optimizer_resnet_checkpoint_sort_key,
        )
        if not checkpoint_files:
            raise ValueError(f"No optimizer-scoped ResNet checkpoints found in {checkpoint_dir}")

        for checkpoint_path in checkpoint_files:
            step, repeat = _optimizer_resnet_checkpoint_sort_key(checkpoint_path)
            if requested_steps is not None and step not in requested_steps:
                continue
            if requested_repeats is not None and repeat not in requested_repeats:
                continue

            _, save_path = save_resnet_optimizer_checkpoint_summary(
                model_name,
                dataset,
                step,
                repeat,
                data_root=data_root,
                results_dir=results_dir,
            )
            saved_paths.append(save_path)
            if verbose:
                print(f"Saved optimizer-scoped ResNet summary: {save_path}")

    return saved_paths

def free_energy_kernel(K, lam = 1, jitter = 1e-6, verbose=False):
    '''
    Compute free energy for noiseless Bayesian GLM, with gram matrix given by K. 
    K: torch.Tensor of shape (n,n). Should be positive semi-definite.
    lam is regularizer, jitter is added to eigenvalues for numerical stability.

    returns mean free energy, and eigenvalues of K.
    '''
    n, _ = K.shape

    # Compute log-determinant with numerical stability
    try:
        logdet = torch.linalg.slogdet(K + jitter * torch.eye(n, device=K.device))[1] # Computed with LU Factorization, more stable than eigvals for large matrices

        # Compute trace of resolvent, using eigs.
        e = torch.linalg.eigvalsh(K)
        inverse_trace = torch.sum(1 / (e + jitter))
    except RuntimeError as e:
        print(f"Error computing free energy: {e}")
        print("Adding more jitter for numerical stability.")
        K += 1e-3 * torch.eye(n, device=K.device)
        e = torch.linalg.eigvalsh(K)
        logdet = torch.linalg.slogdet(K)[1]
        inverse_trace = torch.sum(1 / (e))

    # Compute free energy
    F1 = inverse_trace * lam / 2
    F2 = 1/2 * logdet
    F3 = -1/2*np.log(lam/2/np.pi)

    if verbose:
        print(f'F1: {F1}, F2: {F2}, F3: {F3}.')

    return (F1 + F2 + F3) / n, e

def weight_gram(network):
    weight_gram_list = []
    for m in network.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach()
            print(f'Layer {m} has weight shape {W.shape}')
            print(f'Gram matrix shape: {(W @ W.T).shape}')
            weight_gram_list.append(W @ W.T)
    return weight_gram_list

def weight_gram_sl(network):
     
     '''
     For resnet, take conv2 from last block. 
     Assume second last-layer is labelled fc1 for other networks. '''
     if network._get_name() == 'ResNet':
        W = list(network.layer4.children())[0].conv2.weight.detach().flatten(1, -1)
     else:
        W = network.fc1.weight.detach()
     return W @ W.T


def weight_spectra_wtw(network):
    spectra = {}
    for name, module in network.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue

        weight = module.weight.detach()
        weight_matrix = weight.flatten(1) if isinstance(module, torch.nn.Conv2d) else weight
        gram = weight_matrix.T @ weight_matrix
        eigvals = torch.linalg.eigvalsh(gram)

        spectra[name] = {
            "eigenvalues": eigvals.cpu().numpy(),
            "aspect_ratio": weight_matrix.shape[1] / weight_matrix.shape[0],
            "weight_shape": tuple(weight.shape),
            "matrix_shape": tuple(weight_matrix.shape),
        }

    return spectra
    

def _per_example_channel_standardize_and_rescale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""Normalize a single image tensor and rescale to ||x||_2 = sqrt(d).

	Args:
		x: Tensor with shape (C, H, W) and dtype float.
		eps: Small constant to avoid division by zero.
	"""
	if x.ndim != 3:
		raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(x.shape)}")

	# Per-channel mean/std over spatial dimensions.
	mean = x.mean(dim=(1, 2), keepdim=True)
	std = x.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(eps)
	x = (x - mean) / std

	# Rescale so flattened vector has norm sqrt(d), d = C*H*W.
	d = x.numel()
	target_norm = math.sqrt(d)
	current_norm = torch.linalg.vector_norm(x.reshape(-1), ord=2).clamp_min(eps)
	x = x * (target_norm / current_norm)
	return x


def get_cifar10_loaders(
	*,
	data_dir: str = "data",
	batch_size: int = 128,
	pin_memory: bool | None = None,
):
	"""Create CIFAR-10 train/test dataloaders with the required normalization."""

	if pin_memory is None:
		pin_memory = torch.cuda.is_available()

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(_per_example_channel_standardize_and_rescale),
		]
	)

	train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
	test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		pin_memory=pin_memory,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=batch_size,
		shuffle=False,
		pin_memory=pin_memory,
	)

	return train_loader, test_loader

### Plotting functions for heavy tails notebook
def extract_key(file):
    # Result files are named like <model>_<dataset>_b<batch>_<step>_<repeat>[...].pt,
    # and may also include a trailing layer suffix such as _layer1.
    stem = Path(file).stem
    match = re.search(r'_(\d+)_\d+(?:_layer\d+)?$', stem)
    if match:
            return int(match.group(1))
    raise ValueError(f"Could not extract epoch key from filename: {file}")

def get_keys(files):
    keys = []
    for file in files:
        key = extract_key(file)
        keys.append(key)
    return list(set(keys))

def get_key_files(files, key):
    return [f for f in files if f'_{key}_' in f]

def get_key_eigs(files, key, eig_type='weight'):
        key_files = get_key_files(files, key)
        eigs = []
        for file in key_files:
            res_dict = torch.load(file, weights_only=False, map_location='cpu')
            if eig_type == 'weight':
                eigs.append(res_dict['weight_eigvals'])
            elif eig_type == 'ck':
                eigs.append(res_dict['ck_eigvals'])
            elif eig_type == 'ntk':
                eigs.append(res_dict['ntk_eigvals'])
        return np.concatenate(eigs)

def get_free_energy(files, key, energy_type='ck'):
        key_files = get_key_files(files, key)
        energies = []
        for file in key_files:
            res_dict = torch.load(file, weights_only=False, map_location='cpu')
            if energy_type == 'ck':
                energies.append(res_dict['ck_free_energy'])
            elif energy_type == 'ntk':
                energies.append(res_dict['ntk_free_energy'])
            elif energy_type == 'weight':
                energies.append(res_dict['weight_free_energy'])
        return np.array(energies)

def get_test_acc(files, key, acc_key='test_acc'):
        key_files = get_key_files(files, key)
        accuracies = []
        for file in key_files:
            res_dict = torch.load(file, weights_only=False, map_location='cpu')
            accuracies.append(res_dict[acc_key])
        return np.array(accuracies)

def get_train_loss(files, key):
        key_files = get_key_files(files, key)
        losses = []
        for file in key_files:
            res_dict = torch.load(file, weights_only=False, map_location='cpu')
            losses.append(res_dict['train_loss'])
        return np.array(losses)

def get_train_acc(files, key):
        key_files = get_key_files(files, key)
        accs = []
        for file in key_files:
            res_dict = torch.load(file, weights_only=False, map_location='cpu')
            accs.append(res_dict['train_acc'])
        return np.array(accs)

    
def marchenko_pastur_pdf(lmbda, beta):
    """
    Return the Marchenko-Pastur density evaluated at lmbda for the ratio beta = M/N.
    """
    lmbda_min = (1 - np.sqrt(beta))**2
    lmbda_max = (1 + np.sqrt(beta))**2

    # MP density is zero outside [lmbda_min, lmbda_max]
    # Inside that interval:
    #   rho(lambda) = [1 / (2πβλ)] * sqrt((λ_max - λ)(λ - λ_min))
    
    pdf = np.where(
        (lmbda > lmbda_min) & (lmbda < lmbda_max),
        (1.0 / (2.0 * np.pi * beta * lmbda)) *
        np.sqrt((lmbda_max - lmbda) * (lmbda - lmbda_min)),
        0.0
    )
    return pdf

def _mp_log_density(gamma, x):
    x_minus = (1.0 - np.sqrt(gamma)) ** 2
    x_plus = (1.0 + np.sqrt(gamma)) ** 2
    rho = np.sqrt((x_plus - x) * (x - x_minus)) / (2.0 * np.pi * gamma * x)
    return np.log(rho), x_minus, x_plus


def _log_neg_hypu(a, b, z, kappa_switch=80.0):
    """
    Return 2 log |U(a, -b, -z)|.

    For small/moderate kappa, use mpmath.
    For large kappa in the scaled regime used by _logpdf, use the MP-matched asymptotic.
    """
    a_arr, b_arr, z_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(z, dtype=float),
    )

    out = np.empty_like(a_arr, dtype=float)
    flat_a = a_arr.ravel()
    flat_b = b_arr.ravel()
    flat_z = z_arr.ravel()
    flat_out = out.ravel()

    for i, (ai, bi, zi) in enumerate(zip(flat_a, flat_b, flat_z)):
        # kappa = 2.0 * ai

        # if kappa < kappa_switch:
        #     try:
        val = 2.0 * float(mpm.log(mpm.fabs(mpm.hyperu(ai, -bi, -zi))))
        flat_out[i] = val
        #         if np.isfinite(val):
        #             flat_out[i] = val
        #             continue
        #     except Exception:
        #         pass

        # flat_out[i] = float(approx_log_neg_hypu(ai, bi, zi))

    return out

def approx_log_neg_hypu(a, b, z, edge_factor=3.0):
    """
    Approximate 2 log |U(a, -b, -z)| for the exact scaled regime used in helpers.py::_logpdf.

    Assumes:
        a > 0, b > -1, z > 0,
        gamma = a / (a + b + 1) in (0, 1),
        x = z / (a + b + 1) fixed as a -> inf.
    """
    a_arr, b_arr, z_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(z, dtype=float),
    )

    out = np.empty_like(a_arr, dtype=float)
    flat_a = a_arr.ravel()
    flat_b = b_arr.ravel()
    flat_z = z_arr.ravel()
    flat_out = out.ravel()

    for i, (ai, bi, zi) in enumerate(zip(flat_a, flat_b, flat_z)):
        s = ai + bi + 1.0
        gamma = ai / s
        x = zi / s

        if not (ai > 0 and bi > -1 and zi > 0 and 0.0 < gamma < 1.0):
            flat_out[i] = np.nan
            continue

        x_minus = (1.0 - np.sqrt(gamma)) ** 2
        x_plus = (1.0 + np.sqrt(gamma)) ** 2

        # Leading asymptotics are not uniform exactly at the MP edges.
        edge_pad = max(1e-8, edge_factor * (2.0 * ai) ** (-2.0 / 3.0))
        x_eval = np.clip(x, x_minus + edge_pad, x_plus - edge_pad)

        log_rho_mp, _, _ = _mp_log_density(gamma, x_eval)

        flat_out[i] = (
            -gammaln(ai + 1.0)
            -gammaln(s)
            + np.log(s)
            + bi * np.log(zi)
            - zi
            - log_rho_mp
        )

    return out


class marchenko_pastur_gen(rv_continuous):
    """
    Marchenko-Pastur distribution implementation.

    Parameters:
        gam (float): Ratio of dimensions (p / n), where gam > 1.
        scale (float): Scaling factor (default is 1).
    """

    def __init__(self, gam, scale=1):
        c = gam
        if not (0 < c <= 1):
            raise ValueError("Parameter 'gam' must be >= 1.")
        self.c = c
        self.scale = scale
        super().__init__(a=scale * (1 - np.sqrt(c))**2, b=scale * (1 + np.sqrt(c))**2)

    def _pdf(self, x):
        """Probability density function of the Marchenko-Pastur distribution."""
        c, scale = self.c, self.scale
        support_min = scale * (1 - np.sqrt(c))**2
        support_max = scale * (1 + np.sqrt(c))**2

        if np.any(x < support_min) or np.any(x > support_max):
            return 0.0

        sqrt_term = np.sqrt((scale * (1 + np.sqrt(c))**2 - x) * (x - scale * (1 - np.sqrt(c))**2))
        return (1 / (2 * np.pi * c * x * scale)) * sqrt_term

    def _cdf(self, x):
        """Cumulative density function (numerical integration)."""
        from scipy.integrate import quad

        def integrand(t):
            return self._pdf(t)

        result = np.zeros_like(x, dtype=float)
        for i, val in enumerate(x):
            if val > self.a:
                result[i], _ = quad(integrand, self.a, val)
        return result

class _htmp(rv_continuous):
    r"""
    A continuous random variable representing the two-parameter
    high-temperature Marchenko-Pastur (HTMP) distribution.

    Parameters:
        gam (float): Ratio of dimensions (p / n), where gam > 1.
        kap (float): Kappa shape parameter, where kap > 0.

    Parameters
    ----------
    kappa_max : float, optional
        Maximum allowable value for the `kap` (kappa) parameter to ensure
        numerical stability. Default is 200.

    Methods
    -------
    logpdf(x, gam, kap)
        Compute the logarithm of the probability density function (PDF).
    pdf(x, gam, kap)
        Compute the probability density function (PDF).
    cdf(x, gam, kap)
        Compute the cumulative distribution function (CDF).
    fit(data)
        Fit the distribution to data.
    stieltjes(x, gam, kap)
        Compute the Stieltjes transform for the distribution.
    """

    def __init__(self, kappa_max=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = 0
        self.b = np.inf
        self.kappa_max = kappa_max

    def _get_ab(self, gam, kap):
        a = kap/2 * (1 / gam - 1) - 1
        b = kap/2
        return a,b
    

    def _logpdf(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        y = b/gam * x
        const = -loggamma(b+1) - loggamma(a+b+1) + np.log(b/gam)
        return const + a*np.log(y) - y - _log_neg_hypu(b, a, y)

    def _logpxf(self, x, gam, kap):
        return self._logpdf(x, gam, kap)

    def pdf(self, x, gam, kap):
        return np.exp(self._logpdf(x, gam, kap))

    def _argcheck(self, *args):
        gam = args[0]
        kap = args[1]
        if gam > 1 and kap > 0 and kap < self.kappa_max:
            return True
        return False

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (2, 2)
        loc, scale = (0, 1)
        return args + (loc, scale)

    def _cdf_mpm(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        const = -mpm.loggamma(b+1) - mpm.loggamma(a+b+1) + np.log(b/gam)
        integ = lambda t: (b/gam*t)**a * mpm.exp(-b/gam*t) / \
                    mpm.fabs(mpm.hyperu(b,-a,-b/gam*t))**2
        return mpm.quad(integ, [0, x]) * mpm.exp(const)

    def _stieltjes(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        numer = mpm.hyperu(b+1,1-a,-x)
        denom = mpm.hyperu(b,-a,-x)
        return numer / denom

    def _cdf(self, x, gam, kap):
        return np.vectorize(self._cdf_mpm, otypes=(float,))(x,gam,kap)

    def stieltjes(self, x, gam, kap):
        return np.vectorize(self._stieltjes, otypes=(complex,))(x,gam,kap)

    def _shape_info(self):
        return [_ShapeInfo("gam", False, (1,np.inf), (False,False)),
                _ShapeInfo("kap", False, (0,self.kappa_max), (False,False))]

htmp = _htmp()
    
gamma_dict = {'lenet': {'weight': 120 / (16*5*5), 'ck': 84 / 1000, 'ntk': 10000 / 63000},
                'minialexnet': {'weight': 1000/(192 * 4 * 4), 'ck': 1 / 50},
                'resnet9': {'weight': 512/4608, 'ck': 512/50000},
                'resnet18': {'weight': 512/4608, 'ck': 512/50000},
                'resnet34': {'weight': 512/4608, 'ck': 512/50000},
                'resnet50': {'weight': 512/4608, 'ck': 4*512/50000},
                'pythia-70m-deduped': {'weight': 0.25, 'ck': 512/50000},
                'pythia-160m-deduped': {'weight': 0.25, 'ck': 512/50000},
                'pythia-410m-deduped': {'weight': 0.25, 'ck': 512/50000},
                'pythia-1b-deduped': {'weight': 0.25, 'ck': 512/50000},
}


kappa_dict = {
    'lenet': {
        'subsample': {
            1000: np.inf, 800: np.inf, 250: np.inf, 100: np.inf, 50: np.inf, 5: np.nan
        },
        'full': {
        'weight': {
            1000: np.inf, 800: np.inf, 250: 1.1, 100: 0.9, 50: np.nan, 5: np.nan
        },
        },
    },
    'minialexnet': {
        'subsample': {
        'weight': {
            1000: np.inf, 800: np.inf, 250: np.inf, 100: np.inf, 50: np.inf, 5: np.nan
        },
        },
        'full': {
        'weight': {
            1000: np.inf, 800: np.inf, 250: 5.5, 100: 2.3, 50: np.nan, 5: np.nan
        }
        }
    }
}

def free_energy_eigs(eigvals, n, lam=1, jitter=1e-9):
    log_det = torch.sum(torch.log((eigvals + jitter))) / (n)
    trace = torch.sum(1 / (eigvals + jitter)) / (n)
    const = 1/2 * np.log(2 * np.pi / lam)
    F = lam / 2 * trace + log_det / 2 + const
    return F, trace, log_det

def free_energy_terms(g, ld, lam):
    g = np.array(g); ld = np.array(ld)
    const = -1/2 * np.log(lam / (2 * np.pi))
    return lam * g / 2 + ld / 2 + const

def htmp_eig_pdf(lam, gamma, kappa, beta):
    a = kappa / (2 * gamma * beta)
    s = 1.0 / (a * lam)
    return htmp.pdf(s, gamma, kappa) / (a * lam**2)

def loglog_pdf_loss(eigs, gamma, kappa, beta, nbins=80, eps=1e-300):
    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs) & (eigs > 0)]

    edges = np.logspace(np.log10(eigs.min()), np.log10(eigs.max()), nbins + 1)
    counts, edges = np.histogram(eigs, bins=edges, density=False)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centers (correct for log bins)

    p_emp = counts / (len(eigs) * widths)      # empirical density estimate
    p_mod = htmp_eig_pdf(centers, gamma, kappa, beta)

    mask = (counts > 0) & np.isfinite(p_mod) & (p_mod > 0) & np.isfinite(p_emp) & (p_emp > 0)
    return np.mean((np.log(p_emp[mask] + eps) - np.log(p_mod[mask] + eps))**2)    


def ternary_search(f, left, right, its, verbose=False, input_name = 'input', output_name = 'function', log_scale=False):
    if log_scale:
        left_input, right_input = np.log10(left), np.log10(right)
    else:
        left_input, right_input = left, right

    # Ternary Search
    for _ in range(its):
        left_third = left_input + (right_input - left_input) / 3
        right_third = right_input - (right_input - left_input) / 3

        # Left function value
        left_val = 10**left_third if log_scale else left_third
        right_val = 10**right_third if log_scale else right_third
        left_f = f(left_val)
        right_f = f(right_val)

        if left_f > right_f:
            left_input = left_third
        else:
            right_input = right_third

        input_log = (left_input + right_input) / 2
        input = 10**input_log if log_scale else input_log

        # Print info
        if verbose:
            print(f'\n{input_name}: {input:.3}; {output_name}: [{left_f:.4},{right_f:.4}]') 

        if abs(right_input - left_input) <= 1e-2:
            if verbose:
                print('Converged.')
            break

    return input

def ternary_search_2d(f, x_left, x_right, x_its, y_left, y_right, y_its, verbose=False, log_scale_x=False, log_scale_y=True):
    '''
    Assume f(x,y). 
    '''
    def f_outer(y):
        x_star = ternary_search(lambda x: f(x, y), x_left, x_right, x_its, verbose=verbose, input_name='x', output_name='f', log_scale=log_scale_x)
        return f(x_star, y)
    
    if y_its == 1:
        y_star = y_left
    else:
        y_star = ternary_search(f_outer, y_left, y_right, y_its, verbose=verbose, input_name='y', output_name='f', log_scale=log_scale_y)
    x_star = ternary_search(lambda x: f(x, y_star), x_left, x_right, x_its, verbose=verbose, input_name='x', output_name='f', log_scale=log_scale_x)
    return x_star, y_star, f(x_star, y_star)
    

def grid_search_2d(f, x_min, x_max, x_its, y_min, y_max, y_its, verbose=False, log_scale_x=True, log_scale_y=True):
    if log_scale_x:
        x_values = np.logspace(x_min, x_max, x_its)
    else:
        x_values = np.linspace(x_min, x_max, x_its)
    if log_scale_y:
        y_values = np.logspace(y_min, y_max, y_its)
    else:
        y_values = np.linspace(y_min, y_max, y_its)

    best_x = None
    best_y = None
    best_f = float("inf")

    for x in x_values:
        for y in y_values:
            f_val = f(x, y)
            # if verbose:
            #     print(f"Evaluated (x={x:.4f}, y={y:.4f}) with f={f_val:.4f}")
            if f_val < best_f:
                best_f = f_val
                best_x = x
                best_y = y
                if verbose:
                    print(f"New best found: f({x:.4}, {y:.4}) = {f_val:.4}")

    return best_x, best_y, best_f

def stieltjes_weights(kappa, gamma, beta, x):
        return beta * scipy.special.hyperu(kappa/2+1, 2 - kappa / (2*gamma) + kappa / 2, beta * x) \
            / scipy.special.hyperu(kappa/2, 1 - kappa / (2*gamma) + kappa / 2, beta * x)

def stieltjes_feature(kappa, gamma, beta, x):
        return 1/x - beta / x**2 * scipy.special.hyperu(kappa/2+1, 2 - kappa / (2*gamma) + kappa / 2, beta / x) \
            / scipy.special.hyperu(kappa/2, 1 - kappa / (2*gamma) + kappa / 2, beta / x)

def wass_distance_grid(eigs, gamma, kappas, betas, bins=50, inverse=False, stieltjes=False, lp_ord=1, moment_penalty_weight=0.0):
    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs)]
    kappas = np.asarray(kappas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    if eigs.size == 0 or kappas.size == 0 or betas.size == 0:
        return np.full((kappas.size, betas.size), float('inf'))

    if stieltjes:
        x_min = 1e-4
        x_max = 1e2*np.max(eigs)
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), bins)


        empirical_stieltjes = np.mean(1.0 / (eigs[None, :] + x_vals[:, None]), axis=1)
        if not inverse:
            theoretical_stieltjes = np.asarray(
                np.real_if_close(
                    stieltjes_weights(
                        kappas[:, None, None],
                        gamma,
                        betas[None, :, None],
                        x_vals[None, None, :],
                    )
                ),
                dtype=float,
            )
        elif inverse:
            theoretical_stieltjes = np.asarray(
                np.real_if_close(
                    stieltjes_feature(
                        kappas[:, None, None],
                        gamma,
                        betas[None, :, None],
                        x_vals[None, None, :] * gamma,
                    )*gamma
                ),
                dtype=float,
            )
    
        valid = np.isfinite(theoretical_stieltjes)
        # Where theory is non-finite, treat as theory=0 (full residual = empirical value)
        # so that bad/degenerate parameters are never rewarded with a near-zero distance.
        diffs = np.where(valid, empirical_stieltjes[None, None, :] - theoretical_stieltjes, empirical_stieltjes[None, None, :])
        distances = np.linalg.norm(diffs, ord=lp_ord, axis=2) / np.linalg.norm(empirical_stieltjes, ord=lp_ord)
        distances[~np.any(valid, axis=2)] = float('inf')
        if moment_penalty_weight > 0:
            distances = _add_moment_penalty_grid(distances, eigs, gamma, kappas, betas, moment_penalty_weight)
        return distances

    if inverse:
        bin_edges = np.logspace(-4, np.log10(eigs.max()), bins)
    else:
        bin_edges = np.linspace(eigs.min(), eigs.max(), bins)
        # bin_edges = np.concatenate([1e-5 * np.array([1]), bin_edges[0:]])  # Ensure positive bins for log scale

    # print(bin_edges)
    p_e, x_e = np.histogram(eigs, bins=bin_edges, density=True)
    support = x_e[:-1]
    a = kappas[:, None, None] / (2 * gamma * betas[None, :, None])

    if inverse:
        spectral_density = lambda x, kappa: htmp.pdf(1 / a / x, gamma, kappa) / (a * x**2)
        p = spectral_density(support[None, None, :] * gamma, kappas[:, None, None]) * gamma
        # p = htmp.pdf(1 / a / support[None, None, :], gamma, kappas[:, None, None]) / (a * support[None, None, :]**2)
    else:
        p = htmp.pdf(support[None, None, :] / a, gamma, kappas[:, None, None]) / a

    p = np.asarray(np.real_if_close(p), dtype=float)
    valid = np.isfinite(p)
    diffs = np.where(valid, p_e[None, None, :] - p, 0.0)
    distances = np.linalg.norm(diffs, ord=lp_ord, axis=2) / np.linalg.norm(p_e, ord=lp_ord)
    distances[~np.any(valid, axis=2)] = float('inf')
    if moment_penalty_weight > 0:
        distances = _add_moment_penalty_grid(distances, eigs, gamma, kappas, betas, moment_penalty_weight)
    return distances

def _add_moment_penalty_grid(distances, eigs, gamma, kappas, betas, moment_penalty_weight):
    eigs_pos = eigs[eigs > 0]
    if eigs_pos.size == 0:
        return distances
    emp_mean = np.mean(eigs_pos)
    if emp_mean <= 0:
        return distances
    theory_mean = kappas[:, None] / (2 * gamma * betas[None, :])  # shape (n_kappa, n_beta)
    rel_err = ((emp_mean - theory_mean) / emp_mean) ** 2
    penalty = np.where(theory_mean > 0, moment_penalty_weight * rel_err, 0.0)
    return distances + penalty

def wass_distance(eigs, gamma, kappa, beta, bins=50, inverse=False, stieltjes=False, lp_ord=1, moment_penalty_weight=0.0):
    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs)]
    if eigs.size == 0:
        return float('inf')

    if stieltjes:
        eigs_pos = eigs[eigs > 0]
        if eigs_pos.size == 0:
            return float('inf')

        # Evaluate empirical/theoretical Stieltjes transforms on a positive x-grid.
        x_min = max(1e-8, 1e-2 * np.min(eigs_pos))
        x_max = max(10.0 * x_min, 1e2*np.max(eigs_pos))
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), bins)

        empirical_stieltjes = np.array([np.mean(1.0 / (eigs_pos + x)) for x in x_vals])
        theoretical_stieltjes = np.real_if_close(stieltjes_weights(kappa, gamma, beta, x_vals))

        valid = np.isfinite(empirical_stieltjes) & np.isfinite(theoretical_stieltjes)
        if not np.any(valid):
            return float('inf')
        dist = np.linalg.norm(empirical_stieltjes[valid] - theoretical_stieltjes[valid], ord=lp_ord) / bins
        if moment_penalty_weight > 0:
            dist = dist + _moment_penalty(eigs, gamma, kappa, beta, moment_penalty_weight)
        return dist

    # Create log-bins
    n=bins
    if inverse:
        bins = np.logspace(-4, np.log10(eigs.max()), bins)
    else:
        # bins = np.linspace(1e-5, eigs.max().item(), bins)
        bins = np.logspace(-5, np.log10(eigs.max()), bins)
    p_e, x_e = np.histogram(eigs, bins=bins, density=True)
    a = kappa / (2 * gamma * beta)
    if inverse:
        p = htmp.pdf(1 / a / x_e[:-1], gamma, kappa) / (a * x_e[:-1]**2)
        dist = np.linalg.norm(p_e - p, ord=lp_ord)
    else:
        p = htmp.pdf(x_e[:-1] / a, gamma, kappa) / a
        dist = np.linalg.norm(p_e - p, ord=lp_ord) / n
    if moment_penalty_weight > 0:
        dist = dist + _moment_penalty(eigs, gamma, kappa, beta, moment_penalty_weight)
    return dist

def _moment_penalty(eigs, gamma, kappa, beta, moment_penalty_weight):
    eigs_pos = eigs[eigs > 0]
    if eigs_pos.size == 0:
        return 0.0
    emp_mean = np.mean(eigs_pos)
    theory_mean = kappa / (2 * gamma * beta)
    if emp_mean <= 0 or theory_mean <= 0:
        return 0.0
    return moment_penalty_weight * ((emp_mean - theory_mean) / emp_mean) ** 2
    
    
def _warn_if_at_boundary(kappa, beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta = False, rtol=1e-3):
    """Print a warning if the optimal parameters are at the boundary of the search range."""
    at_boundary = []
    if np.isclose(kappa, kappa_min, rtol=rtol):
        at_boundary.append(f"kappa={kappa:.4g} is at kappa_min={kappa_min:.4g}")
    if np.isclose(kappa, kappa_max, rtol=rtol):
        at_boundary.append(f"kappa={kappa:.4g} is at kappa_max={kappa_max:.4g}")
    if not ignore_beta:
        if np.isclose(beta, beta_min, rtol=rtol):
            at_boundary.append(f"beta={beta:.4g} is at beta_min={beta_min:.4g}")
        if np.isclose(beta, beta_max, rtol=rtol):
            at_boundary.append(f"beta={beta:.4g} is at beta_max={beta_max:.4g}")
    if at_boundary:
        print(f"Warning: minimum distance found at parameter boundary ({'; '.join(at_boundary)}). "
              "A wider search range is probably required for a better fit.")


def find_best_htmp(
    eigs,
    gamma,
    search_method='grid',
    method='pdf',
    kappa_min=1e-3,
    kappa_max=1e1,
    num_kappas=100,
    beta_min=1e-2,
    beta_max=1e2,
    num_betas=None,
    n_eval = 50,
    inverse=False,
    lp_ord=1,
    vectorized_grid=True,
    kappa_log_scale=True,
    moment_penalty_weight=0.0,
    beta_log_scale=True,
    verbose=False
):
    method = method.lower()
    if method not in ('pdf', 'stieltjes', 'ks'):
        raise ValueError("method must be one of: 'pdf', 'stieltjes', 'ks'")
    stieltjes = method == 'stieltjes'

    if method == 'ks':
        from htmp_cdf import compute_htmp_ks_distance

        def objective(kappa, beta):
            return compute_htmp_ks_distance(
                eigs,
                kappa,
                gamma,
                beta,
                tau=0.0,
                inverse=inverse,
                return_details=False,
            )

        if search_method == 'grid':
            if kappa_log_scale:
                kappa_values = np.logspace(np.log10(kappa_min), np.log10(kappa_max), num_kappas)
            else:
                kappa_values = np.linspace(kappa_min, kappa_max, num_kappas)
            if beta_log_scale:
                beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num_betas if num_betas is not None else 1)
            else:
                beta_values = np.linspace(beta_min, beta_max, num_betas if num_betas is not None else 1)

            best_kappa = None
            best_beta = None
            best_distance = float('inf')
            for kappa in kappa_values:
                for beta in beta_values:
                    distance = objective(kappa, beta)
                    if distance < best_distance:
                        best_distance = distance
                        best_kappa = kappa
                        best_beta = beta

            if verbose:
                print(f"Best grid point: kappa={best_kappa:.4g}, beta={best_beta:.4g}, distance={best_distance:.6g}")
            _warn_if_at_boundary(best_kappa, best_beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta=(num_betas is None or num_betas == 1 ))
            return best_kappa, best_beta, best_distance

        if search_method == 'ternary':
            best_kappa, best_beta, best_distance = ternary_search_2d(
                objective,
                x_left=kappa_min, x_right=kappa_max, x_its=num_kappas,
                y_left=beta_min, y_right=beta_max, y_its=num_betas if num_betas is not None else 1,
                verbose=verbose,
                log_scale_x=kappa_log_scale,
                log_scale_y=beta_log_scale,
            )
            _warn_if_at_boundary(best_kappa, best_beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta=(num_betas is None or num_betas == 1 ))
            return best_kappa, best_beta, best_distance

        raise ValueError(f"Invalid search method: {search_method}. Choose 'grid' or 'ternary'.")

    if search_method == 'grid':
        if vectorized_grid:
            if kappa_log_scale:
                kappa_values = np.logspace(np.log10(kappa_min), np.log10(kappa_max), num_kappas)
            else:
                kappa_values = np.linspace(kappa_min, kappa_max, num_kappas)
            if beta_log_scale:
                beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num_betas if num_betas is not None else 1)
            else:
                beta_values = np.linspace(beta_min, beta_max, num_betas if num_betas is not None else 1)
            distance_grid = wass_distance_grid(
                eigs,
                gamma,
                kappa_values,
                beta_values,
                bins=n_eval,
                inverse=inverse,
                stieltjes=stieltjes,
                lp_ord=lp_ord,
                moment_penalty_weight=moment_penalty_weight,
            )
            best_idx = np.unravel_index(np.argmin(distance_grid), distance_grid.shape)
            best_kappa = kappa_values[best_idx[0]]
            best_beta = beta_values[best_idx[1]]
            best_distance = distance_grid[best_idx]
            if verbose:
                print(f"Best grid point: kappa={best_kappa:.4g}, beta={best_beta:.4g}, distance={best_distance:.6g}")
            _warn_if_at_boundary(best_kappa, best_beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta=(num_betas is None or num_betas == 1 ))
            return best_kappa, best_beta, best_distance

        if kappa_log_scale:
            x_min_kappa = np.log10(kappa_min)
            x_max_kappa = np.log10(kappa_max)
        else:
            x_min_kappa = kappa_min
            x_max_kappa = kappa_max

        if beta_log_scale:
            y_min_beta = np.log10(beta_min)
            y_max_beta = np.log10(beta_max)
        else:
            y_min_beta = beta_min
            y_max_beta = beta_max
        
        best_kappa, best_beta, best_distance = grid_search_2d(
            lambda kappa, beta: wass_distance(
                eigs,
                gamma,
                kappa,
                beta,
                n_eval,
                inverse=inverse,
                stieltjes=stieltjes,
                lp_ord=lp_ord,
                moment_penalty_weight=moment_penalty_weight,
            ),
            x_min=x_min_kappa,
            x_max=x_max_kappa,
            x_its=num_kappas,
            y_min=y_min_beta,
            y_max=y_max_beta,
            y_its=num_betas if num_betas is not None else 1,
            verbose=verbose,
            log_scale_x=kappa_log_scale,
            log_scale_y=beta_log_scale,
        )
        _warn_if_at_boundary(best_kappa, best_beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta=(num_betas is None or num_betas == 1 ))
        return best_kappa, best_beta, best_distance
    elif search_method == 'ternary':
        y_left_beta = beta_min
        y_right_beta = beta_max
        best_kappa, best_beta, best_distance = ternary_search_2d(
            lambda kappa, beta: wass_distance(eigs, gamma, kappa, beta, n_eval, inverse=inverse, stieltjes=stieltjes, lp_ord=lp_ord, moment_penalty_weight=moment_penalty_weight),
            x_left=kappa_min, x_right=kappa_max, x_its=num_kappas,
            y_left=y_left_beta, y_right=y_right_beta, y_its=num_betas if num_betas is not None else 1,
            verbose=verbose,
            log_scale_x=kappa_log_scale,
            log_scale_y=beta_log_scale,
        )
        _warn_if_at_boundary(best_kappa, best_beta, kappa_min, kappa_max, beta_min, beta_max, ignore_beta=(num_betas is None or num_betas == 1 ))
        return best_kappa, best_beta, best_distance
    
    else:
        raise ValueError(f"Invalid search method: {search_method}. Choose 'grid' or 'ternary'.")


def compute_correlation(x, y, method='pearson'):
    method = method.lower()
    if method == 'pearson':
        return np.corrcoef(x, y)[0, 1]
    if method == 'spearman':
        return stats.spearmanr(x, y).statistic
    if method == 'kendall':
        res = stats.kendalltau(x, y)
        return res.statistic, res.pvalue
    raise ValueError("method must be one of: 'pearson', 'spearman', 'kendall'")