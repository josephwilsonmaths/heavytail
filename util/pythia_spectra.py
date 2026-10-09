"""Memory-conscious, restartable extraction of Pythia weight spectra."""

import json
from pathlib import Path
import shutil
import time
import warnings

import torch
from safetensors import safe_open


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_weights(path, names):
    """Keep only one shard open; mmap avoids eagerly loading all its tensors."""
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                yield name, handle.get_tensor(name)
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as exc:
            # Older torch.save files do not support memory mapping.
            if "mmap can only be used" not in str(exc):
                raise
            warnings.warn(f"{path.name} cannot be memory-mapped; loading one shard into RAM.")
            state = torch.load(path, map_location="cpu", weights_only=True)
        try:
            for name in names:
                yield name, state[name]
        finally:
            del state


def save_checkpoint_summary(
    model_name,
    step,
    *,
    model_cache_root=".",
    results_dir="results/htmp/epoch",
    delete_model_cache=False,
    access_token=None,
    device="cpu",
    spectral_dtype=torch.float32,
    resume=True,
    verbose=True,
):
    """Save the original W.T @ W spectra without constructing a language model.

    Supports indexed and single-file PyTorch/safetensors checkpoints. Downloads
    are sequential and remain cached on failure. Completed matrices are saved
    atomically to a .partial.pt file; a completed .pt is reused when resume=True.
    Use resume=False to recompute (also when changing spectral_dtype). Only load
    summaries in a trusted results_dir: their legacy format contains NumPy arrays.
    """
    short_name = str(model_name).removeprefix("EleutherAI/")
    if not short_name.startswith("pythia-"):
        short_name = "pythia-" + short_name
    if "/" in short_name or "\\" in short_name:
        raise ValueError("Expected a Pythia model name, not a filesystem path.")
    step = int(step)
    device = torch.device(device)
    if spectral_dtype not in (torch.float32, torch.float64):
        raise ValueError("spectral_dtype must be torch.float32 or torch.float64.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use device='cpu'.")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{short_name}_pile_b2m_{step}_0.pt"
    partial_path = save_path.with_suffix(".partial.pt")

    def log(message):
        if verbose:
            print(f"[{short_name} step{step}] {message}", flush=True)

    if resume and save_path.exists():
        result = torch.load(save_path, map_location="cpu", weights_only=False)
        if result.get("model") != short_name or result.get("epoch") != step:
            raise ValueError(f"Summary metadata does not match {save_path}.")
        log(f"Already saved: {save_path}")
        return result, save_path

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = f"EleutherAI/{short_name}"
    cache_root = Path(model_cache_root).resolve()
    cache_dir = cache_root / short_name / f"step{step}"
    # Resolve the requested step once, then pin every file to this exact commit.
    log("Reading checkpoint metadata")
    info = HfApi().model_info(repo_id, revision=f"step{step}", token=access_token)
    filenames = {entry.rfilename for entry in info.siblings}

    def download(name):
        log(f"Downloading/reusing {name}")
        return Path(hf_hub_download(
            repo_id, name, revision=info.sha, cache_dir=str(cache_dir), token=access_token,
        ))

    config = json.loads(download("config.json").read_text(encoding="utf-8"))
    n_layers = int(config["num_hidden_layers"])
    targets = {}
    for idx in range(n_layers):
        targets[f"gpt_neox.layers.{idx}.attention.query_key_value.weight"] = ("QueryKey", idx)
        targets[f"gpt_neox.layers.{idx}.mlp.dense_h_to_4h.weight"] = ("Dense", idx)
    targets["embed_out.weight"] = ("EmbedOut", None)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name in filenames:
            weight_map = json.loads(download(index_name).read_text(encoding="utf-8"))["weight_map"]
            missing = targets.keys() - weight_map.keys()
            if missing:
                raise KeyError(f"Checkpoint is missing required weights: {sorted(missing)}")
            break
    else:
        single = next((name for name in ("model.safetensors", "pytorch_model.bin") if name in filenames), None)
        if single is None:
            raise FileNotFoundError(f"No supported weights at {repo_id}/step{step}.")
        weight_map = dict.fromkeys(targets, single)

    result = {"QueryKey": {}, "Dense": {}, "EmbedOut": {},
              "model": short_name, "dataset": "pile", "epoch": step}
    if resume and partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if (partial["revision"] != info.sha or partial["dtype"] != str(spectral_dtype)
                or partial["summary"]["model"] != short_name or partial["summary"]["epoch"] != step):
            raise ValueError("Partial result settings differ; use resume=False to recompute.")
        result = partial["summary"]

    shards = {}
    completed = 0
    for name, (group, idx) in targets.items():
        done = bool(result[group]) if idx is None else idx in result[group]
        if done:
            completed += 1
        else:
            shards.setdefault(weight_map[name], []).append(name)
    log(f"{completed}/{len(targets)} spectra already complete; computing on {device}")

    for shard, names in shards.items():
        path = download(shard)
        weights = _read_weights(path, names)
        try:
            for name, raw_weight in weights:
                started = time.perf_counter()
                log(f"Computing {name} ({completed + 1}/{len(targets)})")
                with torch.no_grad():
                    weight = raw_weight.to(device=device, dtype=spectral_dtype)
                    gram = weight.T @ weight
                    eigvals = torch.linalg.eigvalsh(gram).cpu().numpy()
                    summary = {"eigvals": eigvals, "aspect_ratio": weight.shape[1] / weight.shape[0]}
                    del gram, weight, raw_weight
                group, idx = targets[name]
                if idx is None:
                    result[group] = summary
                else:
                    result[group][idx] = summary
                _atomic_save({"revision": info.sha, "dtype": str(spectral_dtype), "summary": result}, partial_path)
                completed += 1
                log(f"Saved spectrum in {time.perf_counter() - started:.1f}s")
        finally:
            weights.close()

    _atomic_save(result, save_path)
    partial_path.unlink(missing_ok=True)
    log(f"Saved {save_path}")
    if delete_model_cache and cache_dir.exists():
        # Do not follow a user-created directory link outside the cache root.
        expected = cache_root / short_name / f"step{step}"
        if cache_dir.resolve() != expected:
            warnings.warn(f"Skipping cleanup of redirected cache directory: {cache_dir}")
        else:
            try:
                shutil.rmtree(cache_dir)
            except OSError as exc:
                warnings.warn(f"Summary saved, but cache cleanup failed: {exc}")
    return result, save_path
