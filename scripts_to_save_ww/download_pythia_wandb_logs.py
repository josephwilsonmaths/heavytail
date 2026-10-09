"""Download the public Pythia deduped-v2 training logs from W&B.

The original EleutherAI runs were restarted several times, so one released
model can be represented by multiple W&B run IDs.  The run manifest below
records the segments that form the released 410M, 1B, and 12B deduped-v2
training histories.  Later segments take precedence at overlapping steps.

Raw W&B history is saved as JSONL.  The script also writes stitched per-model
CSVs, a combined CSV, and a CSV aligned to the locally saved spectral-summary
checkpoints in ``results/htmp/epoch``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
WANDB_ENTITY = "eleutherai"
WANDB_PROJECT = "pythia"

# Ordered from oldest to newest.  When restarted runs overlap, the newer run
# is the one that fed the next checkpoint and therefore replaces older rows.
PYTHIA_RUN_SEGMENTS = {
    "pythia-410m-deduped": [
        {"run_id": "15ft9mwt", "min_step": 0, "max_step": 3447},
        {"run_id": "2pum2cd0", "min_step": 3000, "max_step": 3240},
        {"run_id": "irm38y25", "min_step": 3001, "max_step": 36043},
        {"run_id": "2pmqy4bd", "min_step": 35001, "max_step": 143000},
    ],
    "pythia-1b-deduped": [
        {"run_id": "2sysbatj", "min_step": 0, "max_step": 79990},
    ],
    "pythia-12b-deduped": [
        {"run_id": "1hkl964p", "min_step": 0, "max_step": 9014},
        {"run_id": "2kpltosr", "min_step": 9001, "max_step": 55801},
        {"run_id": "6b4idzqw", "min_step": 55000, "max_step": 62054},
        {"run_id": "3gehdaci", "min_step": 74000, "max_step": 74982},
    ],
}

HISTORY_QUERY = """
query SampledHistoryPage(
  $entity: String!,
  $project: String!,
  $run: String!,
  $spec: JSONString!
) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      sampledHistory(specs: [$spec])
    }
  }
}
"""

HISTORY_KEYS = [
    "_step",
    "_timestamp",
    "_runtime",
    "train/lm_loss",
    "train/learning_rate",
    "train/loss_scale",
]

CSV_COLUMNS = [
    "model",
    "step",
    "train_loss",
    "learning_rate",
    "loss_scale",
    "timestamp",
    "runtime_seconds",
    "wandb_run_id",
    "wandb_url",
]


def _graphql(query: str, variables: dict, retries: int = 5) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        WANDB_GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "heavytail-pythia-log-downloader"},
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
            if result.get("errors"):
                raise RuntimeError(json.dumps(result["errors"], indent=2))
            return result["data"]
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"W&B request failed after {retries} attempts: {exc}") from exc
            time.sleep(2**attempt)

    raise AssertionError("unreachable")


def download_run_history(run_id: str, min_step: int, max_step: int, page_size: int) -> list[dict]:
    rows = []
    page_start = int(min_step)
    stop = int(max_step) + 1

    while page_start < stop:
        page_stop = min(page_start + page_size, stop)
        data = _graphql(
            HISTORY_QUERY,
            {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "run": run_id,
                "spec": json.dumps(
                    {
                        "keys": HISTORY_KEYS,
                        "minStep": page_start,
                        # W&B treats maxStep as inclusive.  Subtract one so a
                        # 10,000-row page is not silently sampled down from
                        # 10,001 points (which would drop one training step).
                        "maxStep": page_stop - 1,
                        "samples": page_stop - page_start,
                    },
                    separators=(",", ":"),
                ),
            },
        )
        history = data["project"]["run"]["sampledHistory"][0]
        rows.extend(history)
        page_start = page_stop

    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _csv_row(model: str, raw_row: dict, run_id: str) -> dict:
    return {
        "model": model,
        "step": int(raw_row["_step"]),
        "train_loss": raw_row.get("train/lm_loss"),
        "learning_rate": raw_row.get("train/learning_rate"),
        "loss_scale": raw_row.get("train/loss_scale"),
        "timestamp": raw_row.get("_timestamp"),
        "runtime_seconds": raw_row.get("_runtime"),
        "wandb_run_id": run_id,
        "wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str] = CSV_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _saved_checkpoint_steps(checkpoint_dir: Path, model: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(model)}_pile_b2m_(\d+)_\d+$")
    steps = set()
    for file_path in checkpoint_dir.glob(f"{model}_pile_b2m_*.pt"):
        match = pattern.match(file_path.stem)
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def _contiguous_ranges(steps: list[int]) -> list[list[int]]:
    if not steps:
        return []
    ranges = []
    start = previous = steps[0]
    for step in steps[1:]:
        if step != previous + 1:
            ranges.append([start, previous])
            start = step
        previous = step
    ranges.append([start, previous])
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pythia/training_logs"),
        help="Directory for downloaded and stitched logs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/htmp/epoch"),
        help="Directory containing the saved Pythia spectral summaries.",
    )
    parser.add_argument("--page-size", type=int, default=10_000)
    args = parser.parse_args()

    if args.page_size <= 0 or args.page_size > 10_000:
        parser.error("--page-size must be between 1 and 10000")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "wandb_raw"
    combined_rows = []
    checkpoint_rows = []
    manifest = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "downloaded_from": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}",
        "models": {},
    }

    for model, segments in PYTHIA_RUN_SEGMENTS.items():
        print(f"Downloading {model}...")
        rows_by_step = {}
        run_summaries = []

        for segment in segments:
            run_id = segment["run_id"]
            print(f"  {run_id}: steps {segment['min_step']}..{segment['max_step']}")
            raw_rows = download_run_history(
                run_id,
                segment["min_step"],
                segment["max_step"],
                args.page_size,
            )
            _write_jsonl(raw_dir / f"{model}_{run_id}.jsonl", raw_rows)

            kept_rows = 0
            for raw_row in raw_rows:
                if "_step" not in raw_row or raw_row.get("train/lm_loss") is None:
                    continue
                csv_row = _csv_row(model, raw_row, run_id)
                rows_by_step[csv_row["step"]] = csv_row
                kept_rows += 1

            run_summaries.append(
                {
                    **segment,
                    "downloaded_history_rows": len(raw_rows),
                    "rows_with_train_loss": kept_rows,
                    "wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
                }
            )

        model_rows = [rows_by_step[step] for step in sorted(rows_by_step)]
        _write_csv(args.output_dir / f"{model}_train_log.csv", model_rows)
        combined_rows.extend(model_rows)

        saved_steps = _saved_checkpoint_steps(args.checkpoint_dir, model)
        for step in saved_steps:
            row = rows_by_step.get(step)
            checkpoint_rows.append(
                {
                    "model": model,
                    "epoch": step,
                    "train_loss": "" if row is None else row["train_loss"],
                    "wandb_run_id": "" if row is None else row["wandb_run_id"],
                    "wandb_url": "" if row is None else row["wandb_url"],
                }
            )

        available_steps = sorted(rows_by_step)
        manifest["models"][model] = {
            "runs": run_summaries,
            "stitched_rows": len(model_rows),
            "available_step_ranges": _contiguous_ranges(available_steps),
            "saved_checkpoint_steps": saved_steps,
            "saved_checkpoints_with_loss": [step for step in saved_steps if step in rows_by_step],
            "saved_checkpoints_missing_loss": [step for step in saved_steps if step not in rows_by_step],
        }

    combined_rows.sort(key=lambda row: (row["model"], row["step"]))
    _write_csv(args.output_dir / "pythia_deduped_v2_train_logs.csv", combined_rows)
    _write_csv(
        args.output_dir / "pythia_deduped_v2_train_loss_checkpoints.csv",
        checkpoint_rows,
        ["model", "epoch", "train_loss", "wandb_run_id", "wandb_url"],
    )
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Saved logs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
