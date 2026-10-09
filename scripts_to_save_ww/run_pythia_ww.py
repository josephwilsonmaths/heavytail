"""Run from the repository root: python -m scripts_to_save_ww.run_pythia_ww."""

import argparse
import json
import os
from pathlib import Path

from threadpoolctl import threadpool_limits

from util.pythia_ww import (DEFAULT_MODELS, run_model, build_comparison,
                       correlation_tables, export_comparison)


def main():
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    parser.add_argument('--step', type=int, default=143000)
    parser.add_argument('--output-dir', default='results/pythia/ww')
    parser.add_argument('--cache-dir', default='results/pythia/ww/checkpoint_cache')
    parser.add_argument('--spectra-dir', default='results/pythia')
    parser.add_argument('--evals-root', default='evals')
    parser.add_argument('--task', default='lambada_openai')
    parser.add_argument('--shot', default='zero-shot')
    parser.add_argument('--metric', default='acc')
    parser.add_argument('--jitter', type=float, default=0.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--no-memory-guard', action='store_true')
    args = parser.parse_args()
    failures = []
    with threadpool_limits(limits=args.threads):
        if not args.export_only:
            for model in args.models:
                try:
                    run_model(model, step=args.step, output_dir=args.output_dir,
                              cache_dir=args.cache_dir, memory_guard=not args.no_memory_guard)
                except Exception as exc:
                    failures.append({'model': model, 'error': str(exc)})
                    print(f'[FAILED] {model}: {exc}', flush=True)
        failure_path = Path(args.output_dir) / 'failures.json'
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2))
        if failures:
            raise SystemExit('Incomplete WW results; rerun to resume. See failures.json.')
        data = build_comparison(args.models, step=args.step, output_dir=args.output_dir,
                                spectra_dir=args.spectra_dir, evals_root=args.evals_root,
                                task=args.task, shot=args.shot, metric=args.metric, jitter=args.jitter)
        correlations, bfe_correlations = correlation_tables(data)
        path = Path(args.output_dir) / f'pythia_step{args.step}_{args.task}_{args.shot}_{args.metric}.xlsx'
        export_comparison(data, correlations, bfe_correlations, path)
        print(correlations.to_string(index=False), flush=True)
        print(f'Saved {path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
