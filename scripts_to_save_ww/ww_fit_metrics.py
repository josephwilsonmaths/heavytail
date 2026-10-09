"""Cache PL alpha and exponentially truncated PL Lambda in WW result files."""

import json
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_or_fit_ww_metrics(summary_path, model_factory, lambda_method='paper_e_tpl'):
    """Preserve old-WW metrics and cache post hoc lambda.

    Default: paper E-TPL peak cutoff and equal layer average, no WW import.
    Optional ww_pl_tail: reuse WW PL bounds and average matrix/slice fits.
    The model factory is called only for missing measurements.
    """
    if lambda_method == 'paper_e_tpl':
        return load_or_fit_paper_lambda(summary_path, model_factory)
    if lambda_method != 'ww_pl_tail':
        raise ValueError('lambda_method must be paper_e_tpl or ww_pl_tail')
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if all(summary.get(key) is not None for key in ('alpha', 'lambda')):
        return summary

    import weightwatcher as ww

    import powerlaw
    import weightwatcher.weightwatcher as ww_core
    from types import SimpleNamespace

    records = []
    original_powerlaw = getattr(ww_core, 'powerlaw', powerlaw)

    # Intercept only WW's reference, preserving its exact spectra, scaling,
    # slice selection, SVD and PL bounds. Restore it even if analysis fails.
    def record_fit(data, fit):
        alpha = getattr(fit, 'alpha', None)
        if alpha is None:
            alpha = fit.power_law.alpha
        record = {'fit_index': len(records), 'alpha': float(alpha),
                  'xmin': float(fit.xmin), 'xmax': fit.xmax,
                  'num_evals': len(data)}
        try:
            # Modern WW may return its own PL-only fit object. Refit the
            # same tail with powerlaw without changing WW's PL result.
            if hasattr(fit, 'truncated_power_law'):
                tpl = fit.truncated_power_law
            else:
                tpl = powerlaw.Fit(data, xmin=fit.xmin, xmax=fit.xmax,
                                   discrete=False, verbose=False).truncated_power_law
            value = float(tpl.Lambda)
            if not np.isfinite(value) or value < 0 or getattr(tpl, 'noise_flag', False):
                raise ValueError('Invalid or unconverged truncated power-law fit')
            record.update(Lambda=value, alpha_tpl=float(tpl.alpha), status='success')
        except Exception as exc:
            record.update(Lambda=None, status='failed', error=str(exc))
        records.append(record)
        return fit

    def capture_fit(data, *args, **kwargs):
        return record_fit(data, original_powerlaw.Fit(data, *args, **kwargs))

    class PostHocWatcher(ww.WeightWatcher):
        def analyze_weights(self, weights, layerid, *args, **kwargs):
            first = len(records)
            result = super().analyze_weights(weights, layerid, *args, **kwargs)
            for record in records[first:]:
                record['layer_id'] = layerid
            return result

    watcher = PostHocWatcher(model=model_factory())
    if 'alphas' in inspect.signature(watcher.analyze).parameters:
        ww_core.powerlaw = SimpleNamespace(Fit=capture_fit)
        try:
            watcher.analyze(alphas=True, spectralnorms=True, plot=False,
                            normalize=False, glorot_fix=False)
        finally:
            ww_core.powerlaw = original_powerlaw
    else:
        original_pl_fit = ww_core.pl_fit

        def capture_pl_fit(*args, **kwargs):
            data = kwargs.get('data', args[0] if args else None)
            return record_fit(data, original_pl_fit(*args, **kwargs))

        ww_core.pl_fit = capture_pl_fit
        try:
            watcher.analyze(fit='PL', plot=False)
        finally:
            ww_core.pl_fit = original_pl_fit

    details_dir = summary_path.parent.parent / 'details'
    details_dir.mkdir(parents=True, exist_ok=True)
    stem = summary_path.stem.removesuffix('_summary')
    tpl_details = pd.DataFrame(records)
    tpl_details.to_csv(details_dir / f'{stem}_tpl_details.csv', index=False)
    values = [r['Lambda'] for r in records if r['status'] == 'success']
    if not values:
        raise RuntimeError(f'No successful post hoc Lambda fits for {stem}; see saved details.')
    if summary.get('alpha') is None:
        summary['alpha'] = float(watcher.get_summary()['alpha'])
    summary['lambda'] = float(np.mean(values))
    summary['lambda_fit'] = 'powerlaw.truncated_power_law'
    summary['lambda_bounds'] = 'original WW PL xmin and xmax'
    summary['lambda_aggregation'] = 'mean of successful matrix/slice fits'
    summary['lambda_num_fits'] = len(values)
    summary['lambda_num_failed'] = len(records) - len(values)
    summary['lambda_ww_version'] = str(getattr(ww, '__version__', 'unknown'))
    summary['lambda_powerlaw_version'] = str(getattr(powerlaw, '__version__', 'unknown'))
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    # Rebuild the existing combined CSV so cached models keep their new fields.
    root = summary_path.parent.parent
    csv_name = 'summary_metrics_repo_v1.csv' if stem.endswith('_v1') else 'summary_metrics.csv'
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(summary_path.parent.glob('*_summary.json'))]
    pd.DataFrame(rows).to_csv(root / csv_name, index=False)
    return summary


def fit_paper_e_tpl(eigenvalues):
    """Reproduce the E_TPL fitting call in WW 0.5.6 used by Yang et al.

    Source: https://github.com/nsfzyzz/Generalization_metrics_for_NLP
    E_TPL maps to TPL + XMIN_PEAK in weightwatcher==0.5.6. Its 100-bin
    log10 histogram supplies a +/-5% xmin search range, xmax is max(evals),
    and values <=1e-5 are excluded. The reference powerlaw==1.5 call passed
    distribution='truncated_power_law' but left xmin_distribution at its
    default 'power_law'; we explicitly retain that behavior across versions.
    """
    import powerlaw

    e = np.asarray(eigenvalues, dtype=float).reshape(-1)
    e = e[np.isfinite(e) & (e > 1e-5)]
    if len(e) < 3:
        raise ValueError('Fewer than three eigenvalues above 1e-5')
    counts, edges = np.histogram(np.log10(e), bins=100)
    peak = float(10 ** edges[np.argmax(counts)])
    # Keep suppression local to fitting; errors and saved quality flags remain.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(e, xmin=(0.95 * peak, 1.05 * peak),
                           xmax=float(e.max()), discrete=False, verbose=False,
                           xmin_distribution='power_law')
        tpl = fit.truncated_power_law
        value = float(tpl.Lambda)
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid E-TPL lambda')
    return {'Lambda': value, 'beta': float(tpl.alpha),
            'xmin': float(fit.xmin), 'xmax': float(fit.xmax),
            'xmin_peak': peak, 'num_evals': len(e), 'D': float(tpl.D),
            'noise_flag': bool(getattr(tpl, 'noise_flag', False))}


def load_or_fit_paper_lambda(summary_path, weights_factory):
    """Fit missing paper E-TPL lambda without importing WeightWatcher.

    weights_factory returns a CPU PyTorch model or state dict. For CV models,
    use full SVD of dense matrices and each Conv2D channel slice; pool slices
    within a layer, scaling conv eigenvalues by kernel area / 2, as in WW
    0.5.6. Average the layer lambdas equally. Existing old-WW alpha is retained.
    """
    import powerlaw
    import torch

    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding='utf-8'))
    if summary.get('lambda_method') == 'paper_e_tpl_v1' and summary.get('lambda') is not None:
        return summary
    weights = weights_factory()
    state = weights.state_dict() if hasattr(weights, 'state_dict') else weights
    records = []
    for name, tensor in state.items():
        # BatchNorm and biases are excluded, as in the reference WW analysis.
        if not name.endswith('weight') or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim not in (2, 4):
            continue
        W = tensor.detach().cpu().numpy().astype(float)
        record = {'layer': name, 'shape': str(tuple(W.shape))}
        try:
            if W.ndim == 2:
                evals = np.linalg.svd(W, compute_uv=False) ** 2
            else:
                rf = W.shape[2] * W.shape[3]
                evals = np.concatenate([
                    np.linalg.svd(W[:, :, i, j], compute_uv=False) ** 2 * (rf / 2.0)
                    for i in range(W.shape[2]) for j in range(W.shape[3])])
            record.update(fit_paper_e_tpl(evals), status='success')
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            record.update(Lambda=None, status='failed', error=str(exc))
        records.append(record)
    stem = path.stem.removesuffix('_summary')
    details_dir = path.parent.parent / 'details'
    details_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(details_dir / f'{stem}_e_tpl_details.csv', index=False)
    values = [r['Lambda'] for r in records if r['status'] == 'success']
    if not values:
        raise RuntimeError(f'No valid E-TPL fits for {stem}; see {details_dir}.')
    if summary.get('lambda') is not None and summary.get('lambda_method') != 'paper_e_tpl_v1':
        summary['lambda_pl_tail'] = summary['lambda']
    summary.update({
        'lambda': float(np.mean(values)), 'lambda_e_tpl': float(np.mean(values)),
        'lambda_method': 'paper_e_tpl_v1',
        'lambda_fit': 'powerlaw.truncated_power_law (WW 0.5.6 E_TPL procedure)',
        'lambda_bounds': '100-bin log10 peak +/-5% xmin range; xmax=max(evals)',
        'lambda_aggregation': 'equal mean of valid layer fits; Conv2D slices pooled per layer',
        'lambda_num_fits': len(values), 'lambda_num_failed': len(records) - len(values),
        'lambda_num_flagged': sum(bool(r.get('noise_flag')) for r in records),
        'lambda_powerlaw_version': str(getattr(powerlaw, '__version__', 'unknown')),
        'lambda_reference_ww_version': '0.5.6',
        'lambda_reference_powerlaw_version': '1.5',
        'lambda_backend': 'direct powerlaw; no WeightWatcher import',
        'lambda_reference': 'https://github.com/nsfzyzz/Generalization_metrics_for_NLP',
    })
    path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    csv_name = 'summary_metrics_repo_v1.csv' if stem.endswith('_v1') else 'summary_metrics.csv'
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(path.parent.glob('*_summary.json'))]
    pd.DataFrame(rows).to_csv(path.parent.parent / csv_name, index=False)
    return summary
