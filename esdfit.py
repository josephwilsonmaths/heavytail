import os
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
import mpmath
import numpy as np
import powerlaw
import tqdm
from helpers import *
from matplotlib.ticker import FormatStrFormatter
import texplot

class ESDFit(object):
    def __init__(self, model='minialexnet', dataset='cifar10', kernel='weight', batch_size=None, subsample=False, file_dir = 'results/htmp/epoch', weight_type=None, layer=None):
        self.model = model
        self.dataset = dataset
        self.kernel = kernel
        self.batch_size = batch_size
        self.subsample = subsample
        self.file_dir = file_dir
        self.weight_type = weight_type
        self.layer = layer

        self.fileloader = fileloading(model, dataset, kernel, batch_size, subsample, file_dir, weight_type=weight_type, layer=layer)

        self.epoch_list = self.fileloader.get_epoch_list()
        self._fitted_epochs_by_kernel = {}
        self._kappa_opt_by_kernel = {}
        self._beta_opt_by_kernel = {}
        self._tau_opt_by_kernel = {}
        self._dist_opt_by_kernel = {}

        self.lam = None
        self.free_energies = {}
        self.free_energy_vars = {}
        self.free_energy_stds = {}
        self.free_energy_ci95 = {}
        self.free_energy_counts = {}
        self.theoretical_free_energies = {}
        self.empirical_traces = {}
        self.empirical_trace_vars = {}
        self.empirical_trace_stds = {}
        self.empirical_trace_ci95 = {}
        self.empirical_log_dets = {}
        self.empirical_log_det_vars = {}
        self.empirical_log_det_stds = {}
        self.empirical_log_det_ci95 = {}
        self.theoretical_traces = {}
        self.theoretical_log_dets = {}
        self.traces = {}
        self.log_dets = {}
        self.train_loss_scaled_free_energies = {}
        self.train_loss_scaled_traces = {}
        self.train_losses = {}

        self.inverse_htmp = (kernel == 'ntk' or kernel == 'ck')

    def _get_active_fit_scope_key(self):
        return self._get_fit_scope_key() if hasattr(self, '_get_fit_scope_key') else self.kernel

    def _get_kernel_store(self, store_by_kernel, default_factory):
        scope_key = self._get_active_fit_scope_key()
        return store_by_kernel.setdefault(scope_key, default_factory())

    def _set_kernel_store(self, store_by_kernel, value, coerce_fn):
        scope_key = self._get_active_fit_scope_key()
        store_by_kernel[scope_key] = coerce_fn(value)

    def _coerce_kernel_dict(self, value):
        return dict(value) if value is not None else {}

    def _coerce_kernel_list(self, value):
        return list(value) if value is not None else []

    def _normalize_saved_kernel_store(self, saved_fit, scoped_key, flat_key, coerce_fn):
        normalized_store = {}

        scoped_values = saved_fit.get(scoped_key, {}) or {}
        for kernel, value in scoped_values.items():
            normalized_store[kernel] = coerce_fn(value)

        if flat_key in saved_fit:
            flat_kernel = saved_fit.get('kernel', self.kernel)
            normalized_store[flat_kernel] = coerce_fn(saved_fit.get(flat_key))

        return normalized_store

    @property
    def fitted_epochs_by_kernel(self):
        return self._fitted_epochs_by_kernel

    @property
    def kappa_opt_by_kernel(self):
        return self._kappa_opt_by_kernel

    @property
    def beta_opt_by_kernel(self):
        return self._beta_opt_by_kernel

    @property
    def tau_opt_by_kernel(self):
        return self._tau_opt_by_kernel

    @property
    def dist_opt_by_kernel(self):
        return self._dist_opt_by_kernel

    @property
    def fitted_epochs(self):
        return self._get_kernel_store(self._fitted_epochs_by_kernel, list)

    @fitted_epochs.setter
    def fitted_epochs(self, value):
        self._set_kernel_store(self._fitted_epochs_by_kernel, value, self._coerce_kernel_list)

    @property
    def kappa_opt(self):
        return self._get_kernel_store(self._kappa_opt_by_kernel, dict)

    @kappa_opt.setter
    def kappa_opt(self, value):
        self._set_kernel_store(self._kappa_opt_by_kernel, value, self._coerce_kernel_dict)

    @property
    def beta_opt(self):
        return self._get_kernel_store(self._beta_opt_by_kernel, dict)

    @beta_opt.setter
    def beta_opt(self, value):
        self._set_kernel_store(self._beta_opt_by_kernel, value, self._coerce_kernel_dict)

    @property
    def tau_opt(self):
        return self._get_kernel_store(self._tau_opt_by_kernel, dict)

    @tau_opt.setter
    def tau_opt(self, value):
        self._set_kernel_store(self._tau_opt_by_kernel, value, self._coerce_kernel_dict)

    @property
    def dist_opt(self):
        return self._get_kernel_store(self._dist_opt_by_kernel, dict)

    @dist_opt.setter
    def dist_opt(self, value):
        self._set_kernel_store(self._dist_opt_by_kernel, value, self._coerce_kernel_dict)

    def _skip_epoch_fit(self, epoch):
        return epoch == 0 and not self.inverse_htmp

    def _resolve_fit_epochs(self, epochs_list):
        if epochs_list is not None:
            self.fitted_epochs = epochs_list
        else:
            self.fitted_epochs = self.epoch_list
        return self.fitted_epochs

    def _validate_fit_config(self, beta_min, beta_max, num_betas, tau_min, tau_max, num_taus):
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive.")
        if num_betas is None or num_betas < 1:
            raise ValueError("num_betas must be at least 1.")
        if tau_min < 0 or tau_max < 0:
            raise ValueError("tau_min and tau_max must be non-negative.")
        if tau_max < tau_min:
            raise ValueError("tau_max must be greater than or equal to tau_min.")
        if num_taus is None or num_taus < 1:
            raise ValueError("num_taus must be at least 1.")

    def _get_beta_grid(self, beta_min, beta_max, num_betas, beta_log_scale=True):
        if beta_log_scale:
            return np.logspace(np.log10(beta_min), np.log10(beta_max), num_betas)
        return np.linspace(beta_min, beta_max, num_betas)

    def _get_tau_grid(self, tau_min, tau_max, num_taus):
        if tau_max == tau_min:
            return [tau_min]
        return np.logspace(np.log10(tau_min), np.log10(tau_max), num_taus)

    def _shift_eigs(self, eigs, tau=0.0):
        if torch.is_tensor(eigs):
            eigs = eigs.detach().cpu().numpy()
        else:
            eigs = np.asarray(eigs)
        return eigs + tau

    def _get_epoch_tau_from_eigs(self, eigs):
        if torch.is_tensor(eigs):
            eigs = eigs.detach().cpu().numpy()
        else:
            eigs = np.asarray(eigs)
        finite_mask = np.isfinite(eigs)
        if not np.any(finite_mask):
            return 0.0
        return float(np.min(eigs[finite_mask]))

    def _shift_eigs_to_zero(self, eigs, tau=None):
        if torch.is_tensor(eigs):
            eigs = eigs.detach().cpu().numpy()
        else:
            eigs = np.asarray(eigs)
        tau_value = self._get_epoch_tau_from_eigs(eigs) if tau is None else float(tau)
        return eigs - tau_value, tau_value

    def _beta_from_tau(self, tau, c=1.0):
        tau_value = self._as_float(tau)
        c_value = self._as_float(c)
        if tau_value <= 0 or not np.isfinite(tau_value):
            raise ValueError(
                f"beta_from_tau requires a positive finite tau, got tau={tau_value}."
            )
        if c_value <= 0 or not np.isfinite(c_value):
            raise ValueError(
                f"beta_from_tau requires a positive finite c, got c={c_value}."
            )
        return c_value / tau_value

    def _load_epoch_eigs(self, epochs, verbose=False):
        epochs_to_fit = [epoch for epoch in epochs if not self._skip_epoch_fit(epoch)]
        if verbose:
            print("Loading eigenvalues...")
        return {epoch: self.fileloader.get_eigs_epoch(epoch) for epoch in epochs_to_fit}

    def _get_gamma(self, epoch=None):
        if self.fileloader._uses_saved_aspect_ratio():
            if epoch is None:
                raise ValueError("epoch must be provided when resolving gamma for a checkpoint-summary model.")
            return self.fileloader.get_aspect_ratio_epoch(epoch)
        return gamma_dict[self.model][self.kernel]

    def _get_gamma_by_epoch(self, epochs):
        return {epoch: self._get_gamma(epoch) for epoch in epochs}

    def get_aspect_ratio_epoch(self, epoch):
        return self.fileloader.get_aspect_ratio_epoch(epoch)

    def _fit_epoch_parameters(self,
                              eigs,
                              gamma,
                              search_method,
                              kappa_min,
                              kappa_max,
                              num_kappas,
                              n_eval,
                              beta_min,
                              beta_max,
                              num_betas,
                              tau_values,
                              fixed_beta=None,
                              fixed_tau=None,
                              beta_from_tau=False,
                              c=1.0,
                              stieltjes=False,
                              vectorized_grid=True,
                              kappa_log_scale=True,
                              beta_log_scale=True,
                              moment_penalty_weight=0.0,
                              inverse=None,
                              verbose=False):
        inverse = self.inverse_htmp if inverse is None else inverse
        shifted_eigs, tau_epoch = self._shift_eigs_to_zero(eigs, tau=fixed_tau)

        if beta_from_tau:
            fixed_beta = self._beta_from_tau(tau_epoch, c=c)

        search_beta_min = fixed_beta if fixed_beta is not None else beta_min
        search_beta_max = fixed_beta if fixed_beta is not None else beta_max
        search_num_betas = 1 if fixed_beta is not None else num_betas

        kappa_opt, beta_opt, dist_opt = find_best_htmp(
            shifted_eigs,
            gamma,
            search_method=search_method,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            num_kappas=num_kappas,
            n_eval=n_eval,
            beta_min=search_beta_min,
            beta_max=search_beta_max,
            num_betas=search_num_betas,
            inverse=inverse,
            stieltjes=stieltjes,
            vectorized_grid=vectorized_grid,
            kappa_log_scale=kappa_log_scale,
            beta_log_scale=beta_log_scale,
            moment_penalty_weight=moment_penalty_weight,
            verbose=verbose
        )

        return {
            'kappa': kappa_opt,
            'beta': beta_opt,
            'tau': tau_epoch,
            'dist': dist_opt,
        }

    def _fit_shared_parameters(self,
                               eigs_by_epoch,
                               gamma_by_epoch,
                               search_method,
                               kappa_min,
                               kappa_max,
                               num_kappas,
                               n_eval,
                               beta_min,
                               beta_max,
                               num_betas,
                               tau_values,
                               shared_beta_values,
                               shared_tau_values,
                               shared_lp_ord,
                               beta_from_tau=False,
                               c=1.0,
                               stieltjes=False,
                               vectorized_grid=True,
                               kappa_log_scale=True,
                               beta_log_scale=True,
                               moment_penalty_weight=0.0,
                               inverse=None,
                               verbose=False,
                               extra_verbose=False):
        best_shared_result = None

        for shared_beta in shared_beta_values:
            for shared_tau in shared_tau_values:
                epoch_results = {}
                epoch_distances = []

                for epoch, eigs in eigs_by_epoch.items():
                    epoch_result = self._fit_epoch_parameters(
                        eigs,
                        gamma_by_epoch[epoch],
                        search_method,
                        kappa_min,
                        kappa_max,
                        num_kappas,
                        n_eval,
                        beta_min,
                        beta_max,
                        num_betas,
                        tau_values,
                        fixed_beta=shared_beta,
                        fixed_tau=shared_tau,
                        beta_from_tau=beta_from_tau,
                        c=c,
                        stieltjes=stieltjes,
                        vectorized_grid=vectorized_grid,
                        kappa_log_scale=kappa_log_scale,
                        beta_log_scale=beta_log_scale,
                        moment_penalty_weight=moment_penalty_weight,
                        inverse=inverse,
                        verbose=extra_verbose
                    )
                    epoch_results[epoch] = epoch_result
                    epoch_distances.append(epoch_result['dist'])

                aggregate_dist = np.linalg.norm(np.asarray(epoch_distances), ord=shared_lp_ord)
                if verbose:
                    print(
                        f"Shared candidate: β={shared_beta if shared_beta is not None else 'per-epoch'}, "
                        f"τ={shared_tau if shared_tau is not None else 'per-epoch'}, "
                        f"||d||_{shared_lp_ord}={aggregate_dist:.6g}"
                    )

                shared_result = {
                    'shared_beta': shared_beta,
                    'shared_tau': shared_tau,
                    'aggregate_dist': aggregate_dist,
                    'epoch_results': epoch_results,
                }
                if best_shared_result is None or aggregate_dist < best_shared_result['aggregate_dist']:
                    best_shared_result = shared_result

        if best_shared_result is None:
            raise RuntimeError("Failed to find valid shared HTMP parameters.")
        return best_shared_result

    def _store_skipped_epoch(self, epoch, shared_beta=None, shared_tau=None, beta_from_tau=False, c=1.0, verbose=False):
        eigs = self.fileloader.get_eigs_epoch(epoch)
        tau_epoch = self._get_epoch_tau_from_eigs(eigs) if shared_tau is None else shared_tau
        beta_epoch = self._beta_from_tau(tau_epoch, c=c) if beta_from_tau else shared_beta
        self.kappa_opt[epoch] = float('inf')
        self.beta_opt[epoch] = beta_epoch
        self.tau_opt[epoch] = tau_epoch
        self.dist_opt[epoch] = None
        if verbose:
            print(
                f"Fitting HTMP for Epoch: {epoch}, κ*: inf, β*: {beta_epoch}, "
                f"τ*: {tau_epoch}, ||.||ₚ*: None"
            )

    def _store_epoch_result(self, epoch, epoch_result, verbose=False):
        self.kappa_opt[epoch] = epoch_result['kappa']
        self.beta_opt[epoch] = epoch_result['beta']
        self.tau_opt[epoch] = epoch_result['tau']
        self.dist_opt[epoch] = epoch_result['dist']
        if verbose:
            print(
                f"Fitting HTMP for Epoch: {epoch}, κ*: {epoch_result['kappa']:.3f}, "
                f"β*: {epoch_result['beta']:.3f}, τ*: {epoch_result['tau']:.3f}, "
                f"||.||ₚ*: {epoch_result['dist']:.3f}"
            )

    def _get_fit_scope_key(self):
        parts = [str(self.kernel)]
        if self.weight_type is not None:
            parts.append(f"weight_type={self.weight_type}")
        if self.layer is not None:
            parts.append(f"layer={self.layer}")
        return '|'.join(parts)

    def _append_fit_scope_to_filename(self, filename):
        if self.weight_type is not None:
            filename += f'_{self.weight_type}'
        if self.layer is not None:
            filename += f'_layer{self.layer}'
        return filename

    def _get_figure_file_stem(self, file_dir='figures/htmp_fits'):
        filename = f'{self.model}_{self.dataset}'
        if self.subsample:
            filename += '_subsampled'
        filename += f'_{self.kernel}'
        filename += f'_b{self.batch_size}'
        filename = self._append_fit_scope_to_filename(filename)
        return os.path.join(file_dir, filename)

    def _get_legacy_fit_params_path(self, file_dir='results/htmp/fits'):
        filename = f'{self.model}_{self.dataset}'
        if self.subsample:
            filename += '_subsampled'
        filename += f'_{self.kernel}'
        filename += f'_b{self.batch_size}'
        return os.path.join(file_dir, f'{filename}_fit.pt')

    def _get_fit_params_path(self, file_dir='results/htmp/fits'):
        filename = f'{self.model}_{self.dataset}'
        if self.subsample:
            filename += '_subsampled'
        filename += f'_{self.kernel}'
        filename += f'_b{self.batch_size}'
        filename = self._append_fit_scope_to_filename(filename)
        return os.path.join(file_dir, f'{filename}_fit.pt')

    def save_htmp_fit(self, file_path=None, file_dir='results/htmp/fits', verbose=False):
        save_path = file_path or self._get_fit_params_path(file_dir)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        saved_fit = {}
        if os.path.exists(save_path):
            saved_fit = torch.load(save_path, weights_only=False, map_location='cpu')

        fitted_epochs_by_kernel = self._normalize_saved_kernel_store(
            saved_fit,
            'fitted_epochs_by_kernel',
            'fitted_epochs',
            self._coerce_kernel_list
        )
        kappa_opt_by_kernel = self._normalize_saved_kernel_store(
            saved_fit,
            'kappa_opt_by_kernel',
            'kappa_opt',
            self._coerce_kernel_dict
        )
        beta_opt_by_kernel = self._normalize_saved_kernel_store(
            saved_fit,
            'beta_opt_by_kernel',
            'beta_opt',
            self._coerce_kernel_dict
        )
        tau_opt_by_kernel = self._normalize_saved_kernel_store(
            saved_fit,
            'tau_opt_by_kernel',
            'tau_opt',
            self._coerce_kernel_dict
        )
        dist_opt_by_kernel = self._normalize_saved_kernel_store(
            saved_fit,
            'dist_opt_by_kernel',
            'dist_opt',
            self._coerce_kernel_dict
        )

        scope_key = self._get_fit_scope_key()
        fitted_epochs_by_kernel[scope_key] = list(self.fitted_epochs)
        kappa_opt_by_kernel[scope_key] = dict(self.kappa_opt)
        beta_opt_by_kernel[scope_key] = dict(self.beta_opt)
        tau_opt_by_kernel[scope_key] = dict(self.tau_opt)
        dist_opt_by_kernel[scope_key] = dict(self.dist_opt)

        torch.save(
            {
                'model': self.model,
                'dataset': self.dataset,
                'kernel': self.kernel,
                'fit_scope_key': scope_key,
                'weight_type': self.weight_type,
                'layer': self.layer,
                'batch_size': self.batch_size,
                'subsample': self.subsample,
                'source_file_dir': self.file_dir,
                'fitted_epochs': list(self.fitted_epochs),
                'kappa_opt': dict(self.kappa_opt),
                'beta_opt': dict(self.beta_opt),
                'tau_opt': dict(self.tau_opt),
                'dist_opt': dict(self.dist_opt),
                'fitted_epochs_by_kernel': fitted_epochs_by_kernel,
                'kappa_opt_by_kernel': kappa_opt_by_kernel,
                'beta_opt_by_kernel': beta_opt_by_kernel,
                'tau_opt_by_kernel': tau_opt_by_kernel,
                'dist_opt_by_kernel': dist_opt_by_kernel,
            },
            save_path
        )

        if verbose:
            print(f"Saved HTMP fit parameters to {save_path}")
        return save_path

    def load_htmp_fit(self, file_path=None, file_dir='results/htmp/fits', verbose=False):
        load_path = file_path or self._get_fit_params_path(file_dir)
        legacy_path = None if file_path is not None else self._get_legacy_fit_params_path(file_dir)
        if not os.path.exists(load_path):
            if legacy_path is not None and os.path.exists(legacy_path):
                load_path = legacy_path
            else:
                if verbose:
                    print(f"No saved HTMP fit parameters found at {load_path}")
                return False

        saved_fit = torch.load(load_path, weights_only=False, map_location='cpu')

        current_scope_key = self._get_fit_scope_key()
        legacy_kernel_key = saved_fit.get('kernel', self.kernel)

        def _normalize_scoped_store(scoped_key, flat_key, coerce_fn):
            normalized_store = {}

            scoped_values = saved_fit.get(scoped_key, {}) or {}
            for key, value in scoped_values.items():
                normalized_store[key] = coerce_fn(value)

            if flat_key in saved_fit:
                flat_scope_key = saved_fit.get('fit_scope_key') or legacy_kernel_key
                normalized_store[flat_scope_key] = coerce_fn(saved_fit.get(flat_key))

            return normalized_store

        self._fitted_epochs_by_kernel = _normalize_scoped_store(
            'fitted_epochs_by_kernel',
            'fitted_epochs',
            self._coerce_kernel_list
        )
        self._kappa_opt_by_kernel = _normalize_scoped_store(
            'kappa_opt_by_kernel',
            'kappa_opt',
            self._coerce_kernel_dict
        )
        self._beta_opt_by_kernel = _normalize_scoped_store(
            'beta_opt_by_kernel',
            'beta_opt',
            self._coerce_kernel_dict
        )

        tau_opt_by_kernel = _normalize_scoped_store(
            'tau_opt_by_kernel',
            'tau_opt',
            self._coerce_kernel_dict
        )
        self._tau_opt_by_kernel = {}
        for key, kappa_opt in self._kappa_opt_by_kernel.items():
            kernel_tau_opt = tau_opt_by_kernel.get(key, {})
            self._tau_opt_by_kernel[key] = {
                epoch: kernel_tau_opt.get(epoch, 0.0) for epoch in kappa_opt
            }

        dist_opt_by_kernel = _normalize_scoped_store(
            'dist_opt_by_kernel',
            'dist_opt',
            self._coerce_kernel_dict
        )
        self._dist_opt_by_kernel = {}
        for key, kappa_opt in self._kappa_opt_by_kernel.items():
            kernel_dist_opt = dist_opt_by_kernel.get(key, {})
            self._dist_opt_by_kernel[key] = {
                epoch: kernel_dist_opt.get(epoch, None) for epoch in kappa_opt
            }

        has_current_kernel_fit = current_scope_key in self._kappa_opt_by_kernel
        if not has_current_kernel_fit and current_scope_key != legacy_kernel_key and legacy_kernel_key in self._kappa_opt_by_kernel:
            self._fitted_epochs_by_kernel[current_scope_key] = list(self._fitted_epochs_by_kernel[legacy_kernel_key])
            self._kappa_opt_by_kernel[current_scope_key] = dict(self._kappa_opt_by_kernel[legacy_kernel_key])
            self._beta_opt_by_kernel[current_scope_key] = dict(self._beta_opt_by_kernel[legacy_kernel_key])
            self._tau_opt_by_kernel[current_scope_key] = dict(self._tau_opt_by_kernel.get(legacy_kernel_key, {}))
            self._dist_opt_by_kernel[current_scope_key] = dict(self._dist_opt_by_kernel.get(legacy_kernel_key, {}))
            has_current_kernel_fit = True

            if verbose:
                print(
                    f"Loaded legacy HTMP fit parameters from {load_path} into scope {current_scope_key}"
                )
        elif verbose:
            if has_current_kernel_fit:
                print(f"Loaded HTMP fit parameters from {load_path} for scope {current_scope_key}")
            else:
                print(f"Loaded HTMP fit parameters from {load_path}, but no saved parameters were found for scope {current_scope_key}")
        return has_current_kernel_fit

    def fit_htmp(self, 
                 search_method='grid',
                 kappa_min=1e-2, 
                 kappa_max=1e1, 
                 num_kappas=200, 
                 n_eval=50, 
                 beta_min = 1, 
                 beta_max = 100, 
                 num_betas = 1, 
                 tau_min = 0.0,
                 tau_max = 0.0,
                 num_taus = 1,
                 verbose=False,
                 extra_verbose=False,
                 stieltjes=False,
                 vectorized_grid=True,
                 kappa_log_scale=True,
                 moment_penalty_weight=0.0,
                 shared_beta_grid=False,
                 shared_tau_grid=False,
                 beta_lp_ord=2,
                 shared_lp_ord=None,
                 epochs_list = None,
                 save_fit=False,
                 save_file_path=None,
                 save_file_dir='results/htmp/fits',
                 beta_from_tau=False,
                 c=1.0,
                 beta_log_scale=True):
        if beta_from_tau and shared_beta_grid:
            raise ValueError("beta_from_tau cannot be combined with shared_beta_grid.")

        fitted_epochs = self._resolve_fit_epochs(epochs_list)
        self._validate_fit_config(beta_min, beta_max, num_betas, tau_min, tau_max, num_taus)
        shared_lp_ord = beta_lp_ord if shared_lp_ord is None else shared_lp_ord

        eigs_by_epoch = self._load_epoch_eigs(fitted_epochs, verbose=verbose)
        gamma_by_epoch = self._get_gamma_by_epoch(eigs_by_epoch.keys())
        if verbose:
            print(f"Fitting HTMP parameters for {len(eigs_by_epoch)} epochs using {search_method} search...")
            if shared_tau_grid or tau_min != 0.0 or tau_max != 0.0 or num_taus != 1:
                print("Tau grid arguments are ignored: tau is set per epoch to the minimum eigenvalue.")
            if beta_from_tau:
                print(f"Beta grid arguments are ignored: beta is set per epoch to {c} / tau.")

        if shared_beta_grid or shared_tau_grid:
            shared_beta_values = self._get_beta_grid(beta_min, beta_max, num_betas, beta_log_scale=beta_log_scale) if shared_beta_grid else [None]
            shared_tau_values = [None]
            best_shared_result = self._fit_shared_parameters(
                eigs_by_epoch,
                gamma_by_epoch,
                search_method,
                kappa_min,
                kappa_max,
                num_kappas,
                n_eval,
                beta_min,
                beta_max,
                num_betas,
                [None],
                shared_beta_values,
                shared_tau_values,
                shared_lp_ord,
                beta_from_tau=beta_from_tau,
                c=c,
                stieltjes=stieltjes,
                vectorized_grid=vectorized_grid,
                kappa_log_scale=kappa_log_scale,
                beta_log_scale=beta_log_scale,
                moment_penalty_weight=moment_penalty_weight,
                inverse=self.inverse_htmp,
                verbose=verbose,
                extra_verbose=extra_verbose
            )

            for epoch in fitted_epochs:
                if self._skip_epoch_fit(epoch):
                    self._store_skipped_epoch(
                        epoch,
                        shared_beta=best_shared_result['shared_beta'],
                        shared_tau=best_shared_result['shared_tau'],
                        beta_from_tau=beta_from_tau,
                        c=c,
                        verbose=verbose
                    )
                    continue
                self._store_epoch_result(epoch, best_shared_result['epoch_results'][epoch], verbose=verbose)

            if verbose:
                print(
                    f"Shared optimum: β*: {best_shared_result['shared_beta']}, "
                    f"τ*: {best_shared_result['shared_tau']}, "
                    f"aggregated ||d||_{shared_lp_ord}: {best_shared_result['aggregate_dist']:.6g}"
                )
            if save_fit:
                self.save_htmp_fit(file_path=save_file_path, file_dir=save_file_dir, verbose=verbose)
            return

        for epoch in fitted_epochs:
            if self._skip_epoch_fit(epoch):
                self._store_skipped_epoch(epoch, beta_from_tau=beta_from_tau, c=c, verbose=verbose)
                continue

            epoch_result = self.fit_htmp_epoch(
                search_method,
                epoch,
                kappa_min,
                kappa_max,
                num_kappas,
                n_eval,
                beta_min,
                beta_max,
                num_betas,
                tau_min,
                tau_max,
                num_taus,
                stieltjes,
                vectorized_grid,
                kappa_log_scale,
                moment_penalty_weight,
                self.inverse_htmp,
                extra_verbose,
                beta_from_tau=beta_from_tau,
                c=c,
                beta_log_scale=beta_log_scale
            )
            self._store_epoch_result(
                epoch,
                {
                    'kappa': epoch_result[0],
                    'beta': epoch_result[1],
                    'tau': epoch_result[2],
                    'dist': epoch_result[3],
                },
                verbose=verbose
            )
        if save_fit:
            self.save_htmp_fit(file_path=save_file_path, file_dir=save_file_dir, verbose=verbose)
        return
    
    def fit_htmp_epoch(self, 
                       search_method,
                       epoch, 
                       kappa_min=1e-2, 
                       kappa_max=1e1, 
                       num_kappas=200, 
                       n_eval=50, 
                       beta_min = 1, 
                       beta_max = 100, 
                       num_betas = 10, 
                       tau_min = 0.0,
                       tau_max = 0.0,
                       num_taus = 1,
                       stieltjes=False,
                       vectorized_grid=True,
                       kappa_log_scale=True,
                       moment_penalty_weight=0.0,
                       inverse=False, 
                       verbose=False,
                       beta_from_tau=False,
                       c=1.0,
                       beta_log_scale=True):
        eigs = self.fileloader.get_eigs_epoch(epoch)
        gamma = self._get_gamma(epoch)
        epoch_result = self._fit_epoch_parameters(
            eigs,
            gamma,
            search_method,
            kappa_min,
            kappa_max,
            num_kappas,
            n_eval,
            beta_min,
            beta_max,
            num_betas,
            [None],
            fixed_beta=None,
            fixed_tau=None,
            beta_from_tau=beta_from_tau,
            c=c,
            stieltjes=stieltjes,
            vectorized_grid=vectorized_grid,
            kappa_log_scale=kappa_log_scale,
            beta_log_scale=beta_log_scale,
            moment_penalty_weight=moment_penalty_weight,
            inverse=inverse,
            verbose=verbose
        )
        return epoch_result['kappa'], epoch_result['beta'], epoch_result['tau'], epoch_result['dist']

    def _store_missing_theoretical_free_energy_terms(self, epoch):
        self.theoretical_free_energies[epoch] = np.nan
        self.theoretical_traces[epoch] = np.nan
        self.theoretical_log_dets[epoch] = np.nan

    

    def _summarize_repeat_values(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {'mean': np.nan, 'var': np.nan, 'std': np.nan, 'ci95': np.nan, 'count': 0}

        mean = float(np.mean(values))
        var = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        std = float(np.sqrt(var))
        ci95 = float(1.96 * std / np.sqrt(values.size)) if values.size > 1 else 0.0
        return {'mean': mean, 'var': var, 'std': std, 'ci95': ci95, 'count': int(values.size)}

    def compute_free_energy(self, lam=0.5, tau=0, weight_inverse=False):
        self.free_energies = {}
        self.free_energy_vars = {}
        self.free_energy_stds = {}
        self.free_energy_ci95 = {}
        self.free_energy_counts = {}
        self.theoretical_free_energies = {}
        self.empirical_traces = {}
        self.empirical_trace_vars = {}
        self.empirical_trace_stds = {}
        self.empirical_trace_ci95 = {}
        self.empirical_log_dets = {}
        self.empirical_log_det_vars = {}
        self.empirical_log_det_stds = {}
        self.empirical_log_det_ci95 = {}
        self.theoretical_traces = {}
        self.theoretical_log_dets = {}
        self.traces = self.empirical_traces
        self.log_dets = self.empirical_log_dets
        self.weight_inverse = weight_inverse

        for epoch in self.epoch_list:
            repeat_eigs = self.fileloader.get_eigs_epoch_values(epoch)
            free_energy_values = []
            trace_values = []
            log_det_values = []

            for eigs in repeat_eigs:
                f, t, l = self.empirical_free_energy_eigs(eigs, len(eigs), lam=lam, tau=0, weight_inverse=weight_inverse)
                free_energy_values.append(f)
                trace_values.append(t)
                log_det_values.append(l)

            free_energy_stats = self._summarize_repeat_values(free_energy_values)
            trace_stats = self._summarize_repeat_values(trace_values)
            log_det_stats = self._summarize_repeat_values(log_det_values)

            self.free_energies[epoch] = free_energy_stats['mean']
            self.free_energy_vars[epoch] = free_energy_stats['var']
            self.free_energy_stds[epoch] = free_energy_stats['std']
            self.free_energy_ci95[epoch] = free_energy_stats['ci95']
            self.free_energy_counts[epoch] = free_energy_stats['count']
            self.empirical_traces[epoch] = trace_stats['mean']
            self.empirical_trace_vars[epoch] = trace_stats['var']
            self.empirical_trace_stds[epoch] = trace_stats['std']
            self.empirical_trace_ci95[epoch] = trace_stats['ci95']
            self.empirical_log_dets[epoch] = log_det_stats['mean']
            self.empirical_log_det_vars[epoch] = log_det_stats['var']
            self.empirical_log_det_stds[epoch] = log_det_stats['std']
            self.empirical_log_det_ci95[epoch] = log_det_stats['ci95']

            kappa = self.kappa_opt.get(epoch, None)
            beta = self.beta_opt.get(epoch, None)
            if kappa is None or beta is None:
                self._store_missing_theoretical_free_energy_terms(epoch)
                continue

            kappa_val = self._as_float(kappa)
            beta_val = self._as_float(beta)
            if not np.isfinite(kappa_val) or not np.isfinite(beta_val):
                self._store_missing_theoretical_free_energy_terms(epoch)
                continue

            gamma = self._get_gamma(epoch)
            tau_val = self._as_float(self.tau_opt.get(epoch, tau if tau is not None else 0.0))
            theoretical_free_energy, theoretical_trace, theoretical_log_det = self.theoretical_free_energy(
                kappa_val,
                gamma,
                beta_val,
                lam,
                tau_val,
                weight_inverse=self.weight_inverse
            )
            self.theoretical_free_energies[epoch] = theoretical_free_energy
            self.theoretical_traces[epoch] = theoretical_trace
            self.theoretical_log_dets[epoch] = theoretical_log_det

        self.lam = lam

    def _as_float(self, x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    def plot_trace_and_log_det(self,
                               variable='epochs',
                               start_index=0,
                               plot_theoretical=True,
                               plot_sum=False,
                               fig_width=10,
                               fig_height=6,
                               fontsize=12,
                               save_fig=True,
                               show_fig=True,
                               file_dir='figures/htmp_fits',
                               texplot_enabled=False,
                               legend_on=True,
                               title=True):
        epochs = self.epoch_list[start_index:]
        if any(epoch not in self.traces or epoch not in self.log_dets for epoch in epochs):
            print("Please run .compute_free_energy() before plotting.")
            return

        if variable not in ['epochs', 'kappa', 'beta', 'kappa_beta_ratio']:
            raise ValueError("variable must be one of 'epochs', 'kappa', 'beta', or 'kappa_beta_ratio'.")

        plot_rows = []
        skipped_epochs = []
        for epoch in epochs:
            if variable == 'epochs':
                x_val = epoch
            else:
                kappa = self.kappa_opt.get(epoch, None)
                beta = self.beta_opt.get(epoch, None)

                if variable == 'kappa':
                    if kappa is None:
                        skipped_epochs.append((epoch, 'missing kappa'))
                        continue
                    kappa_val = self._as_float(kappa)
                    if not np.isfinite(kappa_val):
                        skipped_epochs.append((epoch, 'non-finite kappa'))
                        continue
                    x_val = kappa_val
                elif variable == 'beta':
                    if beta is None:
                        skipped_epochs.append((epoch, 'missing beta'))
                        continue
                    beta_val = self._as_float(beta)
                    if not np.isfinite(beta_val):
                        skipped_epochs.append((epoch, 'non-finite beta'))
                        continue
                    x_val = beta_val
                else:
                    if kappa is None or beta is None:
                        skipped_epochs.append((epoch, 'missing kappa/beta'))
                        continue
                    kappa_val = self._as_float(kappa)
                    beta_val = self._as_float(beta)
                    if not np.isfinite(kappa_val):
                        skipped_epochs.append((epoch, 'non-finite kappa'))
                        continue
                    if not np.isfinite(beta_val) or beta_val == 0:
                        skipped_epochs.append((epoch, 'invalid beta'))
                        continue
                    x_val = kappa_val / beta_val

            plot_rows.append(
                (
                    x_val,
                    epoch,
                    self.traces[epoch],
                    self.log_dets[epoch],
                    self.theoretical_traces.get(epoch, np.nan),
                    self.theoretical_log_dets.get(epoch, np.nan),
                )
            )

        if len(plot_rows) == 0:
            print(f"No valid epochs available to plot against {variable}.")
            return

        if variable in ['kappa', 'beta', 'kappa_beta_ratio']:
            plot_rows.sort(key=lambda row: row[0])

        xs = np.array([row[0] for row in plot_rows])
        epochs_plotted = [row[1] for row in plot_rows]
        empirical_traces = np.array([row[2] for row in plot_rows])
        empirical_log_dets = np.array([row[3] for row in plot_rows])
        theoretical_traces = np.array([row[4] for row in plot_rows])
        theoretical_log_dets = np.array([row[5] for row in plot_rows])
        empirical_trace_ci95 = np.array([self.empirical_trace_ci95.get(epoch, np.nan) for epoch in epochs_plotted])
        empirical_log_det_ci95 = np.array([self.empirical_log_det_ci95.get(epoch, np.nan) for epoch in epochs_plotted])
        empirical_sums = empirical_traces + empirical_log_dets
        theoretical_sums = theoretical_traces + theoretical_log_dets

        if variable == 'epochs':
            xlabel = 'Epoch'
        elif variable == 'kappa':
            xlabel = 'κ'
        elif variable == 'beta':
            xlabel = 'β'
        else:
            xlabel = 'κ / β'

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        fig, ax_trace = plt.subplots(figsize=(fig_width, fig_height))
        ax_log_det = ax_trace.twinx()
        ax_sum = None
        if plot_sum:
            ax_sum = ax_trace.twinx()
            ax_sum.spines['right'].set_position(('outward', 80))

        trace_lines = []
        log_det_lines = []
        sum_lines = []
        trace_bands = []
        log_det_bands = []

        trace_lines.extend(
            ax_trace.plot(
                xs,
                empirical_traces,
                marker='X',
                color='tab:blue',
                label='Empirical Trace'
            )
        )
        trace_bands.append(
            ax_trace.fill_between(
                xs,
                empirical_traces - empirical_trace_ci95,
                empirical_traces + empirical_trace_ci95,
                color='tab:blue',
                alpha=0.15,
                label='Empirical Trace 95% CI'
            )
        )
        if plot_theoretical:
            trace_lines.extend(
                ax_trace.plot(
                    xs,
                    theoretical_traces,
                    marker='o',
                    linestyle='--',
                    color='tab:blue',
                    label='Theoretical Trace'
                )
            )

        log_det_lines.extend(
            ax_log_det.plot(
                xs,
                empirical_log_dets,
                marker='X',
                color='tab:green',
                label='Empirical Log-Det'
            )
        )
        log_det_bands.append(
            ax_log_det.fill_between(
                xs,
                empirical_log_dets - empirical_log_det_ci95,
                empirical_log_dets + empirical_log_det_ci95,
                color='tab:green',
                alpha=0.15,
                label='Empirical Log-Det 95% CI'
            )
        )
        if plot_theoretical:
            log_det_lines.extend(
                ax_log_det.plot(
                    xs,
                    theoretical_log_dets,
                    marker='o',
                    linestyle='--',
                    color='tab:green',
                    label='Theoretical Log-Det'
                )
            )

        if plot_sum:
            sum_lines.extend(
                ax_sum.plot(
                    xs,
                    empirical_sums,
                    marker='X',
                    color='tab:red',
                    label='Empirical Trace + Log-Det'
                )
            )
            if plot_theoretical:
                sum_lines.extend(
                    ax_sum.plot(
                        xs,
                        theoretical_sums,
                        marker='o',
                        linestyle='--',
                        color='tab:red',
                        label='Theoretical Trace + Log-Det'
                    )
                )
            print(f"Log-Det at epoch {start_index}: empirical={empirical_log_dets[start_index]:.3g}")
            print(f"Min free energy epoch: {epochs_plotted[np.argmin(empirical_sums)]}, min empirical free energy: {np.min(empirical_sums):.3g}")

        ax_trace.set_xlabel(xlabel, fontsize=fontsize)
        ax_trace.set_ylabel('Trace', color='tab:blue', fontsize=fontsize)
        ax_log_det.set_ylabel('-Log-Determinant', color='tab:green', fontsize=fontsize)
        ax_trace.tick_params(axis='both', labelsize=fontsize * 0.7)
        ax_log_det.tick_params(axis='y', labelcolor='tab:green', labelsize=fontsize * 0.7)
        ax_trace.tick_params(axis='y', labelcolor='tab:blue')
        if plot_sum:
            ax_sum.set_ylabel('Free Energy', color='tab:red', fontsize=fontsize)
            ax_sum.tick_params(axis='y', labelcolor='tab:red', labelsize=fontsize * 0.7)
            if title:
                ax_trace.set_title(f'Free Energy vs. {xlabel}', fontsize=fontsize)
        else:
            if title:
                ax_trace.set_title(f'Trace and Log-Determinant vs. {xlabel}', fontsize=fontsize)

        lines = trace_lines + trace_bands + log_det_lines + log_det_bands + sum_lines
        labels = [line.get_label() for line in lines]
        if legend_on:
            plt.legend(lines, labels, loc='best')

        fig.tight_layout()

        if skipped_epochs and variable != 'epochs':
            skipped_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_epochs)
            print(f"Skipping epochs for {xlabel} plot: {skipped_epochs_str}")

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            plt.savefig(f'{plt_file}_trace_and_log_det_vs_{variable}.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        texplot.reset_theme()

    def plot_kappa_vs_epoch(self,
                            start_index=0,
                            title=True,
                            fig_width=10,
                            fig_height=6,
                            fontsize=12,
                            save_fig=True,
                            show_fig=True,
                            file_dir='figures/htmp_fits',
                            texplot_enabled=False,
                            axes=None,
                            color_plot=False):
        epochs = self.epoch_list[start_index:]

        plot_rows = []
        skipped_epochs = []
        for epoch in epochs:
            kappa = self.kappa_opt.get(epoch, None)
            if kappa is None:
                skipped_epochs.append((epoch, 'missing kappa'))
                continue

            kappa_val = self._as_float(kappa)
            if not np.isfinite(kappa_val):
                skipped_epochs.append((epoch, 'non-finite kappa'))
                continue

            plot_rows.append((epoch, kappa_val))

        if len(plot_rows) == 0:
            print('No valid fitted kappa values available to plot.')
            return

        x_epochs = np.array([row[0] for row in plot_rows])
        y_kappas = np.array([row[1] for row in plot_rows])

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        if axes is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        else:
            ax = axes
            print(f"Plotting κ/β ratio vs. epoch on provided axes.")
        if color_plot:
            color = color_plot
        else:
            color = 'tab:blue'
            
        ax.plot(x_epochs, y_kappas, marker='o', linestyle='-', color=color, label='$\kappa_t$')
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('κ', fontsize=fontsize)
        if title:
            fig.suptitle(fr'{self.model.upper()} {self.dataset.upper()} ESD; $\beta={self.beta_opt[epoch]:.3f}$, $\gamma={self._get_gamma(epoch):.3f}$', fontsize=fontsize)
        # ax.set_title('Fitted κ vs. Epoch')
        # ax.legend(loc='best')
        ax.tick_params(axis='both')

        if skipped_epochs:
            skipped_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_epochs)
            print(f"Skipping epochs for κ plot: {skipped_epochs_str}")

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            plt.savefig(f'{plt_file}_kappa_vs_epoch.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        return ax

    def plot_kappa_beta_ratio_vs_epoch(self,
                                       start_index=0,
                                       title=True,
                                       fig_width=10,
                                       fig_height=6,
                                       fontsize=12,
                                       save_fig=True,
                                       show_fig=True,
                                       file_dir='figures/htmp_fits',
                                       texplot_enabled=False,
                                       axes=None,
                                       color_plot=False):
        epochs = self.epoch_list[start_index:]

        plot_rows = []
        skipped_epochs = []
        for epoch in epochs:
            kappa = self.kappa_opt.get(epoch, None)
            beta = self.beta_opt.get(epoch, None)
            if kappa is None or beta is None:
                skipped_epochs.append((epoch, 'missing kappa/beta'))
                continue

            kappa_val = self._as_float(kappa)
            beta_val = self._as_float(beta)
            if not np.isfinite(kappa_val):
                skipped_epochs.append((epoch, 'non-finite kappa'))
                continue
            if not np.isfinite(beta_val) or beta_val == 0:
                skipped_epochs.append((epoch, 'invalid beta'))
                continue

            plot_rows.append((epoch, kappa_val / beta_val))

        if len(plot_rows) == 0:
            print('No valid fitted κ/β ratios available to plot.')
            return

        x_epochs = np.array([row[0] for row in plot_rows])
        y_ratios = np.array([row[1] for row in plot_rows])

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        if axes is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        else:
            ax = axes
            print(f"Plotting κ/β ratio vs. epoch on provided axes.")
        if color_plot:
            color = color_plot
        else:
            color = 'tab:blue'
        ax.plot(x_epochs, y_ratios, marker='o', linestyle='-', color=color, label=f'{self.model.upper()}')
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('κ / β', fontsize=fontsize)
        ax.tick_params(axis='both')

        if title:
            ax.set_title(f'{self.model.upper()}', fontweight='bold', fontsize=fontsize, y=0.98)

        if skipped_epochs:
            skipped_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_epochs)
            print(f"Skipping epochs for κ/β plot: {skipped_epochs_str}")
            

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            if axes is None:
                fig.tight_layout()
            plt.savefig(f'{plt_file}_kappa_beta_ratio_vs_epoch.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        texplot.reset_theme()

        return ax
    
    def plot_esd(self,
                 epoch_plot_interval=5,
                 num_plots_per_row=5,
                 fig_width = 15,
                 fig_height_per_row = 3,
                 log_bins=False,
                 log_y=False,
                 n_bins=100,
                 epochs_list = None,
                 x_left=None,
                 x_right=None,
                 plot_power=False,
                 plot_htmp=True,
                 save_fig=True,
                 show_fig=True,
                 title = False,
                 title_new = None,
                 fontsize=12,
                 file_dir = 'figures/htmp_fits'):

        if epochs_list is not None:
            epochs_plotting = epochs_list
        else:
            epochs_plotting = [epoch for epoch in self.epoch_list if epoch % epoch_plot_interval == 0]


        if plot_htmp:
            missing_epochs = [epoch for epoch in epochs_plotting if epoch not in self.kappa_opt]
            if missing_epochs:
                print(f"Warning: Missing fitted HTMP parameters for epochs: {missing_epochs}.")
                print("Please run .fit_htmp() before plotting.")
                return 
        
        num_plots_per_row_ = min(num_plots_per_row, len(epochs_plotting))
        f, axes = plt.subplots(int(np.ceil(len(epochs_plotting) / num_plots_per_row_)), 
                               num_plots_per_row_, 
                               figsize=(fig_width, fig_height_per_row * int(np.ceil(len(epochs_plotting) / num_plots_per_row_))),
                               layout="constrained")
        f.subplots_adjust(hspace=0.4, top=0.8, left=0.14)
        axes_list = (axes.flatten() if len(epochs_plotting) > 1 else [axes])

        def format_fit_label(epoch_value):
            kappa = self.kappa_opt.get(epoch_value, None)
            beta = self.beta_opt.get(epoch_value, None)

            if kappa is None:
                kappa_text = 'N/A'
            else:
                kappa_value = self._as_float(kappa)
                kappa_text = r'\infty' if np.isinf(kappa_value) else f'{kappa_value:.3f}'

            if beta is None:
                beta_text = 'N/A'
            else:
                beta_value = self._as_float(beta)
                beta_text = r'\infty' if np.isinf(beta_value) else f'{beta_value:.3f}'

            return fr'$\kappa={kappa_text}$' + '\n' + fr'$\beta={beta_text}$'
        
        for idx, epoch in tqdm.tqdm(enumerate(epochs_plotting)):
            tau = self.tau_opt.get(epoch, 0.0) or 0.0
            eigs = self.fileloader.get_eigs_epoch(epoch)
            ax = axes_list[idx]
                
            if log_bins:
                min_eig = np.min(eigs)
                max_eig = np.max(eigs)
                print(f"Min eig = {min_eig:.3g}, Max eig = {max_eig:.3g}")
                if min_eig > 0:
                    bins = np.logspace(np.log10(min_eig), np.log10(max_eig), n_bins)
                    ax.set_xscale('log')
                else:
                    print(f"Epoch {epoch}: non-positive eigenvalues found; falling back to linear bins.")
                    bins = (n_bins if not self.subsample else n_bins // 3)
                    bins = np.linspace(min_eig, max_eig, bins)
                # ax.set_yscale('log')
            else:
                bins = (n_bins if not self.subsample else n_bins // 3)
                bins = np.linspace(np.min(eigs), np.max(eigs), bins)

            if log_y:
                ax.set_yscale('log')

            ax.hist(eigs, bins=bins, density=True, color='grey')
            ax.set_title(f'{epoch}', fontsize=fontsize)
            
            # Plot powerlaw fit
            if plot_power:
                try:
                    fit = powerlaw.Fit(eigs, verbose=0)
                    fit.power_law.plot_pdf(ax=ax, color='red', label=f'α: {fit.power_law.alpha:.3f}')
                    ax.legend(fontsize=fontsize-2)
                except Exception as e:
                    print(f"Error fitting powerlaw: {e}")

            # # Plot fitted HTMP density
            if plot_htmp:
                eigs_shifted = eigs - tau
                eigs_shifted_pos = eigs_shifted[eigs_shifted > 0]
                if log_bins:
                    if eigs_shifted_pos.size == 0:
                        lmbda_vals = np.linspace(np.min(eigs), np.max(eigs), 200)
                    else:
                        min_shifted = np.min(eigs_shifted_pos)
                        max_shifted = np.max(eigs_shifted)
                        lmbda_vals = np.logspace(np.log10(min_shifted)-1, np.log10(max_shifted), 100) + tau
                else:
                    lmbda_vals = np.linspace(np.min(eigs), np.max(eigs), 200)

                if epoch == 0 and not self.inverse_htmp:
                    mp_vals = marchenko_pastur_pdf(lmbda_vals, self._get_gamma(epoch))
                    ax.plot(lmbda_vals, mp_vals, 'r:', linewidth=2)
                else:
                    gamma = self._get_gamma(epoch)
                    
                    if not self.inverse_htmp:
                        a = self.kappa_opt[epoch] / (2 * gamma * self.beta_opt[epoch])
                        shifted_support = lmbda_vals - tau
                        density = np.zeros_like(lmbda_vals)
                        valid = shifted_support > 0
                        density[valid] = htmp.pdf(shifted_support[valid] / a, gamma, self.kappa_opt[epoch]) / a
                        ax.plot(lmbda_vals, density, 'r:', linewidth=2)
                    elif self.inverse_htmp:
                        a = self.kappa_opt[epoch] / (2 * gamma * self.beta_opt[epoch])
                        spectral_density = lambda x : htmp.pdf(1 / a / x, gamma, self.kappa_opt[epoch]) / (a * x**2)
                        shifted_support = lmbda_vals - tau
                        density = np.zeros_like(lmbda_vals)
                        valid = shifted_support > 0
                        density[valid] = spectral_density(shifted_support[valid] * gamma) * gamma
                        ax.plot(lmbda_vals, density, 'r:', linewidth=2)
                ax.legend(
                    handles=[Line2D([], [], linestyle='none')],
                    labels=[format_fit_label(epoch)],
                    fontsize=fontsize-2,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    borderpad=0,
                    loc='upper right'
                )

            ax.tick_params(axis='both', labelsize=fontsize)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if x_right is not None:
                ax.set_xlim(left = x_left, right=x_right)

        if title:
            if title_new is not None:
                f.text(-0.02, 0.5, title_new, fontsize=fontsize, fontweight='bold', rotation=90, va='center', ha='left')
            else:
                f.text(-0.02, 0.5, self.model.upper(), fontsize=fontsize, fontweight='bold', rotation=90, va='center', ha='left')
            
        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            plt.savefig(f'{plt_file}_esd.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

    def plot_test_acc_and_train_loss(self,
                                     start_index=0,
                                     log_scale_loss=False,
                                     loss_val = None,
                                     save_fig=True,
                                     show_fig=True,
                                     file_dir='figures/htmp_fits'):
        epochs = self.epoch_list[start_index:]
        if len(epochs) == 0:
            print('No epochs available to plot.')
            return

        test_acc_stats = [self.fileloader.get_test_acc_epoch_stats(epoch) for epoch in epochs]
        test_accs = np.array([stats['mean'] for stats in test_acc_stats])
        test_acc_ci95 = np.array([stats['ci95'] for stats in test_acc_stats])
        train_losses = np.array([self.fileloader.get_train_loss_epoch(epoch) for epoch in epochs])

        fig, ax_test = plt.subplots(figsize=(10, 6))
        ax_loss = ax_test.twinx()

        test_line = ax_test.plot(
            epochs,
            test_accs,
            marker='o',
            color='tab:red',
            label='Test Accuracy'
        )
        test_band = ax_test.fill_between(
            epochs,
            test_accs - test_acc_ci95,
            test_accs + test_acc_ci95,
            color='tab:red',
            alpha=0.15,
            label='Test Accuracy 95% CI'
        )
        loss_line = ax_loss.plot(
            epochs,
            train_losses,
            marker='x',
            color='tab:blue',
            label='Train Loss'
        )
        if loss_val is not None:
            loss_line = ax_loss.plot(
                epochs,
                [loss_val] * len(epochs),
                linestyle='--',
                color='tab:blue',
                label=f'Train Loss = {loss_val}'
            )

        ax_test.set_xlabel('Epoch')
        ax_test.set_ylabel('Test Accuracy', color='tab:red')
        ax_loss.set_ylabel('Train Loss', color='tab:blue')
        ax_test.tick_params(axis='y', labelcolor='tab:red')
        ax_loss.tick_params(axis='y', labelcolor='tab:blue')
        ax_test.set_title('Test Accuracy and Train Loss vs. Epoch')
        if log_scale_loss:
            ax_loss.set_yscale('log')

        lines = test_line + loss_line + [test_band]
        labels = [line.get_label() for line in lines]
        ax_test.legend(lines, labels, loc='best')

        fig.tight_layout()

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            plt.savefig(f'{plt_file}_test_acc_and_train_loss_vs_epoch.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()
    
    def plot_test_vs_free(self,
                          variable='epochs',
                          neg_free=False,
                          start_index=0,
                          plot_theoretical_free=False,
                          single_plot=False,
                          fig_width=10,
                          fig_height=6,
                          fontsize=12,
                          task='arc_easy',
                          metric='acc',
                          shot='zero-shot',
                          evals_root='evals',
                          suite='pythia-v1',
                          deduped=None,
                          save_fig=True,
                          show_fig=True,
                          file_dir='figures/htmp_fits',
                          log_x=False,
                          texplot_enabled=False,
                          optimal_ratio=None,
                          marker_size=8,
                          title=True,
                          tight_layout=True):
        if self.fileloader._is_moonlight_model():
            raise NotImplementedError("Moonlight models do not yet support plotting test accuracy against free energy.")

        epochs = self.epoch_list[start_index:]
        if any(epoch not in self.free_energies for epoch in epochs):
            print("Please run .compute_free_energy() before plotting.")
            return

        test_acc_stats = [
            self.fileloader.get_test_acc_epoch_stats(
                epoch,
                task=task,
                metric=metric,
                shot=shot,
                evals_root=evals_root,
                suite=suite,
                deduped=deduped,
            )
            for epoch in epochs
        ]
        test_accs = np.array([stats['mean'] for stats in test_acc_stats])
        test_acc_ci95 = np.array([stats['ci95'] for stats in test_acc_stats])
        mult = (-1 if neg_free else 1)
        free_energies = np.array([mult * self.free_energies[epoch] for epoch in epochs])
        free_energy_ci95 = np.array([self.free_energy_ci95.get(epoch, np.nan) for epoch in epochs])

        theoretical_free_energies = None
        if plot_theoretical_free:
            if any(epoch not in self.theoretical_free_energies for epoch in epochs):
                print("Theoretical free energies are not available for all epochs. Run .compute_free_energy() after fitting HTMP parameters.")
                return
            theoretical_free_energies = [mult * self.theoretical_free_energies[epoch] for epoch in epochs]

        if variable == 'epochs':
            xs = epochs
            xlabel = 'Epoch'
        elif variable == 'kappa':
            xs = [self.kappa_opt[epoch] for epoch in epochs]
            xlabel = 'κ'
        elif variable == 'kappa_beta_ratio':
            xs = [self.kappa_opt[epoch] / self.beta_opt[epoch] for epoch in epochs]
            xlabel = 'κ / β'
        elif variable == 'bartlett_ratio':
            xs = self.lam * np.array([self.kappa_opt[epoch] / self.beta_opt[epoch] for epoch in epochs]) / (2 * self._get_gamma())
            xlabel = 'Bartlett Ratio'

        corr = np.corrcoef(test_accs, free_energies)[0, 1]

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        if single_plot:
            

            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
            ax2 = ax1.twinx()

            test_line = ax1.plot(xs, test_accs, marker='o', markerfacecolor='none', color='red', label='Accuracy (Mean)')
            test_band = ax1.fill_between(
                xs,
                test_accs - test_acc_ci95,
                test_accs + test_acc_ci95,
                color='red',
                alpha=0.15,
                label='Accuracy (95% CI)'
            )

            free_line = ax2.plot(xs, free_energies, marker='x', color='green', label='Free Energy (Mean)')
            free_band = ax2.fill_between(
                xs,
                free_energies - free_energy_ci95,
                free_energies + free_energy_ci95,
                color='green',
                alpha=0.15,
                label='Free Energy (95% CI)'
            )
            if plot_theoretical_free:
                theoretical_line = ax2.plot(
                    xs,
                    theoretical_free_energies,
                    marker='s',
                    linestyle='--',
                    color='blue',
                    label='Theoretical Free Energy'
                )
            else:
                theoretical_line = []

            ax1.set_xlabel(xlabel, fontsize=fontsize)
            ax1.set_ylabel('Test Accuracy', fontsize=fontsize)
            ax2.set_ylabel('Free Energy', fontsize=fontsize)
            ax1.tick_params(axis='y')
            ax2.tick_params(axis='y')
            if title:
                ax1.set_title(f'{self.model.upper()} ($r$: {corr:.3f}, $\lambda$: {self.lam:.2f})', fontweight='bold', fontsize=fontsize)

            if log_x:
                ax1.set_xscale('log')
                ax2.set_xscale('log')

            if optimal_ratio is not None:
                # Place a marker on the free energy at the optimal kappa_beta_ratio, if provided
                ax2.scatter(
                    optimal_ratio,
                    theoretical_free_energies[np.argmin(np.abs(np.array(xs) - optimal_ratio))],
                    color='purple',
                    label=f'Optimal Ratio = {optimal_ratio:.3f}',
                    marker='X',
                    s=marker_size,
                )

            lines = test_line + [test_band] + free_line + [free_band] + theoretical_line
            labels = [line.get_label() for line in test_line] + [test_band.get_label()] + [line.get_label() for line in free_line] + [free_band.get_label()] + [line.get_label() for line in theoretical_line]
            ax1.legend(lines, labels, loc='best')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
            test_line = ax1.plot(xs, test_accs, marker='o', markerfacecolor='none', color='red', label='Mean')
            test_band = ax1.fill_between(
                xs,
                test_accs - test_acc_ci95,
                test_accs + test_acc_ci95,
                color='red',
                alpha=0.15,
                label='95% CI'
            )
            ax1.set_ylabel('Test Accuracy')
            ax1.set_xlabel(xlabel)
            ax1.tick_params(axis='y')
            ax1.legend(test_line + [test_band], [line.get_label() for line in test_line] + [test_band.get_label()], loc='best')
            if log_x:
                ax1.set_xscale('log')

            free_line = ax2.plot(xs, free_energies, marker='x', color='green', label='Empirical Mean')
            free_band = ax2.fill_between(
                xs,
                free_energies - free_energy_ci95,
                free_energies + free_energy_ci95,
                color='green',
                alpha=0.15,
                label='Empirical 95% CI'
            )
            if plot_theoretical_free:
                theoretical_line = ax2.plot(xs, theoretical_free_energies, marker='s', linestyle='--', color='blue', label='Limiting')
            else:
                theoretical_line = []
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Negative Free Energy' if neg_free else 'Free Energy')
            ax2.tick_params(axis='y')
            ax2.legend(free_line + [free_band] + theoretical_line, [line.get_label() for line in free_line] + [free_band.get_label()] + [line.get_label() for line in theoretical_line], loc='best')
            if log_x:
                ax2.set_xscale('log')

            fig.suptitle(f'{self.model.upper()} ($r$: {corr:.3f})', fontweight='bold', fontsize=fontsize, y=0.94)

        plt_file = self._get_figure_file_stem(file_dir)
        if self.weight_inverse:
            plt_file += '_weight_inverse'

        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if tight_layout:
            fig.tight_layout()
        if save_fig:
            plt.savefig(f'{plt_file}_test_vs_free_energy_vs_{variable}.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        print(f"Correlation between Test Accuracy and {'-Free Energy' if neg_free else 'Free Energy'}: {corr:.3f}")
        print(f'Maximum Test Accuracy: {np.max(test_accs):.3f} at {xlabel}={xs[np.argmax(test_accs)]:.3g}, Free Energy={free_energies[np.argmax(test_accs)]:.5f}')
        texplot.reset_theme()

    def empirical_free_energy_eigs(self, eigvals, n, lam=1, tau=1e-9, weight_inverse=False):
        if weight_inverse:
            log_det = - np.sum(np.log((eigvals + lam*tau))) / (n)
            trace = np.sum(eigvals + lam*tau) / (n)
            const = 1/2 * np.log(2 * np.pi / lam)
            F = lam / 2 * trace + log_det / 2 + const
            return F, lam*trace, log_det - np.log(lam)
        else:
            log_det = np.sum(np.log((eigvals + lam*tau))) / (n)
            trace = np.sum(1 / (eigvals + lam*tau)) / (n)
            const = 1/2 * np.log(2 * np.pi / lam)
            F = lam / 2 * trace + log_det / 2 + const
            return F, lam*trace, log_det - np.log(lam)
        
    def theoretical_free_energy(self, kappa, gamma, beta, lam, tau, weight_inverse=False):
        if not self.inverse_htmp:
            if weight_inverse:
                log_det = self.weight_log_det(kappa, gamma, beta, tau)
                trace = self.weight_first_moment(kappa, gamma, beta, tau)
                const = 1/2 * np.log(2 * np.pi / lam)
                free_energy = lam / 2 * trace - log_det / 2 + const
                return free_energy, lam*trace, -log_det - np.log(lam)
            else:
                log_det = self.weight_log_det(kappa, gamma, beta, tau)
                trace = self.weight_stieltjes(kappa, gamma, beta, tau)
                const = 1/2 * np.log(2 * np.pi / lam)
                free_energy = lam / 2 * trace + log_det / 2 + const
                return free_energy, lam*trace, log_det - np.log(lam)
        else:
            raise NotImplementedError("Theoretical free energy computation for inverse HTMP is not implemented yet.")
    
    def weight_stieltjes(self, kappa, gamma, beta, x):
        return beta * scipy.special.hyperu(kappa/2+1, 2 - kappa / (2*gamma) + kappa / 2, beta * x) \
            / scipy.special.hyperu(kappa/2, 1 - kappa / (2*gamma) + kappa / 2, beta * x)
    
    def weight_log_det(self, kappa, gamma, beta, x):
        if x == 0:
            return -2/kappa * scipy.special.loggamma(kappa/2/gamma - kappa/2) + 2/kappa * scipy.special.loggamma(kappa/2/gamma) - np.log(beta)
        else:
            return - 2/kappa * np.log(scipy.special.hyperu(kappa/2, 1 - kappa / (2*gamma) + kappa / 2, beta * x)) - np.log(beta)
        
    def weight_first_moment(self, kappa, gamma, beta, x):
        return kappa / (2 * gamma * beta) + x
    
    ### TODO: Add feature matrices versions of the above for inverse HTMP case


class fileloading(object):
    def __init__(self, model='minialexnet', dataset='cifar10', kernel='weight', batch_size=None, subsample=False, file_dir = 'results/htmp/epoch', weight_type=None, layer=None):
        self.model = model
        self.dataset = dataset
        self.kernel = kernel
        self.batch_size = batch_size
        self.subsample = subsample
        self.file_dir = file_dir
        self.weight_type = weight_type
        self.layer = layer

        self.filename = f'{model}_{dataset}'
        if subsample:
            self.filename += '_subsampled'
        self.filename += f'_b{batch_size}'
        if not subsample:
            self.files = [f for f in sorted(glob.glob(f'{self.file_dir}/{self.filename}*.pt'), key=extract_key, reverse=True) if '_subsample' not in f]
        else:
            self.files = [f for f in sorted(glob.glob(f'{self.file_dir}/{self.filename}*.pt'), key=extract_key, reverse=True) if '_subsample' in f]
        self.files = self._filter_files_for_scope(self.files)
        self.epoch_list = sorted(get_keys(self.files), reverse=False)
        self._pythia_eval_cache = {}

    def get_epoch_list(self):
        return self.epoch_list

    def _filter_files_for_scope(self, files):
        if not self._is_moonlight_model() or self.layer is None:
            return files

        layer_suffix = f'_layer{self.layer}.pt'
        scoped_files = [file for file in files if file.endswith(layer_suffix)]
        if scoped_files:
            return scoped_files

        # Backward compatibility for older Moonlight summaries saved before layer-specific filenames.
        legacy_files = [file for file in files if '_layer' not in Path(file).stem]
        return legacy_files if legacy_files else files

    def _resolve_saved_layer_key(self, layer_dict):
        if self.layer is None:
            raise ValueError('layer must be set before resolving a saved layer key.')

        candidates = [self.layer, str(self.layer)]
        if self._is_moonlight_model():
            legacy_layer = self.layer - 1
            candidates.extend([legacy_layer, str(legacy_layer)])

        for candidate in candidates:
            if candidate in layer_dict:
                return candidate

        available = ', '.join(map(str, sorted(layer_dict.keys(), key=lambda x: int(x))))
        raise KeyError(f"Layer {self.layer} not found under {self.weight_type}. Available layers: {available}")

    def _get_new_format_eigvals(self, res_dict):
        if self.weight_type is None:
            return None

        if self.weight_type == 'EmbedOut':
            return np.asarray(res_dict['EmbedOut']['eigvals'], dtype=float)

        if self.weight_type not in ('Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2'):
            raise ValueError("weight_type must be one of None, 'Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2', or 'EmbedOut'.")

        layer_dict = res_dict[self.weight_type]
        if self.layer is None:
            ordered_keys = sorted(layer_dict.keys(), key=lambda x: int(x))
            return np.concatenate([np.asarray(layer_dict[k]['eigvals'], dtype=float) for k in ordered_keys])

        layer_key = self._resolve_saved_layer_key(layer_dict)
        return np.asarray(layer_dict[layer_key]['eigvals'], dtype=float)

    def _get_new_format_aspect_ratio(self, res_dict):
        if self.weight_type is None:
            raise ValueError("weight_type must be set to load a saved aspect ratio from a checkpoint summary.")

        if self.weight_type == 'EmbedOut':
            return float(res_dict['EmbedOut']['aspect_ratio'])

        if self.weight_type not in ('Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2'):
            raise ValueError("weight_type must be one of 'Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2', or 'EmbedOut'.")

        if self.layer is None:
            raise ValueError(f"layer must be set to load a saved aspect ratio for {self.weight_type}.")

        layer_dict = res_dict[self.weight_type]
        layer_key = self._resolve_saved_layer_key(layer_dict)
        return float(layer_dict[layer_key]['aspect_ratio'])

    def _get_eigvals_from_file(self, file):
        res_dict = torch.load(file, weights_only=False, map_location='cpu')

        new_format_eigs = self._get_new_format_eigvals(res_dict)
        if new_format_eigs is not None:
            return new_format_eigs

        eig_key = {
            'weight': 'weight_eigvals',
            'ck': 'ck_eigvals',
            'ntk': 'ntk_eigvals',
        }[self.kernel]
        return np.asarray(res_dict[eig_key], dtype=float)

    def _get_aspect_ratio_from_file(self, file):
        res_dict = torch.load(file, weights_only=False, map_location='cpu')

        try:
            return self._get_new_format_aspect_ratio(res_dict)
        except KeyError as exc:
            raise KeyError(
                "Saved aspect ratios are only available for checkpoint summary files with weight_type/layer metadata."
            ) from exc

    def get_eigs_epoch(self, epoch):
        key_files = get_key_files(self.files, epoch)
        eigvals_by_repeat = [self._get_eigvals_from_file(file) for file in key_files]
        if len(eigvals_by_repeat) == 0:
            return np.array([])
        return np.concatenate(eigvals_by_repeat)

    def get_aspect_ratio_epoch(self, epoch):
        key_files = get_key_files(self.files, epoch)
        if len(key_files) == 0:
            return np.nan

        aspect_ratios = [self._get_aspect_ratio_from_file(file) for file in key_files]
        first_aspect_ratio = float(aspect_ratios[0])
        if not all(np.isclose(first_aspect_ratio, aspect_ratio) for aspect_ratio in aspect_ratios[1:]):
            raise ValueError(f"Found inconsistent saved aspect ratios for epoch {epoch}.")
        return first_aspect_ratio

    def get_eigs_epoch_values(self, epoch):
        key_files = get_key_files(self.files, epoch)
        return [self._get_eigvals_from_file(file) for file in key_files]

    def get_test_acc_epoch_values(self, epoch):
        return np.asarray(get_test_acc(self.files, epoch), dtype=float)

    def _is_pythia_model(self):
        model_name = str(self.model).strip().lower()
        if model_name.startswith('eleutherai/'):
            model_name = model_name.split('/', 1)[1]
        return model_name.startswith('pythia-')

    def _is_moonlight_model(self):
        model_name = str(self.model).strip().lower()
        return model_name.startswith('moonlight_')

    def _is_optimizer_resnet_model(self):
        model_name = str(self.model).strip().lower()
        return bool(re.fullmatch(r'resnet(?:9|18|34|50)_.+', model_name))

    def _uses_saved_aspect_ratio(self):
        return self._is_pythia_model() or self._is_moonlight_model() or self._is_optimizer_resnet_model()

    def _resolve_pythia_model_size_and_deduped(self, deduped):
        model_name = str(self.model).strip().lower()
        if model_name.startswith('eleutherai/'):
            model_name = model_name.split('/', 1)[1]
        if not model_name.startswith('pythia-'):
            raise ValueError(f'Model {self.model} is not a Pythia model.')

        model_size = model_name[len('pythia-'):]
        inferred_deduped = model_size.endswith('-deduped')
        if inferred_deduped:
            model_size = model_size[:-len('-deduped')]

        deduped_value = inferred_deduped if deduped is None else bool(deduped)
        return model_size, deduped_value

    def _get_pythia_eval_metric_df(self, task, metric, shot, evals_root, suite, deduped):
        model_size, deduped_value = self._resolve_pythia_model_size_and_deduped(deduped)
        cache_key = (model_size, task, metric, shot, evals_root, suite, deduped_value)
        if cache_key not in self._pythia_eval_cache:
            self._pythia_eval_cache[cache_key] = load_pythia_eval_metric(
                model_size=model_size,
                task=task,
                metric=metric,
                evals_root=evals_root,
                suite=suite,
                deduped=deduped_value,
                shot=shot,
            )
        return self._pythia_eval_cache[cache_key]

    def get_test_acc_epoch_stats(self,
                                 epoch,
                                 task='arc_easy',
                                 metric='acc',
                                 shot='zero-shot',
                                 evals_root='evals',
                                 suite='pythia-v1',
                                 deduped=None):
        if self._is_moonlight_model():
            raise NotImplementedError("Moonlight checkpoint summaries do not include evaluation accuracy yet.")

        if self._is_pythia_model():
            eval_df = self._get_pythia_eval_metric_df(
                task=task,
                metric=metric,
                shot=shot,
                evals_root=evals_root,
                suite=suite,
                deduped=deduped,
            )
            step_rows = eval_df.loc[eval_df['step'] == int(epoch)]
            if step_rows.empty:
                return {'mean': np.nan, 'std': np.nan, 'ci95': np.nan, 'count': 0}

            row = step_rows.iloc[-1]
            stderr = float(row['stderr']) if ('stderr' in row and np.isfinite(row['stderr'])) else np.nan
            return {
                'mean': float(row['value']),
                'std': stderr,
                'ci95': stderr,
                'count': 1,
            }

        accuracies = self.get_test_acc_epoch_values(epoch)
        if accuracies.size == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci95': np.nan, 'count': 0}

        mean = float(np.mean(accuracies))
        std = float(np.std(accuracies, ddof=1)) if accuracies.size > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(accuracies.size)) if accuracies.size > 1 else 0.0
        return {'mean': mean, 'std': std, 'ci95': ci95, 'count': int(accuracies.size)}
    
    def get_test_acc_epoch(self, epoch):
        return self.get_test_acc_epoch_stats(epoch)['mean']
    
    def get_train_loss_epoch(self, epoch):
        return get_train_loss(self.files, epoch).mean().item()
    

    


    
    




        