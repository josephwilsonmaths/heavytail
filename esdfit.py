import os
import numbers
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
from htmp_cdf import compute_htmp_ks_distance, compute_mp_ks_distance, evaluate_htmp_cdf
from matplotlib.ticker import FormatStrFormatter
import texplot

class ESDFit(object):
    def __init__(self, model='minialexnet', dataset='cifar10', kernel='weight', batch_size=None, subsample=False, file_dir = 'results/htmp/epoch', weight_type=None, layer=None, trim_bottom_eigs=0, trim_top_eigs=0, permuted_labels=False, nonzero_eigs_only=False):
        self.model = model
        self.dataset = dataset
        self.kernel = kernel
        self.batch_size = batch_size
        self.subsample = subsample
        self.file_dir = file_dir
        self.weight_type = weight_type
        self.layer = layer
        self.trim_bottom_eigs = int(trim_bottom_eigs)
        self.trim_top_eigs = int(trim_top_eigs)
        self.permuted_labels = permuted_labels
        self.nonzero_eigs_only = nonzero_eigs_only

        self.fileloader = fileloading(
            model,
            dataset,
            kernel,
            batch_size,
            subsample,
            file_dir,
            weight_type=weight_type,
            layer=layer,
            trim_bottom_eigs=trim_bottom_eigs,
            trim_top_eigs=trim_top_eigs,
            permuted_labels=permuted_labels,
            nonzero_eigs_only=nonzero_eigs_only,
        )

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

    def get_available_layers(self):
        return self.fileloader.get_available_layers()

    def print_available_layers(self):
        layers = self.get_available_layers()
        if len(layers) == 0:
            print('No saved layers available for this ESDFit configuration.')
            return layers

        print('Available layers:')
        for layer in layers:
            print(layer)
        return layers

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

    def _normalize_fit_epochs(self, epochs):
        normalized_epochs = []

        def append_epochs(value):
            if isinstance(value, numbers.Integral):
                normalized_epochs.append(int(value))
                return

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    append_epochs(value.item())
                    return
                for item in value.tolist():
                    append_epochs(item)
                return

            if isinstance(value, (list, tuple, set, range)):
                for item in value:
                    append_epochs(item)
                return

            raise TypeError(
                "epochs_list must contain integers or iterables of integers. "
                f"Received {type(value).__name__}."
            )

        append_epochs(epochs)
        return normalized_epochs

    def _resolve_fit_epochs(self, epochs_list):
        if epochs_list is not None:
            self.fitted_epochs = self._normalize_fit_epochs(epochs_list)
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
    def _beta_from_mean_eigs(self, kappa, eigs, gamma):
        kappa_value = self._as_float(kappa)
        gamma_value = self._as_float(gamma)
        eigs_value = np.asarray(eigs, dtype=float)
        mean_eigs = float(np.mean(eigs_value))
        if kappa_value <= 0 or not np.isfinite(kappa_value):
            raise ValueError(f"beta_from_mean_eigs requires a positive finite kappa, got kappa={kappa_value}.")
        if gamma_value <= 0 or not np.isfinite(gamma_value):
            raise ValueError(f"beta_from_mean_eigs requires a positive finite gamma, got gamma={gamma_value}.")
        if mean_eigs <= 0 or not np.isfinite(mean_eigs):
            raise ValueError(f"beta_from_mean_eigs requires a positive finite mean eig, got mean={mean_eigs}.")
        return kappa_value / (2.0 * gamma_value * mean_eigs)

    def _load_epoch_eigs(self, epochs, verbose=False):
        epochs_to_fit = [epoch for epoch in epochs if not self._skip_epoch_fit(epoch)]
        if verbose:
            print("Loading eigenvalues...")
        return {epoch: self.fileloader.get_eigs_epoch(epoch) for epoch in epochs_to_fit}

    def _get_gamma(self, epoch=None):
        # nonzero_eigs_only trims eigenvalues using the per-layer saved aspect ratio,
        # so the gamma for fitting must also come from that saved ratio.
        if self.fileloader._uses_saved_aspect_ratio() or (self.nonzero_eigs_only and epoch is not None):
            if epoch is None:
                raise ValueError("epoch must be provided when resolving gamma for a checkpoint-summary model.")
            return self.fileloader.get_aspect_ratio_epoch(epoch)
        return gamma_dict[self.model][self.kernel]

    def _get_gamma_by_epoch(self, epochs):
        return {epoch: self._get_gamma(epoch) for epoch in epochs}

    def get_aspect_ratio_epoch(self, epoch):
        return self.fileloader.get_aspect_ratio_epoch(epoch)

    def _resolve_epoch_fit_params(self, epoch, kappa=None, beta=None, tau=None, inverse=None):
        kappa_value = self.kappa_opt.get(epoch, None) if kappa is None else kappa
        beta_value = self.beta_opt.get(epoch, None) if beta is None else beta
        tau_value = self.tau_opt.get(epoch, 0.0) if tau is None else tau
        inverse_value = self.inverse_htmp if inverse is None else inverse

        if kappa_value is None or beta_value is None:
            raise ValueError(
                f"Missing fitted HTMP parameters for epoch {epoch}. Run .fit_htmp() first or provide kappa and beta explicitly."
            )

        return (
            self._as_float(kappa_value),
            self._as_float(beta_value),
            self._as_float(tau_value),
            bool(inverse_value),
        )

    def evaluate_htmp_cdf_epoch(self,
                                epoch,
                                points,
                                kappa=None,
                                beta=None,
                                tau=None,
                                inverse=None,
                                dps=50,
                                branch_epsilon=1e-12):
        gamma = self._get_gamma(epoch)
        kappa_value, beta_value, tau_value, inverse_value = self._resolve_epoch_fit_params(
            epoch,
            kappa=kappa,
            beta=beta,
            tau=tau,
            inverse=inverse,
        )
        return evaluate_htmp_cdf(
            points,
            kappa_value,
            gamma,
            beta_value,
            tau=tau_value,
            inverse=inverse_value,
            dps=dps,
            branch_epsilon=branch_epsilon,
        )

    def compute_ks_distance_epoch(self,
                                  epoch,
                                  eigs=None,
                                  kappa=None,
                                  beta=None,
                                  tau=None,
                                  inverse=None,
                                  dps=50,
                                  branch_epsilon=1e-12,
                                  return_details=False):
        eigs_value = self.fileloader.get_eigs_epoch(epoch) if eigs is None else eigs

        if epoch == 0 and not self.inverse_htmp:
            return compute_mp_ks_distance(
                eigs_value,
                self._get_gamma(epoch),
                scale=1.0,
                tau=0.0,
                return_details=return_details,
            )

        gamma = self._get_gamma(epoch)
        kappa_value, beta_value, tau_value, inverse_value = self._resolve_epoch_fit_params(
            epoch,
            kappa=kappa,
            beta=beta,
            tau=tau,
            inverse=inverse,
        )
        return compute_htmp_ks_distance(
            eigs_value,
            kappa_value,
            gamma,
            beta_value,
            tau=tau_value,
            inverse=inverse_value,
            dps=dps,
            branch_epsilon=branch_epsilon,
            return_details=return_details,
        )

    def print_htmp_fit_summary(self,
                               epoch,
                               eigs=None,
                               include_ks=True,
                               dps=50,
                               branch_epsilon=1e-12):
        gamma = self._get_gamma(epoch)
        kappa_value, beta_value, tau_value, inverse_value = self._resolve_epoch_fit_params(epoch)
        dist_value = self.dist_opt.get(epoch, None)

        print(f"Epoch: {epoch}")
        print(f"gamma: {gamma:.6g}")
        print(f"kappa: {kappa_value:.6g}")
        print(f"beta: {beta_value:.6g}")
        print(f"tau: {tau_value:.6g}")
        print(f"inverse_htmp: {inverse_value}")
        print(f"fit_distance: {dist_value}")

        if include_ks:
            ks_distance = self.compute_ks_distance_epoch(
                epoch,
                eigs=eigs,
                dps=dps,
                branch_epsilon=branch_epsilon,
            )
            print(f"ks_distance: {ks_distance:.6g}")
            return ks_distance
        return None

    def _fit_epoch_parameters(self,
                              eigs,
                              gamma,
                              search_method,
                              method,
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
                              beta_from_mean=False,
                              c=1.0,
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
        if beta_from_mean:
            mean_scale = 1.0 / (2.0 * float(gamma) * float(np.mean(shifted_eigs)))

            def objective(kappa):
                beta = kappa * mean_scale
                if method == 'ks':
                    from htmp_cdf import compute_htmp_ks_distance

                    return compute_htmp_ks_distance(
                        shifted_eigs,
                        kappa,
                        gamma,
                        beta,
                        tau=0.0,
                        inverse=inverse,
                        return_details=False,
                    )

                return wass_distance(
                    shifted_eigs,
                    gamma,
                    kappa,
                    beta,
                    bins=n_eval,
                    inverse=inverse,
                    stieltjes=(method == 'stieltjes'),
                    lp_ord=1,
                    moment_penalty_weight=moment_penalty_weight,
                )

            if search_method == 'grid':
                if kappa_log_scale:
                    kappa_values = np.logspace(np.log10(kappa_min), np.log10(kappa_max), num_kappas)
                else:
                    kappa_values = np.linspace(kappa_min, kappa_max, num_kappas)

                best_kappa = None
                best_beta = None
                best_distance = float('inf')
                for kappa in kappa_values:
                    beta = kappa * mean_scale
                    distance = objective(kappa)
                    if distance < best_distance:
                        best_distance = distance
                        best_kappa = kappa
                        best_beta = beta

                if verbose:
                    print(f"Best grid point: kappa={best_kappa:.4g}, beta={best_beta:.4g}, distance={best_distance:.6g}")
                return {
                    'kappa': best_kappa,
                    'beta': best_beta,
                    'tau': tau_epoch,
                    'dist': best_distance,
                }

            if search_method == 'ternary':
                if kappa_log_scale:
                    best_kappa = ternary_search(objective, kappa_min, kappa_max, num_kappas, verbose=verbose, input_name='kappa', output_name='distance', log_scale=True)
                else:
                    best_kappa = ternary_search(objective, kappa_min, kappa_max, num_kappas, verbose=verbose, input_name='kappa', output_name='distance', log_scale=False)
                best_beta = best_kappa * mean_scale
                best_distance = objective(best_kappa)
                return {
                    'kappa': best_kappa,
                    'beta': best_beta,
                    'tau': tau_epoch,
                    'dist': best_distance,
                }

            raise ValueError(f"Invalid search method: {search_method}. Choose 'grid' or 'ternary'.")

        search_beta_min = fixed_beta if fixed_beta is not None else beta_min
        search_beta_max = fixed_beta if fixed_beta is not None else beta_max
        search_num_betas = 1 if fixed_beta is not None else num_betas

        kappa_opt, beta_opt, dist_opt = find_best_htmp(
            shifted_eigs,
            gamma,
            search_method=search_method,
            method=method,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            num_kappas=num_kappas,
            n_eval=n_eval,
            beta_min=search_beta_min,
            beta_max=search_beta_max,
            num_betas=search_num_betas,
            inverse=inverse,
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
                               method,
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
                               beta_from_mean=False,
                               c=1.0,
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
                        method,
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
                        beta_from_mean=beta_from_mean,
                        c=c,
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
        if epoch == 0 and not self.inverse_htmp:
            beta_epoch = float('inf')
        else:
            beta_epoch = self._beta_from_tau(tau_epoch, c=c) if beta_from_tau else shared_beta
        if beta_epoch is None:
            beta_epoch = float('nan')
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
        if self.permuted_labels:
            parts.append('permuted_labels=True')
        if self.nonzero_eigs_only:
            parts.append('nonzero_eigs_only=True')
        if self.weight_type is not None:
            parts.append(f"weight_type={self.weight_type}")
        if self.layer is not None:
            parts.append(f"layer={self.layer}")
        if self.trim_bottom_eigs > 0:
            parts.append(f"trim_bottom={self.trim_bottom_eigs}")
        if self.trim_top_eigs > 0:
            parts.append(f"trim_top={self.trim_top_eigs}")
        return '|'.join(parts)

    def _append_fit_scope_to_filename(self, filename):
        if self.permuted_labels:
            filename += '_permuted_labels'
        if self.nonzero_eigs_only:
            filename += '_nonzero'
        if self.weight_type is not None:
            filename += f'_{self.weight_type}'
        if self.layer is not None:
            filename += f'_layer{self.layer}'
        if self.trim_bottom_eigs > 0:
            filename += f'_trimbottom{self.trim_bottom_eigs}'
        if self.trim_top_eigs > 0:
            filename += f'_trimtop{self.trim_top_eigs}'
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
                'trim_bottom_eigs': self.trim_bottom_eigs,
                'trim_top_eigs': self.trim_top_eigs,
                'permuted_labels': self.permuted_labels,
                'nonzero_eigs_only': self.nonzero_eigs_only,
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
            saved_permuted = saved_fit.get('permuted_labels', False)
            if bool(saved_permuted) != bool(self.permuted_labels):
                import warnings
                warnings.warn(
                    f"load_htmp_fit: fit file '{load_path}' was saved with "
                    f"permuted_labels={saved_permuted} but this ESDFit has "
                    f"permuted_labels={self.permuted_labels}. "
                    "Refusing legacy migration to avoid mixing normal and permuted-label fit parameters.",
                    stacklevel=2,
                )
                return False
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
                 beta_from_mean=False,
                 c=1.0,
                 beta_log_scale=True,
                 method='pdf'):
        if beta_from_tau and shared_beta_grid:
            raise ValueError("beta_from_tau cannot be combined with shared_beta_grid.")
        if beta_from_mean and shared_beta_grid:
            raise ValueError("beta_from_mean cannot be combined with shared_beta_grid.")
        if beta_from_tau and beta_from_mean:
            raise ValueError("beta_from_tau cannot be combined with beta_from_mean.")

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
            if beta_from_mean:
                print("Beta grid arguments are ignored: beta is set per epoch to kappa / (2 * gamma * mean(eigs)).")
            if method != 'pdf':
                print(f"HTMP fit method: {method}")

        if shared_beta_grid or shared_tau_grid:
            shared_beta_values = self._get_beta_grid(beta_min, beta_max, num_betas, beta_log_scale=beta_log_scale) if shared_beta_grid else [None]
            shared_tau_values = [None]
            best_shared_result = self._fit_shared_parameters(
                eigs_by_epoch,
                gamma_by_epoch,
                search_method,
                method,
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
                beta_from_mean=beta_from_mean,
                c=c,
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
                vectorized_grid,
                kappa_log_scale,
                moment_penalty_weight,
                self.inverse_htmp,
                extra_verbose,
                beta_from_tau=beta_from_tau,
                beta_from_mean=beta_from_mean,
                c=c,
                beta_log_scale=beta_log_scale,
                method=method
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
                       vectorized_grid=True,
                       kappa_log_scale=True,
                       moment_penalty_weight=0.0,
                       inverse=False, 
                       verbose=False,
                       beta_from_tau=False,
                       beta_from_mean=False,
                       c=1.0,
                       beta_log_scale=True,
                       method='pdf'):
        eigs = self.fileloader.get_eigs_epoch(epoch)
        gamma = self._get_gamma(epoch)
        epoch_result = self._fit_epoch_parameters(
            eigs,
            gamma,
            search_method,
            method,
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
            beta_from_mean=beta_from_mean,
            c=c,
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
                               single_axis=False,
                               x_log_scale=False,
                               fig_width=10,
                               fig_height=6,
                               fontsize=12,
                               ticksize=10,
                               legendsize=10,
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
            xlabel = r'$h = \kappa / \beta$'

        if x_log_scale:
            if np.any(xs <= 0):
                raise ValueError("x_log_scale=True requires all plotted x values to be positive.")

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        fig, ax_trace = plt.subplots(figsize=(fig_width, fig_height))
        ax_log_det = ax_trace if single_axis else ax_trace.twinx()
        ax_sum = ax_trace if single_axis and plot_sum else None
        if plot_sum and not single_axis:
            ax_sum = ax_trace.twinx()
            ax_sum.spines['right'].set_position(('outward', 80))

        trace_lines = []
        log_det_lines = []
        sum_lines = []
        trace_bands = []
        log_det_bands = []
        legend_lines = []

        trace_lines.extend(
            ax_trace.plot(
                xs,
                empirical_traces,
                marker='X',
                color='tab:blue',
                label='Trace'
            )
        )
        legend_lines.extend(trace_lines[:1])
        trace_bands.append(
            ax_trace.fill_between(
                xs,
                empirical_traces - empirical_trace_ci95,
                empirical_traces + empirical_trace_ci95,
                color='tab:blue',
                alpha=0.15,
                label='_nolegend_'
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
                    label='_nolegend_'
                )
            )

        log_det_lines.extend(
            ax_log_det.plot(
                xs,
                empirical_log_dets,
                marker='X',
                color='tab:green',
                label='Log-Determinant'
            )
        )
        legend_lines.extend(log_det_lines[:1])
        log_det_bands.append(
            ax_log_det.fill_between(
                xs,
                empirical_log_dets - empirical_log_det_ci95,
                empirical_log_dets + empirical_log_det_ci95,
                color='tab:green',
                alpha=0.15,
                label='_nolegend_'
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
                    label='_nolegend_'
                )
            )

        if plot_sum:
            sum_lines.extend(
                ax_sum.plot(
                    xs,
                    empirical_sums,
                    marker='X',
                    color='tab:red',
                    label='Trace + Log-Determinant'
                )
            )
            legend_lines.extend(sum_lines[:1])
            if plot_theoretical:
                sum_lines.extend(
                    ax_sum.plot(
                        xs,
                        theoretical_sums,
                        marker='o',
                        linestyle='--',
                        color='tab:red',
                        label='_nolegend_'
                    )
                )
            print(f"Log-Det at epoch {start_index}: empirical={empirical_log_dets[start_index]:.3g}")
            print(f"Min free energy epoch: {epochs_plotted[np.argmin(empirical_sums)]}, min empirical free energy: {np.min(empirical_sums):.3g}")

        ax_trace.set_xlabel(xlabel, fontsize=fontsize)
        if x_log_scale:
            ax_trace.set_xscale('log')
        if single_axis:
            ax_trace.set_ylabel('Value', fontsize=fontsize)
        else:
            ax_trace.set_ylabel('Trace', color='tab:blue', fontsize=fontsize)
            ax_log_det.set_ylabel('-Log-Determinant', color='tab:green', fontsize=fontsize)
        ax_trace.tick_params(axis='both', labelsize=ticksize)
        if single_axis:
            ax_trace.tick_params(axis='y', labelsize=ticksize)
        else:
            ax_log_det.tick_params(axis='y', labelcolor='tab:green', labelsize=ticksize)
            ax_trace.tick_params(axis='y', labelcolor='tab:blue', labelsize=ticksize)
        if plot_sum and not single_axis:
            ax_sum.set_ylabel('Free Energy', color='tab:red', fontsize=fontsize)
            ax_sum.tick_params(axis='y', labelcolor='tab:red', labelsize=ticksize)
            if title:
                ax_trace.set_title(f'Free Energy vs. {xlabel}', fontsize=fontsize)
        else:
            if title:
                ax_trace.set_title(f'Trace and Log-Determinant vs. {xlabel}', fontsize=fontsize)

        if legend_on:
            labels = [line.get_label() for line in legend_lines]
            plt.legend(legend_lines, labels, loc='best', fontsize=legendsize)

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
                                       c_lw=2,
                                       save_fig=True,
                                       show_fig=True,
                                       file_dir='figures/htmp_fits',
                                       texplot_enabled=False,
                                       axes=None,
                                       color_plot=False,
                                       plot_c=False,
                                       c_color='tab:red',
                                       c_label=r'$c_t=\tau_t\beta_t$',
                                       c_twin_axis=True,
                                       c_axis_label=r'$c=\tau\beta$',
                                       separate_plots=False,
                                       alignment='horizontal',
                                       vline_epochs=None,
                                       vline_width=1.0,
                                       lam=None,
                                       plot_h_over_h_star=False):
        epochs = self.epoch_list[start_index:]

        if plot_h_over_h_star:
            if lam is None:
                raise ValueError("lam must be provided when plot_h_over_h_star=True.")
            lam_val = float(lam)
            if not np.isfinite(lam_val) or lam_val == 0:
                raise ValueError("lam must be finite and non-zero when plot_h_over_h_star=True.")
        else:
            lam_val = None

        plot_rows = []
        skipped_epochs = []
        skipped_c_epochs = []
        skipped_h_ratio_epochs = []
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

            c_val = np.nan
            tau = self.tau_opt.get(epoch, None)
            if tau is None:
                skipped_c_epochs.append((epoch, 'missing tau'))
                tau_val = np.nan
            else:
                tau_val = self._as_float(tau)
                if np.isfinite(tau_val):
                    c_val = tau_val * beta_val
                else:
                    skipped_c_epochs.append((epoch, 'non-finite tau'))

            h_ratio_val = np.nan
            if plot_h_over_h_star:
                gamma_val = self._as_float(self._get_gamma(epoch))
                if not np.isfinite(gamma_val):
                    skipped_h_ratio_epochs.append((epoch, 'non-finite gamma'))
                elif not np.isfinite(tau_val):
                    skipped_h_ratio_epochs.append((epoch, 'invalid tau for c'))
                else:
                    h_val = kappa_val / beta_val
                    c_epoch = beta_val * tau_val
                    h_star = 2 * gamma_val * kappa_val / (lam_val * (kappa_val + 2 * gamma_val * c_epoch))
                    if np.isfinite(h_star) and h_star != 0:
                        h_ratio_val = h_val / h_star
                    else:
                        skipped_h_ratio_epochs.append((epoch, 'invalid h*'))

            plot_rows.append((epoch, kappa_val / beta_val, c_val, kappa_val, h_ratio_val))

        if len(plot_rows) == 0:
            print('No valid fitted κ/β ratios available to plot.')
            return

        x_epochs = np.array([row[0] for row in plot_rows])
        y_ratios = np.array([row[1] for row in plot_rows])
        y_cs = np.array([row[2] for row in plot_rows])
        y_kappas = np.array([row[3] for row in plot_rows])
        y_h_over_h_star = np.array([row[4] for row in plot_rows])
        valid_vline_epochs = []
        skipped_vline_epochs = []
        if vline_epochs is not None:
            for epoch_val in vline_epochs:
                try:
                    epoch_float = float(epoch_val)
                    if np.isfinite(epoch_float):
                        valid_vline_epochs.append(epoch_float)
                    else:
                        skipped_vline_epochs.append(epoch_val)
                except (TypeError, ValueError):
                    skipped_vline_epochs.append(epoch_val)

        if texplot_enabled:
            texplot.set_theme()
        else:
            texplot.reset_theme()

        if color_plot:
            color = color_plot
        else:
            color = 'tab:blue'

        if separate_plots:
            if alignment not in ['vertical', 'horizontal']:
                raise ValueError("alignment must be either 'vertical' or 'horizontal'.")

            n_subplots = 4 if plot_h_over_h_star else 3

            if axes is None:
                if alignment == 'vertical':
                    fig, axes_arr = plt.subplots(n_subplots, 1, figsize=(fig_width, fig_height), sharex=True)
                else:
                    fig, axes_arr = plt.subplots(1, n_subplots, figsize=(fig_width, fig_height), sharex=True)
            else:
                axes_arr = np.asarray(axes).reshape(-1)
                if axes_arr.size != n_subplots:
                    raise ValueError(f"When separate_plots=True, axes must contain exactly {n_subplots} matplotlib axes.")
                print(f"Plotting κ/β ratio, c, κ{', and h/h*' if plot_h_over_h_star else ''} vs. epoch on provided axes ({alignment}).")

            ax_ratio, ax_c, ax_kappa = axes_arr[:3]
            ax_h = axes_arr[3] if plot_h_over_h_star else None

            ax_ratio.plot(x_epochs, y_ratios, marker='o', linestyle='-', color=color, label=f'{self.model.upper()}')
            ax_ratio.set_ylabel('κ / β', fontsize=fontsize)
            ax_ratio.tick_params(axis='both')
            ax_ratio.legend(loc='best')

            finite_c_mask = np.isfinite(y_cs)
            if np.any(finite_c_mask):
                ax_c.plot(
                    x_epochs[finite_c_mask],
                    y_cs[finite_c_mask],
                    linestyle=':',
                    color=c_color,
                    label=c_label,
                    linewidth=c_lw,
                )
                ax_c.legend(loc='best')
            else:
                print('No valid fitted c values available to plot.')
            ax_c.set_ylabel(c_axis_label, fontsize=fontsize)
            ax_c.tick_params(axis='both')

            ax_kappa.plot(x_epochs, y_kappas, marker='o', linestyle='-', color='tab:green', label=r'$\kappa_t$')
            ax_kappa.set_ylabel('κ', fontsize=fontsize)
            ax_kappa.tick_params(axis='both')
            ax_kappa.legend(loc='best')

            if plot_h_over_h_star:
                finite_h_mask = np.isfinite(y_h_over_h_star)
                if np.any(finite_h_mask):
                    ax_h.plot(
                        x_epochs[finite_h_mask],
                        y_h_over_h_star[finite_h_mask],
                        marker='o',
                        linestyle='-',
                        color='tab:purple',
                        label=r'$h/h^*$',
                    )
                    ax_h.legend(loc='best')
                else:
                    print('No valid fitted h/h* values available to plot.')
                ax_h.set_ylabel(r'$h/h^*$', fontsize=fontsize)
                ax_h.tick_params(axis='both')

            if alignment == 'vertical':
                if plot_h_over_h_star:
                    ax_h.set_xlabel('Epoch', fontsize=fontsize)
                else:
                    ax_kappa.set_xlabel('Epoch', fontsize=fontsize)
            else:
                ax_ratio.set_xlabel('Epoch', fontsize=fontsize)
                ax_c.set_xlabel('Epoch', fontsize=fontsize)
                ax_kappa.set_xlabel('Epoch', fontsize=fontsize)
                if plot_h_over_h_star:
                    ax_h.set_xlabel('Epoch', fontsize=fontsize)

            if title:
                if axes is None:
                    fig.suptitle(f'{self.model.upper()}', fontweight='bold', fontsize=fontsize)
                else:
                    ax_ratio.set_title(f'{self.model.upper()}', fontweight='bold', fontsize=fontsize, y=0.98)

            for vline_epoch in valid_vline_epochs:
                ax_ratio.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                ax_c.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                ax_kappa.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                if plot_h_over_h_star:
                    ax_h.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
        else:
            if axes is None:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            else:
                ax = axes
                print(f"Plotting κ/β ratio vs. epoch on provided axes.")

            ax_c_twin = None
            ax.plot(x_epochs, y_ratios, marker='o', linestyle='-', color=color, label=f'{self.model.upper()}')
            if plot_c:
                finite_c_mask = np.isfinite(y_cs)
                if np.any(finite_c_mask):
                    if c_twin_axis:
                        ax_c_twin = ax.twinx()
                        ax_c_twin.plot(
                            x_epochs[finite_c_mask],
                            y_cs[finite_c_mask],
                            linestyle=':',
                            color=c_color,
                            label=c_label,
                        )
                        ax_c_twin.set_ylabel(c_axis_label, fontsize=fontsize, color=c_color)
                        ax_c_twin.tick_params(axis='y', labelcolor=c_color)
                        # Add an empty proxy line on the primary axis so ax.legend() includes c.
                        ax.plot([], [], linestyle=':', color=c_color, label=c_label, linewidth=c_lw, markersize=c_lw)
                    else:
                        ax.plot(
                            x_epochs[finite_c_mask],
                            y_cs[finite_c_mask],
                            linestyle=':',
                            color=c_color,
                            label=c_label,
                        )
                else:
                    print('No valid fitted c values available to plot.')

            ax.set_xlabel('Epoch', fontsize=fontsize)
            ax.set_ylabel('κ / β', fontsize=fontsize)
            ax.tick_params(axis='both')

            if title:
                ax.set_title(f'{self.model.upper()}', fontweight='bold', fontsize=fontsize, y=0.98)

            for vline_epoch in valid_vline_epochs:
                ax.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                if ax_c_twin is not None:
                    ax_c_twin.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)

        if skipped_epochs:
            skipped_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_epochs)
            print(f"Skipping epochs for κ/β plot: {skipped_epochs_str}")
        if (plot_c or separate_plots) and skipped_c_epochs:
            skipped_c_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_c_epochs)
            print(f"Skipping epochs for c plot: {skipped_c_epochs_str}")
        if plot_h_over_h_star and skipped_h_ratio_epochs:
            skipped_h_ratio_epochs_str = ', '.join(f'{epoch} ({reason})' for epoch, reason in skipped_h_ratio_epochs)
            print(f"Skipping epochs for h/h* plot: {skipped_h_ratio_epochs_str}")
        if skipped_vline_epochs:
            skipped_vlines_str = ', '.join(str(epoch) for epoch in skipped_vline_epochs)
            print(f"Skipping invalid vertical line epochs: {skipped_vlines_str}")
            

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            if axes is None:
                fig.tight_layout()
            if separate_plots:
                if plot_h_over_h_star:
                    plt.savefig(f'{plt_file}_kappa_beta_ratio_c_kappa_h_ratio_vs_epoch.pdf', bbox_inches='tight', format='pdf')
                else:
                    plt.savefig(f'{plt_file}_kappa_beta_ratio_c_kappa_vs_epoch.pdf', bbox_inches='tight', format='pdf')
            else:
                plt.savefig(f'{plt_file}_kappa_beta_ratio_vs_epoch.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        texplot.reset_theme()

        if separate_plots:
            return axes_arr
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
            ks_distance = None

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

            if kappa is not None:
                try:
                    ks_distance = self.compute_ks_distance_epoch(epoch_value)
                except Exception:
                    ks_distance = None

            ks_text = 'N/A' if ks_distance is None or not np.isfinite(ks_distance) else f'{ks_distance:.3f}'

            return (
                fr'$\kappa={kappa_text}$'
                + '\n'
                + fr'$\beta={beta_text}$'
                + '\n'
                + fr'$d_{{\text{{KS}}}}={ks_text}$'
            )
        
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
            ax.set_title(f'Epoch = {epoch}', fontsize=fontsize)
            
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
                    gamma = self._get_gamma(epoch)
                    mp_vals = marchenko_pastur_pdf(lmbda_vals, gamma)
                    # When only nonzero eigenvalues are shown the histogram integrates to 1,
                    # but the MP density integrates to gamma (<1); rescale to match.
                    # if self.nonzero_eigs_only and gamma < 1:
                    #     mp_vals = mp_vals / gamma
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
    
    def plot_train_acc_and_train_loss(self,
                                      start_index=0,
                                      log_scale_loss=False,
                                      interp_acc=None,
                                      fig_width=10,
                                      fig_height=6,
                                      fontsize=12,
                                      ticksize=10,
                                      legendsize=10,
                                      save_fig=True,
                                      show_fig=True,
                                      file_dir='figures/htmp_fits'):
        epochs = self.epoch_list[start_index:]
        if len(epochs) == 0:
            print('No epochs available to plot.')
            return

        train_accs = np.array([self.fileloader.get_train_acc_epoch(epoch) for epoch in epochs])
        train_losses = np.array([self.fileloader.get_train_loss_epoch(epoch) for epoch in epochs])

        fig, ax_acc = plt.subplots(figsize=(fig_width, fig_height))
        ax_loss = ax_acc.twinx()

        acc_line = ax_acc.plot(
            epochs, train_accs, marker='o', color='tab:red', label='Train Accuracy'
        )
        loss_line = ax_loss.plot(
            epochs, train_losses, marker='x', color='tab:blue', label='Train Loss'
        )

        ax_acc.set_xlabel('Epoch', fontsize=fontsize)
        ax_acc.set_ylabel('Train Accuracy', color='tab:red', fontsize=fontsize)
        ax_loss.set_ylabel('Train Loss', color='tab:blue', fontsize=fontsize)
        ax_acc.tick_params(axis='y', labelcolor='tab:red', labelsize=ticksize)
        ax_acc.tick_params(axis='x', labelsize=ticksize)
        ax_loss.tick_params(axis='y', labelcolor='tab:blue', labelsize=ticksize)
        ax_acc.set_title('Train Accuracy and Train Loss vs. Epoch', fontsize=fontsize)
        if log_scale_loss:
            ax_loss.set_yscale('log')

        lines = acc_line + loss_line
        if interp_acc is not None:
            reached = [e for e, a in zip(epochs, train_accs) if a >= interp_acc]
            if reached:
                interp_epoch = reached[0]
                vline = ax_acc.axvline(
                    x=interp_epoch, linestyle=':', color='grey',
                    label=f'Interpolation; Acc = {interp_acc}'
                )
                lines = lines + [vline]
                print(f'First epoch reaching train acc {interp_acc}: {interp_epoch}')
            else:
                print(f'Train accuracy never reached {interp_acc} in the plotted epochs.')

        labels = [line.get_label() for line in lines]
        ax_acc.legend(lines, labels, loc='center right', fontsize=legendsize)

        fig.tight_layout()

        plt_file = self._get_figure_file_stem(file_dir)
        os.makedirs(os.path.dirname(plt_file), exist_ok=True)
        if save_fig:
            plt.savefig(f'{plt_file}_train_acc_and_train_loss_vs_epoch.pdf', bbox_inches='tight', format='pdf')
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
                          ticksize=10,
                          legendsize=10,
                          legendbox=True,
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
                          legend_on=True,
                          tight_layout=True,
                          acc_key='test_acc',
                          vline_epochs=None,
                          vline_width=1.0):
        if self.fileloader._is_moonlight_model():
            raise NotImplementedError("Moonlight models do not yet support plotting test accuracy against free energy.")

        epochs = self.epoch_list[start_index:]
        if any(epoch not in self.free_energies for epoch in epochs):
            print("Please run .compute_free_energy() before plotting.")
            return

        _acc_key_labels = {
            'test_acc': 'Test Accuracy',
            'permuted_test_acc': 'Permuted Test Accuracy',
            'test_loss': 'Test Cross-Entropy',
            'permuted_test_loss': 'Permuted Test Cross-Entropy',
        }
        acc_label = _acc_key_labels.get(acc_key, acc_key)

        test_acc_stats = [
            self.fileloader.get_test_acc_epoch_stats(
                epoch,
                task=task,
                metric=metric,
                shot=shot,
                evals_root=evals_root,
                suite=suite,
                deduped=deduped,
                acc_key=acc_key,
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

        valid_vline_epochs = []
        skipped_vline_epochs = []
        if vline_epochs is not None:
            for epoch_val in vline_epochs:
                try:
                    epoch_float = float(epoch_val)
                    if np.isfinite(epoch_float):
                        valid_vline_epochs.append(epoch_float)
                    else:
                        skipped_vline_epochs.append(epoch_val)
                except (TypeError, ValueError):
                    skipped_vline_epochs.append(epoch_val)

        

        # corr = np.corrcoef(test_accs, free_energies)[0, 1]
        corr = compute_correlation(test_accs, free_energies, method='spearman')
        kendall_tau, kendall_p = compute_correlation(test_accs, free_energies, method='kendall')
        pearson_r = compute_correlation(test_accs, free_energies, method='pearson')
        print(f'---- Corr ----')
        print(f'SpearmanCorr: {corr:.3f}')
        print(f'Kendall Tau: {kendall_tau:.3f}, p-value: {kendall_p}')
        print(f'Pearson r: {pearson_r:.3f}')

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
            ax1.set_ylabel(acc_label, fontsize=fontsize)
            ax2.set_ylabel('Free Energy', fontsize=fontsize)
            ax1.tick_params(axis='y', labelsize=ticksize)
            ax1.tick_params(axis='x', labelsize=ticksize)
            ax2.tick_params(axis='y', labelsize=ticksize)
            ax2.tick_params(axis='x', labelsize=ticksize)
            # Only have 5 ticks for x and y axis
            ax1.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=5)
            ax2.locator_params(axis='y', nbins=5)
            if title:
                ax1.set_title(fr'{self.model.upper()} ($\rho$: {corr:.3f}, $\lambda$: {self.lam:.2f})', fontweight='bold', fontsize=fontsize)

            if log_x:
                ax1.set_xscale('log')
                ax2.set_xscale('log')

            if optimal_ratio is not None:
                # Place a marker on the free energy at the optimal kappa_beta_ratio, if provided
                ax2.scatter(
                    optimal_ratio,
                    theoretical_free_energies[np.nanargmin(np.abs(np.array(xs, dtype=float) - optimal_ratio))],
                    color='purple',
                    label=f'Optimal Ratio = {optimal_ratio:.3f}',
                    marker='X',
                    s=marker_size,
                    zorder=5,
                )

            for vline_epoch in valid_vline_epochs:
                ax1.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                ax2.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)

            lines = test_line + [test_band] + free_line + [free_band] + theoretical_line
            labels = [line.get_label() for line in test_line] + [test_band.get_label()] + [line.get_label() for line in free_line] + [free_band.get_label()] + [line.get_label() for line in theoretical_line]
            if legend_on:
                ax1.legend(lines, labels, loc='best', fontsize=legendsize, frameon=legendbox)
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
            ax1.set_ylabel(acc_label)
            ax1.set_xlabel(xlabel)
            ax1.tick_params(axis='y', labelsize=ticksize)
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

            for vline_epoch in valid_vline_epochs:
                ax1.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)
                ax2.axvline(vline_epoch, linestyle=':', color='black', linewidth=vline_width)

            fig.suptitle(f'{self.model.upper()} ($r$: {corr:.3f})', fontweight='bold', fontsize=fontsize, y=0.94)

        if skipped_vline_epochs:
            skipped_vlines_str = ', '.join(str(epoch) for epoch in skipped_vline_epochs)
            print(f"Skipping invalid vertical line epochs: {skipped_vlines_str}")

        plt_file = self._get_figure_file_stem(file_dir)
        if self.weight_inverse:
            plt_file += '_weight_inverse'

        os.makedirs(os.path.dirname(plt_file), exist_ok=True)

        # Write correlation and max test accuracy info to a text file
        with open(f'{plt_file}_test_vs_free_energy_vs_{variable}.txt', 'w') as f:
            f.write(f"Correlation between {acc_label} and {'-Free Energy' if neg_free else 'Free Energy'}: {corr:.3f}\n")
            f.write(f"Spearman Correlation: {corr:.3f}\n")
            f.write(f"Kendall Tau: {kendall_tau:.3f}, p-value: {kendall_p}\n")
            f.write(f"Pearson r: {pearson_r:.3f}\n")
            try:
                f.write(f"Maximum {acc_label}: {np.max(test_accs):.3f} at {xlabel}={xs[np.argmax(test_accs)]:.3g}, Free Energy={free_energies[np.argmax(test_accs)]:.5f}\n")
            except Exception as e:
                f.write(f"Error computing maximum {acc_label}: {e}\n")

        if tight_layout:
            fig.tight_layout()
        if save_fig:
            plt.savefig(f'{plt_file}_test_vs_free_energy_vs_{variable}.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()

        print(f"Correlation between {acc_label} and {'-Free Energy' if neg_free else 'Free Energy'}: {corr:.3f}")
        print(f'Maximum {acc_label}: {np.max(test_accs):.3f} at {xlabel}={xs[np.argmax(test_accs)]:.3g}, Free Energy={free_energies[np.argmax(test_accs)]:.5f}')
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

    def _plot_lambda_cv_seed(self, seed_idx, T_s, h_s, h_star, t_pred, val_s, test_s,
                             figure_dir='figures/htmp_fits', save_fig=True, show_fig=False):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        h_vals   = [h_s[t]               for t in T_s]
        val_vals = [val_s.get(t, np.nan)  for t in T_s]
        te_vals  = [test_s.get(t, np.nan) for t in T_s]

        ax1.plot(T_s, h_vals, 'b-o', markersize=4, label='$h_t$')
        ax1.axhline(h_star, color='cornflowerblue', linestyle='--', label='$h^*$')
        ax1.set_ylabel(r'$h = \kappa / \beta$', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.plot(T_s, val_vals, 'r-o', markersize=4, label='Val acc')
        ax2.plot(T_s, te_vals,  'g-o', markersize=4, label='Test acc')
        ax2.set_ylabel('Accuracy')

        if t_pred is not None:
            ax1.axvline(t_pred, color='purple', linestyle=':', linewidth=1.5,
                        label=f'Predicted stop (epoch {t_pred})')

        ax1.set_xlabel('Epoch')
        ax1.set_title(rf'Seed {seed_idx} — $\lambda$-CV stopping rule')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        fig.tight_layout()

        os.makedirs(figure_dir, exist_ok=True)
        stem = self._get_figure_file_stem(figure_dir)
        if save_fig:
            plt.savefig(f'{stem}_lambda_cv_seed{seed_idx}.pdf', bbox_inches='tight', format='pdf')
        if show_fig:
            plt.show()
        plt.close(fig)

    def lambda_cv(
        self,
        c,
        kappa,
        min_epoch,
        lambda_grid=None,
        save_dir='results/htmp/cv',
        figure_dir='figures/htmp_fits',
        save_fig=True,
        show_fig=False,
    ):
        """Leave-one-seed-out cross-validation for lambda.

        Separates self.fileloader.files by seed (the repeat index in each filename),
        fits HTMP quantities per seed per epoch (tau = min eigenvalue, beta = c/tau,
        h = kappa*tau/c), then runs leave-one-seed-out CV over all epochs >= min_epoch.

        Args:
            c: architecture constant relating tau and beta (beta = c / tau).
            kappa: fixed kappa value used in the stopping-rule formula.
            min_epoch: lower bound (inclusive) on the epoch window.
            lambda_grid: 1-D array of candidate lambda values. Defaults to logspace(-3, 3, 61).
            save_dir: directory for output CSV files.
            figure_dir: directory for per-seed plots.
            save_fig: whether to save plots to disk.
            show_fig: whether to display plots interactively.

        Returns:
            (results_df, grid_df): DataFrames also saved as CSV under save_dir.
        """
        import pandas as pd

        if lambda_grid is None:
            lambda_grid = np.logspace(-3, 3, 61)
        lambda_grid = np.asarray(lambda_grid, dtype=float)

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        # ---- Separate checkpoint files by seed (repeat index) ----
        def _seed_of(file):
            m = re.search(r'_(\d+)_(\d+)(?:_layer\d+)?$', Path(file).stem)
            return int(m.group(2)) if m else None

        seed_to_files = {}
        for f in self.fileloader.files:
            s = _seed_of(f)
            if s is not None:
                seed_to_files.setdefault(s, []).append(f)

        if len(seed_to_files) < 2:
            raise ValueError("lambda_cv requires at least 2 seeds in the loaded files.")

        epochs = sorted(t for t in self.epoch_list if t >= min_epoch)
        if not epochs:
            raise ValueError(f"No epochs >= min_epoch={min_epoch}.")

        gamma = self._get_gamma(epochs[0])

        # ---- Fit HTMP quantities per seed per epoch ----
        # h_{r,t} = kappa * tau_{r,t} / c  where tau_{r,t} = min eigenvalue for seed r at epoch t
        seed_data = {}
        for seed, seed_files in sorted(seed_to_files.items()):
            h_rt      = {}
            val_accs  = {}
            test_accs = {}

            for t in epochs:
                t_files = get_key_files(seed_files, t)
                if not t_files:
                    continue

                eigs_parts = []
                for f in t_files:
                    try:
                        eigs_parts.append(self.fileloader._get_eigvals_from_file(f))
                    except Exception:
                        pass
                if not eigs_parts:
                    continue
                eigs = self.fileloader._trim_aggregated_eigs(np.concatenate(eigs_parts))

                tau = self._get_epoch_tau_from_eigs(eigs)
                if not (np.isfinite(tau) and tau > 0):
                    continue
                h_rt[t] = kappa * tau / c

                for acc_key, acc_dict in [('val_acc', val_accs), ('test_acc', test_accs)]:
                    try:
                        vals = get_test_acc(seed_files, t, acc_key=acc_key)
                        acc_dict[t] = float(np.mean(vals)) if len(vals) > 0 else np.nan
                    except Exception:
                        acc_dict[t] = np.nan

            if h_rt:
                seed_data[seed] = {'h': h_rt, 'val_accs': val_accs, 'test_accs': test_accs}

        valid_seeds = sorted(seed_data.keys())
        if len(valid_seeds) < 2:
            raise ValueError("lambda_cv requires at least 2 seeds with valid data.")

        # h*(lambda) = 2*gamma*kappa / (lambda*(kappa + 2*gamma*c)) — same for all seeds
        results_rows = []
        grid_rows    = []

        for s in valid_seeds:
            cal_seeds = [r for r in valid_seeds if r != s]
            T_s = sorted(seed_data[s]['h'].keys())

            n_lam       = len(lambda_grid)
            n_cal       = len(cal_seeds)
            per_regret  = np.full((n_lam, n_cal), np.nan)
            mean_regret = np.full(n_lam, np.nan)

            for li, lam in enumerate(lambda_grid):
                h_star = 2.0 * gamma * kappa / (lam * (kappa + 2.0 * gamma * c))
                for ci, r in enumerate(cal_seeds):
                    T_r   = sorted(seed_data[r]['h'].keys())
                    h_r   = seed_data[r]['h']
                    val_r = seed_data[r]['val_accs']

                    t_stop = next((t for t in T_r if h_r[t] <= h_star), None)
                    if t_stop is None:
                        t_stop = min(T_r, key=lambda t, _h=h_r, _hs=h_star: abs(_h[t] - _hs))

                    valid_vals = [val_r.get(t, np.nan) for t in T_r
                                  if np.isfinite(val_r.get(t, np.nan))]
                    val_stop = val_r.get(t_stop, np.nan)
                    if valid_vals and np.isfinite(val_stop):
                        per_regret[li, ci] = max(valid_vals) - val_stop

                valid = per_regret[li][np.isfinite(per_regret[li])]
                if valid.size:
                    mean_regret[li] = float(valid.mean())

            finite_mask = np.isfinite(mean_regret)
            best_li = int(np.where(finite_mask)[0][np.argmin(mean_regret[finite_mask])]) if finite_mask.any() else 0
            best_lam = float(lambda_grid[best_li])

            for li, lam in enumerate(lambda_grid):
                for ci, r in enumerate(cal_seeds):
                    grid_rows.append({
                        'held_out_seed':    s,
                        'calibration_seed': r,
                        'lambda':           lam,
                        'regret':           per_regret[li, ci],
                        'mean_regret':      mean_regret[li],
                    })

            # ---- Evaluate held-out seed ----
            h_star_s = 2.0 * gamma * kappa / (best_lam * (kappa + 2.0 * gamma * c))
            h_s   = seed_data[s]['h']
            val_s = seed_data[s]['val_accs']
            test_s = seed_data[s]['test_accs']

            true_crossing = False
            t_pred = next((t for t in T_s if h_s[t] <= h_star_s), None)
            if t_pred is not None:
                true_crossing = True
            else:
                t_pred = min(T_s, key=lambda t, _h=h_s, _hs=h_star_s: abs(_h[t] - _hs))

            def _best(acc_dict, _T=T_s):
                cands = {t: acc_dict.get(t, np.nan) for t in _T
                         if np.isfinite(acc_dict.get(t, np.nan))}
                return (max(cands, key=cands.get), max(cands.values())) if cands else (np.nan, np.nan)

            best_val_epoch,  max_val  = _best(val_s)
            best_test_epoch, max_test = _best(test_s)
            val_at_pred  = val_s.get(t_pred, np.nan)
            test_at_pred = test_s.get(t_pred, np.nan)
            val_regret   = (max_val  - val_at_pred)  if (np.isfinite(max_val)  and np.isfinite(val_at_pred))  else np.nan
            test_regret  = (max_test - test_at_pred) if (np.isfinite(max_test) and np.isfinite(test_at_pred)) else np.nan

            # Spearman: -F_t^{best_lam} vs. val/test acc over T_s
            free_s = {}
            for t in T_s:
                t_files = get_key_files(seed_to_files[s], t)
                eigs_parts = []
                for f in t_files:
                    try:
                        eigs_parts.append(self.fileloader._get_eigvals_from_file(f))
                    except Exception:
                        pass
                if eigs_parts:
                    eigs = np.concatenate(eigs_parts)
                    try:
                        F, _, _ = self.empirical_free_energy_eigs(eigs, len(eigs), lam=best_lam)
                        free_s[t] = float(F)
                    except Exception:
                        pass

            corr_epochs = [t for t in T_s
                           if np.isfinite(free_s.get(t, np.nan)) and np.isfinite(val_s.get(t, np.nan))]
            spearman_val = spearman_test = np.nan
            if len(corr_epochs) >= 3:
                neg_F   = [-free_s[t]            for t in corr_epochs]
                v_vals  = [val_s[t]              for t in corr_epochs]
                te_vals = [test_s.get(t, np.nan) for t in corr_epochs]
                spearman_val = compute_correlation(neg_F, v_vals, method='spearman')
                if all(np.isfinite(v) for v in te_vals):
                    spearman_test = compute_correlation(neg_F, te_vals, method='spearman')

            results_rows.append({
                'held_out_seed':    s,
                'lambda':           best_lam,
                'kappa':            kappa,
                'h_star':           h_star_s,
                'predicted_epoch':  t_pred,
                'true_crossing':    true_crossing,
                'best_val_epoch':   best_val_epoch,
                'best_test_epoch':  best_test_epoch,
                'val_acc_at_pred':  val_at_pred,
                'test_acc_at_pred': test_at_pred,
                'max_val_acc':      max_val,
                'max_test_acc':     max_test,
                'val_regret':       val_regret,
                'test_regret':      test_regret,
                'spearman_val':     spearman_val,
                'spearman_test':    spearman_test,
            })

            self._plot_lambda_cv_seed(
                seed_idx=s, T_s=T_s, h_s=h_s, h_star=h_star_s,
                t_pred=t_pred, val_s=val_s, test_s=test_s,
                figure_dir=figure_dir, save_fig=save_fig, show_fig=show_fig,
            )

        results_df = pd.DataFrame(results_rows)
        grid_df    = pd.DataFrame(grid_rows)
        results_df.to_csv(os.path.join(save_dir, 'lambda_seed_cv_results.csv'), index=False)
        grid_df.to_csv(   os.path.join(save_dir, 'lambda_seed_cv_grid.csv'),    index=False)
        return results_df, grid_df


class fileloading(object):
    def __init__(self, model='minialexnet', dataset='cifar10', kernel='weight', batch_size=None, subsample=False, file_dir = 'results/htmp/epoch', weight_type=None, layer=None, trim_bottom_eigs=0, trim_top_eigs=0, permuted_labels=False, nonzero_eigs_only=False):
        self.model = model
        self.dataset = dataset
        self.kernel = kernel
        self.batch_size = batch_size
        self.subsample = subsample
        self.file_dir = file_dir
        self.weight_type = weight_type
        self.layer = layer
        self.trim_bottom_eigs = self._normalize_trim_count(trim_bottom_eigs, 'trim_bottom_eigs')
        self.trim_top_eigs = self._normalize_trim_count(trim_top_eigs, 'trim_top_eigs')
        self.permuted_labels = permuted_labels
        self.nonzero_eigs_only = nonzero_eigs_only

        self.filename = f'{model}_{dataset}'
        if subsample:
            self.filename += '_subsampled'
        if permuted_labels:
            self.filename += '_permuted_labels'
        self.filename += f'_b{batch_size}'
        if not subsample:
            self.files = [f for f in sorted(glob.glob(f'{self.file_dir}/{self.filename}*.pt'), key=extract_key, reverse=True) if '_subsample' not in f]
        else:
            self.files = [f for f in sorted(glob.glob(f'{self.file_dir}/{self.filename}*.pt'), key=extract_key, reverse=True) if '_subsample' in f]
        # Ensure permuted-label and normal files are never mixed regardless of directory layout.
        if permuted_labels:
            self.files = [f for f in self.files if '_permuted_labels' in Path(f).stem]
        else:
            self.files = [f for f in self.files if '_permuted_labels' not in Path(f).stem]
        self.files = self._filter_files_for_scope(self.files)
        self.epoch_list = sorted(get_keys(self.files), reverse=False)
        self._pythia_eval_cache = {}

    def get_epoch_list(self):
        return self.epoch_list

    def _normalize_trim_count(self, value, name):
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        return value

    def _has_trimmed_spectrum(self):
        return self.trim_bottom_eigs > 0 or self.trim_top_eigs > 0

    def _trim_aggregated_eigs(self, eigs):
        eigs = np.asarray(eigs, dtype=float)
        total_trim = self.trim_bottom_eigs + self.trim_top_eigs
        if total_trim == 0:
            return eigs
        if total_trim >= eigs.size:
            raise ValueError(
                f"Cannot trim {self.trim_bottom_eigs} bottom and {self.trim_top_eigs} top eigenvalues from an array of size {eigs.size}."
            )

        sorted_eigs = np.sort(eigs)
        end_index = sorted_eigs.size - self.trim_top_eigs if self.trim_top_eigs > 0 else sorted_eigs.size
        return sorted_eigs[self.trim_bottom_eigs:end_index]

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

        candidates = [self.layer]
        if not isinstance(self.layer, str):
            candidates.append(str(self.layer))
        if self._is_moonlight_model() and isinstance(self.layer, numbers.Integral):
            legacy_layer = self.layer - 1
            candidates.extend([legacy_layer, str(legacy_layer)])

        ordered_keys = self._sorted_layer_keys(layer_dict)
        if isinstance(self.layer, numbers.Integral) and 0 <= int(self.layer) < len(ordered_keys):
            candidates.append(ordered_keys[int(self.layer)])

        for candidate in candidates:
            if candidate in layer_dict:
                return candidate

        available = ', '.join(map(str, ordered_keys))
        raise KeyError(f"Layer {self.layer} not found under {self.weight_type}. Available layers: {available}")

    def _sorted_layer_keys(self, layer_dict):
        keys = list(layer_dict.keys())
        if all(str(key).isdigit() for key in keys):
            return sorted(keys, key=lambda x: int(x))
        return sorted(keys, key=str)

    def _load_reference_result_dict(self):
        if len(self.files) == 0:
            raise FileNotFoundError(f'No checkpoint files found for pattern {self.filename} in {self.file_dir}.')
        return torch.load(self.files[0], weights_only=False, map_location='cpu')

    def get_available_layers(self):
        res_dict = self._load_reference_result_dict()

        if 'weight_spectra_wtw' in res_dict:
            layer_dict = self._filter_weight_spectra_layers(res_dict['weight_spectra_wtw'])
            return [str(key) for key in self._sorted_layer_keys(layer_dict)]

        if self.weight_type in ('Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2') and self.weight_type in res_dict:
            return [str(key) for key in self._sorted_layer_keys(res_dict[self.weight_type])]

        return []

    def _filter_weight_spectra_layers(self, layer_dict):
        if self.weight_type is None:
            return layer_dict

        if self.weight_type == 'FC':
            filtered = {key: value for key, value in layer_dict.items() if key == 'linear' or key.startswith('fc')}
        elif self.weight_type == 'Conv2':
            filtered = {key: value for key, value in layer_dict.items() if key.endswith('conv2') or key == 'conv2'}
        else:
            raise ValueError(
                "weight_type must be None, 'FC', or 'Conv2' for checkpoints saved with weight_spectra_wtw. "
                "Use layer='<module_name>' to select a specific ResNet layer."
            )

        if len(filtered) == 0:
            raise KeyError(f"No layers matched weight_type={self.weight_type} in saved weight_spectra_wtw.")
        return filtered

    def _get_weight_spectra_wtw_eigvals(self, res_dict):
        if 'weight_spectra_wtw' not in res_dict:
            return None

        layer_dict = self._filter_weight_spectra_layers(res_dict['weight_spectra_wtw'])
        if self.layer is None:
            ordered_keys = self._sorted_layer_keys(layer_dict)
            return np.concatenate([np.asarray(layer_dict[key]['eigenvalues'], dtype=float) for key in ordered_keys])

        layer_key = self._resolve_saved_layer_key(layer_dict)
        return np.asarray(layer_dict[layer_key]['eigenvalues'], dtype=float)

    def _get_weight_spectra_wtw_aspect_ratio(self, res_dict):
        if 'weight_spectra_wtw' not in res_dict:
            return None

        layer_dict = self._filter_weight_spectra_layers(res_dict['weight_spectra_wtw'])
        if self.layer is None:
            raise ValueError("layer must be set to load a saved aspect ratio from weight_spectra_wtw.")

        layer_key = self._resolve_saved_layer_key(layer_dict)
        return float(layer_dict[layer_key]['aspect_ratio'])

    def _get_nonzero_eig_count(self, eigvals, aspect_ratio):
        eigvals = np.asarray(eigvals, dtype=float)
        if eigvals.ndim != 1:
            raise ValueError(f"Expected a 1D eigenvalue array, got shape {eigvals.shape}.")

        gamma = float(aspect_ratio)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"Expected a positive finite aspect ratio, got {aspect_ratio}.")

        total_count = eigvals.size
        if gamma <= 1:
            return total_count

        nonzero_count = int(round(total_count / gamma))
        return max(1, min(total_count, nonzero_count))

    def _get_effective_aspect_ratio(self, aspect_ratio):
        gamma = float(aspect_ratio)
        if not self.nonzero_eigs_only:
            return gamma
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"Expected a positive finite aspect ratio, got {aspect_ratio}.")
        return gamma if gamma <= 1 else 1.0 / gamma

    def _keep_nonzero_eigvals(self, eigvals, res_dict):
        if not self.nonzero_eigs_only:
            return np.asarray(eigvals, dtype=float)
        if self.layer is None:
            raise ValueError("nonzero_eigs_only requires layer to be set so the saved aspect ratio can be resolved.")

        aspect_ratio = self._get_new_format_aspect_ratio(res_dict)
        if aspect_ratio is None:
            raise ValueError("nonzero_eigs_only requires checkpoints with saved aspect ratios.")

        eigvals = np.sort(np.asarray(eigvals, dtype=float))
        nonzero_count = self._get_nonzero_eig_count(eigvals, aspect_ratio)
        return eigvals[-nonzero_count:]

    def _get_new_format_eigvals(self, res_dict):
        weight_spectra_eigs = self._get_weight_spectra_wtw_eigvals(res_dict)
        if weight_spectra_eigs is not None:
            return weight_spectra_eigs

        if self.weight_type is None:
            return None

        if self.weight_type == 'EmbedOut':
            return np.asarray(res_dict['EmbedOut']['eigvals'], dtype=float)

        if self.weight_type == 'FC':
            return np.asarray(res_dict['FC']['eigvals'], dtype=float)

        if self.weight_type not in ('Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2'):
            raise ValueError("weight_type must be one of None, 'Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2', 'FC', or 'EmbedOut'.")

        layer_dict = res_dict[self.weight_type]
        if self.layer is None:
            ordered_keys = sorted(layer_dict.keys(), key=lambda x: int(x))
            return np.concatenate([np.asarray(layer_dict[k]['eigvals'], dtype=float) for k in ordered_keys])

        layer_key = self._resolve_saved_layer_key(layer_dict)
        return np.asarray(layer_dict[layer_key]['eigvals'], dtype=float)

    def _get_new_format_aspect_ratio(self, res_dict):
        weight_spectra_aspect_ratio = self._get_weight_spectra_wtw_aspect_ratio(res_dict)
        if weight_spectra_aspect_ratio is not None:
            return weight_spectra_aspect_ratio

        if self.weight_type is None:
            raise ValueError("weight_type must be set to load a saved aspect ratio from a checkpoint summary.")

        if self.weight_type == 'EmbedOut':
            return float(res_dict['EmbedOut']['aspect_ratio'])

        if self.weight_type == 'FC':
            return float(res_dict['FC']['aspect_ratio'])

        if self.weight_type not in ('Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2'):
            raise ValueError("weight_type must be one of 'Dense', 'QueryKey', 'KeyValue', 'Query', 'Conv2', 'FC', or 'EmbedOut'.")

        if self.layer is None:
            raise ValueError(f"layer must be set to load a saved aspect ratio for {self.weight_type}.")

        layer_dict = res_dict[self.weight_type]
        layer_key = self._resolve_saved_layer_key(layer_dict)
        return float(layer_dict[layer_key]['aspect_ratio'])

    def _get_eigvals_from_file(self, file):
        res_dict = torch.load(file, weights_only=False, map_location='cpu')

        new_format_eigs = self._get_new_format_eigvals(res_dict)
        if new_format_eigs is not None:
            return self._keep_nonzero_eigvals(new_format_eigs, res_dict)

        eig_key = {
            'weight': 'weight_eigvals',
            'ck': 'ck_eigvals',
            'ntk': 'ntk_eigvals',
        }[self.kernel]
        return self._keep_nonzero_eigvals(np.asarray(res_dict[eig_key], dtype=float), res_dict)

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
        eigs = np.concatenate(eigvals_by_repeat)
        return self._trim_aggregated_eigs(eigs)

    def get_aspect_ratio_epoch(self, epoch):
        key_files = get_key_files(self.files, epoch)
        if len(key_files) == 0:
            return np.nan

        aspect_ratios = [self._get_aspect_ratio_from_file(file) for file in key_files]
        first_aspect_ratio = float(aspect_ratios[0])
        if not all(np.isclose(first_aspect_ratio, aspect_ratio) for aspect_ratio in aspect_ratios[1:]):
            raise ValueError(f"Found inconsistent saved aspect ratios for epoch {epoch}.")
        return self._get_effective_aspect_ratio(first_aspect_ratio)

    def get_eigs_epoch_values(self, epoch):
        key_files = get_key_files(self.files, epoch)
        eigvals_by_repeat = [self._get_eigvals_from_file(file) for file in key_files]
        if not self._has_trimmed_spectrum():
            return eigvals_by_repeat
        if len(eigvals_by_repeat) == 0:
            return []
        return [self._trim_aggregated_eigs(np.concatenate(eigvals_by_repeat))]

    def get_test_acc_epoch_values(self, epoch, acc_key='test_acc'):
        return np.asarray(get_test_acc(self.files, epoch, acc_key=acc_key), dtype=float)

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
        return self.weight_type is not None or self._is_pythia_model() or self._is_moonlight_model() or self._is_optimizer_resnet_model()

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
                                 deduped=None,
                                 acc_key='test_acc'):
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

        accuracies = self.get_test_acc_epoch_values(epoch, acc_key=acc_key)
        if accuracies.size == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci95': np.nan, 'count': 0}

        mean = float(np.mean(accuracies))
        std = float(np.std(accuracies, ddof=1)) if accuracies.size > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(accuracies.size)) if accuracies.size > 1 else 0.0
        return {'mean': mean, 'std': std, 'ci95': ci95, 'count': int(accuracies.size)}
    
    def get_test_acc_epoch(self, epoch):
        return self.get_test_acc_epoch_stats(epoch)['mean']

    def get_test_loss_epoch_values(self, epoch, loss_key='test_loss'):
        return np.asarray(get_test_acc(self.files, epoch, acc_key=loss_key), dtype=float)

    def get_test_loss_epoch_stats(self,
                                  epoch,
                                  task='arc_easy',
                                  metric='acc',
                                  shot='zero-shot',
                                  evals_root='evals',
                                  suite='pythia-v1',
                                  deduped=None,
                                  loss_key='test_loss'):
        return self.get_test_acc_epoch_stats(
            epoch,
            task=task,
            metric=metric,
            shot=shot,
            evals_root=evals_root,
            suite=suite,
            deduped=deduped,
            acc_key=loss_key,
        )

    def get_test_loss_epoch(self, epoch):
        return self.get_test_loss_epoch_stats(epoch)['mean']

    def get_test_loss_epoch_all(self, epoch):
        return self.get_test_loss_epoch_values(epoch)
    
    def get_train_loss_epoch(self, epoch):
        return get_train_loss(self.files, epoch).mean().item()

    def get_train_loss_epoch_all(self, epoch):
        return get_train_loss(self.files, epoch)

    def get_train_acc_epoch(self, epoch):
        return get_train_acc(self.files, epoch).mean().item()

    def get_train_acc_epoch_all(self, epoch):
        return get_train_acc(self.files, epoch)

    


    
    




        