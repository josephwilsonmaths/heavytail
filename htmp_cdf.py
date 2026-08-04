import argparse
import time

import mpmath
import numpy as np
from scipy import special as scipy_special

from helpers import htmp


def _as_1d_float_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _standard_beta(kappa, gamma):
    return float(kappa) / (2.0 * float(gamma))


def htmp_parameters(kappa, gamma, beta):
    kappa = float(kappa)
    gamma = float(gamma)
    beta = float(beta)
    a = 0.5 * kappa
    c = 0.5 * kappa / gamma
    delta = c - a
    b = 1.0 - delta
    return a, b, c, delta


def _validate_htmp_grid(points):
    points = _as_1d_float_array(points)
    if points.size == 0:
        return points
    if not np.all(np.isfinite(points)):
        raise ValueError("x_grid must be finite.")
    if np.any(points < 0.0):
        raise ValueError("x_grid must be nonnegative.")
    if np.any(np.diff(points) < 0.0):
        raise ValueError("x_grid must be sorted in increasing order.")
    return points


def htmp_boundary_fast(x, kappa, gamma, beta, integer_tolerance=1e-7):
    x = _as_1d_float_array(x)
    if x.size == 0:
        return np.array([], dtype=np.complex128)
    if np.any(~np.isfinite(x)):
        raise ValueError("x must be finite.")
    if np.any(x < 0.0):
        raise ValueError("x must be nonnegative.")

    a, _, _, delta = htmp_parameters(kappa, gamma, beta)
    m = int(round(delta))
    if m >= 1 and abs(delta - m) <= integer_tolerance:
        return htmp_boundary_integer(x, kappa, gamma, beta, m)

    sin_delta = np.sin(np.pi * delta)
    if abs(sin_delta) < integer_tolerance:
        raise ValueError(
            "delta is too close to an integer for the fast HTMP connection formula."
        )

    t = float(beta) * x
    a_term = scipy_special.hyp1f1(a, 1.0 - delta, -t) * scipy_special.rgamma(a + delta) * scipy_special.rgamma(1.0 - delta)
    b_term = np.power(t, delta) * scipy_special.hyp1f1(a + delta, 1.0 + delta, -t) * scipy_special.rgamma(a) * scipy_special.rgamma(1.0 + delta)

    real_part = (a_term - np.cos(np.pi * delta) * b_term) / sin_delta
    imag_part = b_term
    scale = np.maximum(np.abs(real_part), np.abs(imag_part))
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.pi * (real_part / scale + 1j * imag_part / scale)


def htmp_boundary_integer(x, kappa, gamma, beta, m, max_terms=512, tol=1e-14):
    x = _as_1d_float_array(x)
    if x.size == 0:
        return np.array([], dtype=np.complex128)
    if np.any(~np.isfinite(x)):
        raise ValueError("x must be finite.")
    if np.any(x < 0.0):
        raise ValueError("x must be nonnegative.")

    a = 0.5 * float(kappa)
    t = float(beta) * x

    P = np.zeros_like(t, dtype=float)
    S = np.zeros_like(t, dtype=float)
    I = np.zeros_like(t, dtype=float)

    positive_mask = t > 0.0
    if np.any(positive_mask):
        tp = t[positive_mask]
        log_prefactor = (
            m * np.log(tp)
            - scipy_special.gammaln(m + 1.0)
            - scipy_special.gammaln(a)
        )
        prefactor = np.exp(log_prefactor)

        if m > 1:
            P_series = np.zeros_like(tp, dtype=float)
            tp_pow = np.ones_like(tp, dtype=float)
            for r in range(m):
                coef = scipy_special.gamma(m - r) * scipy_special.poch(a, r) / scipy_special.gamma(r + 1.0)
                P_series += coef * tp_pow
                tp_pow *= tp
            P[positive_mask] = P_series / scipy_special.gamma(a + m)
        else:
            P[positive_mask] = 1.0 / scipy_special.gamma(a + m)

        series = np.zeros_like(tp, dtype=float)
        s = np.ones_like(tp, dtype=float)
        d = (
            scipy_special.digamma(a + m)
            - scipy_special.digamma(1.0)
            - scipy_special.digamma(m + 1.0)
        )
        series += s * (np.log(tp) + d)

        for k in range(max_terms):
            s *= (-(a + m + k) * tp) / ((m + 1.0 + k) * (k + 1.0))
            d += (
                1.0 / (a + m + k)
                - 1.0 / (k + 1.0)
                - 1.0 / (m + k + 1.0)
            )
            term = s * (np.log(tp) + d)
            series += term
            if np.all(np.abs(term) <= tol * np.maximum(1.0, np.abs(series))):
                break
        else:
            raise RuntimeError("Integer HTMP series did not converge.")

        I[positive_mask] = np.pi * prefactor * scipy_special.hyp1f1(a + m, m + 1.0, -tp)
        R = P[positive_mask] - prefactor * series
        boundary = np.empty_like(t, dtype=np.complex128)
        boundary[positive_mask] = R + 1j * I[positive_mask]
    else:
        boundary = np.empty_like(t, dtype=np.complex128)

    boundary[~positive_mask] = 1.0 + 0.0j
    return boundary


def htmp_phase_fast(x_grid, kappa, gamma, beta, integer_tolerance=1e-7, max_phase_step=np.pi / 2):
    points = _validate_htmp_grid(x_grid)
    if points.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, {"phase_steps": empty, "prepended_zero": False}

    prepended_zero = points[0] != 0.0
    if prepended_zero:
        eval_points = np.concatenate(([0.0], points))
    else:
        eval_points = points

    boundary = htmp_boundary_fast(eval_points, kappa, gamma, beta, integer_tolerance=integer_tolerance)
    if not np.all(np.isfinite(boundary)):
        raise RuntimeError("Fast HTMP boundary evaluation produced non-finite values.")

    real_part = np.real(boundary)
    imag_part = np.imag(boundary)
    scale = np.maximum(np.abs(real_part), np.abs(imag_part))
    scale = np.where(scale > 0.0, scale, 1.0)
    principal_phase = np.arctan2(imag_part / scale, real_part / scale)
    continuous_phase = np.unwrap(principal_phase)
    continuous_phase = continuous_phase - continuous_phase[0]

    phase_steps = np.diff(continuous_phase)
    if phase_steps.size and np.max(np.abs(phase_steps)) >= max_phase_step:
        raise RuntimeError(
            "Computed HTMP phase steps are too large for stable unwrapping. Refine the grid or use the fallback."
        )

    if prepended_zero:
        principal_phase = principal_phase[1:]
        continuous_phase = continuous_phase[1:]

    diagnostics = {
        "phase_steps": phase_steps,
        "max_phase_step": float(np.max(np.abs(phase_steps))) if phase_steps.size else 0.0,
        "prepended_zero": prepended_zero,
    }
    return principal_phase, continuous_phase, diagnostics


def htmp_cdf_fallback(x_grid, kappa, gamma, beta, dps=50, branch_epsilon=1e-12):
    points = _validate_htmp_grid(x_grid)
    if points.size == 0:
        return np.array([], dtype=float)
    return _phase_htmp_cdf_nonnegative(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)


def htmp_cdf_fast(
    x_grid,
    kappa,
    gamma,
    beta,
    *,
    integer_tolerance=1e-7,
    max_phase_step=np.pi / 2,
    enforce_monotonicity=True,
    dps=50,
    branch_epsilon=1e-12,
):
    points = _validate_htmp_grid(x_grid)
    if points.size == 0:
        return np.array([], dtype=float)

    prepended_zero = points[0] != 0.0
    if prepended_zero:
        points_for_phase = np.concatenate(([0.0], points))
    else:
        points_for_phase = points

    _, _, _, delta = htmp_parameters(kappa, gamma, beta)
    if abs(np.sin(np.pi * delta)) < integer_tolerance:
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)

    try:
        _, continuous_phase, diagnostics = htmp_phase_fast(
            points_for_phase,
            kappa,
            gamma,
            beta,
            integer_tolerance=integer_tolerance,
            max_phase_step=max_phase_step,
        )
    except Exception:
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)

    if diagnostics["phase_steps"].size and np.max(np.abs(diagnostics["phase_steps"])) >= max_phase_step:
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)

    cdf = 2.0 * continuous_phase / (np.pi * float(kappa))
    if not np.all(np.isfinite(cdf)):
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)

    cdf_steps = np.diff(cdf)
    if cdf_steps.size and np.min(cdf_steps) < -1e-8:
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)
    if enforce_monotonicity:
        cdf = np.maximum.accumulate(cdf)

    if np.min(cdf) < -1e-8 or np.max(cdf) > 1.0 + 1e-8:
        return htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps, branch_epsilon=branch_epsilon)

    cdf = np.clip(cdf, 0.0, 1.0)
    if prepended_zero:
        cdf = cdf[1:]
    return cdf


def validate_fast_htmp(x_points, kappa, gamma, beta, dps=60):
    points = _validate_htmp_grid(x_points)
    fast_cdf = htmp_cdf_fast(points, kappa, gamma, beta)
    direct_cdf = htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps)
    relative_error = np.max(np.abs(fast_cdf - direct_cdf) / np.maximum(np.abs(direct_cdf), 1e-15)) if points.size else 0.0
    return {
        "points": points,
        "fast_cdf": fast_cdf,
        "direct_cdf": direct_cdf,
        "max_abs_error": float(np.max(np.abs(fast_cdf - direct_cdf))) if points.size else 0.0,
        "max_relative_error": float(relative_error),
    }


def benchmark_htmp_cdf(x_grid, kappa, gamma, beta, repeats=3, dps=60):
    points = _validate_htmp_grid(x_grid)
    if points.size == 0:
        raise ValueError("x_grid must contain at least one point for benchmarking.")

    timings = {}
    for name, func in (
        ("fast", lambda: htmp_cdf_fast(points, kappa, gamma, beta)),
        ("direct_mpmath", lambda: _phase_htmp_cdf_nonnegative(points, kappa, gamma, beta, dps=dps)),
        ("fallback", lambda: htmp_cdf_fallback(points, kappa, gamma, beta, dps=dps)),
    ):
        best = float("inf")
        result = None
        for _ in range(int(repeats)):
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
            if elapsed < best:
                best = elapsed
        timings[name] = best
        timings[f"{name}_result"] = result

    timings["speedup_fast_vs_direct_mpmath"] = timings["direct_mpmath"] / timings["fast"] if timings["fast"] > 0.0 else np.inf
    timings["speedup_fast_vs_fallback"] = timings["fallback"] / timings["fast"] if timings["fast"] > 0.0 else np.inf
    return timings


def benchmark_htmp_distance_speed(
    x_grid,
    kappas,
    betas,
    gammas,
    repeats=3,
    bins=50,
    lp_ord=1,
    inverse=False,
    dps=50,
    branch_epsilon=1e-12,
    integer_tolerance=1e-7,
    verbose=True,
):
    points = _validate_htmp_grid(x_grid)
    if points.size == 0:
        raise ValueError("x_grid must contain at least one point for benchmarking.")

    kappas = np.asarray(list(kappas), dtype=float)
    betas = np.asarray(list(betas), dtype=float)
    gammas = np.asarray(list(gammas), dtype=float)
    if kappas.size == 0 or betas.size == 0 or gammas.size == 0:
        raise ValueError("kappas, betas, and gammas must each contain at least one value.")

    from helpers import wass_distance

    records = []
    for gamma in gammas:
        for kappa in kappas:
            delta = float(kappa) * (1.0 - float(gamma)) / (2.0 * float(gamma))
            if np.isclose(delta, np.round(delta), atol=integer_tolerance, rtol=0.0):
                if verbose:
                    print(
                        f"Skipping kappa={kappa:.6g}, gamma={gamma:.6g} because delta={delta:.6g} is too close to an integer."
                    )
                continue

            for beta in betas:
                ks_best = float("inf")
                wass_best = float("inf")

                for _ in range(int(repeats)):
                    start = time.perf_counter()
                    ks_distance = compute_htmp_ks_distance(
                        points,
                        kappa,
                        gamma,
                        beta,
                        inverse=inverse,
                        dps=dps,
                        branch_epsilon=branch_epsilon,
                    )
                    ks_elapsed = time.perf_counter() - start
                    if ks_elapsed < ks_best:
                        ks_best = ks_elapsed

                    start = time.perf_counter()
                    wass_distance(
                        points,
                        gamma,
                        kappa,
                        beta,
                        bins=bins,
                        inverse=inverse,
                        stieltjes=False,
                        lp_ord=lp_ord,
                    )
                    wass_elapsed = time.perf_counter() - start
                    if wass_elapsed < wass_best:
                        wass_best = wass_elapsed

                record = {
                    "kappa": float(kappa),
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "delta": float(delta),
                    "ks_time": ks_best,
                    "wasserstein_time": wass_best,
                    "speedup_wasserstein_over_ks": wass_best / ks_best if ks_best > 0.0 else np.inf,
                    "ks_distance": float(ks_distance),
                }
                records.append(record)

                if verbose:
                    print(
                        "kappa={kappa:.6g}, beta={beta:.6g}, gamma={gamma:.6g}, delta={delta:.6g}, "
                        "ks={ks_time:.6f}s, wasserstein={wasserstein_time:.6f}s, speedup={speedup_wasserstein_over_ks:.3f}x".format(
                            **record
                        )
                    )

    if not records:
        raise ValueError("All parameter combinations were skipped by the integer-delta filter.")

    ks_times = np.array([record["ks_time"] for record in records], dtype=float)
    wass_times = np.array([record["wasserstein_time"] for record in records], dtype=float)
    summary = {
        "points": points,
        "records": records,
        "num_records": len(records),
        "median_ks_time": float(np.median(ks_times)),
        "median_wasserstein_time": float(np.median(wass_times)),
        "mean_ks_time": float(np.mean(ks_times)),
        "mean_wasserstein_time": float(np.mean(wass_times)),
        "median_speedup_wasserstein_over_ks": float(np.median(wass_times / ks_times)),
    }
    return summary


def evaluate_mp_cdf(points, gamma, scale=1.0, tau=0.0):
    points = _as_1d_float_array(points)
    gamma = float(gamma)
    scale = float(scale)
    tau = float(tau)

    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must satisfy 0 < gamma <= 1, got {gamma}.")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")

    x = (points - tau) / scale
    cdf = np.zeros_like(x, dtype=float)

    if gamma == 1.0:
        support_min = 0.0
        support_max = 4.0
        right_mask = x > support_max
        middle_mask = (x >= support_min) & (x <= support_max)
        cdf[right_mask] = 1.0
        if np.any(middle_mask):
            theta = np.arccos(np.clip(1.0 - x[middle_mask] / 2.0, -1.0, 1.0))
            cdf[middle_mask] = (theta + np.sin(theta)) / np.pi
        return np.clip(cdf, 0.0, 1.0)

    sqrt_gamma = np.sqrt(gamma)
    support_min = (1.0 - sqrt_gamma) ** 2
    support_max = (1.0 + sqrt_gamma) ** 2
    right_mask = x > support_max
    middle_mask = (x >= support_min) & (x <= support_max)
    cdf[right_mask] = 1.0

    if np.any(middle_mask):
        x_mid = x[middle_mask]
        theta = np.arccos(np.clip((1.0 + gamma - x_mid) / (2.0 * sqrt_gamma), -1.0, 1.0))
        r_gamma = (1.0 + sqrt_gamma) / (1.0 - sqrt_gamma)
        tangent_term = np.tan(theta / 2.0)
        cdf[middle_mask] = (
            ((1.0 + gamma) / (2.0 * np.pi * gamma)) * theta
            + np.sin(theta) / (np.pi * sqrt_gamma)
            - ((1.0 - gamma) / (np.pi * gamma)) * np.arctan(r_gamma * tangent_term)
        )

    return np.clip(cdf, 0.0, 1.0)


def _phase_htmp_cdf_nonnegative(points, kappa, gamma, beta, dps=50, branch_epsilon=1e-12):
    points = _as_1d_float_array(points)
    if points.size == 0:
        return np.array([], dtype=float)

    if np.any(points < 0):
        raise ValueError("_phase_htmp_cdf_nonnegative expects nonnegative points.")

    kappa = float(kappa)
    gamma = float(gamma)
    beta = float(beta)
    a = kappa / 2.0
    b = 1.0 - kappa / (2.0 * gamma) + a
    sorted_idx = np.argsort(points)
    sorted_points = points[sorted_idx]

    with mpmath.workdps(dps):
        anchor = complex(mpmath.hyperu(a, b, -1j * branch_epsilon))
        values = [complex(mpmath.hyperu(a, b, -beta * x - 1j * branch_epsilon)) for x in sorted_points]

    phases = np.unwrap(np.angle(np.asarray([anchor] + values, dtype=np.complex128)))
    cdf_sorted = (2.0 / (np.pi * kappa)) * (phases[1:] - phases[0])
    cdf_sorted = np.clip(np.real_if_close(cdf_sorted), 0.0, 1.0)

    cdf = np.empty_like(cdf_sorted)
    cdf[sorted_idx] = cdf_sorted
    return cdf


def evaluate_htmp_cdf(points, kappa, gamma, beta, tau=0.0, inverse=False, dps=50, branch_epsilon=1e-12):
    points = _as_1d_float_array(points)
    cdf = np.zeros_like(points, dtype=float)

    shifted = points - float(tau)
    positive_mask = shifted > 0

    if not np.any(positive_mask):
        return cdf

    if np.isinf(kappa):
        if inverse:
            raise ValueError("inverse=True is not supported for kappa=inf.")
        cdf[positive_mask] = evaluate_mp_cdf(shifted[positive_mask], gamma)
        return cdf

    kappa = float(kappa)
    gamma = float(gamma)
    beta = float(beta)

    if not inverse:
        positive_points = shifted[positive_mask]
        order = np.argsort(positive_points)
        sorted_points = positive_points[order]
        sorted_cdf = htmp_cdf_fast(
            sorted_points,
            kappa,
            gamma,
            beta,
            dps=dps,
            branch_epsilon=branch_epsilon,
        )
        positive_cdf = np.empty_like(sorted_cdf)
        positive_cdf[order] = sorted_cdf
        cdf[positive_mask] = positive_cdf
        return cdf

    scale = kappa / (2.0 * gamma * beta)
    transformed = 1.0 / (scale * gamma * shifted[positive_mask])
    standard_cdf = _phase_htmp_cdf_nonnegative(
        transformed,
        kappa,
        gamma,
        _standard_beta(kappa, gamma),
        dps=dps,
        branch_epsilon=branch_epsilon,
    )
    cdf[positive_mask] = np.clip(1.0 - standard_cdf, 0.0, 1.0)
    return cdf


def compute_htmp_ks_distance(eigenvalues, kappa, gamma, beta, tau=0.0, inverse=False, dps=50, branch_epsilon=1e-12, return_details=False):
    eigs = _as_1d_float_array(eigenvalues)
    eigs = eigs[np.isfinite(eigs)]
    if eigs.size == 0:
        if return_details:
            return float("inf"), {
                "points": np.array([], dtype=float),
                "empirical_lower": np.array([], dtype=float),
                "empirical_upper": np.array([], dtype=float),
                "theoretical_cdf": np.array([], dtype=float),
            }
        return float("inf")

    points = np.sort(eigs)
    theoretical_cdf = evaluate_htmp_cdf(
        points,
        kappa,
        gamma,
        beta,
        tau=tau,
        inverse=inverse,
        dps=dps,
        branch_epsilon=branch_epsilon,
    )
    n_points = points.size
    empirical_upper = np.arange(1, n_points + 1, dtype=float) / n_points
    empirical_lower = np.arange(0, n_points, dtype=float) / n_points
    ks_distance = max(
        np.max(np.abs(theoretical_cdf - empirical_upper)),
        np.max(np.abs(theoretical_cdf - empirical_lower)),
    )

    if return_details:
        return ks_distance, {
            "points": points,
            "empirical_lower": empirical_lower,
            "empirical_upper": empirical_upper,
            "theoretical_cdf": theoretical_cdf,
        }
    return ks_distance


def compute_mp_ks_distance(eigenvalues, gamma, scale=1.0, tau=0.0, return_details=False):
    eigs = _as_1d_float_array(eigenvalues)
    eigs = eigs[np.isfinite(eigs)]
    if eigs.size == 0:
        if return_details:
            return float("inf"), {
                "points": np.array([], dtype=float),
                "empirical_lower": np.array([], dtype=float),
                "empirical_upper": np.array([], dtype=float),
                "theoretical_cdf": np.array([], dtype=float),
            }
        return float("inf")

    points = np.sort(eigs)
    theoretical_cdf = evaluate_mp_cdf(points, gamma, scale=scale, tau=tau)
    n_points = points.size
    empirical_upper = np.arange(1, n_points + 1, dtype=float) / n_points
    empirical_lower = np.arange(0, n_points, dtype=float) / n_points
    ks_distance = max(
        np.max(np.abs(theoretical_cdf - empirical_upper)),
        np.max(np.abs(theoretical_cdf - empirical_lower)),
    )

    if return_details:
        return ks_distance, {
            "points": points,
            "empirical_lower": empirical_lower,
            "empirical_upper": empirical_upper,
            "theoretical_cdf": theoretical_cdf,
        }
    return ks_distance


def evaluate_standard_htmp_cdf(points, kappa, gamma, dps=50, branch_epsilon=1e-12):
    return evaluate_htmp_cdf(
        points,
        kappa,
        gamma,
        _standard_beta(kappa, gamma),
        dps=dps,
        branch_epsilon=branch_epsilon,
    )


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the HTMP CDF on a list of points.")
    parser.add_argument("--points", nargs="+", type=float, required=True, help="Points where the CDF is evaluated.")
    parser.add_argument("--kappa", type=float, required=True, help="HTMP kappa parameter.")
    parser.add_argument("--gamma", type=float, required=True, help="HTMP gamma parameter.")
    parser.add_argument("--beta", type=float, required=True, help="HTMP beta parameter.")
    parser.add_argument("--tau", type=float, default=0.0, help="Optional location shift.")
    parser.add_argument("--inverse", action="store_true", help="Evaluate the inverse-HTMP eigenvalue CDF.")
    parser.add_argument("--dps", type=int, default=50, help="mpmath decimal precision.")
    parser.add_argument("--branch-epsilon", type=float, default=1e-12, help="Imaginary offset used for the lower boundary value.")
    parser.add_argument("--benchmark", action="store_true", help="Run a small speed benchmark instead of printing CDF values.")
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Number of repeats used for the benchmark.")
    return parser


def main():
    args = _build_parser().parse_args()

    if args.benchmark:
        bench = benchmark_htmp_cdf(
            args.points,
            args.kappa,
            args.gamma,
            args.beta,
            repeats=args.benchmark_repeats,
            dps=args.dps,
        )
        print(f"fast: {bench['fast']:.6f}s")
        print(f"direct_mpmath: {bench['direct_mpmath']:.6f}s")
        print(f"fallback: {bench['fallback']:.6f}s")
        print(f"speedup_fast_vs_direct_mpmath: {bench['speedup_fast_vs_direct_mpmath']:.3f}x")
        print(f"speedup_fast_vs_fallback: {bench['speedup_fast_vs_fallback']:.3f}x")
        return

    cdf_values = evaluate_htmp_cdf(
        args.points,
        args.kappa,
        args.gamma,
        args.beta,
        tau=args.tau,
        inverse=args.inverse,
        dps=args.dps,
        branch_epsilon=args.branch_epsilon,
    )

    for point, cdf_value in zip(args.points, cdf_values):
        print(f"x={point:.12g}\tF(x)={cdf_value:.12g}")


if __name__ == "__main__":
    main()