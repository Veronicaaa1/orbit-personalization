from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.linalg import null_space, solve_continuous_lyapunov
from scipy.stats import t as student_t, chi2

Array = np.ndarray
ORBIT_SIZES = np.array([2, 3, 4, 5], dtype=int)
N = int(ORBIT_SIZES.sum())
ORBITS = tuple(np.split(np.arange(N), np.cumsum(ORBIT_SIZES)[:-1]))
ORBIT_LABELS = np.repeat(np.arange(4), ORBIT_SIZES)
T_PAR = (np.eye(4)[ORBIT_LABELS] / np.sqrt(ORBIT_SIZES)).T
T_PERP = null_space(T_PAR).T
P_PAR = T_PAR.T @ T_PAR
P_PERP = np.eye(N) - P_PAR
WITHIN_MASK = (ORBIT_LABELS[:, None] == ORBIT_LABELS[None, :]) & ~np.eye(N, dtype=bool)
BETWEEN_MASK = ORBIT_LABELS[:, None] != ORBIT_LABELS[None, :]

@dataclass(frozen=True)
class System:
    name: str
    C: Array
    Q: Array
    r: Array
    sigma: Array
    eta: float = 1.0
    H: float = 1.0

    def matrices(self) -> tuple[Array, Array, Array]:
        """Return K, a, Sigma in dw = (K w + a)dt + Sigma dB."""
        L = np.diag(self.C.sum(axis=1)) - self.C
        K = -self.eta * np.diag(self.Q) - self.H * L
        a = self.eta * self.Q * self.r
        Sigma = np.diag(self.sigma)
        return K, a, Sigma

@dataclass(frozen=True)
class GaussianLaw:
    mean: Array
    cov: Array
    stability_margin: float

def base_graph(weight: float = 0.1) -> Array:
    """Path blown up into independent sets of sizes 2,3,4,5; no within-set edges."""
    C = np.zeros((N, N), dtype=float)
    for left, right in zip(ORBITS[:-1], ORBITS[1:]):
        C[np.ix_(left, right)] = weight
        C[np.ix_(right, left)] = weight
    return C

REFERENCE_SUPPORT = base_graph() > 0

def reference_system() -> System:
    """Exact-compatible reference shared by all positive-compatibility experiments."""
    return System(
        name="exact",
        C=base_graph(0.1),
        Q=np.ones(N),
        r=np.repeat([1.0, 0.4, -0.2, -1.0], ORBIT_SIZES),
        sigma=np.ones(N),
    )

def validate_system(system: System, require_reference_support: bool = True) -> None:
    if not all(np.isfinite(x).all() for x in (system.C, system.Q, system.r, system.sigma)):
        raise ValueError("System coefficients must be finite.")
    if not np.isfinite(system.eta) or not np.isfinite(system.H):
        raise ValueError("eta and H must be finite.")
    if system.C.shape != (N, N):
        raise ValueError("C must be 14x14.")
    for vec in (system.Q, system.r, system.sigma):
        if vec.shape != (N,):
            raise ValueError("Q, r and sigma must have length 14.")
    if not np.allclose(system.C, system.C.T, atol=1e-13, rtol=0):
        raise ValueError("Communication matrix must be symmetric.")
    if np.any(system.C < 0) or not np.allclose(np.diag(system.C), 0):
        raise ValueError("Communication weights must be nonnegative with zero diagonal.")
    if require_reference_support and np.any(system.C[~REFERENCE_SUPPORT] != 0):
        raise ValueError("This experiment does not introduce new host-graph edges.")
    if np.any(system.Q <= 0) or np.any(system.sigma < 0):
        raise ValueError("Require Q_i>0 and sigma_i>=0.")
    if system.eta <= 0 or system.H < 0:
        raise ValueError("Require eta>0 and H>=0.")

    # Check connectivity of active positive-weight graph.
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(system.C[i] > 0):
            j = int(j)
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if len(seen) != N:
        raise ValueError("Active communication graph must stay connected.")

def symmetry_generator_defect(system: System) -> float:
    """Max defect under adjacent swaps generating S2 x S3 x S4 x S5."""
    errs: list[float] = []
    for orbit in ORBITS:
        for i, j in zip(orbit[:-1], orbit[1:]):
            perm = np.arange(N)
            perm[i], perm[j] = perm[j], perm[i]
            errs.append(float(np.linalg.norm(system.C[np.ix_(perm, perm)] - system.C, "fro")))
            errs.extend(float(np.max(np.abs(x[perm] - x))) for x in (system.Q, system.r, system.sigma))
    return max(errs) if errs else 0.0

def joint_direction(seed: int) -> tuple[Array, Array]:
    """Generate normalized communication and local-parameter directions.

    The communication direction is symmetric, supported on the reference graph,
    and centered blockwise so that total weight between adjacent orbit blocks is
    unchanged exactly along this linear perturbation path. The local perturbations are centered within each
    reference orbit before normalization.
    """
    rng = np.random.default_rng(seed)

    E = np.zeros((N, N), dtype=float)
    for left, right in zip(ORBITS[:-1], ORBITS[1:]):
        block = rng.uniform(-1.0, 1.0, size=(len(left), len(right)))
        block -= block.mean()
        E[np.ix_(left, right)] = block
        E[np.ix_(right, left)] = block.T
    fro = np.linalg.norm(E, "fro")
    if fro == 0:
        raise RuntimeError("Degenerate communication perturbation direction.")
    E /= fro

    # columns: dQ, dsigma, dr
    D = rng.uniform(-1.0, 1.0, size=(N, 3))
    for orbit in ORBITS:
        D[orbit] -= D[orbit].mean(axis=0)

    # Scale components so local minimizer perturbations are visually meaningful.
    D *= np.array([0.30, 0.10, 0.60])
    node_l1 = np.abs(D).sum(axis=1)
    D /= node_l1.max()
    return E, D

def joint_perturb(reference: System, epsilon: float, seed: int) -> System:
    """Joint perturbation with exact total manuscript defect epsilon.

    Half of the total defect is assigned to communication and half to local
    quantities:

        ||C_eps-C_0||_F = epsilon/2,
        max_i (|dQ_i|+|dsigma_i|+|dr_i|) = epsilon/2.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")
    E, D = joint_direction(seed)
    half = epsilon / 2.0
    out = replace(
        reference,
        name=f"approx_joint_eps_{epsilon:g}",
        C=reference.C + half * E,
        Q=reference.Q + half * D[:, 0],
        sigma=reference.sigma + half * D[:, 1],
        r=reference.r + half * D[:, 2],
    )
    validate_system(out)
    actual = manuscript_defect(out, reference)
    if not np.isclose(actual, epsilon, rtol=1e-10, atol=2e-13):
        raise AssertionError(f"Joint defect normalization failed: {actual} vs {epsilon}")
    return out

def manuscript_defect(system: System, reference: System) -> float:
    local = (
        np.abs(system.Q - reference.Q)
        + np.abs(system.sigma - reference.sigma)
        + np.abs(system.r - reference.r)
    )
    return float(np.linalg.norm(system.C - reference.C, "fro") + local.max())

def psd_root(A: Array) -> Array:
    A = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A)
    tol = 1e-11 * max(1.0, np.linalg.norm(A, 2))
    if eigvals.min() < -tol:
        raise ArithmeticError(f"Matrix not PSD; minimum eigenvalue {eigvals.min():.3e}")
    return (eigvecs * np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T

def stationary_law(system: System) -> GaussianLaw:
    validate_system(system)
    K, a, Sigma = system.matrices()
    largest = float(np.linalg.eigvalsh(K).max())
    margin = -largest
    if margin <= 0:
        raise ValueError(f"{system.name}: drift is not Hurwitz.")
    mean = np.linalg.solve(-K, a)
    cov = solve_continuous_lyapunov(K, -(Sigma @ Sigma.T))
    cov = (cov + cov.T) / 2.0
    psd_root(cov)
    residual = np.linalg.norm(K @ cov + cov @ K.T + Sigma @ Sigma.T, "fro")
    if residual > 1e-9:
        raise ArithmeticError(f"Lyapunov residual too large: {residual:.3e}")
    return GaussianLaw(mean=mean, cov=cov, stability_margin=margin)

def gaussian_w2(m1: Array, A: Array, m2: Array, B: Array) -> float:
    """2-Wasserstein distance between Gaussian laws, numerically stable."""
    R = psd_root(A)
    S = psd_root(B)
    U, _, Vt = np.linalg.svd(S.T @ R)
    cov_dist2 = np.linalg.norm(R - S @ (U @ Vt), "fro") ** 2
    mean_dist2 = np.linalg.norm(m1 - m2) ** 2
    return float(np.sqrt(mean_dist2 + cov_dist2))

def stationary_metrics(law: GaussianLaw) -> dict[str, float]:
    mean = law.mean
    var = np.maximum(np.diag(law.cov), 0.0)
    d_par = np.abs(mean[:, None] - mean[None, :]) + np.abs(var[:, None] - var[None, :])
    d_mean = np.abs(mean[:, None] - mean[None, :])
    d_w2 = np.sqrt(
        d_mean**2
        + (np.sqrt(var)[:, None] - np.sqrt(var)[None, :]) ** 2
    )

    out: dict[str, float] = {}
    for name, D in [("par", d_par), ("mean", d_mean), ("w2", d_w2)]:
        within = float(D[WITHIN_MASK].max())
        between = float(D[BETWEEN_MASK].min())
        out[f"D_in_{name}"] = within
        out[f"D_out_{name}"] = between
        out[f"margin_{name}"] = between - within

    # A pathwise/transverse energy diagnostic; not the theorem's D_in.
    out["transverse_stationary_energy_per_node"] = float(
        (np.linalg.norm(P_PERP @ mean) ** 2 + np.trace(P_PERP @ law.cov)) / N
    )
    return out


def stationary_risk_metrics(system: System, law: GaussianLaw) -> dict[str, float]:
    """Exact stationary quadratic risks and a direct coefficient certificate.

    This scalar implementation requires Q_i>0, symmetric nonnegative C and
    H>=0. The certificate is evaluated directly at the supplied configuration;
    for the reference it is g_0 from the corollary. At perturbed configurations
    it is NOT g_0-C_risk*epsilon and does not estimate a theorem constant.
    The independent baseline keeps eta, Q, r and sigma fixed. The shared
    oracle optimizes one scalar predictor against all quadratic objectives.
    """
    validate_system(system)
    K, _, _ = system.matrices()
    L = np.diag(system.C.sum(axis=1)) - system.C
    S = system.H * L
    error = law.mean - system.r
    bias = float(np.dot(system.Q, error**2) / (2*N))
    variance = float(np.dot(system.Q, np.diag(law.cov)) / (2*N))
    risk = bias + variance
    independent = float(np.dot(system.sigma, system.sigma) / (4*N*system.eta))
    shared_parameter = float(np.dot(system.Q, system.r) / system.Q.sum())
    shared = float(np.dot(system.Q, (system.r-shared_parameter)**2) / (2*N))
    variance_reduction = float(np.trace(S @ law.cov) / (2*N*system.eta))
    beta = float(np.linalg.eigvalsh(-(K+K.T)/2).max())
    d = float(np.min(system.sigma**2))
    b_star = -S @ system.r
    bias_bound = float(np.sum(b_star**2/system.Q) / (2*N*system.eta**2))
    variance_lower = float(d*np.trace(S) / (4*N*system.eta*beta))
    g = variance_lower - bias_bound
    gain_independent = independent - risk
    gain_shared = shared - risk
    out = {
        'risk_bias': bias,
        'risk_variance': variance,
        'risk_total': risk,
        'risk_independent': independent,
        'risk_shared_oracle': shared,
        'shared_oracle_parameter': shared_parameter,
        'risk_variance_reduction': variance_reduction,
        'risk_gain_independent': gain_independent,
        'risk_gain_shared': gain_shared,
        'risk_d': d,
        'risk_beta': beta,
        'risk_bias_upper_bound': bias_bound,
        'risk_variance_reduction_lower_bound': variance_lower,
        'risk_g_coefficient': g,
        'risk_shared_lower_bound': g+shared-independent,
        'risk_joint_certificate': g-max(0., independent-shared),
        'risk_identity_residual': abs(gain_independent-(variance_reduction-bias)),
    }
    scale = max(1., abs(independent), abs(risk), abs(shared))
    if out['risk_identity_residual'] > 1e-10*scale:
        raise ArithmeticError('The exact stationary-risk identity failed.')
    if gain_independent+1e-10*scale < g:
        raise ArithmeticError('The direct coefficient risk lower bound failed.')
    return out


def terminal_risk_summary(system: System, samples: Array,
                          meanT: Array, covT: Array) -> dict[str, float]:
    """Terminal quadratic-risk estimate and approximate 95% t interval.

    A sampling unit is one complete N-node SDE realization. Node correlation
    is retained by first averaging the N losses within each realization.
    The t interval is an asymptotic Monte Carlo interval for finite-time risk,
    not a finite-sample Gaussian-risk interval or a stationary-law guarantee.
    """
    losses = np.mean(system.Q*(samples-system.r)**2, axis=1)/2
    n = len(losses)
    estimate = float(losses.mean())
    se = float(losses.std(ddof=1)/np.sqrt(n))
    half = float(student_t.ppf(.975, n-1)*se)
    analytic = float((np.dot(system.Q, (meanT-system.r)**2)
                      + np.dot(system.Q, np.diag(covT)))/(2*N))
    return {'terminal_risk_analytic': analytic,
            'terminal_risk_mc': estimate,
            'terminal_risk_mc_se': se,
            'terminal_risk_mc_ci95_lower': estimate-half,
            'terminal_risk_mc_ci95_upper': estimate+half,
            'terminal_risk_mc_replicates': n}


def noise_sweep(reference: System) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Exact common-noise scan with algebraic sufficient-condition boundaries.

    With all other coefficients fixed, g(s)=a*s^2-b and E_ind(s)=c*s^2.
    The present reference has a<c. Thus simultaneous strict certification is
    b/a < s^2 < (H_one-b)/(c-a). Boundaries are included in the CSV but are
    excluded from the strict sufficient-condition interval.
    """
    unit = replace(reference, sigma=np.ones(N))
    unit_risk = stationary_risk_metrics(unit, stationary_law(unit))
    a = unit_risk['risk_variance_reduction_lower_bound']
    b = unit_risk['risk_bias_upper_bound']
    c = unit_risk['risk_independent']
    shared = unit_risk['risk_shared_oracle']
    if not (a > 0 and c > a and shared > b):
        raise ValueError('This common-noise interval calculation needs 0<a<c and H_one>b.')
    low2, high2 = b/a, (shared-b)/(c-a)
    if low2 >= high2:
        raise ValueError('The reference has no common-noise sufficient-condition interval.')
    low, high = float(np.sqrt(low2)), float(np.sqrt(high2))
    points = np.unique(np.r_[np.linspace(.05, 1.3, 126), .12, 1., low, high])
    rows = []
    for s in points:
        system = replace(reference, name=f'noise_{s:.8g}', sigma=np.full(N, s))
        law = stationary_law(system)
        rows.append({'sigma': float(s), **stationary_risk_metrics(system, law),
                     'D_out_par': stationary_metrics(law)['D_out_par']})
    info = {'coefficient_a': a, 'bias_bound_b': b, 'independent_coefficient_c': c,
            'sigma_squared_lower_open': low2, 'sigma_squared_upper_open': high2,
            'sigma_lower_open': low, 'sigma_upper_open': high}
    return rows, info

def finite_transition(system: System, h: float) -> tuple[Array, Array]:
    """Exact OU transition x_{t+h}=m+F(x_t-m)+eta_h."""
    if not np.isfinite(h) or h < 0:
        raise ValueError("Transition time must be finite and nonnegative.")
    K, _, Sigma = system.matrices()
    eigvals, eigvecs = np.linalg.eigh(K)
    rates = eigvals[:, None] + eigvals[None, :]
    F = (eigvecs * np.exp(eigvals * h)) @ eigvecs.T
    G = -np.expm1(rates * h) / (-rates)
    BBT_eig = eigvecs.T @ (Sigma @ Sigma.T) @ eigvecs
    W = eigvecs @ (BBT_eig * G) @ eigvecs.T
    return F, (W + W.T) / 2.0

def simulate_exact_ou(
    system: System,
    law: GaussianLaw,
    w0: Array,
    runs: int,
    T: float,
    dt: float,
    seed: int,
) -> dict[str, Array]:
    """Exact-in-time OU simulation on a discrete observation grid."""
    steps = max(1, int(np.ceil(T / dt)))
    h = T / steps
    F, W = finite_transition(system, h)
    R = psd_root(W)
    # Integrate the affine forcing independently of the stationary solution.
    K, a, _ = system.matrices()
    values, vectors = np.linalg.eigh(K)
    g = vectors @ ((np.expm1(values * h) / values) * (vectors.T @ a))
    rng = np.random.default_rng(seed)

    X = np.broadcast_to(w0, (runs, N)).copy()
    path = np.empty((steps + 1, N))
    path[0] = X[0]

    for step in range(1, steps + 1):
        X = X @ F.T + g + rng.standard_normal((runs, N)) @ R.T
        path[step] = X[0]

    return {
        "endpoints": X,
        "path": path,
        "path_time": np.linspace(0.0, T, steps + 1),
    }

def finite_law(system: System, law: GaussianLaw, w0: Array, T: float) -> tuple[Array, Array]:
    F, W = finite_transition(system, T)
    K, a, _ = system.matrices()
    values, vectors = np.linalg.eigh(K)
    forcing = vectors @ ((np.expm1(values * T) / values) * (vectors.T @ a))
    mean = F @ w0 + forcing
    return mean, W

def mc_intervals(samples: Array, alpha_family: float = 0.05) -> list[dict[str, float]]:
    """Bonferroni simultaneous intervals for 14 means and 14 variances."""
    n = samples.shape[0]
    means = samples.mean(axis=0)
    variances = samples.var(axis=0, ddof=1)

    # 28 total intervals: 14 means + 14 variances.
    alpha_each = alpha_family / (2 * N)
    tcrit = student_t.ppf(1 - alpha_each / 2, n - 1)
    mean_half = tcrit * np.sqrt(variances / n)
    var_lo = (n - 1) * variances / chi2.ppf(1 - alpha_each / 2, n - 1)
    var_hi = (n - 1) * variances / chi2.ppf(alpha_each / 2, n - 1)

    rows: list[dict[str, float]] = []
    for i in range(N):
        rows.append(
            {
                "node": i + 1,
                "orbit": int(ORBIT_LABELS[i] + 1),
                "mean": float(means[i]),
                "mean_lower": float(means[i] - mean_half[i]),
                "mean_upper": float(means[i] + mean_half[i]),
                "variance": float(variances[i]),
                "variance_lower": float(var_lo[i]),
                "variance_upper": float(var_hi[i]),
            }
        )
    return rows

def nonlinear_gradient(x: Array, system: System) -> Array:
    """Gradient of a strongly convex nonquadratic loss.

    L_i(u) = log(1+exp(u-r_i)) + log(1+exp(-(u-r_i)))
             + 0.5*(Q_i-0.5)*(u-r_i)^2.

    Then L_i'(r_i)=0 and L_i''(r_i)=Q_i.
    """
    v = x - system.r
    return np.tanh(v / 2.0) + (system.Q - 0.5) * v

def simulate_nonlinear(
    system: System,
    w0: Array,
    runs: int,
    T: float,
    dt: float,
    seed: int,
    snapshot_dt: float | None = None,
) -> dict[str, Array | float]:
    """Euler--Maruyama with a paired step-halving check.

    Reported endpoints are from the finer dt/2 path. Coarse and fine paths use
    consistent Brownian increments, so their difference is a useful numerical
    discretization diagnostic. If snapshot_dt is supplied, record fine-path
    ensembles at requested times rounded to the nearest coarse-step boundary.
    Recording snapshots does not consume random draws or change either path.
    """
    if np.min(system.Q) <= 0.5:
        raise ValueError("This nonlinear construction needs all Q_i>0.5.")

    L = np.diag(system.C.sum(axis=1)) - system.C
    steps = max(1, int(np.ceil(T / dt)))
    h = T / steps

    # Simple stability guard for explicit EM.
    global_lipschitz_bound = system.eta * np.max(system.Q) + system.H * np.linalg.eigvalsh(L).max()
    if h * global_lipschitz_bound >= 0.5:
        raise ValueError("Nonlinear EM step too large; reduce --nonlinear-dt.")

    keep_steps = np.empty(0, dtype=int)
    snapshots = None
    if snapshot_dt is not None:
        if not np.isfinite(snapshot_dt) or snapshot_dt <= 0:
            raise ValueError('snapshot_dt must be finite and strictly positive.')
        # Do not change the integration grid to meet an observation request.
        requested = time_grid(T, snapshot_dt)
        keep_steps = np.unique(np.clip(np.rint(requested / h).astype(int), 0, steps))
        keep_steps = np.unique(np.r_[0, keep_steps, steps])
        snapshots = np.empty((len(keep_steps), runs, N))
    keep_index = 1
    rng = np.random.default_rng(seed)
    coarse = np.broadcast_to(w0, (runs, N)).copy()
    fine = coarse.copy()
    path = np.empty((steps + 1, N))
    path[0] = fine[0]
    if snapshots is not None:
        snapshots[0] = fine

    for step in range(1, steps + 1):
        dB1 = np.sqrt(h / 2.0) * rng.standard_normal((runs, N))
        dB2 = np.sqrt(h / 2.0) * rng.standard_normal((runs, N))

        drift_c = -system.eta * nonlinear_gradient(coarse, system) - system.H * coarse @ L.T
        coarse += h * drift_c + system.sigma * (dB1 + dB2)

        drift_f1 = -system.eta * nonlinear_gradient(fine, system) - system.H * fine @ L.T
        fine += (h / 2.0) * drift_f1 + system.sigma * dB1
        drift_f2 = -system.eta * nonlinear_gradient(fine, system) - system.H * fine @ L.T
        fine += (h / 2.0) * drift_f2 + system.sigma * dB2

        path[step] = fine[0]
        if snapshots is not None and keep_index < len(keep_steps) and step == keep_steps[keep_index]:
            snapshots[keep_index] = fine
            keep_index += 1

    delta = coarse - fine
    result = {
        "endpoints": fine,
        "coarse_endpoints": coarse,
        "path": path,
        "path_time": np.linspace(0.0, T, steps + 1),
        "step_halving_rms_per_node": float(np.sqrt(np.mean(delta**2))),
        "step_halving_max_mean_difference": float(np.max(np.abs(delta.mean(0)))),
        "step_halving_max_mean_SE": float(np.max(delta.std(0, ddof=1)/np.sqrt(runs))),
        "step_halving_max_variance_difference": float(np.max(np.abs(
            coarse.var(0, ddof=1)-fine.var(0, ddof=1)))),
        "actual_coarse_dt": h,
        "actual_fine_dt": h / 2.0,
    }
    if snapshots is not None:
        snapshot_times = keep_steps * h
        snapshot_times[-1] = T
        result.update({'times': snapshot_times, 'snapshots': snapshots})
    return result

def empirical_1d_w2_matrix(samples: Array) -> Array:
    """Empirical 1-D W2 for equal sample counts via sorted quantiles."""
    ordered = np.sort(samples, axis=0)
    return np.sqrt(np.mean((ordered[:, :, None] - ordered[:, None, :]) ** 2, axis=0))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Summary rows from different experiments intentionally contain different
    # diagnostics.  Use the union of keys so one CSV can hold all retained cases.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------------------------------------------------------
# Diagnostics aligned with the uploaded manuscript
# -----------------------------------------------------------------------------


def compare_laws(system: System, reference: System,
                 law: GaussianLaw, law0: GaussianLaw) -> dict[str, float]:
    """Quantities in theorem (ii)--(iv); all use the original node coordinates."""
    K, a, Sigma = system.matrices()
    K0, a0, Sigma0 = reference.matrices()
    mean_error = float(np.linalg.norm(law.mean - law0.mean))
    covariance_error = float(np.linalg.norm(law.cov - law0.cov, 2))
    joint_w2 = gaussian_w2(law.mean, law.cov, law0.mean, law0.cov)
    radius = float(np.max(np.abs(law.mean-law0.mean)
                          + np.abs(np.diag(law.cov)-np.diag(law0.cov))))
    difference_errors = []
    for i, j in zip(*np.where(np.triu(WITHIN_MASK, 1))):
        v = max(0., law.cov[i, i] + law.cov[j, j] - 2*law.cov[i, j])
        v0 = max(0., law0.cov[i, i] + law0.cov[j, j] - 2*law0.cov[i, j])
        dm = (law.mean[i]-law.mean[j]) - (law0.mean[i]-law0.mean[j])
        difference_errors.append(np.hypot(dm, np.sqrt(v)-np.sqrt(v0)))
    comm = float(np.linalg.norm(system.C-reference.C, 'fro'))
    local = manuscript_defect(system, reference)-comm
    metrics = stationary_metrics(law)
    gap0 = stationary_metrics(law0)['D_out_par']
    out = {
        **metrics,
        **stationary_risk_metrics(system, law),
        'epsilon_actual': manuscript_defect(system, reference),
        'communication_defect': comm,
        'local_defect': local,
        'stability_margin': law.stability_margin,
        'joint_W2_to_reference': joint_w2,
        'mean_error': mean_error,
        'covariance_error': covariance_error,
        'stationary_error_sum': joint_w2+mean_error+covariance_error,
        'prototype_parameter_radius': radius,
        'difference_law_max_W2_to_reference': float(max(difference_errors)),
        'a_posteriori_margin_lower_bound': gap0-4*radius,
        'coefficient_error': float(np.linalg.norm(K-K0, 2)
                                   +np.linalg.norm(a-a0)
                                   +np.linalg.norm(Sigma-Sigma0, 'fro')),
        'drift_mode_mixing': float(np.linalg.norm(T_PERP@K@T_PAR.T, 2)),
        'diffusion_mode_mixing': float(np.linalg.norm(T_PERP@Sigma@T_PAR.T, 2)),
        'transverse_affine_forcing': float(np.linalg.norm(T_PERP@a)),
        'symmetry_generator_defect': symmetry_generator_defect(system),
        'lyapunov_residual': float(np.linalg.norm(K@law.cov+law.cov@K.T+Sigma@Sigma.T, 'fro')),
    }
    # An a posteriori triangle-inequality check, NOT an estimated theorem constant.
    if metrics['margin_par']+1e-11 < out['a_posteriori_margin_lower_bound']:
        raise AssertionError('The blockwise triangle-inequality certificate failed.')
    return out


def time_laws(system: System, w0: Array, times: Array) -> tuple[Array, Array]:
    """Finite-time Gaussian parameters, integrated directly without a stationary law."""
    K, a, Sigma = system.matrices()
    lam, E = np.linalg.eigh(K)
    if lam.max() >= 0:
        raise ValueError('This implementation requires a Hurwitz symmetric drift.')
    rates = lam[:, None]+lam[None, :]
    diffusion = E.T@(Sigma@Sigma.T)@E
    winit, forcing = E.T@w0, E.T@a
    means, covs = [], []
    for time in times:
        means.append(E@(np.exp(lam*time)*winit + np.expm1(lam*time)/lam*forcing))
        cov = E@((np.expm1(rates*time)/rates)*diffusion)@E.T
        covs.append((cov+cov.T)/2)
    return np.array(means), np.array(covs)


def orbit_time_diagnostics(reference: System, law0: GaussianLaw, w0: Array,
                           times: Array) -> tuple[list[dict[str, float]],
                                                  list[dict[str, Any]], Array, float]:
    """Exact transient orbit diagnostics for the compatible blow-up graph.

    These are laws from a common deterministic initial state, not noisy sample
    paths. The pair-OU check also verifies the nonzero stationary difference
    variance under the model's independent Brownian motions.
    """
    means, covs = time_laws(reference, w0, times)
    pairs = list(zip(*np.where(np.triu(WITHIN_MASK, 1))))
    rates = reference.eta * reference.Q + reference.H * reference.C.sum(axis=1)
    rows, node_rows = [], []
    pair_formula_error = 0.0
    for time, mean, cov in zip(times, means, covs):
        metrics = stationary_metrics(GaussianLaw(mean, cov, law0.stability_margin))
        metrics['transverse_energy_per_node'] = metrics.pop('transverse_stationary_energy_per_node')
        row = {'time': float(time), **metrics}
        difference_w2 = []
        for i, j in pairs:
            rate = rates[i]
            variance_limit = (reference.sigma[i]**2 + reference.sigma[j]**2) / (2*rate)
            dm = mean[i] - mean[j]
            dv = cov[i, i] + cov[j, j] - 2*cov[i, j]
            exact_dm = (w0[i] - w0[j]) * np.exp(-rate*time)
            exact_dv = -np.expm1(-2*rate*time) * variance_limit
            pair_formula_error = max(pair_formula_error, abs(dm-exact_dm), abs(dv-exact_dv))
            difference_w2.append(np.hypot(dm, np.sqrt(max(0., dv))-np.sqrt(variance_limit)))
        row['max_difference_law_W2_to_stationary'] = float(max(difference_w2))
        row['max_node_mean_error_to_stationary'] = float(np.max(np.abs(mean-law0.mean)))
        for k, orbit in enumerate(ORBITS, 1):
            # Fixed first pair in each orbit, not selected for its appearance.
            i, j = orbit[:2]
            row[f'pair_{k}_second_moment'] = float((mean[i]-mean[j])**2 + cov[i, i] + cov[j, j] - 2*cov[i, j])
            row[f'pair_{k}_stationary_variance'] = float(
                (reference.sigma[i]**2 + reference.sigma[j]**2) / (2*rates[i]))
        rows.append(row)
        node_rows.extend({'time': float(time), 'node': i+1, 'orbit': int(ORBIT_LABELS[i]+1),
                          'mean': float(mean[i]), 'variance': float(cov[i, i])} for i in range(N))
    if pair_formula_error > 1e-11:
        raise AssertionError(f'Within-orbit pair OU identity failed: {pair_formula_error:.3e}')
    return rows, node_rows, means, float(pair_formula_error)


def transient_comparison(system: System, reference: System, w0: Array,
                         times: Array, cached_reference=None) -> list[dict[str, float]]:
    means0, covs0 = time_laws(reference, w0, times) if cached_reference is None else cached_reference
    means, covs = time_laws(system, w0, times)
    eps = manuscript_defect(system, reference)
    rows = []
    for k, time in enumerate(times):
        # Both laws at t=0 are exactly the same point mass.
        distance = 0. if time == 0 else gaussian_w2(means[k], covs[k], means0[k], covs0[k])
        rows.append({'time': float(time), 'joint_W2': distance,
                     'joint_W2_over_epsilon': distance/eps if eps > 0 else 0.})
    return rows


def transverse_relaxation(reference: System, law0: GaussianLaw, w0: Array,
                          times: Array) -> list[dict[str, float]]:
    """Exact-reference moment relaxation, not a pathwise-disagreement decay claim.

    K_perp is symmetric in THIS example. Thus rate alpha itself is available;
    the manuscript states the more general beta < alpha result.
    """
    K, _, _ = reference.matrices()
    Kperp = T_PERP@K@T_PERP.T
    lam, E = np.linalg.eigh(Kperp)
    alpha = float(-lam.max())
    init_mean = T_PERP@w0   # T_perp r = 0 in the exactly compatible reference
    omega = T_PERP@law0.cov@T_PERP.T
    mnorm, cnorm = np.linalg.norm(init_mean), np.linalg.norm(omega, 2)
    rows = []
    for time in times:
        F = (E*np.exp(lam*time))@E.T
        # For deterministic initialization, Cov(t)-Omega = -F Omega F.T.
        rows.append({'time': float(time), 'alpha_perp': alpha,
                     'relative_mean_norm': float(np.linalg.norm(F@init_mean)/mnorm),
                     'relative_covariance_error': float(np.linalg.norm(F@omega@F.T, 2)/cnorm),
                     'mean_envelope': float(np.exp(-alpha*time)),
                     'covariance_envelope': float(np.exp(-2*alpha*time))})
    return rows


def summarize_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ('D_in_par','D_out_par','margin_par','joint_W2_to_reference',
            'mean_error','covariance_error','stationary_error_sum',
            'difference_law_max_W2_to_reference','D_in_w2',
            'D_in_par_over_epsilon','joint_W2_over_epsilon',
            'risk_bias','risk_variance','risk_total','risk_independent',
            'risk_shared_oracle','risk_variance_reduction',
            'risk_gain_independent','risk_gain_shared','risk_g_coefficient',
            'risk_shared_lower_bound','risk_joint_certificate','risk_identity_residual')
    result = []
    for eps in sorted({r['epsilon'] for r in rows}):
        subset = [r for r in rows if r['epsilon'] == eps]
        record = {'epsilon': eps, 'directions': len(subset)}
        for key in keys:
            v = np.array([r[key] for r in subset])
            for suffix, value in (('median', np.median(v)), ('q10', np.quantile(v, .1)),
                                  ('q90', np.quantile(v, .9)), ('min', v.min()), ('max', v.max())):
                record[f'{key}_{suffix}'] = float(value)
        record['positive_margin_count'] = sum(r['margin_par'] > 0 for r in subset)
        record['positive_both_risk_gains_count'] = sum(
            r['risk_gain_independent'] > 0 and r['risk_gain_shared'] > 0 for r in subset)
        record['positive_risk_certificate_count'] = sum(
            r['risk_joint_certificate'] > 0 for r in subset)
        result.append(record)
    return result


def scaling_fits(rows: list[dict[str, Any]], max_epsilon: float) -> list[dict[str, Any]]:
    """Descriptive log-log fits along FIXED directions; no regression CI/p-values."""
    result = []
    for seed in sorted({r['direction_seed'] for r in rows}):
        subset = sorted([r for r in rows if r['direction_seed'] == seed
                         and r['epsilon'] <= max_epsilon], key=lambda r:r['epsilon'])
        for key in ('joint_W2_to_reference','D_in_par','mean_error','covariance_error'):
            usable = [r for r in subset if r[key] > 1e-14]
            if len(usable) < 2:
                continue
            x = np.log([r['epsilon'] for r in usable])
            y = np.log([r[key] for r in usable])
            slope, intercept = np.polyfit(x, y, 1)
            result.append({'direction_seed': seed, 'metric': key, 'points': len(usable),
                           'epsilon_min': usable[0]['epsilon'], 'epsilon_max': usable[-1]['epsilon'],
                           'slope': float(slope), 'intercept': float(intercept)})
    return result


# -----------------------------------------------------------------------------
# Publication figures: compact panels, native Matplotlib color defaults
# -----------------------------------------------------------------------------

PANEL_FIGSIZE = (5.2, 3.6)
RISK_PAIR_FIGSIZE = (10.8, 3.8)
ORBIT_FIGSIZE = (6.4, 4.3)


def new_figure(title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    ax = fig.add_subplot(111)
    style_axes(ax, title, xlabel, ylabel)
    return fig, ax


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)


def save_figure(fig, stem: Path) -> None:
    from io import BytesIO
    import os
    import tempfile
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not fig.get_constrained_layout():
            fig.tight_layout(pad=1.1)
        for extension in ('png', 'pdf'):
            buffer = BytesIO()
            fig.savefig(buffer, format=extension, dpi=300, bbox_inches='tight')
            payload = buffer.getvalue()
            if not payload:
                raise RuntimeError(f'Empty {extension} export for {stem.name}')
            # Finish each export before replacing its destination file.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=stem.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, stem.with_suffix('.' + extension))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    finally:
        plt.close(fig)


def plot_orbit_marginal_w2(times: Array, rows: list[dict[str, float]],
                           law0: GaussianLaw, stem: Path) -> None:
    """Export exact within-orbit and cross-orbit marginal W2 distances."""
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
                         'legend.fontsize': 10, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=ORBIT_FIGSIZE, layout='constrained')
        shown = times <= min(float(times[-1]), 12.)
        within = np.array([r['D_in_w2'] for r in rows])
        between = np.array([r['D_out_w2'] for r in rows])
        ax.semilogy(times[shown], within[shown], color='C0',
                    label='Maximum within-orbit distance')
        ax.semilogy(times[shown], between[shown], color='C1',
                    label='Minimum cross-orbit distance')
        ax.axhline(stationary_metrics(law0)['D_out_w2'], color='C1', ls=':', lw=.9)
        ax.set(title='Within-orbit agreement and cross-orbit separation', xlabel='Time',
               ylabel=r'Marginal $W_2$ distance')
        ax.legend(loc='lower left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        save_figure(fig, stem)


def plot_marginals(samples: Array, law: GaussianLaw | None, title: str,
                   stem: Path, ylim: tuple[float, float] | None = None) -> None:
    fig, ax = new_figure(title, 'Node (fixed orbit labels)', 'Parameter')
    positions = np.arange(1, N+1)
    ax.violinplot([samples[:, i] for i in range(N)], positions=positions,
                  showmeans=True, showextrema=False, widths=.68)
    if law is not None:
        ax.errorbar(positions, law.mean, 2*np.sqrt(np.maximum(np.diag(law.cov), 0)),
                    fmt='.', markersize=5, capsize=2,
                    label='Stationary mean +/- 2 SD')
        ax.legend(fontsize=11, loc='upper right')
    for boundary in np.cumsum(ORBIT_SIZES)[:-1]+.5:
        ax.axvline(boundary, linestyle=':', linewidth=.7, alpha=.65)
    ax.set_xticks(positions)
    if ylim is not None:
        ax.set_ylim(*ylim)
    save_figure(fig, stem)


def draw_band(ax, summary: list[dict[str, Any]], metric: str, label: str,
              marker: str = 'o') -> None:
    x = np.array([r['epsilon'] for r in summary])
    med = np.array([r[f'{metric}_median'] for r in summary])
    low = np.array([r[f'{metric}_q10'] for r in summary])
    high = np.array([r[f'{metric}_q90'] for r in summary])
    ax.plot(x, med, marker=marker, markersize=4, linewidth=1.6, label=label)
    ax.fill_between(x, low, high, alpha=.17)


def plot_sweep(summary: list[dict[str, Any]], figdir: Path) -> None:
    eps = np.array([r['epsilon'] for r in summary])
    for metric, ylabel, title, filename in (
        ('joint_W2_to_reference', r'$W_2(\varrho_\varepsilon,\varrho_0)$',
         'Stationary-law stability', 'stationary_W2'),
        ('D_in_par', r'$D_{\mathrm{in}}(\varepsilon)$',
         'Within-orbit agreement', 'within_orbit_scaling')):
        fig, ax = new_figure(title, r'Total defect $\varepsilon$', ylabel)
        draw_band(ax, summary, metric, 'Median; 10-90% range')
        guide = summary[0][f'{metric}_median']*(eps/eps[0])
        ax.plot(eps, guide, '--', linewidth=1.1, label='Slope-1 guide')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.legend(fontsize=11, loc='upper left')
        save_figure(fig, figdir/filename)

    fig, ax = new_figure('Orbit separation', r'Total defect $\varepsilon$',
                        r'$\mathcal{M}_{\mathrm{orb}}=D_{\mathrm{out}}-D_{\mathrm{in}}$')
    draw_band(ax, summary, 'margin_par', 'Median; 10-90% range')
    ax.plot(eps, [r['margin_par_min'] for r in summary], 's--', markersize=4,
            label='Sampled minimum')
    ax.axhline(0, linestyle=':', linewidth=1.0)
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='center left')
    save_figure(fig, figdir/'separation_margin')


def plot_transients(rows: list[dict[str, Any]], seed: int, stem: Path) -> None:
    fig, ax = new_figure('Transient stability\nOne fixed perturbation direction', 'Time',
                        r'$W_2(w_\varepsilon(t),w_0(t))/\varepsilon$')
    styles = ('-','--','-.',':','-','--')
    for index, eps in enumerate(sorted({r['epsilon'] for r in rows})):
        rr = [r for r in rows if r['epsilon'] == eps and r['direction_seed'] == seed]
        ax.plot([r['time'] for r in rr], [r['joint_W2_over_epsilon'] for r in rr],
                linestyle=styles[index % len(styles)], linewidth=1.15,
                label=rf'$\varepsilon={eps:g}$')
    ax.legend(fontsize=11, ncol=2, loc='lower right')
    save_figure(fig, stem)


def plot_transverse(rows: list[dict[str, float]], stem: Path) -> None:
    fig, ax = new_figure('Transverse moment relaxation', 'Time',
                        'Relative moment error')
    for key, label, ls in (
        ('relative_mean_norm', 'Transverse mean', '-'),
        ('relative_covariance_error', 'Covariance error', '-'),
        ('mean_envelope', r'$e^{-\alpha_\perp t}$ envelope', '--'),
        ('covariance_envelope', r'$e^{-2\alpha_\perp t}$ envelope', ':')):
        ax.semilogy([r['time'] for r in rows], [r[key] for r in rows],
                    linestyle=ls, linewidth=1.4, label=label)
    ax.legend(fontsize=11)
    save_figure(fig, stem)


def plot_risk_robustness(summary: list[dict[str, Any]], stem: Path) -> None:
    """Direction bands are empirical variation, never uncertainty intervals."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=RISK_PAIR_FIGSIZE, sharex=True)
    eps = np.array([r['epsilon'] for r in summary])
    style_axes(axes[0], 'Risk gains under perturbations',
               r'Total defect $\varepsilon$', 'Exact risk reduction')
    for metric, label, color, marker in (
            ('risk_gain_independent', 'vs. independent', 'C0', 'o'),
            ('risk_gain_shared', 'vs. shared oracle', 'C1', 's')):
        med = np.array([r[f'{metric}_median'] for r in summary])
        q10 = np.array([r[f'{metric}_q10'] for r in summary])
        q90 = np.array([r[f'{metric}_q90'] for r in summary])
        axes[0].plot(eps, med, color=color, marker=marker, markersize=4,
                     label=f'Median {label}')
        axes[0].fill_between(eps, q10, q90, color=color, alpha=.2)
        axes[0].plot(eps, [r[f'{metric}_min'] for r in summary],
                     linestyle='--', color=color, label=f'Minimum {label}')
    axes[0].axhline(0, color='0.35', linestyle=':', linewidth=1)
    axes[0].set_ylim(bottom=-.004)
    axes[0].margins(y=.25)
    axes[0].legend(fontsize=11, loc='lower left')
    style_axes(axes[1], 'Direct coefficient certificate',
               r'Total defect $\varepsilon$', 'Joint gain lower bound')
    draw_band(axes[1], summary, 'risk_joint_certificate', 'Median; 10-90% range')
    axes[1].plot(eps, [r['risk_joint_certificate_min'] for r in summary], 's--',
                 color='C0', label='Sampled minimum', markersize=4)
    axes[1].axhline(0, color='0.35', linestyle=':', linewidth=1)
    axes[1].set_ylim(bottom=-.002)
    axes[1].legend(fontsize=11, loc='lower left')
    axes[1].set_xscale('log')
    save_figure(fig, stem)


# -----------------------------------------------------------------------------
# Run, save auditable numerical data, and generate paper-ready captions
# -----------------------------------------------------------------------------


def run_affine_case(system: System, reference: System, law0: GaussianLaw,
                    w0: Array, args: argparse.Namespace, seed: int,
                    data_dir: Path) -> tuple[GaussianLaw, dict[str, Any], dict[str, Any]]:
    law = stationary_law(system)
    sim = simulate_exact_ou(system, law, w0, args.runs, args.T, args.dt, seed)
    meanT, covT = finite_law(system, law, w0, args.T)
    row = {'experiment': system.name, **compare_laws(system, reference, law, law0)}
    row.update(terminal_risk_summary(system, sim['endpoints'], meanT, covT))
    row['terminal_risk_stationarity_bias'] = row['terminal_risk_analytic']-row['risk_total']
    row['terminal_to_stationary_W2'] = gaussian_w2(meanT, covT, law.mean, law.cov)
    if row['terminal_to_stationary_W2'] > 1e-3:
        warnings.warn(f'{system.name}: terminal law is not very close to stationarity; increase --T.')
    intervals = mc_intervals(sim['endpoints'])
    for i, record in enumerate(intervals):
        record.update({'finite_time_mean': float(meanT[i]),
                       'finite_time_variance': float(covT[i,i]),
                       'stationary_mean': float(law.mean[i]),
                       'stationary_variance': float(law.cov[i,i]),
                       'finite_mean_covered': record['mean_lower'] <= meanT[i] <= record['mean_upper'],
                       'finite_variance_covered': record['variance_lower'] <= covT[i,i] <= record['variance_upper']})
    row['MC_intervals_covered_of_28'] = sum(int(r['finite_mean_covered'])
                                          +int(r['finite_variance_covered']) for r in intervals)
    write_csv(data_dir/f'{system.name}_mc_intervals.csv', intervals)
    write_csv(data_dir/f'{system.name}_terminal_risk_replicates.csv', [
        {'replicate': i+1, 'terminal_quadratic_risk': float(loss)}
        for i, loss in enumerate(np.mean(system.Q*(sim['endpoints']-system.r)**2, axis=1)/2)])
    write_csv(data_dir/f'{system.name}_local_parameters.csv', [
        {'node': i+1, 'reference_orbit': int(ORBIT_LABELS[i]+1),
         'Q': float(system.Q[i]), 'r': float(system.r[i]), 'sigma': float(system.sigma[i])}
        for i in range(N)])
    np.savez_compressed(data_dir/f'{system.name}.npz', **sim,
                        C=system.C, Q=system.Q, r=system.r, sigma=system.sigma,
                        mean=law.mean, cov=law.cov, finite_time_mean=meanT,
                        finite_time_cov=covT, orbit_labels=ORBIT_LABELS+1,
                        initial_state=w0)
    print(f"  {system.name}: D_in={row['D_in_par']:.6g}, "
          f"D_out={row['D_out_par']:.6g}, margin={row['margin_par']:.6g}", flush=True)
    return law, sim, row


def run_joint_sweep(reference: System, law0: GaussianLaw, w0: Array,
                    args: argparse.Namespace, data_dir: Path):
    rows, transient_rows = [], []
    times = np.linspace(0, args.T, args.transient_points)
    reference_t = time_laws(reference, w0, times)
    seeds = list(range(args.perturb_seed, args.perturb_seed+args.sweep_directions))
    saved_E, saved_D = [], []
    for seed in seeds:
        E, D = joint_direction(seed)
        saved_E.append(E); saved_D.append(D)
        for eps in args.eps_grid:
            system = joint_perturb(reference, eps, seed)
            law = stationary_law(system)
            row = {'direction_seed': seed, 'epsilon': eps,
                   **compare_laws(system, reference, law, law0)}
            row['D_in_par_over_epsilon'] = row['D_in_par']/eps
            row['joint_W2_over_epsilon'] = row['joint_W2_to_reference']/eps
            rr = transient_comparison(system, reference, w0, times, reference_t)
            row['max_grid_transient_W2_over_epsilon'] = max(r['joint_W2_over_epsilon'] for r in rr)
            rows.append(row)
            transient_rows.extend({'direction_seed': seed, 'epsilon': eps, **r} for r in rr)
    np.savez_compressed(data_dir/'joint_directions.npz', seeds=seeds,
                        communication_directions=np.array(saved_E),
                        local_directions=np.array(saved_D), eps_grid=args.eps_grid)
    summary = summarize_sweep(rows)
    fits = scaling_fits(rows, args.scaling_max_epsilon)
    write_csv(data_dir/'joint_epsilon_sweep.csv', rows)
    write_csv(data_dir/'joint_epsilon_sweep_summary.csv', summary)
    write_csv(data_dir/'descriptive_scaling_slopes.csv', fits)
    write_csv(data_dir/'transient_joint.csv', transient_rows)
    write_csv(data_dir/'transient_grid_summary.csv', [
        {'epsilon': eps,
         'min_over_directions_of_grid_max_W2_over_epsilon': min(
             r['max_grid_transient_W2_over_epsilon'] for r in rows if r['epsilon'] == eps),
         'median_over_directions_of_grid_max_W2_over_epsilon': float(np.median([
             r['max_grid_transient_W2_over_epsilon'] for r in rows if r['epsilon'] == eps])),
         'max_over_directions_of_grid_max_W2_over_epsilon': max(
             r['max_grid_transient_W2_over_epsilon'] for r in rows if r['epsilon'] == eps)}
        for eps in args.eps_grid])
    return rows, summary, fits, transient_rows


def run_nonlinear(reference: System, w0: Array, args: argparse.Namespace,
                  data_dir: Path, fig_dir: Path,
                  temporal_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    system = replace(reference, name='nonquadratic_extension')
    # Keep the former six-experiment script's nonlinear seed for continuity.
    sim = simulate_nonlinear(system, w0, args.runs, args.T, args.nonlinear_dt, args.seed+6,
                             snapshot_dt=getattr(args, 'snapshot_dt', .1))
    W = empirical_1d_w2_matrix(sim['endpoints'])
    row = {'experiment': 'nonquadratic_extension',
           'empirical_D_in_W2': float(W[WITHIN_MASK].max()),
           'empirical_D_out_W2': float(W[BETWEEN_MASK].min()),
           'empirical_margin_W2': float(W[BETWEEN_MASK].min()-W[WITHIN_MASK].max())}
    for key in ('step_halving_rms_per_node','step_halving_max_mean_difference',
                'step_halving_max_mean_SE','step_halving_max_variance_difference',
                'actual_coarse_dt','actual_fine_dt'):
        row[key] = float(sim[key])
    legacy_sim = {key: value for key, value in sim.items() if key not in ('times', 'snapshots')}
    np.savez_compressed(data_dir/'nonquadratic_extension.npz', **legacy_sim,
                        empirical_node_W2=W, C=system.C, Q=system.Q, r=system.r,
                        sigma=system.sigma, orbit_labels=ORBIT_LABELS+1)
    write_csv(data_dir/'nonquadratic_summary.csv', [row])
    plot_marginals(sim['endpoints'], None, 'Nonquadratic: finite-time marginals',
                   fig_dir/'nonquadratic_marginals')
    diagnostics = export_nonlinear_temporal(sim, w0, args, data_dir, fig_dir)
    if temporal_metadata is not None:
        temporal_metadata.update(diagnostics)
    print(f"  Nonquadratic paired step-halving RMS={row['step_halving_rms_per_node']:.4g}", flush=True)
    return row


def write_paper_files(out: Path, args: argparse.Namespace,
                      case_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]],
                      summary: list[dict[str, Any]], fits: list[dict[str, Any]],
                      law0: GaussianLaw, nonlinear_row: dict[str, Any] | None,
                      noise_interval: dict[str, float]) -> None:
    paper = out/'paper'; paper.mkdir(exist_ok=True)
    exact, joint = case_rows
    actual_ou_interval = args.T / max(1, int(np.ceil(args.T / args.dt)))
    sampled_min = min(r['margin_par'] for r in sweep_rows)
    positive = sum(r['margin_par'] > 0 for r in sweep_rows)
    nsystems = len(sweep_rows)
    slope_w = [r['slope'] for r in fits if r['metric'] == 'joint_W2_to_reference']
    slope_d = [r['slope'] for r in fits if r['metric'] == 'D_in_par']
    risk_positive = sum(r['risk_gain_independent'] > 0 and r['risk_gain_shared'] > 0
                        for r in sweep_rows)
    risk_certified = sum(r['risk_joint_certificate'] > 0 for r in sweep_rows)
    min_gain_ind = min(r['risk_gain_independent'] for r in sweep_rows)
    min_gain_shared = min(r['risk_gain_shared'] for r in sweep_rows)
    min_certificate = min(r['risk_joint_certificate'] for r in sweep_rows)
    source_note = ('Theory labels refer to the accompanying revised Theory.tex. '
                   'These are numerical illustrations in the scalar symmetric subclass, '
                   'not proofs of the general theorem or its converse.')
    table = (r'''\begin{tabular}{lrrr}
\hline
Setting & $D_{\rm in}$ & $D_{\rm out}$ & $\mathcal M_{\rm orb}$ \\
\hline
'''
        + f"Exact & {exact['D_in_par']:.3g} & {exact['D_out_par']:.6f} & {exact['margin_par']:.6f} " + r'\\'+'\n'
        + f"Joint ($\\varepsilon={args.representative_epsilon:g}$) & {joint['D_in_par']:.6f} & {joint['D_out_par']:.6f} & {joint['margin_par']:.6f} " + r'\\'+'\n'
        + r'\hline'+'\n'+r'\end{tabular}'+'\n')
    (paper/'numerical_table.tex').write_text(table, encoding='utf-8')
    risk_table = (r'''\begin{tabular}{lrrrrr}
\hline
Setting & Bias & Variance & $\mathcal E$ & $\mathcal E_{\rm ind}$ & $\mathcal H_{\rm one}$ \\
\hline
'''
        + '\n'.join(
            label+' & '+' & '.join(f"{row[key]:.6f}" for key in
                ('risk_bias','risk_variance','risk_total','risk_independent','risk_shared_oracle'))+r' \\'
            for label, row in [('Exact', exact), (rf'Joint ($\varepsilon={args.representative_epsilon:g}$)', joint)])
        + '\n'+r'\hline'+'\n'+r'\end{tabular}'+'\n')
    (paper/'risk_table.tex').write_text(risk_table, encoding='utf-8')
    text = rf'''% {source_note}
\subsection{{Synthetic orbit-compatibility experiments}}
We study a fixed four-orbit blow-up path with orbit sizes $(2,3,4,5)$.
Each adjacent pair of orbits forms a complete bipartite graph; there are no
within-orbit edges. We use $C_0=0.1A$, $\eta=H=1$, $Q_i=1$, and
$\sigma_i=1$. The orbit-wise local minimizers are $(1,0.4,-0.2,-1)$.
This is a scalar, symmetric, positive-curvature instance of the affine dynamics
in Section~\ref{{preliminaries}}. The reference drift is Hurwitz.
We use the original four orbit labels throughout.

\paragraph{{Orbit convergence in the reference construction.}}
Figure~\ref{{fig:orbit-marginal-w2}} displays exact one-dimensional marginal
$W_2$ distances on $[0,{min(args.T, 12.):g}]$, using the same fixed initial
state as the simulation. The curves are calculated directly from analytical
Gaussian moments. The accompanying numerical data also record node moments
and persistent within-orbit difference noise.
For a same-orbit pair in orbit $k$, with unweighted degree $d_k$,
\[
\mathbb E[(w_i(t)-w_j(t))^2]
=(w_i(0)-w_j(0))^2e^{{-2(1+0.1d_k)t}}
+\frac{{1-e^{{-2(1+0.1d_k)t}}}}{{1+0.1d_k}}.
\]
Thus marginal agreement does not mean vanishing differences along noisy
sample paths. Different stationary orbit centers are explicitly verified for
this construction, rather than inferred from orbit membership alone.

\paragraph{{Exact reference and representative joint perturbation.}}
Figure~\ref{{fig:compat-marginals}} compares exact-reference terminal samples from
{args.runs} independent realizations at $T={args.T:g}$, started from the same
fixed non-orbit-symmetric state, with the analytically computed stationary
means and variances. Quadratic trajectories use exact Ornstein--Uhlenbeck
transitions at actual observation interval {actual_ou_interval:g}
($T/\lceil T/\mathtt{{dt}}\rceil$, with requested $\mathtt{{dt}}={args.dt:g}$).
The reference has $D_{{\rm in}}={exact['D_in_par']:.3g}$ (numerical roundoff)
and $D_{{\rm out}}={exact['D_out_par']:.6f}$, illustrating equal within-orbit
stationary marginals and separated orbit prototypes, as in
Theorem~\ref{{thm:symmetry_breaking_main}}.
We then perturb communication weights and all local quantities jointly, assigning
half the total defect to $\|C_\varepsilon-C_0\|_F$ and half to
$\max_i(|\Delta Q_i|+|\Delta\sigma_i|+|\Delta r_i|)$.
At $\varepsilon={args.representative_epsilon:g}$ the within-orbit discrepancy
is {joint['D_in_par']:.6f} and the separation margin is {joint['margin_par']:.6f}.
These results illustrate approximate agreement under joint perturbations,
not pathwise equality of stochastic realizations.

\paragraph{{Stationary risk advantages.}}
The accompanying risk table reports the exact quadratic-risk decomposition
and evaluates the identity in Eq.~\eqref{{eq:dt-communication-gain-identity}}.
At the reference, the communicating risk is ${exact['risk_total']:.8f}$,
with bias ${exact['risk_bias']:.8f}$ and variance ${exact['risk_variance']:.8f}$.
Matched independent training has risk ${exact['risk_independent']:.8f}$;
the best single shared predictor has risk ${exact['risk_shared_oracle']:.8f}$.
The corresponding gains are ${exact['risk_gain_independent']:.8f}$ and
${exact['risk_gain_shared']:.8f}$. The conservative reference certificate is
$g_0={exact['risk_g_coefficient']:.8f}$, so the strict assumptions of
Corollary~\ref{{cor:dt-learning-robustness}} hold. The independent baseline
uses the same local coefficients, step-scale parameter, and noise amplitude;
the shared benchmark is restricted to one scalar predictor.

Terminal Monte Carlo estimates average all node losses within each independent
whole-system realization before estimating standard errors. The approximate
95\% $t$ intervals concern finite-time risk, with stationarity bias recorded
separately. Exact-reference and representative-perturbation estimates are
${exact['terminal_risk_mc']:.6f}$ (SE ${exact['terminal_risk_mc_se']:.6f}$) and
${joint['terminal_risk_mc']:.6f}$ (SE ${joint['terminal_risk_mc_se']:.6f}$), respectively.
These intervals are not simultaneous intervals or confidence bands across
perturbation directions. Analytical stationary risks provide the comparisons.

The accompanying noise-sweep data vary only the common noise amplitude $s$.
The sufficient condition is satisfied on the open interval
$({noise_interval['sigma_lower_open']:.6f},{noise_interval['sigma_upper_open']:.6f})$;
$s=1$ lies inside it. Increasing noise raises absolute risks while increasing
the fluctuations that communication can remove. The scan also includes the
original low-noise value $s=0.12$, where communication does not improve on
matched independent training. Orbit separation is checked independently.

\paragraph{{Quantitative robustness.}}
For each of {len(args.eps_grid)} positive defect values in
$[{min(args.eps_grid):g},{max(args.eps_grid):g}]$, we evaluate
{args.sweep_directions} pre-specified joint perturbation directions.
The same directions are scaled across defect values. Communication perturbations
preserve support and adjacent-block total weight, while local perturbation
directions are centered within each reference orbit. Thus the sampled family
is a subset of the perturbations permitted by the theory.
Figure~\ref{{fig:compat-scaling}} reports the full stationary Gaussian-law
$W_2$ distance and the maximal within-orbit parameter discrepancy.
The slope-one guides are visual references, not fitted theorem bounds.
The response curves are interpreted relative to the $O(\varepsilon)$ upper
bounds in Theorem~\ref{{thm:robust_orbit_auto-personalization}};
linearity or monotonicity is not asserted by that theorem.

Figure~\ref{{fig:compat-margin}} reports the orbit-separation margin of
Corollary~\ref{{cor:orbit_auto-personalization_margin}}.
Among the {nsystems} sampled systems, {positive} have positive margins;
the smallest sampled margin is {sampled_min:.6f}.
Shading denotes the empirical 10th--90th percentile range across perturbation
directions, not a confidence interval. The sampled minimum is not a
worst-case guarantee over all admissible perturbations. We do not infer
a critical defect, a universal constant, or a necessary compatibility condition.

Figure~\ref{{fig:compat-risk-robustness}} reports both exact risk gains over the
same {nsystems} pre-specified systems: {risk_positive} improve on both
benchmarks, with sampled minimum gains ${min_gain_ind:.8f}$ and
${min_gain_shared:.8f}$. Directly evaluating the coefficient bound at each
configuration certifies both gains for {risk_certified} systems; the smallest
joint certificate is ${min_certificate:.8f}$. This coefficient evaluation
does not estimate the perturbation constant in the corollary. Direction bands
and sampled minima describe the finite family only, not all admissible systems.

\paragraph{{Additional diagnostics.}}
We also report exact-reference transverse moment relaxation and finite-time
$W_2(w_\varepsilon(t),w_0(t))/\varepsilon$ on a {args.transient_points}-point
grid in $[0,{args.T:g}]$. The latter is a finite-grid illustration of
Theorem~\ref{{thm:robust_orbit_auto-personalization}}(i), not verification of
the supremum over all $t\ge0$. Persistent transverse noise is consistent
with distributional synchronization and does not imply zero pathwise
within-orbit disagreement.
'''
    if slope_w and slope_d:
        text += (f"\nThe median descriptive small-defect log--log slopes are "
                 f"{np.median(slope_w):.4f} for the joint stationary $W_2$ distance and "
                 f"{np.median(slope_d):.4f} for $D_{{\\rm in}}$, using points with "
                 f"$\\varepsilon\\le {args.scaling_max_epsilon:g}$. These fits do not "
                 "estimate the theorem's universal constants.\n")
    if nonlinear_row is not None:
        text += rf'''
\paragraph{{Nonquadratic extension.}}
Beyond the affine analysis, we use the population objective
\[
L_i(u)=\log(1+e^{{u-r_i}})+\log(1+e^{{-(u-r_i)}})
+\tfrac12(Q_i-\tfrac12)(u-r_i)^2.
\]
Its local minimizer and Hessian at that minimizer match the quadratic reference.
We simulate the exactly compatible system using Euler--Maruyama with paired
steps {nonlinear_row['actual_coarse_dt']:g} and {nonlinear_row['actual_fine_dt']:g}.
The paired terminal RMS discrepancy per node is
{nonlinear_row['step_halving_rms_per_node']:.3g}.
The empirical within- and cross-orbit one-dimensional Wasserstein distances are
{nonlinear_row['empirical_D_in_W2']:.5f} and {nonlinear_row['empirical_D_out_W2']:.5f}.
These are finite-time empirical quantities with sampling and discretization
error; they neither establish Gaussian stationary laws nor test the
joint-perturbation theorem for nonlinear dynamics.
'''
    (paper/'results_text.tex').write_text(text, encoding='utf-8')
    # Single-panel files and horizontal risk pairs share the appendix's compact widths.
    latex = r'''% Add \usepackage{graphicx} to the manuscript preamble.
% Paths are relative to the manuscript project root; edit this one directory.
\providecommand{\CompatFigureDir}{figures}
\providecommand{\CompatPanelWidth}{0.44\linewidth}
\providecommand{\CompatSingleWidth}{0.56\linewidth}
\providecommand{\CompatWideWidth}{0.92\linewidth}

\begin{figure}[t]
\centering
\includegraphics[width=\CompatSingleWidth]{\CompatFigureDir/orbit_marginal_w2.pdf}
\caption{Within-orbit distributional agreement and cross-orbit separation in the same reference system. The largest within-orbit one-dimensional marginal $W_2$ distance tends to zero, while the smallest cross-orbit distance tends to approximately $0.515099$ (dotted line). Curves use exact finite-time Gaussian parameters. The positive cross-orbit limit is verified for this chosen construction; orbit membership alone does not guarantee separation. These marginal Wasserstein distances differ from the parameter distances used in the perturbation study.}
\label{fig:orbit-marginal-w2}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\CompatSingleWidth]{\CompatFigureDir/exact_marginals.pdf}
\caption{Within-orbit agreement in the exact reference. Violins show node-wise terminal samples from independent SDE realizations; short horizontal marks are empirical means. Overlaid points and bars are analytical stationary means and $\pm2$ stationary standard deviations, not confidence intervals. Dotted separators mark the fixed reference orbits. Exact equality applies to the stationary marginals, not to finite-sample violin shapes or noisy sample paths.}
\label{fig:compat-marginals}
\end{figure}

\begin{figure}[t]
\centering
\begin{minipage}{\CompatPanelWidth}\centering
\includegraphics[width=\linewidth]{\CompatFigureDir/stationary_W2.pdf}\\(a) Joint stationary-law distance
\end{minipage}\hfill
\begin{minipage}{\CompatPanelWidth}\centering
\includegraphics[width=\linewidth]{\CompatFigureDir/within_orbit_scaling.pdf}\\(b) Within-orbit discrepancy
\end{minipage}
\caption{Quantitative response to jointly perturbed local quantities and communication. Distances are calculated from analytical stationary Gaussian laws rather than fitted Monte Carlo marginals. Solid curves are medians and shading is the 10th--90th percentile range over fixed perturbation directions. Dashed slope-one guides are not theoretical upper bounds. The curves provide numerical evidence consistent with the $O(\varepsilon)$ stability and agreement bounds for the sampled perturbation family.}
\label{fig:compat-scaling}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\CompatSingleWidth]{\CompatFigureDir/separation_margin.pdf}
\caption{Persistence of reference-orbit separation. The margin $\mathcal M_{\rm orb}=D_{\rm out}-D_{\rm in}$ compares the smallest cross-orbit parameter distance with the largest within-orbit distance. The additional dashed curve is the minimum margin over sampled directions, not the infimum over all admissible perturbations. The dotted zero line marks loss of this strict distance-ordering criterion; no sharp phase transition or theoretical critical defect is estimated.}
\label{fig:compat-margin}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\CompatWideWidth]{\CompatFigureDir/risk_robustness.pdf}
\caption{Risk advantages under jointly perturbed coefficients. Left: exact risk reductions against matched independent training and the best single shared predictor. Solid curves are medians, shaded regions are the 10th--90th percentile ranges, and dashed curves are sampled minima over fixed directions. Right: the direct coefficient certificate $g_\varepsilon-\max\{0,\mathcal E_{\rm ind}^\varepsilon-\mathcal H_{\rm one}^\varepsilon\}$, with the same summaries. This evaluates the coefficient inequality separately at each configuration; it does not fit a perturbation-theorem constant. Bands are not confidence intervals and minima are not uniform worst-case guarantees.}
\label{fig:compat-risk-robustness}
\end{figure}

'''
    (paper/'figures.tex').write_text(latex, encoding='utf-8')
    report = f'''# 本次运行摘要

- 参考系统的轨道距离图 orbit_marginal_w2 生成至 figures/（PDF 与 PNG）；完整矩数据保留在 data/orbit_time_metrics.csv 和 data/node_time_moments.csv，风险数据见 data/experiment_summary.csv 的 Exact 行。
- OU实际观测间隔为 {actual_ou_interval:g}=T/ceil(T/dt)，请求间隔为 {args.dt:g}。
- Exact: D_in={exact['D_in_par']:.8g}, D_out={exact['D_out_par']:.6f}, margin={exact['margin_par']:.6f}。
- Joint (epsilon={args.representative_epsilon:g}): D_in={joint['D_in_par']:.6f}, D_out={joint['D_out_par']:.6f}, margin={joint['margin_par']:.6f}。
- 共 {nsystems} 个预先指定的扰动系统，正 margin 数量 {positive}；最小样本 margin={sampled_min:.6f}。
- 参考四轨道均值：{', '.join(f'{law0.mean[o[0]]:.6f}' for o in ORBITS)}。
- 有限终止时间分布与稳态的 W2：Exact {exact['terminal_to_stationary_W2']:.3g}；Joint {joint['terminal_to_stationary_W2']:.3g}。
- 统一参考噪声 sigma=1。参考风险：通信 {exact['risk_total']:.9f} = 偏差 {exact['risk_bias']:.9f} + 方差 {exact['risk_variance']:.9f}；独立 {exact['risk_independent']:.9f}；最佳共享 {exact['risk_shared_oracle']:.9f}。
- 参考收益：相对独立 {exact['risk_gain_independent']:.9f}，相对共享 {exact['risk_gain_shared']:.9f}；g0={exact['risk_g_coefficient']:.9f}，双收益证书={exact['risk_joint_certificate']:.9f}。
- 扰动样本双收益为正 {risk_positive}/{nsystems}，直接系数证书为正 {risk_certified}/{nsystems}；最小独立收益={min_gain_ind:.9f}，最小共享收益={min_gain_shared:.9f}，最小证书={min_certificate:.9f}。
- 恒等式最大绝对残差：{max(r['risk_identity_residual'] for r in [*case_rows, *sweep_rows]):.3g}。
- 噪声扫描的严格充分条件区间：{noise_interval['sigma_lower_open']:.9f}<s<{noise_interval['sigma_upper_open']:.9f}；保留原s=0.12作为扫描点，非主要案例配置。
- Exact终端风险MC={exact['terminal_risk_mc']:.6f}，SE={exact['terminal_risk_mc_se']:.6f}，近似95%CI=[{exact['terminal_risk_mc_ci95_lower']:.6f},{exact['terminal_risk_mc_ci95_upper']:.6f}]；解析终端风险={exact['terminal_risk_analytic']:.6f}。
- Joint终端风险MC={joint['terminal_risk_mc']:.6f}，SE={joint['terminal_risk_mc_se']:.6f}，近似95%CI=[{joint['terminal_risk_mc_ci95_lower']:.6f},{joint['terminal_risk_mc_ci95_upper']:.6f}]；解析终端风险={joint['terminal_risk_analytic']:.6f}。

这里只报告本次参数与随机方向下的计算结果，不讨论 compatibility 的必要性。
独立基准使用相同eta及局部噪声；共享基准只允许一个参数，不代表所有中心化方法。
收益证书逐个系统直接由系数计算，不是拟合的理论扰动常数。风险属于给定二次模型的稳态超额风险，不是有限样本泛化保证。
MC区间以完整的独立系统实现为采样单位，保留节点相关性；方向分位带不是MC置信区间。
有限时间网格不是对 sup(t>=0) 的证明；随机方向的最小值不是全扰动空间的最坏情形。
先核对 data/*.csv，再使用对应的 figure captions；不要只看预设的橙色解析误差条。
'''
    if slope_w and slope_d:
        report += f'\n小扰动描述性 log-log 斜率中位数：W2={np.median(slope_w):.6f}，D_in={np.median(slope_d):.6f}。\n'
    (paper/'RUN_REPORT_ZH.md').write_text(report, encoding='utf-8')
    manifest = [
        {'placement':'Main Figure 1(b)', 'stem':'orbit_marginal_w2',
         'purpose':'Marginal W2 agreement within orbits and positive limiting cross-orbit distance'},
        {'placement':'Main Figure 1(a)', 'stem':'exact_marginals',
         'purpose':'Exact orbit marginal equality and chosen positive cross-orbit gaps'},
        {'placement':'Appendix E: stationary response (a)', 'stem':'stationary_W2',
         'purpose':'Stationary joint-law stability, theorem (ii)'},
        {'placement':'Appendix E: stationary response (b)', 'stem':'within_orbit_scaling',
         'purpose':'Within-orbit O(epsilon) discrepancy, theorem (iii)'},
        {'placement':'Main Figure 1(c)', 'stem':'separation_margin',
         'purpose':'Persistence of pairwise reference-orbit separation, corollary'},
        {'placement':'Main or supplement', 'stem':'risk_robustness',
         'purpose':'Exact risk gains and directly evaluated coefficient certificates under perturbations'},
        {'placement':'Supplement', 'stem':'transverse_relaxation', 'purpose':'Exact transverse moments and rate envelopes'},
        {'placement':'Supplement or main if theorem (i) is emphasized', 'stem':'transient_stability', 'purpose':'Finite-grid transient law stability'},
    ]
    manifest.extend([
        {'placement':'Appendix E: temporal validation (c)', 'stem':'joint_node_means',
         'purpose':'Exact perturbed node means with node-specific stationary limits'},
        {'placement':'Appendix E: temporal validation (d)', 'stem':'joint_marginal_w2',
         'purpose':'Within/cross reference-orbit marginal W2 with stationary limits'}])
    if nonlinear_row is not None:
        manifest.extend([
            {'placement':'Appendix E: nonquadratic (a)', 'stem':'nonquadratic_node_means',
             'purpose':'Node-wise ensemble means from the same nonlinear simulation'},
            {'placement':'Appendix E: nonquadratic (b)', 'stem':'nonquadratic_marginal_w2',
             'purpose':'Finite-sample empirical marginal W2 across time'},
            {'placement':'Supplement', 'stem':'nonquadratic_marginals', 'purpose':'Finite-time nonlinear extension only'}])
    write_csv(paper/'figure_manifest.csv', manifest)


def time_grid(horizon: float, interval: float) -> np.ndarray:
    """Include the terminal time, allowing a last interval shorter than interval."""
    if not (np.isfinite(horizon) and horizon > 0
            and np.isfinite(interval) and interval > 0):
        raise ValueError('Time horizon and observation interval must be finite and positive.')
    times = np.arange(0., horizon, interval)
    times = times[times < horizon]
    return np.r_[times, horizon]


def marginal_distances(mean: np.ndarray, covariance: np.ndarray) -> tuple[float, float]:
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.))
    W = np.hypot(mean[:, None] - mean[None, :], sd[:, None] - sd[None, :])
    return float(W[WITHIN_MASK].max()), float(W[BETWEEN_MASK].min())


def finish(fig, ax, stem: Path) -> None:
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
    ax.margins(x=.01)
    save_figure(fig, stem)


def plot_node_means(times: np.ndarray, means: np.ndarray, stem: Path,
                    title: str, empirical: bool, stationary: np.ndarray | None = None) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, layout='constrained')
    for k, orbit in enumerate(ORBITS):
        for node in orbit:
            ax.plot(times, means[:, node], color=f'C{k}', lw=1.0, alpha=.8,
                    label=f'Orbit {k + 1}' if node == orbit[0] else None)
            if stationary is not None:
                ax.axhline(stationary[node], color=f'C{k}', ls=':', lw=.65, alpha=.65)
    ax.set(title=title, xlabel='Time', ylabel='Empirical node mean' if empirical else r'$\mathbb{E}[w_i(t)]$')
    ax.legend(ncol=2, loc='lower right')
    finish(fig, ax, stem)


def plot_distances(times: np.ndarray, within: np.ndarray, between: np.ndarray,
                   stem: Path, title: str, empirical: bool,
                   stationary: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, layout='constrained')
    ax.semilogy(times, within, color='C0', lw=1.6, label='Maximum within-orbit distance')
    ax.semilogy(times, between, color='C1', lw=1.6, label='Minimum cross-orbit distance')
    if stationary is not None:
        for k, value in enumerate(stationary):
            ax.axhline(value, color=f'C{k}', ls=':', lw=1.0)
    ax.set(title=title, xlabel='Time',
           ylabel=r'Empirical marginal $W_2$' if empirical else r'Marginal $W_2$ distance')
    ax.legend(loc='upper right')
    finish(fig, ax, stem)



def export_gaussian_temporal(reference: System, joint: System,
                             reference_stationary: GaussianLaw, stationary: GaussianLaw,
                             w0: Array, args: argparse.Namespace,
                             data_dir: Path, figure_dir: Path) -> dict[str, Any]:
    """Export exact transient marginal diagnostics from the current run's coefficients."""
    import matplotlib.pyplot as plt
    times = time_grid(args.T, args.gaussian_dt)
    means, covariances = time_laws(joint, w0, times)
    reference_means, reference_covariances = time_laws(reference, w0, times)
    exact_rows, node_rows = [], []
    for t, m, c, mr, cr in zip(times, means, covariances, reference_means, reference_covariances):
        within, between = marginal_distances(m, c)
        ref_within, ref_between = marginal_distances(mr, cr)
        exact_rows.append({'time': float(t), 'D_in_W2': within, 'D_out_W2': between,
                           'margin_W2': between - within,
                           'reference_D_in_W2': ref_within, 'reference_D_out_W2': ref_between,
                           'max_node_mean_error_to_stationary': float(np.max(np.abs(m - stationary.mean)))})
        node_rows.extend({'time': float(t), 'node': i + 1, 'orbit': int(ORBIT_LABELS[i] + 1),
                          'mean': float(m[i]), 'variance': float(c[i, i]),
                          'stationary_mean': float(stationary.mean[i]),
                          'reference_mean': float(mr[i]), 'reference_variance': float(cr[i, i])}
                         for i in range(N))
    terminal_mean, terminal_cov = finite_law(joint, stationary, w0, args.T)
    terminal_mean_error = float(np.max(np.abs(means[-1] - terminal_mean)))
    terminal_covariance_error = float(np.max(np.abs(covariances[-1] - terminal_cov)))
    if not (np.allclose(means[-1], terminal_mean, rtol=1e-10, atol=1e-12)
            and np.allclose(covariances[-1], terminal_cov, rtol=1e-10, atol=1e-12)):
        raise AssertionError('Exact time laws disagree with the terminal Gaussian parameters.')
    write_csv(data_dir / 'joint_temporal_metrics.csv', exact_rows)
    write_csv(data_dir / 'joint_node_moments.csv', node_rows)
    np.savez_compressed(data_dir / 'joint_time_laws.npz', times=times, means=means,
                        covariances=covariances, reference_means=reference_means,
                        reference_covariances=reference_covariances,
                        stationary_mean=stationary.mean, stationary_covariance=stationary.cov,
                        reference_stationary_mean=reference_stationary.mean,
                        reference_stationary_covariance=reference_stationary.cov,
                        initial_state=w0, orbit_labels=ORBIT_LABELS + 1)
    joint_limits = marginal_distances(stationary.mean, stationary.cov)
    with plt.rc_context(TEMPORAL_STYLE):
        plot_node_means(times, means, figure_dir / 'joint_node_means',
                        'Perturbed node means', False, stationary.mean)
        plot_distances(times, np.array([row['D_in_W2'] for row in exact_rows]),
                        np.array([row['D_out_W2'] for row in exact_rows]),
                        figure_dir / 'joint_marginal_w2', 'Perturbed marginal distances',
                        False, joint_limits)
    return {'grid_size': len(times), 'requested_interval': args.gaussian_dt,
            'terminal_time': float(times[-1]),
            'terminal_mean_maximum_error': terminal_mean_error,
            'terminal_covariance_maximum_error': terminal_covariance_error,
            'stationary_D_in_W2': joint_limits[0], 'stationary_D_out_W2': joint_limits[1],
            'terminal_metrics': exact_rows[-1],
            'figures': ['figures/joint_node_means.pdf', 'figures/joint_marginal_w2.pdf'],
            'interpretation': 'Exact Gaussian moments and marginal W2, not Monte Carlo estimates or the parameter metric d_par.'}


TEMPORAL_STYLE = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
                  'xtick.labelsize': 14, 'ytick.labelsize': 14,
                  'legend.fontsize': 11, 'pdf.fonttype': 42}


def export_nonlinear_temporal(simulation: dict[str, Any], w0: Array,
                              args: argparse.Namespace, data_dir: Path,
                              figure_dir: Path) -> dict[str, Any]:
    """Analyze snapshots recorded by the original, single nonlinear simulation."""
    import matplotlib.pyplot as plt
    snapshots = simulation['snapshots']
    if not np.array_equal(snapshots[-1], simulation['endpoints']):
        raise AssertionError('The last nonlinear snapshot must equal the fine endpoints.')
    snapshot_means = snapshots.mean(axis=1)
    snapshot_se = snapshots.std(axis=1, ddof=1) / np.sqrt(args.runs)
    nonlinear_rows, nonlinear_node_rows = [], []
    for index, (t, samples) in enumerate(zip(simulation['times'], snapshots)):
        W = empirical_1d_w2_matrix(samples)
        within = float(W[WITHIN_MASK].max())
        between = float(W[BETWEEN_MASK].min())
        mean_gaps = np.abs(snapshot_means[index, :, None] - snapshot_means[index, None, :])
        nonlinear_rows.append({'time': float(t), 'empirical_D_in_W2': within,
                               'empirical_D_out_W2': between, 'empirical_margin_W2': between - within,
                               'maximum_within_orbit_mean_gap': float(mean_gaps[WITHIN_MASK].max()),
                               'minimum_cross_orbit_mean_gap': float(mean_gaps[BETWEEN_MASK].min()),
                               'maximum_node_mean_SE': float(snapshot_se[index].max())})
        nonlinear_node_rows.extend({'time': float(t), 'node': i + 1,
                                    'orbit': int(ORBIT_LABELS[i] + 1),
                                    'mean': float(snapshot_means[index, i]),
                                    'mean_SE': float(snapshot_se[index, i])}
                                   for i in range(N))
    write_csv(data_dir / 'nonquadratic_temporal_metrics.csv', nonlinear_rows)
    write_csv(data_dir / 'nonquadratic_node_moments.csv', nonlinear_node_rows)
    np.savez_compressed(data_dir / 'nonquadratic_snapshots.npz', **simulation,
                        node_means=snapshot_means, node_mean_SE=snapshot_se,
                        orbit_labels=ORBIT_LABELS + 1, initial_state=w0)
    with plt.rc_context(TEMPORAL_STYLE):
        plot_node_means(simulation['times'], snapshot_means, figure_dir / 'nonquadratic_node_means',
                        'Nonquadratic node means', True)
        plot_distances(simulation['times'],
                        np.array([row['empirical_D_in_W2'] for row in nonlinear_rows]),
                        np.array([row['empirical_D_out_W2'] for row in nonlinear_rows]),
                        figure_dir / 'nonquadratic_marginal_w2', 'Nonquadratic marginal distances', True)
    return {'snapshot_count': len(simulation['times']),
            'requested_snapshot_interval': getattr(args, 'snapshot_dt', .1),
            'snapshot_schedule': 'Requests round to the nearest coarse-step boundary; duplicates removed and both endpoints included. Actual times saved in nonquadratic_snapshots.npz.',
            'actual_coarse_dt': simulation['actual_coarse_dt'],
            'actual_fine_dt': simulation['actual_fine_dt'],
            'seed': args.seed + 6, 'independent_whole_system_replicates': args.runs,
            'recorded_in_original_simulation': True,
            'last_snapshot_equals_fine_endpoints': True,
            'initial_metrics': nonlinear_rows[0], 'terminal_metrics': nonlinear_rows[-1],
            'figures': ['figures/nonquadratic_node_means.pdf', 'figures/nonquadratic_marginal_w2.pdf'],
            'interpretation': [
                'Snapshots use fine Euler paths from independent whole-system replicates; nodes within a replicate are dependent.',
                'Node-mean SE is computed across replicates separately at each time and node; no simultaneous confidence claim is made.',
                'Empirical W2 uses sorted equal-size marginal samples; its within-orbit residual includes finite-sample error.',
                'No smoothing, time averaging, bootstrap bands, or exact stationary nonlinear law is used.',
                'Finite-horizon diagnostics do not prove stationarity, limiting equality, monotonic convergence, or an affine Gaussian theorem for nonlinear drift.',
                'Positive noise persists: means and marginal laws, not individual sample paths, are compared.']}


def write_temporal_paper_files(out: Path, args: argparse.Namespace,
                                temporal: dict[str, Any]) -> None:
    """Optional figure snippets; the complete revised Section E already includes them."""
    paper = out / 'paper'
    def panel(stem: str, caption: str, width: str = '0.44') -> str:
        return (r'\begin{subfigure}[b]{' + width + r'\linewidth}\centering' + '\n'
                + r'\includegraphics[width=\linewidth]{figures/' + stem + '.pdf}\n'
                + r'\caption{' + caption + '}\n' + r'\end{subfigure}' + '\n')
    latex = r'''% Requires graphicx, caption, subcaption. Copy PDFs to the manuscript figures/ folder.
% Optional snippets: do not insert again if using Section_E_revised.tex.
\begin{figure}[!htbp]
\centering
\captionsetup{font=small,skip=4pt}
\captionsetup[subfigure]{font=small,labelfont=bf,skip=2pt}
'''
    latex += panel('transverse_relaxation', 'Reference transverse relaxation') + r'\hfill' + '\n'
    latex += panel('transient_stability', 'Perturbed joint-law stability') + '\n' + r'\medskip' + '\n'
    latex += panel('joint_node_means', 'Perturbed node means') + r'\hfill' + '\n'
    latex += panel('joint_marginal_w2', 'Perturbed marginal distances')
    latex += (r'\caption{Temporal validation from exact affine moments. '
              r'(a) Reference transverse errors and exponential bounds. '
              r'(b) Joint-law $W_2$ distance divided by $\varepsilon$, for direction seed '
              + f'{args.perturb_seed} and {len(args.eps_grid)} defect magnitudes, at {args.transient_points} times in $[0,{args.T:g}]$. '
              + r'(c)--(d) Node means and maximum within-orbit and minimum cross-orbit marginal $W_2$ distances for '
              + rf'$\varepsilon={args.representative_epsilon:g}$, seed {args.perturb_seed}, at {temporal["gaussian"]["grid_size"]} times with requested spacing ${args.gaussian_dt:g}$. '
              + r'Colors indicate reference orbits; dotted lines are stationary limits. Approximate compatibility allows nonzero within-orbit limits.}'
              + '\n' + r'\label{fig:app_temporal_validation}' + '\n' + r'\end{figure}' + '\n')
    if args.nonquadratic:
        nonlinear = temporal['nonlinear']
        latex += '\n' + r'''\begin{figure}[!htbp]
\centering
\captionsetup{font=small,skip=4pt}
\captionsetup[subfigure]{font=small,labelfont=bf,skip=2pt}
'''
        latex += panel('nonquadratic_node_means', 'Node-wise sample means') + r'\hfill' + '\n'
        latex += panel('nonquadratic_marginal_w2', 'Empirical marginal distances') + '\n' + r'\medskip' + '\n'
        latex += panel('nonquadratic_marginals', 'Terminal node marginals', '0.64')
        latex += (r'\caption{Nonquadratic marginal grouping from '
                  + f'{args.runs} independent realizations, using fine Euler--Maruyama step ${nonlinear["actual_fine_dt"]:g}$ and seed ${args.seed+6}$. '
                  + r'(a) Node-wise means; colors indicate reference orbits. '
                  + r'(b) Maximum within-orbit and minimum cross-orbit empirical marginal $W_2$ distances. '
                  + f'Panels (a)--(b) use {nonlinear["snapshot_count"]} snapshots in $[0,{args.T:g}]$, '
                  + r'with requested times rounded to coarse-step boundaries and no smoothing or uncertainty bands. '
                  + f'(c) Endpoint samples at $T={args.T:g}$. '
                  + r'Curves retain Monte Carlo fluctuations; within-orbit empirical distances include finite-sample error.}'
                  + '\n' + r'\label{fig:app_nonquadratic}' + '\n'
                  + r'\label{fig:app_nonquadratic_marginals}' + '\n' + r'\end{figure}' + '\n')
    (paper/'temporal_figures.tex').write_text(latex, encoding='utf-8')
    with (paper/'RUN_REPORT_ZH.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## 合并后的时间诊断\n\n'
                     f'- Gaussian 时间网格：{temporal["gaussian"]["grid_size"]} 个时刻；请求间隔 {args.gaussian_dt:g}。\n'
                     '- 新增 joint_node_means 与 joint_marginal_w2；使用本次运行的精确矩。\n')
        if args.nonquadratic:
            stream.write(f'- 非二次快照：{temporal["nonlinear"]["snapshot_count"]} 个时刻；在原模拟循环内记录，不进行第二次模拟。\n'
                         '- 新增 nonquadratic_node_means 与 nonquadratic_marginal_w2；实际时间与所有样本保存在 data/nonquadratic_snapshots.npz。\n')
        stream.write('- 时间诊断数据位于 data/，对应图位于 figures/；完整图清单见 paper/figure_manifest.csv。\n'
                     '- metadata.json 的 temporal_diagnostics 记录实际设定和本次内部一致性检查。\n'
                     '- paper/temporal_figures.tex 为可选插图片段；完整 E 节已含对应组合图，勿重复插入。\n')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results_compatibility', help='A new or empty output directory')
    p.add_argument('--runs', type=int, default=600, help='Independent SDE realizations per displayed case')
    p.add_argument('--T', type=float, default=30., help='Time horizon')
    p.add_argument('--dt', type=float, default=.02, help='Exact OU observation interval, not an Euler step')
    p.add_argument('--representative-epsilon', type=float, default=.1)
    p.add_argument('--eps-grid', type=float, nargs='+', default=[.001,.003,.01,.03,.1,.2])
    p.add_argument('--sweep-directions', type=int, default=20,
                   help='Fixed directions per epsilon (default: 20 x 6 = 120 systems)')
    p.add_argument('--perturb-seed', type=int, default=101)
    p.add_argument('--seed', type=int, default=30000, help='Simulation seed base')
    p.add_argument('--init-seed', type=int, default=2027)
    p.add_argument('--transient-points', type=int, default=101, help='Finite-time diagnostic grid size')
    p.add_argument('--scaling-max-epsilon', type=float, default=.03, help='Maximum defect for descriptive slopes')
    experiment_mode = p.add_mutually_exclusive_group()
    experiment_mode.add_argument('--nonquadratic', dest='nonquadratic', action='store_true',
                   help='Run the matched nonlinear extension with sigma=1 (enabled by default; no affine risk certificate is claimed for it)')
    experiment_mode.add_argument('--quadratic-only', dest='nonquadratic', action='store_false',
                   help='Run only the quadratic experiments')
    p.set_defaults(nonquadratic=True)
    p.add_argument('--nonlinear-dt', type=float, default=.01, help='Coarse EM step; fine step is half')
    p.add_argument('--gaussian-dt', type=float, default=.05,
                   help='Exact Gaussian moment observation spacing, not an integrator step')
    p.add_argument('--snapshot-dt', type=float, default=.1,
                   help='Nonlinear snapshot spacing; requests round to nearest coarse-step times')
    args = p.parse_args(argv)
    numbers = [args.T, args.dt, args.representative_epsilon, args.scaling_max_epsilon,
               args.nonlinear_dt, args.gaussian_dt, args.snapshot_dt, *args.eps_grid]
    if not all(np.isfinite(x) and x > 0 for x in numbers):
        p.error('All times and defect values must be finite and strictly positive.')
    if args.runs < 3 or args.sweep_directions < 1 or args.transient_points < 2:
        p.error('Require runs>=3, sweep-directions>=1, transient-points>=2.')
    if min(args.seed, args.perturb_seed, args.init_seed) < 0:
        p.error('Seeds must be nonnegative integers.')
    args.eps_grid = sorted(set(args.eps_grid))
    return args


def run_all(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'{out} already exists and is not empty. Choose a new --out; previous results are preserved.')
    reference = reference_system()
    validate_system(reference)
    if symmetry_generator_defect(reference) > 1e-12:
        raise AssertionError('Reference symmetry failed.')
    law0 = stationary_law(reference)
    joint = replace(joint_perturb(reference, args.representative_epsilon, args.perturb_seed), name='approx_joint')
    # Fail BEFORE expensive simulation if any planned perturbation is inadmissible.
    for seed in range(args.perturb_seed, args.perturb_seed+args.sweep_directions):
        for eps in args.eps_grid:
            try:
                joint_perturb(reference, eps, seed)
            except ValueError as exc:
                raise ValueError(f'Invalid joint perturbation at epsilon={eps}, seed={seed}. '
                                 'Reduce the defect; no clipping or resampling is performed.') from exc
    w0 = np.random.default_rng(args.init_seed).normal(0, 1.5, N)
    if np.linalg.norm(T_PERP@w0) < 1e-8:
        raise ValueError('Initial state is accidentally orbit-symmetric; change --init-seed.')
    data_dir, figdir = out/'data', out/'figures'
    data_dir.mkdir(parents=True, exist_ok=True); figdir.mkdir(exist_ok=True)
    np.savez_compressed(data_dir/'reference_setup.npz', C=reference.C,
                        Q=reference.Q, r=reference.r, sigma=reference.sigma,
                        initial_state=w0, orbit_labels=ORBIT_LABELS+1,
                        T_parallel=T_PAR, T_perp=T_PERP)
    print('Experiment 1/3: exact compatibility', flush=True)
    exactlaw, exactsim, exactrow = run_affine_case(reference, reference, law0, w0, args, args.seed+1, data_dir)
    print('Experiment 2/3: representative joint approximate compatibility', flush=True)
    jointlaw, jointsim, jointrow = run_affine_case(joint, reference, law0, w0, args, args.seed+2, data_dir)
    temporal = {'gaussian': export_gaussian_temporal(
        reference, joint, law0, jointlaw, w0, args, data_dir, figdir)}
    case_rows = [exactrow, jointrow]
    write_csv(data_dir/'experiment_summary.csv', case_rows)
    # Orbit marginal distances use the run's shared initial state and horizon.
    orbit_times = np.unique(np.r_[np.linspace(0, min(args.T, 12.), 241), min(args.T, 6.), args.T])
    orbit_rows, node_rows, _, pair_error = orbit_time_diagnostics(
        reference, law0, w0, orbit_times)
    write_csv(data_dir/'orbit_time_metrics.csv', orbit_rows)
    write_csv(data_dir/'node_time_moments.csv', node_rows)
    plot_orbit_marginal_w2(orbit_times, orbit_rows, law0, figdir/'orbit_marginal_w2')
    noise_rows, noise_interval = noise_sweep(reference)
    write_csv(data_dir/'risk_noise_sweep.csv', noise_rows)
    (data_dir/'risk_noise_certificate_interval.json').write_text(
        json.dumps(noise_interval, indent=2, allow_nan=False), encoding='utf-8')
    ymin = min(exactsim['endpoints'].min(), jointsim['endpoints'].min())
    ymax = max(exactsim['endpoints'].max(), jointsim['endpoints'].max())
    pad = .10*(ymax-ymin)
    ylim = (float(ymin-pad), float(ymax+pad))
    plot_marginals(exactsim['endpoints'], exactlaw, 'Exact compatibility',
                   figdir/'exact_marginals', ylim)
    print('Experiment 3/3: joint epsilon sweep and finite-time diagnostics', flush=True)
    sweep, summary, fits, transients = run_joint_sweep(reference, law0, w0, args, data_dir)
    plot_sweep(summary, figdir)
    plot_risk_robustness(summary, figdir/'risk_robustness')
    plot_transients(transients, args.perturb_seed, figdir/'transient_stability')
    transverse = transverse_relaxation(reference, law0, w0, np.linspace(0, min(args.T,8.),161))
    write_csv(data_dir/'reference_transverse_relaxation.csv', transverse)
    plot_transverse(transverse, figdir/'transverse_relaxation')
    nonlin = None
    if args.nonquadratic:
        print('Nonquadratic extension: population objective', flush=True)
        temporal['nonlinear'] = {}
        nonlin = run_nonlinear(reference, w0, args, data_dir, figdir, temporal['nonlinear'])
    write_paper_files(out, args, case_rows, sweep, summary, fits, law0, nonlin, noise_interval)
    write_temporal_paper_files(out, args, temporal)
    source_path = Path(__file__)
    metadata = {
        'script_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
        'arguments': vars(args), 'orbit_sizes': ORBIT_SIZES.tolist(),
        'temporal_diagnostics': temporal,
        'actual_ou_observation_interval': args.T / max(1, int(np.ceil(args.T / args.dt))),
        'orbit_convergence_diagnostics': {
            'figures': ['figures/orbit_marginal_w2.pdf'],
            'moment_data': 'data/node_time_moments.csv',
            'distance_and_pair_data': 'data/orbit_time_metrics.csv',
            'initial_state_seed': args.init_seed,
            'display_horizon': min(args.T, 12.),
            'time_grid_points': len(orbit_times), 'terminal_time': args.T,
            'pair_OU_formula_max_error': pair_error,
            'terminal_metrics': orbit_rows[-1]},
        'reference_edges': int(np.count_nonzero(reference.C)//2),
        'reference_full_stability_margin': law0.stability_margin,
        'reference_transverse_margin': transverse[0]['alpha_perp'],
        'reference_orbit_means': [float(law0.mean[o[0]]) for o in ORBITS],
        'reference_orbit_variances': [float(law0.cov[o[0],o[0]]) for o in ORBITS],
        'reference_sigma': reference.sigma.tolist(),
        'reference_risk_metrics': stationary_risk_metrics(reference, law0),
        'common_noise_certificate_interval': noise_interval,
        'sampled_systems': len(sweep),
        'sampled_positive_margins': sum(r['margin_par']>0 for r in sweep),
        'minimum_sampled_margin': min(r['margin_par'] for r in sweep),
        'sampled_positive_both_risk_gains': sum(
            r['risk_gain_independent']>0 and r['risk_gain_shared']>0 for r in sweep),
        'sampled_positive_direct_risk_certificates': sum(
            r['risk_joint_certificate']>0 for r in sweep),
        'minimum_sampled_gain_independent': min(r['risk_gain_independent'] for r in sweep),
        'minimum_sampled_gain_shared': min(r['risk_gain_shared'] for r in sweep),
        'minimum_sampled_joint_risk_certificate': min(r['risk_joint_certificate'] for r in sweep),
        'maximum_risk_identity_residual': max(r['risk_identity_residual'] for r in [*case_rows, *sweep]),
        'figure_fonts': {'title': 18, 'axis_label': 16, 'major_tick': 14, 'minor_tick': 13, 'legend': 11},
        'figure_layout': {'panel_inches': PANEL_FIGSIZE, 'risk_pair_inches': RISK_PAIR_FIGSIZE,
                          'orbit_individual_inches': ORBIT_FIGSIZE,
                          'risk_pair_orientation': 'horizontal', 'latex_panel_width': '0.44\\linewidth',
                          'latex_single_width': '0.56\\linewidth', 'latex_wide_width': '0.92\\linewidth'},
        'notes': [
            'The orbit marginal W2 figure uses exact finite-time laws from the same deterministic initial state.',
            'The affine experiments evaluate positive implications, not necessity or a converse.',
            'Compatibility alone does not force distinct cross-orbit prototypes; their gaps are checked explicitly.',
            'Stationary-law sweeps are analytical computations, not fitted Monte Carlo distributions.',
            'All direction seeds are pre-specified; invalid perturbations raise errors rather than being clipped or resampled.',
            'The 10--90% bands describe sampled-direction variation and are not confidence intervals.',
            'The minimum over sampled directions is not a universal worst case.',
            'No monotonicity, sharp critical defect, or theorem constant is estimated.',
            'Terminal-law bias is reported separately from Monte Carlo uncertainty.',
            'Terminal-risk t intervals are approximate 95% intervals using independent whole-system replicates, not independent nodes.',
            'The independent-risk baseline fixes eta and all local coefficients, including diffusion.',
            'The shared oracle optimizes a single parameter, not arbitrary centralized multitask predictors.',
            'risk_g_coefficient and risk_joint_certificate directly evaluate a bound at each configuration; they do not estimate C_risk.',
            'The common-noise certificate boundaries are algebraic sufficient-condition boundaries, not necessary risk thresholds.',
            'All affine and nonlinear reference runs use sigma=1; sigma=0.12 appears only in the noise scan.',
            'The finite-time grid cannot verify a supremum over an infinite time interval.',
            'Nonlinear results, when included, are finite-time empirical evidence outside the affine theorem.',
            'The simulation uses prescribed population drift and diffusion, not training on a finite dataset.',
        ]}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Sampled positive margins: {metadata['sampled_positive_margins']}/{len(sweep)}; "
          f"minimum={metadata['minimum_sampled_margin']:.6f}", flush=True)
    print(f"Sampled positive risk gains: {metadata['sampled_positive_both_risk_gains']}/{len(sweep)}; "
          f"direct certificates: {metadata['sampled_positive_direct_risk_certificates']}/{len(sweep)}; "
          f"minimum certificate={metadata['minimum_sampled_joint_risk_certificate']:.6f}", flush=True)
    print(f'Done. Results written to: {out.resolve()}', flush=True)


if __name__ == '__main__':
    try:
        run_all(parse_args())
    except (ValueError, FileExistsError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
