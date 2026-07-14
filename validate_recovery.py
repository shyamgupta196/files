"""
Step 5: Validate the inference method on simulated ("known ground truth")
data.

  - Parameter recovery (posterior mean/median vs. true theta, scatter plots)
  - RMSE & bias per parameter
  - Simulation-Based Calibration (SBC): rank statistic of the true parameter
    within its own posterior sample should be Uniform if the amortized
    posterior is well-calibrated
  - Coverage analysis: for a grid of central credible intervals (e.g. 50%,
    80%, 90%, 95%), check the empirical proportion of simulated datasets for
    which the true parameter falls inside the interval matches the nominal
    level

This script assumes you have already run bayesflow_model.train() (or
train.py) and saved an approximator. It draws NEW test parameter sets from
the prior (not used in training, since training uses online/on-the-fly
simulation) and simulates held-out test datasets with rdm_dr_simulator, so
these diagnostics are computed on genuinely unseen ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt

from rdm_dr_simulator import PARAM_NAMES, sample_prior, simulate_batch


def _normalize_theta(theta):
    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta[:, None]
    return theta


def _normalize_posterior_samples(posterior_samples):
    posterior_samples = np.asarray(posterior_samples)
    if posterior_samples.ndim == 2:
        posterior_samples = posterior_samples[:, :, None]
    return posterior_samples


def _param_labels(theta_true, posterior_samples):
    n_params = min(theta_true.shape[1], posterior_samples.shape[2], len(PARAM_NAMES))
    if n_params < len(PARAM_NAMES):
        print(f"Warning: using {n_params} parameter dimension(s) for validation plots; "
              f"expected {len(PARAM_NAMES)}.")
    return PARAM_NAMES[:n_params]


def generate_validation_set(n_datasets=200, n_obs=300, dt=0.005, t_max=2.5,
                             dr_window=0.25, seed=0):
    rng = np.random.default_rng(seed)
    theta_true = sample_prior(n_datasets, rng=rng)
    data = simulate_batch(theta_true, n_obs=n_obs, dt=dt, t_max=t_max,
                           dr_window=dr_window, rng=rng)
    return theta_true, data


def get_posterior_samples(approximator, data, num_samples=1000):
    """Draw `num_samples` posterior samples for each of the datasets in
    `data` (shape (n_datasets, n_obs, 4)).

    Returns array of shape (n_datasets, num_samples, n_params).
    """
    import bayesflow as bf  # noqa: F401  (import only needed if user wants type hints)

    conditions = {"trials": data}
    samples_dict = approximator.sample(conditions=conditions, num_samples=num_samples)
    # BayesFlow inverts the adapter's `.concatenate(...)` transform on output,
    # so `sample` returns ONE array per parameter (keyed by PARAM_NAMES), each
    # of shape (n_datasets, num_samples, 1) -- NOT a single "inference_variables"
    # array. Stack them back into (n_datasets, num_samples, n_params) in
    # PARAM_NAMES order. (Older/other configs may return "inference_variables"
    # directly; handle that too.)
    if "inference_variables" in samples_dict:
        samples = np.asarray(samples_dict["inference_variables"])
    else:
        n_datasets = data.shape[0]
        cols = [np.asarray(samples_dict[name]).reshape(n_datasets, num_samples, -1)
                for name in PARAM_NAMES]
        samples = np.concatenate(cols, axis=-1)  # (n_datasets, num_samples, n_params)
    return samples


# ---------------------------------------------------------------------------
# Parameter recovery + RMSE/bias
# ---------------------------------------------------------------------------

def parameter_recovery(theta_true, posterior_samples, out_png="recovery.png"):
    theta_true = _normalize_theta(theta_true)
    posterior_samples = _normalize_posterior_samples(posterior_samples)
    labels = _param_labels(theta_true, posterior_samples)

    post_mean = posterior_samples.mean(axis=1)  # (n_datasets, n_params)
    n_params = len(labels)

    rmse = np.sqrt(np.mean((post_mean[:, :n_params] - theta_true[:, :n_params]) ** 2, axis=0))
    bias = np.mean(post_mean[:, :n_params] - theta_true[:, :n_params], axis=0)

    print("Parameter recovery summary (posterior mean vs. true value):")
    for i, name in enumerate(labels):
        print(f"  {name:5s}  RMSE = {rmse[i]:.4f}   bias = {bias[i]:+.4f}")

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(labels):
        ax = axes[i]
        lo = min(theta_true[:, i].min(), post_mean[:, i].min())
        hi = max(theta_true[:, i].max(), post_mean[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(theta_true[:, i], post_mean[:, i], s=10, alpha=0.5)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"posterior mean {name}")
        ax.set_title(f"{name}\nRMSE={rmse[i]:.3f}, bias={bias[i]:+.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    return rmse, bias


# ---------------------------------------------------------------------------
# Simulation-Based Calibration (SBC)
# ---------------------------------------------------------------------------

def sbc_ranks(theta_true, posterior_samples):
    """Rank of the true parameter value among the posterior samples, for
    each dataset and parameter. Under perfect calibration these ranks are
    Uniform(0, num_samples) (Talts et al., 2018)."""
    theta_true = _normalize_theta(theta_true)
    posterior_samples = _normalize_posterior_samples(posterior_samples)
    n_datasets, num_samples, n_params = posterior_samples.shape
    ranks = np.zeros((n_datasets, n_params), dtype=np.int64)
    for p in range(n_params):
        ranks[:, p] = (posterior_samples[:, :, p] < theta_true[:, p:p + 1]).sum(axis=1)
    return ranks  # values in [0, num_samples]


def sbc_plot(ranks, num_samples, out_png="sbc.png"):
    ranks = np.asarray(ranks)
    n_params = ranks.shape[1]
    n_datasets = ranks.shape[0]
    n_bins = 20
    expected = n_datasets / n_bins

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3.5))
    axes = np.atleast_1d(axes)
    for p in range(n_params):
        ax = axes[p]
        ax.hist(ranks[:, p], bins=n_bins, range=(0, num_samples), color="tab:blue", alpha=0.7)
        ax.axhline(expected, color="k", ls="--", lw=1)
        ax.set_title(PARAM_NAMES[p] if p < len(PARAM_NAMES) else f"param_{p}")
        ax.set_xlabel("rank statistic")
        if p == 0:
            ax.set_ylabel("count")
    plt.suptitle("Simulation-Based Calibration (should be ~Uniform)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def sbc_ecdf_plot(ranks, num_samples, out_png="sbc_ecdf.png"):
    """Alternative, often more sensitive, calibration_ecdf-style plot: the
    ECDF of normalized ranks against the diagonal, with a simulated envelope
    (BayesFlow has bf.diagnostics.plots.calibration_ecdf for this -- this is
    a minimal from-scratch reimplementation in case that call fails)."""
    ranks = np.asarray(ranks)
    n_params = ranks.shape[1]
    n_datasets = ranks.shape[0]
    normalized = ranks / num_samples  # in [0,1]

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3.5))
    axes = np.atleast_1d(axes)
    x = np.linspace(0, 1, 200)
    for p in range(n_params):
        ax = axes[p]
        sorted_ranks = np.sort(normalized[:, p])
        ecdf = np.arange(1, n_datasets + 1) / n_datasets
        ax.plot(sorted_ranks, ecdf - x[np.searchsorted(x, sorted_ranks).clip(max=len(x)-1)], color="tab:blue")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_title(PARAM_NAMES[p] if p < len(PARAM_NAMES) else f"param_{p}")
        ax.set_xlabel("normalized rank")
        if p == 0:
            ax.set_ylabel("ECDF - identity")
    plt.suptitle("SBC ECDF deviation (should hover around 0)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def coverage_analysis(theta_true, posterior_samples, levels=(0.5, 0.8, 0.9, 0.95),
                       out_png="coverage.png"):
    theta_true = _normalize_theta(theta_true)
    posterior_samples = _normalize_posterior_samples(posterior_samples)
    n_datasets, num_samples, n_params = posterior_samples.shape
    labels = _param_labels(theta_true, posterior_samples)
    empirical = np.zeros((len(levels), n_params))

    for li, level in enumerate(levels):
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        lo = np.quantile(posterior_samples, lower_q, axis=1)  # (n_datasets, n_params)
        hi = np.quantile(posterior_samples, upper_q, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        empirical[li] = inside.mean(axis=0)

    print("Coverage (nominal vs. empirical), per parameter:")
    for li, level in enumerate(levels):
        row = "  ".join(f"{labels[p]}={empirical[li, p]:.3f}" for p in range(n_params))
        print(f"  nominal {level:.2f}: {row}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for p in range(n_params):
        ax.plot(levels, empirical[:, p], "o-", label=labels[p])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("nominal credible-interval level")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage analysis")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    return empirical


# ---------------------------------------------------------------------------
def run_all(approximator, n_datasets=200, n_obs=300, num_posterior_samples=1000, seed=0):
    theta_true, data = generate_validation_set(n_datasets=n_datasets, n_obs=n_obs, seed=seed)
    posterior_samples = get_posterior_samples(approximator, data, num_samples=num_posterior_samples)

    parameter_recovery(theta_true, posterior_samples)
    ranks = sbc_ranks(theta_true, posterior_samples)
    sbc_plot(ranks, num_posterior_samples)
    sbc_ecdf_plot(ranks, num_posterior_samples)
    coverage_analysis(theta_true, posterior_samples)


if __name__ == "__main__":
    import bayesflow as bf
    import keras
    approximator = keras.saving.load_model("rdm_dr_approximator1.keras")
    run_all(approximator)
