"""
Step 7: Posterior predictive checks (PPC) + posterior analysis on the real
data.

Reproduces the spirit of the paper's Fig. 2 CDF plots: empirical response
time quantiles + double-response proportions (dots) vs. model-predicted
quantiles/proportions from simulating the fitted model at posterior draws
(crosses/lines).

This module only depends on numpy/matplotlib/rdm_dr_simulator, so (unlike
bayesflow_model.py / validate_recovery.py / fit_real_data.py) it can be, and
was, executed and checked in this sandbox using synthetic stand-ins for
"real data" and "posterior samples" -- see the __main__ block and the
project README for the demo output.
"""

import numpy as np
import matplotlib.pyplot as plt

from rdm_dr_simulator import simulate_batch, PARAM_NAMES

QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def empirical_cdf_summary(trials):
    """trials: array (n_obs, 4) = [rt, correct, dr, drt].

    Returns a dict with the quantities needed for a Fig.-2-style plot:
    RT quantiles for correct/error responses, and P(DR|correct), P(DR|error).
    """
    rt, correct, dr, drt = trials[:, 0], trials[:, 1], trials[:, 2], trials[:, 3]
    out = {}
    for label, mask_val in [("correct", 1), ("error", 0)]:
        mask = correct == mask_val
        n = mask.sum()
        if n == 0:
            out[f"{label}_quantiles"] = np.full(len(QUANTILES), np.nan)
            out[f"{label}_prop"] = 0.0
            out[f"{label}_dr_prop"] = np.nan
            continue
        out[f"{label}_quantiles"] = np.quantile(rt[mask], QUANTILES)
        out[f"{label}_prop"] = n / len(rt)
        out[f"{label}_dr_prop"] = dr[mask].mean()
    return out


def posterior_predictive_summaries(posterior_samples, n_obs, n_ppc_draws=50,
                                    dt=0.005, t_max=2.5, dr_window=0.25, seed=0):
    """Simulate `n_ppc_draws` synthetic datasets, one per randomly-chosen
    posterior draw, each with `n_obs` trials, and summarise each the same
    way as `empirical_cdf_summary`. Returns a list of summary dicts."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(posterior_samples.shape[0], size=n_ppc_draws, replace=False)
    theta_draws = posterior_samples[idx]  # (n_ppc_draws, n_params)

    data = simulate_batch(theta_draws, n_obs=n_obs, dt=dt, t_max=t_max,
                           dr_window=dr_window, rng=rng)  # (n_ppc_draws, n_obs, 4)
    summaries = [empirical_cdf_summary(data[i]) for i in range(n_ppc_draws)]
    return summaries


def plot_ppc(trials_real, posterior_samples, n_ppc_draws=50, out_png="ppc.png",
             title=""):
    """Fig.-2-style CDF plot: empirical dots vs. posterior-predictive
    crosses+lines, for correct (green) and error (red) responses, plus
    double-response proportions at x=0."""
    n_obs = trials_real.shape[0]
    emp = empirical_cdf_summary(trials_real)
    ppc_summaries = posterior_predictive_summaries(posterior_samples, n_obs,
                                                     n_ppc_draws=n_ppc_draws)

    fig, ax = plt.subplots(figsize=(6, 6))

    for label, color in [("correct", "tab:green"), ("error", "tab:red")]:
        # empirical dots
        prop = emp[f"{label}_prop"]
        y = QUANTILES * prop
        ax.scatter(emp[f"{label}_quantiles"], y, color=color, marker="o",
                   zorder=5, label=f"empirical ({label})")
        # empirical DR proportion at x=0
        ax.scatter([0.0], [emp[f"{label}_dr_prop"] * prop], color=color,
                   marker="o", zorder=5)

        # posterior-predictive crosses (averaged across draws) + individual
        # draws as thin lines for uncertainty
        all_q = np.array([s[f"{label}_quantiles"] for s in ppc_summaries])
        all_prop = np.array([s[f"{label}_prop"] for s in ppc_summaries])
        all_dr = np.array([s[f"{label}_dr_prop"] for s in ppc_summaries])

        for q, p, d in zip(all_q, all_prop, all_dr):
            y_pred = QUANTILES * p
            ax.plot(np.concatenate([[0.0], q]),
                    np.concatenate([[d * p], y_pred]),
                    color=color, alpha=0.15, lw=1)

        mean_q = np.nanmean(all_q, axis=0)
        mean_p = np.nanmean(all_prop)
        mean_d = np.nanmean(all_dr)
        ax.plot(np.concatenate([[0.0], mean_q]),
                np.concatenate([[mean_d * mean_p], QUANTILES * mean_p]),
                color=color, marker="x", lw=2, label=f"model ({label})")

    ax.set_xlabel("Response time (s)  [double-response proportions at x=0]")
    ax.set_ylabel("Response proportion")
    ax.set_title(title or "Posterior predictive check")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def posterior_pairplot(posterior_samples, out_png="posterior_pairs.png", title=""):
    n_params = posterior_samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(2.2 * n_params, 2.2 * n_params))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(posterior_samples[:, i], bins=30, color="tab:blue")
            elif i > j:
                ax.scatter(posterior_samples[:, j], posterior_samples[:, i], s=2, alpha=0.3)
            else:
                ax.axis("off")
            if i == n_params - 1:
                ax.set_xlabel(PARAM_NAMES[j])
            if j == 0:
                ax.set_ylabel(PARAM_NAMES[i])
    plt.suptitle(title or "Posterior pairwise marginals")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- self-contained demo using synthetic stand-ins ----------------------
    # This lets us exercise & sanity-check the PPC/plotting code in THIS
    # sandbox (no BayesFlow / real CSV needed): we simulate a "real"
    # dataset from a known theta, then pretend a (slightly noisy) cloud of
    # draws around that theta is the "posterior" a trained approximator
    # would have returned, and check the PPC plot looks sensible (model
    # crosses should land close to the empirical dots).
    rng = np.random.default_rng(7)
    true_theta = np.array([3.4, 1.1, 2.6, 0.9, 0.28])
    fake_real_data = simulate_batch(true_theta[None, :], n_obs=2000, dt=0.0015,
                                     t_max=2.5, dr_window=0.25, rng=rng)[0]

    fake_posterior = true_theta[None, :] + rng.normal(0, 0.05, size=(500, 5)) * true_theta[None, :]
    fake_posterior = np.clip(fake_posterior, 1e-3, None)

    plot_ppc(fake_real_data, fake_posterior, n_ppc_draws=30,
              out_png="demo_ppc.png", title="Demo PPC (synthetic stand-in, NOT real data)")
    posterior_pairplot(fake_posterior, out_png="demo_posterior_pairs.png",
                        title="Demo posterior (synthetic stand-in)")
