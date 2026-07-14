"""
Racing Diffusion Model (RDM) extended to double responding.

Implements the *base* model of the 9-model tree in Evans, Dutilh,
Wagenmakers & van der Maas (2020, Cognitive Psychology), i.e. the model with
NO feed-forward inhibition, NO lateral inhibition, and NO leakage (their
"RDM", Tillman & Logan, 2017), extended with the paper's "primary
definition" of double responding (their Fig. 1 / Eqs. 1 and 7-9).

Two accumulators race:
    - accumulator 0 = "correct" accumulator, drift v_c
    - accumulator 1 = "error"   accumulator, drift v_e
(this is the standard correct/error coding used to fit a single-condition
2AFC data set with an LBA/RDM-style model, as in Brown & Heathcote, 2008.)

Model equation (paper's Eq. 1), sigma fixed to 1 for scale identifiability:
    dx_i = v_i * dt + eps_i * sqrt(dt),   eps_i ~ N(0, 1) iid across i

Between-trial variability: ONLY in starting point (paper footnote 1):
    x_i(0) ~ Uniform(0, A),   A < b

Decision: first accumulator to reach threshold b triggers the INITIAL
response at decision time t. Observed RT = t + t0 (non-decision time).

Double response (primary definition, Fig. 1B/C in the paper): accumulation
for BOTH accumulators continues, unchanged, after the initial response.
If the accumulator that did NOT win the race subsequently reaches b within
tmax_DR (250 ms in the paper) of the initial response, a second ("double")
response is triggered. drt is measured from the initial response (t=0 at
the initial response), matching the paper's Eqs. 7-9 / Fig. 2-3.

Parameters (theta), in the order used everywhere in this project:
    v_c   : drift rate, correct accumulator   (> 0, typically 0-6)
    v_e   : drift rate, error accumulator      (> 0, typically 0-6)
    b     : threshold                         (> 0, typically 0.5-5)
    A     : upper bound of starting-point U(0,A); constrained A < b
    t0    : non-decision time (s)              (typically 0.1-0.4)

sigma (within-trial noise sd) is fixed to 1 throughout (standard
LBA/RDM identifiability convention -- v, b, A are all expressed in units
of sigma).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

PARAM_NAMES = ["v_c", "v_e", "b", "A", "t0"]

# Weakly-informative priors chosen to be broadly consistent with the
# empirical ranges reported in the paper (Tables 2, 5, 6): mean RTs
# ~0.37-0.61 s, accuracy ~0.61-1.0, and with fitted 5-parameter LBA/RDM-style
# models in mind. These are intentionally wide relative to the plausible
# posterior region -- feel free to tighten them once you have looked at a
# prior-predictive check (see `prior_predictive_check` below).
PRIOR_RANGES = {
    "v_c": (0.0, 6.0),   # Uniform(0, 6)
    "v_e": (0.0, 6.0),   # Uniform(0, 6)
    "b":   (0.5, 5.0),   # Uniform(0.5, 5)
    "t0":  (0.1, 0.45),  # Uniform(0.1, 0.45)
}
# A is sampled conditional on b to respect A < b (see sample_prior).
A_FRACTION_RANGE = (0.01, 0.9)  # A = frac * b, frac ~ Uniform(0.01, 0.9)


def sample_prior(batch_size, rng=None):
    """Sample a batch of parameter vectors from the joint prior.

    Returns
    -------
    theta : np.ndarray, shape (batch_size, 5)
        Columns ordered as PARAM_NAMES = [v_c, v_e, b, A, t0].
    """
    rng = np.random.default_rng() if rng is None else rng
    v_c = rng.uniform(*PRIOR_RANGES["v_c"], size=batch_size)
    v_e = rng.uniform(*PRIOR_RANGES["v_e"], size=batch_size)
    b = rng.uniform(*PRIOR_RANGES["b"], size=batch_size)
    frac = rng.uniform(*A_FRACTION_RANGE, size=batch_size)
    A = frac * b
    t0 = rng.uniform(*PRIOR_RANGES["t0"], size=batch_size)
    theta = np.stack([v_c, v_e, b, A, t0], axis=1)
    return theta


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_rdm_dr_trials(
    theta,
    n_obs,
    dt=0.001,
    t_max=3.0,
    dr_window=0.25,
    rng=None,
):
    """Simulate `n_obs` iid trials from the RDM + double-response model for
    a single parameter vector `theta` = (v_c, v_e, b, A, t0).

    Uses a vectorised (over trials) Euler-Maruyama scheme on a fixed time
    grid. All `n_obs` trials are advanced together, which is efficient in
    numpy even though the underlying model is a per-trial stopping-time
    process.

    Returns a dict of numpy arrays, each of length n_obs:
        rt      : observed RT of the initial response (s) (t0 already added)
        choice  : 0 = correct accumulator won, 1 = error accumulator won
        correct : 1 if choice == 0 else 0   (kept for readability/CSV parity)
        dr      : 1 if a double response occurred, else 0
        drt     : double-response time (s), measured from the initial
                  response; 0.0 when dr == 0 (masked, not missing -- see
                  note in bayesflow_model.py on how this is fed to the
                  summary network)
        timeout : 1 if neither accumulator reached b within t_max (should be
                  ~0 for sensible parameters; such trials get rt=t_max+t0,
                  choice=-1 as a sentinel and are best discarded)
    """
    rng = np.random.default_rng() if rng is None else rng
    v_c, v_e, b, A, t0 = theta

    n_steps_main = int(round(t_max / dt))
    n_steps_dr = int(round(dr_window / dt))
    total_steps = n_steps_main + n_steps_dr
    sqdt = np.sqrt(dt)

    x_c = rng.uniform(0.0, A, size=n_obs)
    x_e = rng.uniform(0.0, A, size=n_obs)

    # step index (1-based) of first threshold crossing; -1 = not yet crossed
    cross_c = np.full(n_obs, -1, dtype=np.int64)
    cross_e = np.full(n_obs, -1, dtype=np.int64)

    # Already-at-threshold at t=0 is possible in principle if A >= b, but we
    # enforce A < b in the prior, so x_i(0) < b always.

    for step in range(1, total_steps + 1):
        x_c = x_c + v_c * dt + rng.normal(0.0, 1.0, size=n_obs) * sqdt
        x_e = x_e + v_e * dt + rng.normal(0.0, 1.0, size=n_obs) * sqdt

        newly_c = (x_c >= b) & (cross_c < 0)
        newly_e = (x_e >= b) & (cross_e < 0)
        if newly_c.any():
            cross_c[newly_c] = step
        if newly_e.any():
            cross_e[newly_e] = step

        # Early exit once every trial has at least one crossing AND we are
        # already dr_window past the *latest possible* first crossing --
        # not worth the complexity here; we simply run the fixed grid.

    # --- resolve initial response ------------------------------------------------
    inf = total_steps + 10 ** 6
    cc = np.where(cross_c < 0, inf, cross_c)
    ce = np.where(cross_e < 0, inf, cross_e)

    choice = np.where(cc <= ce, 0, 1)  # ties (extremely rare) go to correct acc.
    first_step = np.minimum(cc, ce)
    timeout = (first_step >= inf).astype(np.int64)
    # Avoid indexing issues for timed-out trials; clip to last step.
    first_step_clipped = np.clip(first_step, 1, total_steps)

    decision_time = first_step_clipped * dt
    rt = decision_time + t0

    # --- resolve double response --------------------------------------------------
    loser_cross = np.where(choice == 0, ce, cc)  # crossing step of the OTHER acc.
    has_loser_cross = loser_cross < inf
    dr_delta_steps = loser_cross - first_step_clipped
    within_window = (dr_delta_steps > 0) & (dr_delta_steps <= n_steps_dr)
    dr = (has_loser_cross & within_window & (timeout == 0)).astype(np.int64)
    drt = np.where(dr == 1, dr_delta_steps * dt, 0.0)

    correct = (choice == 0).astype(np.int64)
    choice_out = choice.copy()
    choice_out[timeout == 1] = -1  # sentinel for undecided trials

    return {
        "rt": rt,
        "choice": choice_out,
        "correct": correct,
        "dr": dr,
        "drt": drt,
        "timeout": timeout,
    }


def simulate_batch(theta_batch, n_obs, dt=0.005, t_max=2.5, dr_window=0.25, rng=None):
    """Simulate one dataset of `n_obs` trials for EACH row of theta_batch.

    Fully vectorised over (batch_size * n_obs) simultaneously -- much faster
    than looping over batch_size and calling `simulate_rdm_dr_trials` once
    per parameter vector, which matters when this is called online, once
    per training batch, inside the BayesFlow training loop.

    Parameters
    ----------
    theta_batch : np.ndarray, shape (batch_size, 5)
    n_obs : int
        Number of trials to simulate per parameter vector (per "participant").

    Returns
    -------
    data : np.ndarray, shape (batch_size, n_obs, 4)
        Feature order: [rt, correct, dr, drt_masked]. `choice` is redundant
        with `correct` in this 2-accumulator, correct/error-coded setup, so
        we feed the summary network [rt, correct, dr, drt] (4 features/trial).
        drt_masked = drt if dr==1 else 0.0.
    """
    rng = np.random.default_rng() if rng is None else rng
    batch_size = theta_batch.shape[0]

    v_c = theta_batch[:, 0:1]
    v_e = theta_batch[:, 1:2]
    b = theta_batch[:, 2:3]
    A = theta_batch[:, 3:4]
    t0 = theta_batch[:, 4:5]

    n_steps_main = int(round(t_max / dt))
    n_steps_dr = int(round(dr_window / dt))
    total_steps = n_steps_main + n_steps_dr
    sqdt = np.sqrt(dt)

    shape = (batch_size, n_obs)
    x_c = rng.uniform(0.0, 1.0, size=shape) * A  # broadcasts (batch,1) -> (batch,n_obs)
    x_e = rng.uniform(0.0, 1.0, size=shape) * A

    cross_c = np.full(shape, -1, dtype=np.int64)
    cross_e = np.full(shape, -1, dtype=np.int64)

    for step in range(1, total_steps + 1):
        x_c = x_c + v_c * dt + rng.normal(0.0, 1.0, size=shape) * sqdt
        x_e = x_e + v_e * dt + rng.normal(0.0, 1.0, size=shape) * sqdt

        newly_c = (x_c >= b) & (cross_c < 0)
        newly_e = (x_e >= b) & (cross_e < 0)
        if newly_c.any():
            cross_c[newly_c] = step
        if newly_e.any():
            cross_e[newly_e] = step

    inf = total_steps + 10 ** 6
    cc = np.where(cross_c < 0, inf, cross_c)
    ce = np.where(cross_e < 0, inf, cross_e)

    choice = np.where(cc <= ce, 0, 1)
    first_step = np.minimum(cc, ce)
    timeout = (first_step >= inf)
    first_step_clipped = np.clip(first_step, 1, total_steps)

    decision_time = first_step_clipped * dt
    rt = decision_time + t0  # t0 broadcasts (batch,1) -> (batch,n_obs)

    loser_cross = np.where(choice == 0, ce, cc)
    has_loser_cross = loser_cross < inf
    dr_delta_steps = loser_cross - first_step_clipped
    within_window = (dr_delta_steps > 0) & (dr_delta_steps <= n_steps_dr)
    dr = (has_loser_cross & within_window & (~timeout)).astype(np.float32)
    drt = np.where(dr == 1, dr_delta_steps * dt, 0.0)

    correct = (choice == 0).astype(np.float32)

    data = np.stack([rt, correct, dr, drt], axis=-1).astype(np.float32)
    return data


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(0)
    theta = np.array([3.0, 1.2, 1.6, 0.8, 0.25])
    out = simulate_rdm_dr_trials(theta, n_obs=5000, dt=0.001, rng=rng)
    print("mean RT:", out["rt"].mean())
    print("accuracy:", out["correct"].mean())
    print("P(DR):", out["dr"].mean())
    print("timeouts:", out["timeout"].sum())
