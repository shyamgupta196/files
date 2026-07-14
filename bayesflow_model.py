"""
BayesFlow pipeline for amortized Bayesian inference on the RDM + double-
response model (Evans et al., 2020).

*** IMPORTANT ***
This project's sandbox has no internet access and BayesFlow is not
installed here, so this file could not be executed/tested in this
environment. It is written against the BayesFlow 2.x API as documented at
the time of writing (bayesflow.readthedocs.io / github.com/bayesflow-org/
bayesflow). Install with:

    pip install bayesflow

If your installed version's exact class/module names differ slightly
(BayesFlow's public API has moved around across 2.x releases), the fix is
almost always a one-line import path change -- the modeling logic below
(what the simulator does, how data is represented, network architecture
choices) does not depend on those details. Check `import bayesflow as bf;
help(bf)` / the package's `examples/` folder for your installed version if
an import fails.

Pipeline
--------
1. `make_simulator()`      -- wraps rdm_dr_simulator into a BayesFlow
                               two-stage simulator: prior -> likelihood.
2. `build_adapter()`       -- tells BayesFlow how to turn the raw simulator
                               output (a dict of numpy arrays) into
                               (summary_variables, inference_variables)
                               tensors, incl. standardization.
3. `build_approximator()`  -- a DeepSet summary network (permutation-
                               invariant over trials, handles the variable
                               number of trials per participant) feeding a
                               coupling-flow (normalizing-flow) inference
                               network -> the amortized posterior
                               q(theta | data).
4. `train()`               -- online training: fresh simulated datasets are
                               drawn every batch (the "simulation-based
                               inference" / "neural posterior estimation"
                               regime), so there is no fixed offline
                               training set to store.
"""

import numpy as np

from rdm_dr_simulator import PARAM_NAMES, PRIOR_RANGES, A_FRACTION_RANGE, simulate_batch


# ---------------------------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------------------------

def prior_fn():
    """Single prior draw as a dict of scalars (BayesFlow convention: a
    'meta' function returning named quantities that subsequent simulator
    stages can consume by name)."""
    v_c = np.random.uniform(*PRIOR_RANGES["v_c"])
    v_e = np.random.uniform(*PRIOR_RANGES["v_e"])
    b = np.random.uniform(*PRIOR_RANGES["b"])
    frac = np.random.uniform(*A_FRACTION_RANGE)
    A = frac * b
    t0 = np.random.uniform(*PRIOR_RANGES["t0"])
    return dict(v_c=v_c, v_e=v_e, b=b, A=A, t0=t0)


def make_trial_simulator(n_obs=300, dt=0.005, t_max=2.5, dr_window=0.25):
    """Returns a likelihood function with signature (v_c, v_e, b, A, t0) ->
    dict(trials=array) suitable for bf.simulators.make_simulator, using a
    fixed number of trials `n_obs` per simulated "participant".

    n_obs should be chosen to roughly match the real dataset you plan to
    apply the trained network to (see fit_real_data.py). Because the
    summary network (DeepSet) is permutation-invariant and operates
    trial-by-trial, it can in principle generalize across different n_obs
    at inference time, but training and testing at a similar n_obs is
    still recommended for the best amortization -- e.g. train at n_obs=300
    for Experiment 2 (102-923 trials/session) and separately at
    n_obs=10000 (or a random range of large n_obs) for Experiment 1.
    """

    def likelihood_fn(v_c, v_e, b, A, t0):
        theta = np.array([[v_c, v_e, b, A, t0]], dtype=np.float32)
        data = simulate_batch(theta, n_obs=n_obs, dt=dt, t_max=t_max,
                               dr_window=dr_window)[0]  # (n_obs, 4)
        return dict(trials=data)

    return likelihood_fn


def make_simulator(n_obs=300, dt=0.005, t_max=2.5, dr_window=0.25):
    import bayesflow as bf

    likelihood_fn = make_trial_simulator(n_obs=n_obs, dt=dt, t_max=t_max,
                                          dr_window=dr_window)
    simulator = bf.simulators.make_simulator([prior_fn, likelihood_fn])
    return simulator


# ---------------------------------------------------------------------------
# 2. Adapter: raw simulator dict -> network tensors
# ---------------------------------------------------------------------------

def build_adapter():
    """Defines how raw simulator output is converted for the networks.

    - The 5 scalar parameters are concatenated into one "inference_variables"
      vector (the target of posterior estimation).
    - The (n_obs, 4) per-trial array ["trials"] becomes "summary_variables",
      fed into the permutation-invariant summary network.
    - Standardization (z-scoring) of parameters, and of the RT/DRT columns
      of the trial data, greatly helps flow-based inference networks train
      stably; the dr flag (0/1) and correct flag (0/1) are left as-is.
    """
    import bayesflow as bf

    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(["v_c", "v_e", "b", "A", "t0"], into="inference_variables")
        .rename("trials", "summary_variables")
       # .standardize(include="inference_variables")
        # standardize RT/DRT columns of summary_variables but not the two
        # binary flag columns -- if your BayesFlow version's `standardize`
        # cannot target array *slices*, standardize the whole
        # summary_variables tensor instead (binary columns will just end up
        # with a learned near-identity scaling, which is harmless).
       # .standardize(include="summary_variables")
    )
    return adapter


# ---------------------------------------------------------------------------
# 3. Networks / approximator
# ---------------------------------------------------------------------------

def build_approximator(adapter=None):
    import bayesflow as bf

    if adapter is None:
        adapter = build_adapter()

    # Permutation-invariant summary network over the (variable-length) set
    # of trials. DeepSet is the natural choice here: each trial is an
    # exchangeable iid draw given theta, exactly the assumption our
    # simulator makes.
    summary_network = bf.networks.DeepSet(
        summary_dim=32,
    )

    # Normalizing-flow inference network mapping (summary embedding) ->
    # posterior samples over the 5 RDM+DR parameters. A coupling flow is a
    # solid default; bf.networks.FlowMatching is a good alternative if
    # available in your installed version and you want faster sampling.
    inference_network = bf.networks.CouplingFlow(
        depth=6,
    )

    approximator = bf.ContinuousApproximator(
        summary_network=summary_network,
        inference_network=inference_network,
        adapter=adapter,
    )
    return approximator


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train(
    n_obs=400,
    dt=0.005,
    t_max=2.5,
    dr_window=0.25,
    epochs=30,
    num_batches_per_epoch=200,
    batch_size=32,
    save_path="rdm_dr_approximator.keras",
):
    """Online SBI training: a fresh batch of (theta, simulated data) pairs is
    drawn every step directly from `simulator`, so there is no fixed
    training set -- the network effectively sees a new dataset every
    gradient step, for as many steps as
    epochs * num_batches_per_epoch.
    """
    import bayesflow as bf

    simulator = make_simulator(n_obs=n_obs, dt=dt, t_max=t_max, dr_window=dr_window)
    approximator = build_approximator()

    approximator.compile(optimizer="adam")

    history = approximator.fit(
    simulator=simulator,
    epochs=epochs,
    num_batches=num_batches_per_epoch,
    batch_size=batch_size,
)

    approximator.save(save_path)
    return approximator, history


if __name__ == "__main__":
    approximator, history = train()
