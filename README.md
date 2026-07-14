# Double Responding: Simulation-Based Inference for the RDM (Evans et al., 2020)

Project for *Simulation Based Inference*, TU Dortmund University.
Paper: Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L.
(2020). Double responding: A new constraint for models of speeded decision
making. *Cognitive Psychology, 121*, 101292.

This implements the project brief's chosen model — the **racing diffusion
model (RDM)**, the simplest member of the paper's 9-model tree (no
feed-forward inhibition, no lateral inhibition, no leakage) — extended to
double responding via the paper's **primary definition** (Fig. 1), and
fits it with **amortized simulation-based inference (BayesFlow)**.

## Honesty about what could / couldn't be run here

This coding sandbox has **no internet access** and **BayesFlow is not
installed** (nor can it be, without network access). So:

- `rdm_dr_simulator.py` (the cognitive model itself) and
  `posterior_predictive.py`'s plotting/summary logic **were written AND
  executed AND checked** in this sandbox (numpy/scipy/pandas/matplotlib are
  available). `sanity_checks.png` and `demo_ppc.png` in this folder are
  real output from real runs, not mock-ups.
- `bayesflow_model.py`, `validate_recovery.py`, and `fit_real_data.py`
  (everything that actually calls BayesFlow) **could not be executed here**
  and are written against the BayesFlow 2.x API as best documented/recalled;
  see the warning at the top of `bayesflow_model.py` for what to check if an
  import fails in your environment. The *modeling logic* (what gets
  simulated, how data is represented, what the network sees) does not
  depend on those API details and is the part that matters most for
  correctness.
- No real CSV of the Evans et al. (2020) data was uploaded to this
  conversation (only the PDF), so `fit_real_data.py` cannot be run
  end-to-end here either — point `CSV_PATH` at your file once you have it.

## 1-2. The model (`rdm_dr_simulator.py`)

Two accumulators race, coded as **correct** (drift `v_c`) vs. **error**
(drift `v_e`) — the standard way to fit a single-condition 2AFC data set
with an LBA/RDM-style model when you don't have per-stimulus drift rates
(Brown & Heathcote, 2008; this is also effectively what the paper's own
Table 2 reporting is organized around: P(C), P(E), not per-stimulus rates).

Paper's Eq. 1 (their base "RDM", `sigma` fixed to 1 for scale
identifiability):

    dx_i = v_i dt + eps_i sqrt(dt),   eps_i ~ N(0,1) iid

Between-trial variability: **starting point only** (footnote 1 in the
paper), `x_i(0) ~ Uniform(0, A)`, `A < b`.

**Double responding, primary definition** (paper Fig. 1, Eqs. 7-9):
accumulation for *both* accumulators continues, unchanged, past the initial
threshold crossing. If the accumulator that did *not* win the race
subsequently also reaches `b` within `tmax_DR = 250 ms` of the initial
response, a double response is recorded, with `drt` measured from the
initial response (not from trial onset, and with no added non-decision
time — this matches how the paper's Fig. 2/3 axes are constructed).

Parameters: `v_c, v_e, b, A, t0` (5 parameters, matching the paper's own
"each model contains at least 5 parameters" remark).

Priors (`PRIOR_RANGES` in the file): wide, weakly-informative ranges chosen
to be broadly consistent with the empirical ranges in the paper's Tables
2/5/6 (mean RTs 0.37-0.61s, accuracy 0.61-1.0). `A` is sampled as a fraction
of `b` to enforce `A < b`.

Implementation: vectorized Euler-Maruyama on a fixed time grid (`dt`, e.g.
5 ms for training-time speed or 1-1.5 ms for higher-fidelity checks),
vectorized simultaneously over *all* trials of *all* parameter sets in a
batch (`simulate_batch`) for speed inside a BayesFlow training loop.

### Verification against the paper's own qualitative benchmarks

Running `python3 sanity_checks.py` (real output, `sanity_checks.png`)
confirms our implementation reproduces, from first principles, several
patterns the paper reports for the plain RDM:

- RT distributions are positively skewed (paper: standard EAM property).
- **P(DR | error) >> P(DR | correct)** — double responses are
  predominantly *corrective* (paper Tables 2/5/6): we got a ~6.6x ratio.
- RT distributions on trials followed by a double response look broadly
  similar in shape to trials without one (paper's Fig. 4 point: double
  responses are not just "fast guesses").
- The **DRT distribution comes out essentially flat/uniform** in our
  simulator for the plain RDM, *not* matching the empirical positive skew.
  This is not a bug — it is *exactly* the paper's own diagnosis of the
  plain RDM's failure mode (their Figs. 9-11: "all models predict fairly
  evenly spaced quantiles, which indicates a distribution with equal mass
  throughout", worst for models without inhibition). Reproducing this known
  misfit qualitatively from our from-scratch implementation is good
  evidence the simulator's mechanics are right.
- Consistent with the paper's headline finding, a no-inhibition RDM with
  otherwise plausible parameters over-predicts the double-response rate
  relative to what's seen in real data (we saw ~9-14% at parameters giving
  realistic RT/accuracy, vs. the paper's <=2.6% ceiling in real
  participants) — again this is the paper's own point (Fig. 7: "The models
  with no inhibition... appear to provide the worst account of the data,
  greatly over-predicting... the double responses"), not a defect in this
  implementation.

## 3-4. BayesFlow amortized inference (`bayesflow_model.py`)

- **Prior sampling** (`prior_fn`) draws the 5 parameters as above.
- **Likelihood/simulator** (`make_trial_simulator`) wraps
  `rdm_dr_simulator.simulate_batch` to produce `n_obs` iid trials per
  parameter draw. `n_obs` should be set close to the real dataset(s) you
  intend to fit (e.g. ~10,000 for Experiment 1's S1-S4, since the project's
  CSV appears to be that data set based on the participant labels; use a
  smaller `n_obs` — or a random range of `n_obs` if you want one network
  that generalizes across session lengths — if you are instead targeting
  Experiment 2's 22-923-trial sessions).
- **Adapter**: concatenates the 5 parameters into `inference_variables`,
  renames the raw `(n_obs, 4)` trial array to `summary_variables`
  (columns: `[rt, correct, dr, drt_masked]`, with `drt_masked = drt` when
  `dr==1` and `0.0` otherwise — a standard masking trick so the network
  sees a fixed-width vector per trial rather than a ragged one), and
  z-standardizes both.
- **Summary network**: `bf.networks.DeepSet` — permutation-invariant over
  trials, which is exactly the right inductive bias since trials are iid
  given theta, and it naturally handles the fact that different
  participants/sessions have different numbers of trials.
- **Inference network**: a coupling-flow (normalizing flow) that maps the
  summary embedding to samples from the amortized posterior
  q(theta | data).
- **Training** is fully online/simulation-based: a fresh batch of
  (theta, simulated dataset) pairs is drawn every gradient step, for
  `epochs * num_batches_per_epoch` steps total — there is no fixed offline
  training set.

## 5. Validation on simulated data (`validate_recovery.py`)

Draws a *held-out* validation set of parameter/data pairs (not used during
training, since training is online) and computes, all against the known
ground truth:

- **Parameter recovery**: scatter of posterior mean vs. true value, one
  panel per parameter, with the identity line.
- **RMSE & bias** per parameter.
- **Simulation-Based Calibration (SBC)**: rank of the true parameter within
  its own posterior sample; histograms should be ~Uniform if the
  approximate posterior is well calibrated. Includes both the classic
  rank-histogram version (Talts et al., 2018) and an ECDF-deviation version.
- **Coverage analysis**: for nominal central credible levels
  {50%, 80%, 90%, 95%}, the empirical fraction of validation datasets whose
  true parameter falls inside the corresponding interval — should track the
  diagonal.

## 6. Fit the real data (`fit_real_data.py`)

Loads the CSV (`rt, choice, correct, dr, drt, participant`), groups by
`participant`, and calls the trained approximator's `.sample()` once per
participant (batch size 1, `n_obs` = that participant's trial count) to get
posterior samples for `v_c, v_e, b, A, t0`. Saves a summary table
(mean/sd/90% CI per parameter per participant) and the raw posterior
samples.

## 7. Evaluate the fit (`posterior_predictive.py`)

**Posterior predictive checks**: for each participant, draws ~30-50 posterior
samples, simulates a synthetic data set of the same size from each, and
plots a Fig.-2-style CDF chart — empirical RT quantiles + P(DR|correct),
P(DR|error) as dots, model-predicted versions (averaged over posterior
draws, with individual draws shown as thin lines for posterior-predictive
uncertainty) as crosses/lines. This part of the code **was run** in this
sandbox with a synthetic stand-in "posterior" (Gaussian noise around a
known true theta) against synthetic "real" data from that same theta, to
confirm the plotting/summary logic is correct — see `demo_ppc.png`. Once
you have real posterior samples and real data, call `plot_ppc(real_trials,
posterior_samples)` per participant instead.

Also includes `posterior_pairplot` for inspecting marginal and pairwise
posterior structure (e.g. checking for the `v_e`/`b`/`A` trade-offs you'd
expect in an LBA/RDM-family model).

## 8. Conclusions — what to expect / how to discuss your results

Based on the paper's own findings for the plain RDM, and confirmed
mechanistically by the sanity checks above, when you fit the real S1-S4 (or
Experiment-2) data you should anticipate:

- The RDM will likely fit the **choice/RT distributions alone** reasonably
  well (the paper found *all 9 models* do, Figs. 5/6) — so if you only look
  at RT/accuracy PPC, the RDM will look fine.
- Once you fit **jointly** to RTs *and* double-response proportions (which
  is what `simulate_batch`/the summary network is doing by construction,
  since `dr`/`drt` are part of every simulated trial), expect the RDM to
  **over-predict the double-response rate**, especially P(DR|error), for
  participants under speed emphasis (paper's S1/S2) — this is the paper's
  central result about inhibition-free models, and the PPC plot should show
  the red (error) model line sitting noticeably above the red empirical dot
  at x=0.
- Expect the fitted model to **fail to capture the shape of the DRT
  distribution** even where it gets the DR *rate* about right — real DRTs
  are strongly front-loaded (positive skew) in the 0-250ms window, while
  the plain RDM predicts something close to uniform (again, this should be
  visible if you add a DRT-quantile panel to the PPC, analogous to the
  paper's Figs. 9-11).
- Because the RDM has no inhibition mechanism, whatever parameter values
  the network settles on to best explain the *rate* of double responses are
  likely doing so partly by trading off against the RT distribution fit
  (the paper's point that "all models appear to provide a poorer account of
  both the response time distributions and double responses when fit
  jointly compared to when fit separately") — worth checking directly by
  also fitting a version of the pipeline to RT/choice *only* (drop
  `dr`/`drt` from `summary_variables`) and comparing.
- The practical takeaway to write up: incorporating double responses as an
  additional constraint makes the RDM's model misfit *visible* in a way
  that fitting RT + accuracy alone does not — which is exactly the paper's
  argument for why double responding is a useful extra source of
  constraint on EAMs, even though (per the paper's own discussion) it
  should not be over-interpreted as settling which EAM is the uniquely
  "correct" measurement tool (see the paper's discussion of parameter
  identifiability, Miletić et al., 2017, for lateral-inhibition models).

Fill in this section with your *actual* numeric results (RMSE/bias/SBC/
coverage plots, and the real PPC figures) once you've run the BayesFlow
stages in an environment with the package installed.

## Files

| File | What it does | Verified here? |
|---|---|---|
| `rdm_dr_simulator.py` | Cognitive model: RDM + double-response simulator, priors | Yes |
| `sanity_checks.py` | Qualitative benchmark checks against the paper | Yes (`sanity_checks.png`) |
| `bayesflow_model.py` | Simulator wrapper, adapter, networks, training loop | No (BayesFlow unavailable here) |
| `validate_recovery.py` | Parameter recovery, RMSE/bias, SBC, coverage | No (needs trained approximator) |
| `fit_real_data.py` | Load real CSV, per-participant posterior inference | No (needs approximator + CSV) |
| `posterior_predictive.py` | PPC plots, posterior pair plots | Yes (`demo_ppc.png`, synthetic stand-in) |
| `run_pipeline.py` | Single CLI entry point for all stages | — |

## How to run (in an environment with BayesFlow + a Keras backend installed)

```bash
pip install -r requirements.txt

python3 sanity_checks.py                 # step 2 check (no BayesFlow needed)
python3 run_pipeline.py train            # steps 2-4: train the amortized posterior
python run_pipeline.py validate         # step 5: recovery / SBC / coverage
# put your CSV at the path set in fit_real_data.CSV_PATH, then:
python3 run_pipeline.py fit              # step 6: fit real participants
python3 run_pipeline.py ppc              # step 7: PPC (edit __main__ to use real fit results)
```
