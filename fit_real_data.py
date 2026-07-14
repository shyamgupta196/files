"""
Step 6: Apply the trained amortized posterior to the real data.

Expects a CSV with (at least) these columns, one row per trial, as
described in the project brief:

    rt, choice, correct, dr, drt, participant

    rt         : response time of the initial response (s)
    choice     : which alternative was chosen (raw code, e.g. 1/2) -- not
                 used directly by the model below (we use `correct`
                 instead, matching the correct/error-accumulator coding
                 the RDM+DR simulator uses); kept only for reference/QC
    correct    : 1 if the initial response was correct, 0 if an error
    dr         : 1 if a double response followed within the response
                 deadline, else 0
    drt        : double-response time (s), measured from the initial
                 response; blank/NaN when dr == 0
    participant: participant/session ID, e.g. "S1", "S2", ...

NOTE: no CSV was supplied to me in this conversation (only the PDF of the
paper was uploaded), so this script cannot be run end-to-end here. Point
CSV_PATH at your file (e.g. after uploading it) and run this after
training (bayesflow_model.train()).
"""

import numpy as np
import pandas as pd

CSV_PATH = "double_responding_data.csv"  # <-- change to your uploaded file's path


def load_participant_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["drt"] = df["drt"].fillna(0.0)
    df["dr"] = df["dr"].fillna(0).astype(int)

    participants = {}
    for pid, sub in df.groupby("participant"):
        trials = sub[["rt", "correct", "dr", "drt"]].to_numpy(dtype=np.float32)
        participants[pid] = trials
        n = trials.shape[0]
        acc = sub["correct"].mean()
        p_dr = sub["dr"].mean()
        print(f"{pid}: n_trials={n}  mean RT={sub['rt'].mean():.3f}  "
              f"accuracy={acc:.3f}  P(DR)={p_dr:.4f}")
    return participants


def fit_all_participants(approximator, participants, num_posterior_samples=2000):
    """Returns a dict: participant_id -> posterior_samples array of shape
    (num_posterior_samples, n_params)."""
    from rdm_dr_simulator import PARAM_NAMES
    results = {}
    for pid, trials in participants.items():
        # BayesFlow's `sample` expects a batch dimension; here batch=1
        # (one dataset = one participant's full set of trials).
        conditions = {"trials": trials[None, :, :]}
        samples_dict = approximator.sample(conditions=conditions,
                                            num_samples=num_posterior_samples)
        # BayesFlow inverts the adapter's `.concatenate(...)`, returning ONE
        # array per parameter (keyed by PARAM_NAMES), each (1, num_samples, 1)
        # -- NOT a single "inference_variables" array. Stack them back into
        # (num_posterior_samples, n_params) in PARAM_NAMES order.
        if "inference_variables" in samples_dict:
            samples = np.asarray(samples_dict["inference_variables"])[0]
        else:
            cols = [np.asarray(samples_dict[name])[0].reshape(num_posterior_samples, -1)
                    for name in PARAM_NAMES]
            samples = np.concatenate(cols, axis=-1)  # (num_posterior_samples, n_params)
        results[pid] = samples
        print(f"Fitted {pid}: posterior means = "
              f"{dict(zip(['v_c', 'v_e', 'b', 'A', 't0'], samples.mean(axis=0).round(3)))}")
    return results


def summarize_posteriors(results, out_csv="posterior_summary.csv"):
    from rdm_dr_simulator import PARAM_NAMES
    rows = []
    for pid, samples in results.items():
        row = {"participant": pid}
        for i, name in enumerate(PARAM_NAMES):
            row[f"{name}_mean"] = samples[:, i].mean()
            row[f"{name}_sd"] = samples[:, i].std()
            row[f"{name}_q05"] = np.quantile(samples[:, i], 0.05)
            row[f"{name}_q95"] = np.quantile(samples[:, i], 0.95)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved posterior summary to {out_csv}")
    return out


if __name__ == "__main__":
    import bayesflow as bf

    approximator = bf.approximators.Approximator.load("rdm_dr_approximator.keras")
    participants = load_participant_data(CSV_PATH)
    results = fit_all_participants(approximator, participants)
    summarize_posteriors(results)

    import pickle
    with open("posterior_samples_real_data.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved raw posterior samples to posterior_samples_real_data.pkl")
