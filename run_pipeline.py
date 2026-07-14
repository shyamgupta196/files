"""
End-to-end driver, mirroring the 8 steps in the project brief. Run stage by
stage (each is also runnable/importable on its own):

    python3 run_pipeline.py sanity      # step 2 sanity check (works here, no BayesFlow needed)
    python3 run_pipeline.py train       # steps 2-4 (needs `pip install bayesflow`)
    python3 run_pipeline.py validate    # step 5 (needs a saved approximator)
    python3 run_pipeline.py fit         # step 6 (needs a saved approximator + real CSV)
    python3 run_pipeline.py ppc         # step 7 (needs fit results; demo mode works standalone)
"""
import sys
import keras 

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    stage = sys.argv[1]

    if stage == "sanity":
        import sanity_checks  # noqa: F401  (executes on import)

    elif stage == "train":
        from bayesflow_model import train
        train()

    elif stage == "validate":
        import bayesflow as bf
        from validate_recovery import run_all
        # approximator = bf.approximators.Approximator.load("rdm_dr_approximator1.keras")
        approximator = keras.saving.load_model("rdm_dr_approximator1.keras")
        run_all(approximator)

    elif stage == "fit":
        import bayesflow as bf
        from fit_real_data import load_participant_data, fit_all_participants, summarize_posteriors
        # approximator = bf.approximators.Approximator.load("rdm_dr_approximator1.keras")
        approximator = keras.saving.load_model("rdm_dr_approximator1.keras")
        participants = load_participant_data()
        results = fit_all_participants(approximator, participants)
        summarize_posteriors(results)

    elif stage == "ppc":
        # Real-data posterior predictive checks. Load the real participant
        # trials + fitted posterior samples (re-fitting if no cached pickle),
        # then produce a Fig.-2-style PPC plot and a posterior pair-plot per
        # participant. (`import posterior_predictive` alone only defines the
        # functions -- it does NOT run the module's __main__ demo.)
        import os
        import pickle
        import bayesflow as bf  # noqa: F401  registers ContinuousApproximator for keras load
        from fit_real_data import load_participant_data, fit_all_participants
        from posterior_predictive import plot_ppc, posterior_pairplot

        participants = load_participant_data()

        pkl = "posterior_samples_real_data.pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded cached posterior samples from {pkl}")
        else:
            approximator = keras.saving.load_model("rdm_dr_approximator1.keras")
            results = fit_all_participants(approximator, participants)
            with open(pkl, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved posterior samples to {pkl}")

        for pid, trials in participants.items():
            samples = results[pid]
            plot_ppc(trials, samples, out_png=f"ppc_{pid}.png",
                     title=f"Posterior predictive check -- {pid}")
            posterior_pairplot(samples, out_png=f"posterior_pairs_{pid}.png",
                               title=f"Posterior marginals -- {pid}")

    else:
        print(f"Unknown stage '{stage}'. See module docstring.")


if __name__ == "__main__":
    main()
