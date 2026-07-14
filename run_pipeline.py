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
        approximator = bf.Approximator.load("rdm_dr_approximator.keras")
        run_all(approximator)

    elif stage == "fit":
        import bayesflow as bf
        from fit_real_data import load_participant_data, fit_all_participants, summarize_posteriors
        approximator = bf.Approximator.load("rdm_dr_approximator.keras")
        participants = load_participant_data()
        results = fit_all_participants(approximator, participants)
        summarize_posteriors(results)

    elif stage == "ppc":
        import posterior_predictive  # noqa: F401 (runs its __main__ demo)

    else:
        print(f"Unknown stage '{stage}'. See module docstring.")


if __name__ == "__main__":
    main()
