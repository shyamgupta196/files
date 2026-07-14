"""
Sanity checks for the RDM + double-response simulator, against the
*qualitative* benchmarks reported in Evans et al. (2020):

 1. Double responses are rare overall (Table 2: <=2.6% of trials for the RDM
    fit to real data with sensible parameters -- our simulator should be
    ABLE to produce rates in this range, though the paper's whole point is
    that the plain RDM tends to OVER-predict double responses relative to
    lateral-inhibition models; we are not fitting real data here, just
    checking the simulator's own internal consistency).
 2. P(DR | error) > P(DR | correct) -- double responses are mostly
    "corrective" (Table 2, Table 5/6).
 3. RT distributions are positively skewed (standard EAM property).
 4. Double-response-time (DRT) distributions are also positively skewed,
    concentrated in the first part of the 250ms window (Fig. 3).
 5. RT distributions for trials followed by a DR vs. not are qualitatively
    similar (Fig. 4) -- i.e. double responses are not simply "fast
    guesses".

Run: python3 sanity_checks.py
Produces: sanity_checks.png in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from rdm_dr_simulator import simulate_rdm_dr_trials

rng = np.random.default_rng(42)

# A parameter set chosen to land in a broadly realistic range: mean RT
# ~0.45s, accuracy ~85%, low-ish (few-percent) double-response rate.
theta = np.array([
    3.4,   # v_c
    1.1,   # v_e
    2.6,   # b
    0.9,   # A
    0.28,  # t0
])

out = simulate_rdm_dr_trials(theta, n_obs=20000, dt=0.0015, t_max=2.5, dr_window=0.25, rng=rng)

rt, correct, dr, drt, timeout = out["rt"], out["correct"], out["dr"], out["drt"], out["timeout"]
print(f"timeouts: {timeout.sum()} / {len(rt)}")
keep = timeout == 0
rt, correct, dr, drt = rt[keep], correct[keep], dr[keep], drt[keep]

p_c = correct.mean()
p_dr = dr.mean()
p_dr_given_c = dr[correct == 1].mean()
p_dr_given_e = dr[correct == 0].mean()

print(f"theta = {theta}")
print(f"P(correct)        = {p_c:.4f}")
print(f"mean RT            = {rt.mean():.4f} s")
print(f"P(double response) = {p_dr:.4f}")
print(f"P(DR | correct)     = {p_dr_given_c:.4f}")
print(f"P(DR | error)       = {p_dr_given_e:.4f}")
print(f"  -> corrective ratio P(DR|E)/P(DR|C) = {p_dr_given_e / max(p_dr_given_c, 1e-9):.1f}x")

assert p_dr_given_e > p_dr_given_c, "FAILED: double responses should be predominantly corrective"

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# (1) RT distribution, correct vs error -- should show positive skew
ax = axes[0, 0]
ax.hist(rt[correct == 1], bins=60, alpha=0.6, density=True, label="correct", color="tab:green")
ax.hist(rt[correct == 0], bins=60, alpha=0.6, density=True, label="error", color="tab:red")
ax.set_xlabel("Response time (s)")
ax.set_ylabel("Density")
ax.set_title("Initial-response RT distributions\n(should be positively skewed, cf. Fig. 2)")
ax.legend()

# (2) DRT distribution -- should be positively skewed, concentrated early (cf. Fig. 3)
ax = axes[0, 1]
drt_obs = drt[dr == 1]
ax.hist(drt_obs, bins=40, color="tab:purple", alpha=0.8)
ax.set_xlabel("Double response time (s, relative to initial response)")
ax.set_ylabel("Count")
ax.set_title(f"DRT distribution (n={len(drt_obs)})\n(should be positively skewed, cf. Fig. 3)")

# (3) RT before DR vs no-DR trials -- should look qualitatively similar (cf. Fig. 4)
ax = axes[1, 0]
ax.hist(rt[dr == 0], bins=60, density=True, alpha=0.6, label="no double response", color="tab:blue")
ax.hist(rt[dr == 1], bins=60, density=True, alpha=0.6, label="followed by double response", color="tab:orange")
ax.set_xlabel("Response time (s)")
ax.set_ylabel("Density")
ax.set_title("RT before DR vs. no-DR trials\n(should look similar, cf. Fig. 4)")
ax.legend()

# (4) Summary bar chart: P(DR|C) vs P(DR|E)
ax = axes[1, 1]
ax.bar(["P(DR|correct)", "P(DR|error)"], [p_dr_given_c, p_dr_given_e],
       color=["tab:green", "tab:red"])
ax.set_ylabel("Proportion")
ax.set_title("Double responses are predominantly corrective\n(cf. Table 2/5/6)")

plt.tight_layout()
plt.savefig("sanity_checks.png", dpi=150)
print("Saved sanity_checks.png")
