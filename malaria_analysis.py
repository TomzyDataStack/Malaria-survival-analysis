"""
Parasitaemia & Survival Analysis Pipeline for a Murine Malaria Study
=====================================================================
A reproducible Python workflow for the kind of data collected in an in vivo
antimalarial experiment (e.g. P. berghei-infected mice treated with a drug).

IMPORTANT: the data here are SIMULATED for demonstration. They are NOT real
experimental results. To use real data, replace `simulate_study()` with a
function that reads your own CSV files (see README).

Steps
-----
1. Build a (simulated) dataset: daily parasitaemia + survival time per mouse
2. Summarise parasitaemia per group per day (mean, SD)
3. Test group differences on a chosen day (Kruskal-Wallis, then pairwise
   Mann-Whitney U with Bonferroni correction)
4. Estimate survival (Kaplan-Meier) and compare to control (log-rank test)
5. Save figures, tables and a plain-text summary to the `results/` folder
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # lets the script run without a display
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42                      # fixed seed -> same results every run
DAYS = np.arange(3, 11)        # days post-infection when blood smears are read
FOLLOW_UP = 21                 # mice alive at day 21 are "censored"
GROUPS = {                     # group name -> (growth-rate factor, death-rate factor)
    "Infected control": (1.00, 1.00),
    "Drug - low dose": (0.85, 0.60),
    "Drug - high dose": (0.70, 0.35),
}
N_PER_GROUP = 10
COMPARE_DAY = 7                # day used for the parasitaemia group comparison
OUT = Path("results")


# ----------------------------------------------------------------- 1. data
def simulate_study(rng):
    """Return (parasitaemia_df, survival_df) for the simulated study."""
    para_rows, surv_rows = [], []
    mouse_id = 0
    for group, (growth, death) in GROUPS.items():
        for _ in range(N_PER_GROUP):
            mouse_id += 1
            # logistic-style parasitaemia growth (% infected red cells) + noise
            rate = rng.normal(0.9 * growth, 0.08)
            mid = rng.normal(6.0 / growth, 0.5)
            for d in DAYS:
                p = 60 / (1 + np.exp(-rate * (d - mid))) * rng.normal(1, 0.08)
                para_rows.append({"mouse": mouse_id, "group": group, "day": int(d),
                                  "parasitaemia_pct": round(float(np.clip(p, 0.1, 80)), 2)})
            # time to death: Weibull, slower with effective treatment
            t = rng.weibull(4.0) * 9.0 / death ** 0.5
            died = t < FOLLOW_UP
            surv_rows.append({"mouse": mouse_id, "group": group,
                              "time": round(min(t, FOLLOW_UP), 1), "event": int(died)})
    return pd.DataFrame(para_rows), pd.DataFrame(surv_rows)


# ------------------------------------------------------------ 2. parasitaemia
def summarise_parasitaemia(para):
    return (para.groupby(["group", "day"])["parasitaemia_pct"]
            .agg(mean="mean", sd="std", n="count").round(2).reset_index())


def compare_groups_on_day(para, day, control="Infected control"):
    """Kruskal-Wallis across all groups, then Bonferroni-corrected pairwise tests vs control."""
    sub = para[para["day"] == day]
    samples = {g: d["parasitaemia_pct"].values for g, d in sub.groupby("group")}
    kw = stats.kruskal(*samples.values())
    others = [g for g in samples if g != control]
    rows = []
    for g in others:
        u = stats.mannwhitneyu(samples[control], samples[g], alternative="two-sided")
        rows.append({"comparison": f"{g} vs {control}", "U": u.statistic,
                     "p_raw": u.pvalue, "p_bonferroni": min(u.pvalue * len(others), 1.0)})
    return kw, pd.DataFrame(rows).round(4)


# ----------------------------------------------------------------- 3. survival
def kaplan_meier(time, event):
    """Return event times and survival probabilities (step function)."""
    order = np.argsort(time)
    time, event = np.asarray(time)[order], np.asarray(event)[order]
    surv, at_risk, out_t, out_s = 1.0, len(time), [0.0], [1.0]
    for t in np.unique(time):
        deaths = int(((time == t) & (event == 1)).sum())
        leaving = int((time == t).sum())
        if deaths:
            surv *= 1 - deaths / at_risk
            out_t.append(t)
            out_s.append(surv)
        at_risk -= leaving
    return np.array(out_t), np.array(out_s)


def logrank_test(t1, e1, t2, e2):
    """Two-group log-rank test. Returns (chi-square, p-value)."""
    t = np.concatenate([t1, t2]); e = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    obs_minus_exp, var = 0.0, 0.0
    for tt in np.unique(t[e == 1]):
        at_risk = t >= tt
        n, n1 = at_risk.sum(), (at_risk & (g == 0)).sum()
        d = ((t == tt) & (e == 1)).sum()
        d1 = ((t == tt) & (e == 1) & (g == 0)).sum()
        obs_minus_exp += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = obs_minus_exp ** 2 / var
    return chi2, stats.chi2.sf(chi2, df=1)


# -------------------------------------------------------------------- 4. plots
def plot_parasitaemia(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g, d in summary.groupby("group"):
        ax.errorbar(d["day"], d["mean"], yerr=d["sd"], marker="o", capsize=3, label=g)
    ax.set(xlabel="Days post-infection", ylabel="Parasitaemia (%)",
           title="Parasitaemia over time (mean \u00b1 SD, simulated data)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_survival(surv, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g, d in surv.groupby("group"):
        t, s = kaplan_meier(d["time"], d["event"])
        ax.step(np.append(t, FOLLOW_UP), np.append(s, s[-1]), where="post", label=g)
    ax.set(xlabel="Days post-infection", ylabel="Survival probability", ylim=(0, 1.05),
           title="Kaplan-Meier survival (simulated data)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    para, surv = simulate_study(rng)
    para.to_csv(OUT / "parasitaemia_raw.csv", index=False)
    surv.to_csv(OUT / "survival_raw.csv", index=False)

    summary = summarise_parasitaemia(para)
    summary.to_csv(OUT / "parasitaemia_summary.csv", index=False)
    kw, pairwise = compare_groups_on_day(para, COMPARE_DAY)
    pairwise.to_csv(OUT / "parasitaemia_pairwise_tests.csv", index=False)

    ctrl = surv[surv["group"] == "Infected control"]
    lr_rows = []
    for g in GROUPS:
        if g == "Infected control":
            continue
        d = surv[surv["group"] == g]
        chi2, p = logrank_test(ctrl["time"].values, ctrl["event"].values,
                               d["time"].values, d["event"].values)
        lr_rows.append({"comparison": f"{g} vs Infected control",
                        "chi2": round(chi2, 3), "p_value": round(p, 4)})
    logrank = pd.DataFrame(lr_rows)
    logrank.to_csv(OUT / "survival_logrank_tests.csv", index=False)

    plot_parasitaemia(summary, OUT / "parasitaemia_over_time.png")
    plot_survival(surv, OUT / "survival_curves.png")

    report = [
        "SIMULATED DATA - demonstration only, not real results", "",
        f"Mice per group: {N_PER_GROUP}; groups: {', '.join(GROUPS)}", "",
        f"Parasitaemia on day {COMPARE_DAY}: Kruskal-Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.4f}",
        pairwise.to_string(index=False), "",
        "Survival (log-rank vs infected control):", logrank.to_string(index=False),
    ]
    (OUT / "summary.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nFiles saved in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
