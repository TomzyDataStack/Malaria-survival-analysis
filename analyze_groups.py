"""
Group-labelled survival analysis (curative murine malaria model).

Compares every treatment group with the infected positive control (Group 14,
normal saline) using Kaplan-Meier curves and log-rank tests with Bonferroni
correction. Group 16 (normal saline, negative control) had no deaths, so it
is shown on the plots but is not tested.

Usage:  python analyze_groups.py path/to/file.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from malaria_analysis import kaplan_meier, logrank_test
from analyze_real_survival import load_survival, median_survival

OUT = Path("results_real")

LABELS = {
    "Group 1": "Gabapentin 15 mg/kg",
    "Group 2": "Gabapentin 150 mg/kg",
    "Group 3": "Gabapentin 300 mg/kg",
    "Group 4": "Pregabalin 12.5 mg/kg",
    "Group 5": "Pregabalin 25 mg/kg",
    "Group 6": "Pregabalin 50 mg/kg",
    "Group 7": "Vitamin D3 50 ug/kg",
    "Group 8": "Pregabalin 12.5 mg/kg + Vit D3",
    "Group 9": "Pregabalin 25 mg/kg + Vit D3",
    "Group 10": "Pregabalin 50 mg/kg + Vit D3",
    "Group 11": "Gabapentin 75 mg/kg + Vit D3",
    "Group 12": "Gabapentin 150 mg/kg + Vit D3",
    "Group 13": "Gabapentin 300 mg/kg + Vit D3",
    "Group 14": "Positive control (infected, saline)",
    "Group 15": "Artesunate 10 mg/kg",
    "Group 16": "Negative control (saline)",
}
POS, NEG = "Group 14", "Group 16"

PANELS = {
    "Gabapentin alone": ["Group 1", "Group 2", "Group 3"],
    "Pregabalin alone": ["Group 4", "Group 5", "Group 6"],
    "Pregabalin + Vit D3": ["Group 8", "Group 9", "Group 10", "Group 7"],
    "Gabapentin + Vit D3": ["Group 11", "Group 12", "Group 13", "Group 7"],
}


def main():
    surv, follow_up = load_survival(sys.argv[1], 8)
    surv = surv[surv["group"].isin(LABELS)].copy()      # Groups 17-20 had no data
    surv["label"] = surv["group"].map(LABELS)
    OUT.mkdir(exist_ok=True)

    ctrl = surv[surv["group"] == POS]
    rows = []
    for g in LABELS:
        d = surv[surv["group"] == g]
        row = {"group": g, "treatment": LABELS[g], "n": len(d), "deaths": int(d.event.sum()),
               "mean_survival_day": round(d.time.mean(), 2),
               "median_survival_day": median_survival(d.time, d.event)}
        if g not in (POS, NEG):
            chi2, p = logrank_test(ctrl.time.values, ctrl.event.values, d.time.values, d.event.values)
            row.update(logrank_p=round(p, 4))
        rows.append(row)
    res = pd.DataFrame(rows)
    k = res["logrank_p"].notna().sum()
    res["p_bonferroni"] = (res["logrank_p"] * k).clip(upper=1).round(4)
    res.to_csv(OUT / "group_comparison.csv", index=False)
    print(res.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, (title, groups) in zip(axes.ravel(), PANELS.items()):
        for g in groups + [POS, "Group 15", NEG]:
            d = surv[surv["group"] == g]
            t, s = kaplan_meier(d.time, d.event)
            style = {"ls": "--", "color": "black"} if g == POS else \
                    {"ls": ":", "color": "grey"} if g in ("Group 15", NEG) else {}
            ax.step(np.append(t, follow_up), np.append(s, s[-1]), where="post",
                    label=LABELS[g], **style)
        ax.set_title(title); ax.legend(fontsize=7)
    for ax in axes[1]: ax.set_xlabel("Days")
    for ax in axes[:, 0]: ax.set_ylabel("Survival probability")
    fig.tight_layout(); fig.savefig(OUT / "survival_by_treatment.png", dpi=180)


if __name__ == "__main__":
    main()
