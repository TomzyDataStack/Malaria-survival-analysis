"""
Analyse a real survival spreadsheet laid out as daily death counts per group.

Expected layout (as in project_survival_data_curve.xlsx):
  - a header row with "Day 1", "Day 2", ... "Day N"
  - one row per group starting with a label such as "Group 1"
  - each day cell = number of mice that died on that day (blank = 0)
Mice that never die are treated as censored at the last day observed.

Usage:  python analyze_real_survival.py path/to/file.xlsx [group_size]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from malaria_analysis import kaplan_meier

OUT = Path("results_real")


def load_survival(xlsx, group_size=8):
    raw = pd.read_excel(xlsx, header=None)
    hdr_row = raw.index[raw.apply(lambda r: r.astype(str).str.fullmatch(r"Day \d+").any(), axis=1)][0]
    day_cols = {c: int(str(v).split()[1]) for c, v in raw.loc[hdr_row].items()
                if str(v).startswith("Day ")}
    follow_up = max(day_cols.values())
    rows = []
    for _, r in raw.loc[hdr_row + 1:].iterrows():
        label = str(r[0]).strip()
        if not label.lower().startswith("group"):
            continue
        died = 0
        for c, day in day_cols.items():
            n = pd.to_numeric(r[c], errors="coerce")
            n = 0 if pd.isna(n) else int(n)
            rows += [{"group": label, "time": day, "event": 1}] * n
            died += n
        rows += [{"group": label, "time": follow_up, "event": 0}] * (group_size - died)
    return pd.DataFrame(rows), follow_up


def median_survival(time, event):
    t, s = kaplan_meier(time, event)
    below = np.where(s <= 0.5)[0]
    return float(t[below[0]]) if len(below) else np.nan


def main():
    xlsx = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUT.mkdir(exist_ok=True)
    surv, follow_up = load_survival(xlsx, size)
    surv.to_csv(OUT / "survival_per_mouse.csv", index=False)

    table = (surv.groupby("group", sort=False)
             .apply(lambda d: pd.Series({
                 "n": len(d), "deaths": int(d.event.sum()),
                 "survivors": int((1 - d.event).sum()),
                 "median_survival_day": median_survival(d.time, d.event),
                 "first_death_day": d.loc[d.event == 1, "time"].min(),
                 "last_death_day": d.loc[d.event == 1, "time"].max()}), include_groups=False))
    table.to_csv(OUT / "group_summary.csv")

    t, s = kaplan_meier(surv.time, surv.event)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.step(t, s, where="post", color="black")
    ax.set(xlabel="Days", ylabel="Survival probability", ylim=(0, 1.05),
           title=f"Overall Kaplan-Meier survival (n = {len(surv)})")
    fig.tight_layout(); fig.savefig(OUT / "overall_survival.png", dpi=200)
    print(table.to_string())
    print(f"\nTotal mice: {len(surv)}; deaths: {int(surv.event.sum())}; follow-up: day {follow_up}")


if __name__ == "__main__":
    main()
