# Parasitaemia & Survival Analysis Pipeline (Murine Malaria Study)

A reproducible Python workflow for analysing the type of data produced in an in vivo antimalarial experiment: daily parasitaemia counts and survival times for infected mice given a drug or vehicle.

> **Note:** All data in this repository are **simulated** for demonstration. They are not real experimental results and must not be reported as such.

## What it does

1. Builds a simulated dataset (3 groups, 10 mice each, parasitaemia on days 3-10, survival to day 21)
2. Summarises parasitaemia per group per day (mean, SD)
3. Compares groups on a chosen day (Kruskal-Wallis, then Bonferroni-corrected Mann-Whitney U tests against control)
4. Estimates survival with Kaplan-Meier curves and compares groups to control with the log-rank test
5. Saves figures, tables and a text summary to `results/`

The log-rank implementation was cross-checked against `scipy.stats.logrank`.

## Run it

```bash
pip install -r requirements.txt
python malaria_analysis.py
```

## Use your own data

Replace `simulate_study()` with code that reads your CSV files and returns two tables:

- `para`: columns `mouse, group, day, parasitaemia_pct`
- `surv`: columns `mouse, group, time, event` (event = 1 if the mouse died, 0 if it survived to the end)

## Outputs (in `results/`)

- `parasitaemia_over_time.png`, `survival_curves.png`
- `parasitaemia_summary.csv`, `parasitaemia_pairwise_tests.csv`, `survival_logrank_tests.csv`
- `parasitaemia_raw.csv`, `survival_raw.csv`, `summary.txt`

## Libraries

NumPy, pandas, SciPy, Matplotlib

## Possible extensions

- Add body-weight and clinical-score tracking
- Add a molecular docking step (RDKit / AutoDock Vina) for ligand preparation and batch docking
- Run the pipeline on cloud compute for larger datasets
