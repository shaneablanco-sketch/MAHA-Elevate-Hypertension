"""
MCBS 2023 Hypertension Spending Analysis

Loads MCBS 2023 Fall PUF and Cost Supplement PUF, computes survey-weighted
hypertension prevalence and spending by demographic cell, runs an ecological
weighted regression to estimate the spending premium, and projects annual
Medicare FFS savings under prevalence-reduction scenarios.

Dependencies: pip install pandas numpy statsmodels matplotlib
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NA_CODES = ["", "NA", ".", " ", "D", "R", "N"]

FALL_FILE = "data/sfpuf2023_1_fall.csv"
COST_FILE = "data/cspuf2023.csv"

# ---------------------------------------------------------------------------
# Survey helpers
# ---------------------------------------------------------------------------

def _brr_mean(data: pd.DataFrame, var: str, weight_col: str, rep_pattern: str):
    """Return (weighted_mean, BRR_se) for `var` using 100 replicate weights."""
    rep_cols = [c for c in data.columns if re.match(rep_pattern, c)]
    mask = data[var].notna() & data[weight_col].notna()
    d = data.loc[mask]

    w = d[weight_col].values
    y = d[var].values

    mean_full = np.average(y, weights=w)
    rep_means = np.array([np.average(y, weights=d[rc].values) for rc in rep_cols])
    se = np.sqrt(np.mean((rep_means - mean_full) ** 2))

    return mean_full, se


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    fall_raw = pd.read_csv(FALL_FILE, na_values=NA_CODES, low_memory=False)
    cost_raw = pd.read_csv(COST_FILE, na_values=NA_CODES, low_memory=False)
    return fall_raw, cost_raw


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------
FALL_RENAME = {
    # Identifiers & weights
    "PUF_ID":          "respondent_id",
    "SURVEYYR":        "survey_year",
    "PUFFWGT":         "weight_fall",
    # Enrollment flags
    "ADM_FFS_FLAG_YR": "in_ffs",
    "ADM_MA_FLAG_YR":  "in_ma",
    "ADM_DUAL_FLAG_YR":"dual_status",
    "ADM_PARTD":       "has_part_d",
    "ADM_LIS_FLAG_YR": "has_lis_subsidy",
    # Demographics
    "DEM_AGE":         "age",
    "DEM_SEX":         "sex",
    "DEM_RACE":        "race",
    "DEM_EDU":         "education",
    "DEM_INCOME":      "income",
    "DEM_CBSA":        "metro_status",
    "DEM_MARSTA":      "marital_status",
    "DEM_IPR_IND":     "poverty_ratio",
    # Hypertension — core
    "HLT_OCHBP":       "has_hypertension",
    # Hypertension — management
    "HLT_HYPETOLD":    "bp_told_by_doctor",
    "HLT_HYPEMEDS":    "bp_takes_medication",
    "HLT_HYPEHOME":    "bp_monitors_at_home",
    "HLT_HYPECTRL":    "bp_self_ctrl_reported",
    "HLT_HYPESKIP":    "bp_skipped_meds_cost",
    "HLT_HYPEPAY":     "bp_trouble_paying",
    "HLT_HYPECOND":    "bp_told_serious",
    "HLT_HYPEYRS":     "bp_years_diagnosed",
    "HLT_HYPELONG":    "bp_years_on_meds",
    # Cardiovascular comorbidities
    "HLT_OCMYOCAR":    "had_heart_attack",
    "HLT_OCCHD":       "has_coronary_hd",
    "HLT_OCCFAIL":     "has_heart_failure",
    "HLT_OCHRTCND":    "has_other_heart_cond",
    "HLT_OCSTROKE":    "had_stroke",
    "HLT_OCARTERY":    "has_artery_disease",
    # Metabolic comorbidities
    "HLT_OCBETES":     "has_diabetes",
    "HLT_OCCHOLES":    "has_high_cholesterol",
    # Other conditions
    "HLT_OCCANCER":    "has_cancer",
    "HLT_ALZDEM":      "has_alzheimer_dem",
    "HLT_OCDEPRSS":    "has_depression",
    "HLT_OCKIDNY":     "has_kidney_disease",
    "HLT_OCEMPHYS":    "has_copd",
    "HLT_OCARTHRH":    "has_rheum_arthritis",
    "HLT_OCOSARTH":    "has_osteoarthritis",
    "HLT_OCOSTEOP":    "has_osteoporosis",
    # General health
    "HLT_GENHELTH":    "self_rated_health",
    "HLT_BMI_CAT":     "bmi_category",
    "HLT_FUNC_LIM":    "has_functional_limit",
    "HLT_DISDECSN":    "difficulty_decisions",
    "HLT_DISWALK":     "difficulty_walking",
    "HLT_DISBATH":     "difficulty_bathing",
    # Preventive care
    "PRV_BPTAKEN":     "had_bp_screening",
    "PRV_BLOODTST":    "had_blood_test",
    # Access to care
    "ACC_HCTROUBL":    "trouble_getting_care",
    "ACC_HCDELAY":     "delayed_care_cost",
    "ACC_PAYPROB":     "problems_paying_bills",
    "ACC_PAYOVRTM":    "paying_bills_overtime",
}

COST_RENAME = {
    "PUF_ID":          "respondent_id",
    "CSPUFWGT":        "weight_cost",
    # QC demographics
    "CSP_AGE":         "cs_age",
    "CSP_SEX":         "cs_sex",
    "CSP_RACE":        "cs_race",
    "CSP_INCOME":      "cs_income",
    "CSP_NCHRNCND":    "num_chronic_conditions",
    # Spending by payer
    "PAMTTOT":         "spend_total",
    "PAMTCARE":        "spend_medicare_paid",
    "PAMTCAID":        "spend_medicaid_paid",
    "PAMTMADV":        "spend_ma_plan_paid",
    "PAMTOOP":         "spend_out_of_pocket",
    "PAMTALPR":        "spend_private_insurance",
    "PAMTDISC":        "spend_discounts_writeoffs",
    "PAMTOTH":         "spend_other_payers",
    # Spending by service type
    "PAMTIP":          "spend_inpatient_hosp",
    "PAMTOP":          "spend_outpatient_hosp",
    "PAMTMP":          "spend_physician_services",
    "PAMTPM":          "spend_prescription_drugs",
    "PAMTDU":          "spend_dental",
    "PAMTVU":          "spend_vision",
    "PAMTHU":          "spend_hearing",
    "PAMTHH":          "spend_home_health",
    # Utilization
    "PEVENTS":         "total_healthcare_events",
    "IPAEVNTS":        "num_inpatient_stays",
    "OPAEVNTS":        "num_outpatient_visits",
    "MPAEVNTS":        "num_physician_visits",
    "PMAEVNTS":        "num_rx_fills",
    "DUAEVNTS":        "num_dental_events",
    "VUAEVNTS":        "num_vision_events",
    "HUAEVNTS":        "num_hearing_events",
    "HHAEVNTS":        "num_home_health_events",
}

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_fall(df: pd.DataFrame) -> pd.DataFrame:
    rep_cols = [c for c in df.columns if re.match(r"^PUFF\d{3}$", c)]
    keep = [c for c in list(FALL_RENAME) + rep_cols if c in df.columns]
    fall = df[keep].rename(columns=FALL_RENAME).copy()

    # 1=Yes, 2=No → 1/0, preserve NA
    fall["hypertension"] = (fall["has_hypertension"] == 1).astype("Int64")
    fall["hypertension"] = fall["hypertension"].where(fall["has_hypertension"].notna(), pd.NA)

    return fall


def clean_cost(df: pd.DataFrame) -> pd.DataFrame:
    rep_cols = [c for c in df.columns if re.match(r"^CSPUF\d{3}$", c)]
    keep = [c for c in list(COST_RENAME) + rep_cols if c in df.columns]
    return df[keep].rename(columns=COST_RENAME).copy()


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def compute_prevalence_cells(fall: pd.DataFrame) -> pd.DataFrame:
    ffs = fall[fall["in_ffs"] == 1].copy()
    results = []
    for (age_v, sex_v, race_v), grp in ffs.groupby(["age", "sex", "race"], dropna=False):
        mean_est, se_est = _brr_mean(grp, "hypertension", "weight_fall", r"^PUFF\d{3}$")
        results.append({"age": age_v, "sex": sex_v, "race": race_v,
                        "hyp_prev": mean_est, "se_hyp": se_est})
    return pd.DataFrame(results)


_SPEND_VARS = {
    "spend_total":        "se_spend_total",
    "spend_medicare_paid":"se_spend_medicare",
    "spend_out_of_pocket":"se_spend_oop",
}

def compute_spending_cells(cost: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["cs_age", "cs_sex", "cs_race"]
    all_rows = []
    for (age_v, sex_v, race_v), grp in cost.groupby(group_cols, dropna=False):
        row = {"age": age_v, "sex": sex_v, "race": race_v}
        for var, se_col in _SPEND_VARS.items():
            m, se = _brr_mean(grp, var, "weight_cost", r"^CSPUF\d{3}$")
            row[var] = m
            row[se_col] = se
        all_rows.append(row)
    return pd.DataFrame(all_rows)


def join_cells(prev_cells: pd.DataFrame, spend_cells: pd.DataFrame) -> pd.DataFrame:
    cells = prev_cells.merge(spend_cells, on=["age", "sex", "race"], how="inner")
    cells = cells.dropna(subset=["hyp_prev", "spend_total"])
    print(f"Cells available for ecological regression: {len(cells)} / 24")
    return cells


def ecological_regression(cells: pd.DataFrame):
    import statsmodels.api as sm

    w = 1.0 / cells["se_hyp"] ** 2
    X = sm.add_constant(cells["hyp_prev"])

    def _fit(y_col):
        return sm.WLS(cells[y_col], X, weights=w).fit().params["hyp_prev"]

    premium_total    = _fit("spend_total")
    premium_medicare = _fit("spend_medicare_paid")
    premium_oop      = _fit("spend_out_of_pocket")

    print("\n--- Spending premium per person (full hypertension → no hypertension) ---")
    print(f"  Total spending:    ${premium_total:,.0f}")
    print(f"  Medicare paid:     ${premium_medicare:,.0f}")
    print(f"  Out-of-pocket:     ${premium_oop:,.0f}")

    return premium_total, premium_medicare, premium_oop


def compute_baseline(fall: pd.DataFrame):
    ffs = fall[fall["in_ffs"] == 1].copy()
    mask = ffs["hypertension"].notna() & ffs["weight_fall"].notna()
    d = ffs.loc[mask]
    hyp_prev_est = np.average(d["hypertension"].astype(float), weights=d["weight_fall"])

    n_ffs_total = fall.loc[fall["weight_fall"].notna() & (fall["in_ffs"] == 1), "weight_fall"].sum()
    n_hyp = n_ffs_total * hyp_prev_est

    print(
        f"\nBaseline: {hyp_prev_est * 100:.1f}% of FFS Medicare beneficiaries "
        f"have hypertension (~{round(n_hyp / 1000) * 1000:,.0f} people)"
    )
    return hyp_prev_est, n_ffs_total, n_hyp


def build_scenarios(
    n_hyp: float,
    premium_total: float,
    premium_medicare: float,
    premium_oop: float,
) -> pd.DataFrame:
    rows = []
    for r in [5, 10, 15, 20, 25, 30, 50]:
        persons = round(n_hyp * r / 100, -3)
        rows.append({
            "Reduction":                 f"{r}%",
            "People affected":           f"{persons:,.0f}",
            "Savings/person (total)":    f"${premium_total    * r / 100:,.0f}",
            "Savings/person (Medicare)": f"${premium_medicare * r / 100:,.0f}",
            "Savings/person (OOP)":      f"${premium_oop      * r / 100:,.0f}",
            "Total FFS savings":         f"${persons * premium_total / 1e9:.1f}B",
            "_reduction_pct":            r,
            "_total_savings_billions":   persons * premium_total / 1e9,
        })
    df = pd.DataFrame(rows)
    print()
    print(df.drop(columns=["_reduction_pct", "_total_savings_billions"]).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_savings(scenarios: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = scenarios["_reduction_pct"].values
    y = scenarios["_total_savings_billions"].values

    ax.bar(x, y, width=2.8, color="#2166AC")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.05, f"${yi:.1f}B", ha="center", va="bottom",
                fontsize=9.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}B"))
    ax.set_ylim(0, max(y) * 1.14)

    ax.set_title(
        "Projected Annual Medicare FFS Savings from Reducing Hypertension",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Reduction in hypertension prevalence", fontsize=11)
    ax.set_ylabel("Annual savings (billions USD)", fontsize=11)

    fig.text(
        0.5, 0.01,
        "Ecological model: MCBS 2023 Fall (hypertension prevalence) linked to Cost Supplement (spending)\n"
        "by demographic cell (age × sex × race) · survey-weighted · FFS beneficiaries only",
        ha="center", fontsize=8, color="#888888",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = "mcbs_hypertension_savings.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {out_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fall_raw, cost_raw = load_data()
    fall = clean_fall(fall_raw)
    cost = clean_cost(cost_raw)

    prev_cells  = compute_prevalence_cells(fall)
    spend_cells = compute_spending_cells(cost)
    cells       = join_cells(prev_cells, spend_cells)

    premium_total, premium_medicare, premium_oop = ecological_regression(cells)
    _, _, n_hyp = compute_baseline(fall)
    scenarios   = build_scenarios(n_hyp, premium_total, premium_medicare, premium_oop)

    plot_savings(scenarios)


if __name__ == "__main__":
    main()
