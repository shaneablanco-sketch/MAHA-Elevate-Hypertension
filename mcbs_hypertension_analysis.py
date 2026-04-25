"""
MCBS 2023 Hypertension Spending Analysis

Two complementary models:
  1. Ecological regression — cell-level prevalence vs spending across 24 demographic
     cells (age × sex × race). Estimates population-level spending premium.
  2. Severity-based individual model — among FFS hypertensives, compares spending
     for uncontrolled vs controlled BP (HLT_HYPECTRL). The gap is the per-person
     annual savings from improving BP control.
     savings_for_severity_reduction(pct) returns individual savings for any % improvement.

Dependencies: pip install pandas numpy statsmodels matplotlib
"""

import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NA_CODES = ["", "NA", ".", " ", "D", "R", "N"]
FALL_FILE = "data/sfpuf2023_1_fall.csv"
COST_FILE = "data/cspuf2023.csv"

# Fall PUF_ID = 23_000_000 + sequential; Cost PUF_ID = 92_300_000 + sequential
_FALL_ID_OFFSET = 23_000_000
_COST_ID_OFFSET = 92_300_000

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------
FALL_RENAME = {
    "PUF_ID": "respondent_id",
    "SURVEYYR": "survey_year",
    "PUFFWGT": "weight_fall",
    "ADM_FFS_FLAG_YR": "in_ffs",
    "ADM_MA_FLAG_YR": "in_ma",
    "ADM_DUAL_FLAG_YR": "dual_status",
    "ADM_PARTD": "has_part_d",
    "ADM_LIS_FLAG_YR": "has_lis_subsidy",
    "DEM_AGE": "age",
    "DEM_SEX": "sex",
    "DEM_RACE": "race",
    "DEM_EDU": "education",
    "DEM_INCOME": "income",
    "DEM_CBSA": "metro_status",
    "DEM_MARSTA": "marital_status",
    "DEM_IPR_IND": "poverty_ratio",
    "HLT_OCHBP": "has_hypertension",
    "HLT_HYPETOLD": "bp_told_by_doctor",
    "HLT_HYPEMEDS": "bp_takes_medication",
    "HLT_HYPEHOME": "bp_monitors_at_home",
    "HLT_HYPECTRL": "bp_self_ctrl_reported",
    "HLT_HYPESKIP": "bp_skipped_meds_cost",
    "HLT_HYPEPAY": "bp_trouble_paying",
    "HLT_HYPECOND": "bp_told_serious",
    "HLT_HYPEYRS": "bp_years_diagnosed",
    "HLT_HYPELONG": "bp_years_on_meds",
    "HLT_OCMYOCAR": "had_heart_attack",
    "HLT_OCCHD": "has_coronary_hd",
    "HLT_OCCFAIL": "has_heart_failure",
    "HLT_OCHRTCND": "has_other_heart_cond",
    "HLT_OCSTROKE": "had_stroke",
    "HLT_OCARTERY": "has_artery_disease",
    "HLT_OCBETES": "has_diabetes",
    "HLT_OCCHOLES": "has_high_cholesterol",
    "HLT_OCCANCER": "has_cancer",
    "HLT_ALZDEM": "has_alzheimer_dem",
    "HLT_OCDEPRSS": "has_depression",
    "HLT_OCKIDNY": "has_kidney_disease",
    "HLT_OCEMPHYS": "has_copd",
    "HLT_OCARTHRH": "has_rheum_arthritis",
    "HLT_OCOSARTH": "has_osteoarthritis",
    "HLT_OCOSTEOP": "has_osteoporosis",
    "HLT_GENHELTH": "self_rated_health",
    "HLT_BMI_CAT": "bmi_category",
    "HLT_FUNC_LIM": "has_functional_limit",
    "HLT_DISDECSN": "difficulty_decisions",
    "HLT_DISWALK": "difficulty_walking",
    "HLT_DISBATH": "difficulty_bathing",
    "PRV_BPTAKEN": "had_bp_screening",
    "PRV_BLOODTST": "had_blood_test",
    "ACC_HCTROUBL": "trouble_getting_care",
    "ACC_HCDELAY": "delayed_care_cost",
    "ACC_PAYPROB": "problems_paying_bills",
    "ACC_PAYOVRTM": "paying_bills_overtime",
}

COST_RENAME = {
    "PUF_ID": "respondent_id",
    "CSPUFWGT": "weight_cost",
    "CSP_AGE": "cs_age",
    "CSP_SEX": "cs_sex",
    "CSP_RACE": "cs_race",
    "CSP_INCOME": "cs_income",
    "CSP_NCHRNCND": "num_chronic_conditions",
    "PAMTTOT": "spend_total",
    "PAMTCARE": "spend_medicare_paid",
    "PAMTCAID": "spend_medicaid_paid",
    "PAMTMADV": "spend_ma_plan_paid",
    "PAMTOOP": "spend_out_of_pocket",
    "PAMTALPR": "spend_private_insurance",
    "PAMTDISC": "spend_discounts_writeoffs",
    "PAMTOTH": "spend_other_payers",
    "PAMTIP": "spend_inpatient_hosp",
    "PAMTOP": "spend_outpatient_hosp",
    "PAMTMP": "spend_physician_services",
    "PAMTPM": "spend_prescription_drugs",
    "PAMTDU": "spend_dental",
    "PAMTVU": "spend_vision",
    "PAMTHU": "spend_hearing",
    "PAMTHH": "spend_home_health",
    "PEVENTS": "total_healthcare_events",
    "IPAEVNTS": "num_inpatient_stays",
    "OPAEVNTS": "num_outpatient_visits",
    "MPAEVNTS": "num_physician_visits",
    "PMAEVNTS": "num_rx_fills",
    "DUAEVNTS": "num_dental_events",
    "VUAEVNTS": "num_vision_events",
    "HUAEVNTS": "num_hearing_events",
    "HHAEVNTS": "num_home_health_events",
}

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    fall_raw = pd.read_csv(FALL_FILE, na_values=NA_CODES, low_memory=False)
    cost_raw = pd.read_csv(COST_FILE, na_values=NA_CODES, low_memory=False)
    return fall_raw, cost_raw


def clean_fall(df: pd.DataFrame) -> pd.DataFrame:
    rep_cols = [c for c in df.columns if re.match(r"^PUFF\d{3}$", c)]
    keep = [c for c in list(FALL_RENAME) + rep_cols if c in df.columns]
    fall = df[keep].rename(columns=FALL_RENAME).copy()
    fall["hypertension"] = (fall["has_hypertension"] == 1).astype("Int64")
    fall["hypertension"] = fall["hypertension"].where(
        fall["has_hypertension"].notna(), pd.NA
    )
    return fall


def clean_cost(df: pd.DataFrame) -> pd.DataFrame:
    rep_cols = [c for c in df.columns if re.match(r"^CSPUF\d{3}$", c)]
    keep = [c for c in list(COST_RENAME) + rep_cols if c in df.columns]
    return df[keep].rename(columns=COST_RENAME).copy()


# ---------------------------------------------------------------------------
# Survey helpers
# ---------------------------------------------------------------------------


def _brr_mean(data: pd.DataFrame, var: str, weight_col: str, rep_pattern: str):
    """Weighted mean and BRR standard error for a single variable."""
    rep_cols = [c for c in data.columns if re.match(rep_pattern, c)]
    mask = data[var].notna() & data[weight_col].notna()
    d = data.loc[mask]
    w, y = d[weight_col].values, d[var].values
    mean_full = np.average(y, weights=w)
    rep_means = np.array([np.average(y, weights=d[rc].values) for rc in rep_cols])
    se = np.sqrt(np.mean((rep_means - mean_full) ** 2))
    return mean_full, se


def _brr_wls_coef(
    data: pd.DataFrame,
    outcome: str,
    predictors: list,
    weight_col: str,
    rep_pattern: str,
    target_coef: str,
) -> Tuple[float, float]:
    """
    Survey-weighted OLS coefficient with BRR standard error.
    Returns (estimate, se) for `target_coef`.
    """
    rep_cols = [c for c in data.columns if re.match(rep_pattern, c)]
    cols_needed = [outcome] + predictors + [weight_col] + rep_cols
    d = data[cols_needed].dropna()

    X = sm.add_constant(d[predictors])
    y = d[outcome]

    full_coef = sm.WLS(y, X, weights=d[weight_col]).fit().params[target_coef]
    rep_coefs = np.array(
        [sm.WLS(y, X, weights=d[rc]).fit().params[target_coef] for rc in rep_cols]
    )
    se = np.sqrt(np.mean((rep_coefs - full_coef) ** 2))
    return full_coef, se


# ---------------------------------------------------------------------------
# Step 1 — Ecological regression (cell-level)
# ---------------------------------------------------------------------------


def compute_prevalence_cells(fall: pd.DataFrame) -> pd.DataFrame:
    ffs = fall[fall["in_ffs"] == 1].copy()
    results = []
    for (age_v, sex_v, race_v), grp in ffs.groupby(
        ["age", "sex", "race"], dropna=False
    ):
        mean_est, se_est = _brr_mean(grp, "hypertension", "weight_fall", r"^PUFF\d{3}$")
        results.append(
            {
                "age": age_v,
                "sex": sex_v,
                "race": race_v,
                "hyp_prev": mean_est,
                "se_hyp": se_est,
            }
        )
    return pd.DataFrame(results)


def compute_spending_cells(cost: pd.DataFrame) -> pd.DataFrame:
    spend_vars = {
        "spend_total": "se_spend_total",
        "spend_medicare_paid": "se_spend_medicare",
        "spend_out_of_pocket": "se_spend_oop",
    }
    rows = []
    for (age_v, sex_v, race_v), grp in cost.groupby(
        ["cs_age", "cs_sex", "cs_race"], dropna=False
    ):
        row = {"age": age_v, "sex": sex_v, "race": race_v}
        for var, se_col in spend_vars.items():
            m, se = _brr_mean(grp, var, "weight_cost", r"^CSPUF\d{3}$")
            row[var] = m
            row[se_col] = se
        rows.append(row)
    return pd.DataFrame(rows)


def ecological_regression(fall: pd.DataFrame, cost: pd.DataFrame):
    prev_cells = compute_prevalence_cells(fall)
    spend_cells = compute_spending_cells(cost)
    cells = prev_cells.merge(spend_cells, on=["age", "sex", "race"], how="inner")
    cells = cells.dropna(subset=["hyp_prev", "spend_total"])
    print(f"Cells available for ecological regression: {len(cells)} / 24")

    w = 1.0 / cells["se_hyp"] ** 2
    X = sm.add_constant(cells["hyp_prev"])

    def _fit(y_col):
        return sm.WLS(cells[y_col], X, weights=w).fit().params["hyp_prev"]

    p_total = _fit("spend_total")
    p_medicare = _fit("spend_medicare_paid")
    p_oop = _fit("spend_out_of_pocket")

    print("\n--- Spending premium per person (full hypertension → no hypertension) ---")
    print(f"  Total spending:    ${p_total:,.0f}")
    print(f"  Medicare paid:     ${p_medicare:,.0f}")
    print(f"  Out-of-pocket:     ${p_oop:,.0f}")

    return cells, p_total, p_medicare, p_oop


def compute_baseline(fall: pd.DataFrame):
    ffs = fall[fall["in_ffs"] == 1].copy()
    mask = ffs["hypertension"].notna() & ffs["weight_fall"].notna()
    d = ffs.loc[mask]
    hyp_prev_est = np.average(d["hypertension"].astype(float), weights=d["weight_fall"])
    n_ffs_total = fall.loc[
        fall["weight_fall"].notna() & (fall["in_ffs"] == 1), "weight_fall"
    ].sum()
    n_hyp = n_ffs_total * hyp_prev_est
    print(
        f"\nBaseline: {hyp_prev_est * 100:.1f}% of FFS Medicare beneficiaries "
        f"have hypertension (~{round(n_hyp / 1000) * 1000:,.0f} people)"
    )
    return hyp_prev_est, n_ffs_total, n_hyp


def build_prevalence_scenarios(n_hyp, p_total, p_medicare, p_oop) -> pd.DataFrame:
    rows = []
    for r in [5, 10, 15, 20, 25, 30, 50]:
        persons = round(n_hyp * r / 100, -3)
        rows.append(
            {
                "Reduction": f"{r}%",
                "People affected": f"{persons:,.0f}",
                "Savings/person (total)": f"${p_total * r / 100:,.0f}",
                "Savings/person (Medicare)": f"${p_medicare * r / 100:,.0f}",
                "Savings/person (OOP)": f"${p_oop * r / 100:,.0f}",
                "Total FFS savings": f"${persons * p_total / 1e9:.1f}B",
                "_reduction_pct": r,
                "_total_savings_billions": persons * p_total / 1e9,
            }
        )
    df = pd.DataFrame(rows)
    print("\n--- Ecological prevalence scenario table ---")
    print(
        df.drop(columns=["_reduction_pct", "_total_savings_billions"]).to_string(
            index=False
        )
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — Individual severity model
# ---------------------------------------------------------------------------


def merge_hypertensives(fall: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """
    Join FFS hypertensives from the Fall file to the Cost Supplement.
    The two files share a sequential beneficiary ID but use different prefixes:
      Fall PUF_ID = 23_000_000 + seq  →  seq = PUF_ID - 23_000_000
      Cost PUF_ID = 92_300_000 + seq  →  seq = PUF_ID - 92_300_000
    """
    fall_hyp = (
        fall.loc[(fall["in_ffs"] == 1) & (fall["hypertension"] == 1)]
        .copy()
        .assign(_key=lambda d: d["respondent_id"] - _FALL_ID_OFFSET)
    )
    cost_keyed = cost.assign(_key=lambda d: d["respondent_id"] - _COST_ID_OFFSET)

    merged = fall_hyp.merge(cost_keyed, on="_key", suffixes=("", "_cost")).drop(
        columns="_key"
    )

    # bp_self_ctrl_reported: 1=controlled, 2=uncontrolled → recode to 0/1
    merged["bp_uncontrolled"] = (merged["bp_self_ctrl_reported"] == 2).astype("Int64")
    merged["bp_uncontrolled"] = merged["bp_uncontrolled"].where(
        merged["bp_self_ctrl_reported"].notna(), pd.NA
    )
    merged = merged.dropna(subset=["bp_uncontrolled"])

    n_unc = merged["bp_uncontrolled"].sum()
    n_tot = len(merged)
    print(f"\nHypertensives matched to cost file: {n_tot} records")
    print(
        f"BP uncontrolled: {n_unc} ({100 * n_unc / n_tot:.1f}%)  "
        f"| controlled: {n_tot - n_unc} ({100 * (n_tot - n_unc) / n_tot:.1f}%)"
    )
    return merged


def severity_regression(merged: pd.DataFrame):
    """
    Spending gap: uncontrolled vs controlled BP, adjusted for age/sex/race.
    Returns (gap_total, se_total, gap_medicare, se_medicare, gap_oop, se_oop).
    """
    # Dummy-encode age/sex/race (drop first level to avoid collinearity)
    dummies = pd.get_dummies(
        merged[["age", "sex", "race"]].astype(str), drop_first=True
    )
    predictors_df = pd.concat(
        [
            merged[["bp_uncontrolled"]].astype(float).reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    predictor_names = list(predictors_df.columns)

    rep_pattern = r"^CSPUF\d{3}$"
    rep_cols = [c for c in merged.columns if re.match(rep_pattern, c)]
    weight_col = "weight_cost"

    results = {}
    for outcome in ["spend_total", "spend_medicare_paid", "spend_out_of_pocket"]:
        data_full = (
            pd.concat(
                [
                    merged[[outcome, weight_col]].reset_index(drop=True),
                    merged[rep_cols].reset_index(drop=True),
                    predictors_df,
                ],
                axis=1,
            )
            .dropna()
            .astype(float)
        )  # statsmodels requires float64; dropna first to avoid NA→float issues

        X = sm.add_constant(data_full[predictor_names])
        y = data_full[outcome]
        w = data_full[weight_col]

        full_coef = sm.WLS(y, X, weights=w).fit().params["bp_uncontrolled"]
        rep_coefs = np.array(
            [
                sm.WLS(y, X, weights=data_full[rc]).fit().params["bp_uncontrolled"]
                for rc in rep_cols
            ]
        )
        se = np.sqrt(np.mean((rep_coefs - full_coef) ** 2))
        results[outcome] = (full_coef, se)

    gap_total, se_total = results["spend_total"]
    gap_medicare, se_medicare = results["spend_medicare_paid"]
    gap_oop, se_oop = results["spend_out_of_pocket"]

    print("\n--- Annual spending gap: uncontrolled vs controlled BP (adjusted) ---")
    print(f"  Total spending:    ${gap_total:,.0f}  (SE ${se_total:,.0f})")
    print(f"  Medicare paid:     ${gap_medicare:,.0f}  (SE ${se_medicare:,.0f})")
    print(f"  Out-of-pocket:     ${gap_oop:,.0f}  (SE ${se_oop:,.0f})")

    return gap_total, gap_medicare, gap_oop


def savings_for_severity_reduction(
    reduction_pct: float,
    gap_total: float,
    gap_medicare: float,
    gap_oop: float,
) -> dict:
    """
    Individual-level savings for any % improvement in BP severity.
    A 10% reduction means moving 10% of the way from uncontrolled to controlled BP.
    Input is clamped to [1, 100] to prevent out-of-range results.
    """
    pct = min(max(float(reduction_pct), 1.0), 100.0) / 100.0
    return {
        "reduction_pct": reduction_pct,
        "savings_total": gap_total * pct,
        "savings_medicare": gap_medicare * pct,
        "savings_oop": gap_oop * pct,
    }


def build_severity_scenarios(gap_total, gap_medicare, gap_oop) -> pd.DataFrame:
    rows = []
    for r in [10, 20, 30, 40, 50, 75, 100]:
        s = savings_for_severity_reduction(r, gap_total, gap_medicare, gap_oop)
        rows.append(
            {
                "Severity reduction": f"{r}%",
                "Savings/year (total)": f"${s['savings_total']:,.0f}",
                "Savings/year (Medicare)": f"${s['savings_medicare']:,.0f}",
                "Savings/year (OOP)": f"${s['savings_oop']:,.0f}",
                "_pct": r,
                "_savings": s["savings_total"],
            }
        )
    df = pd.DataFrame(rows)
    print("\n--- Individual severity reduction scenario table ---")
    print(df.drop(columns=["_pct", "_savings"]).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_prevalence_savings(scenarios: pd.DataFrame):
    x = scenarios["_reduction_pct"].values
    y = scenarios["_total_savings_billions"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y, width=2.8, color="#2166AC")

    for xi, yi in zip(x, y):
        va = "bottom" if yi >= 0 else "top"
        yoffs = 0.05 if yi >= 0 else -0.05
        ax.text(
            xi,
            yi + yoffs,
            f"${yi:.1f}B",
            ha="center",
            va=va,
            fontsize=9.5,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}B"))
    ax.axhline(0, color="grey", linewidth=0.6)

    pad = max(abs(y)) * 0.14
    ax.set_ylim(min(y) - pad, max(y) + pad)

    ax.set_title(
        "Projected Annual Medicare FFS Savings from Reducing Hypertension",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Reduction in hypertension prevalence", fontsize=11)
    ax.set_ylabel("Annual savings (billions USD)", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Ecological model: MCBS 2023 Fall (hypertension prevalence) linked to "
        "Cost Supplement (spending)\n"
        "by demographic cell (age × sex × race) · survey-weighted · FFS beneficiaries only",
        ha="center",
        fontsize=8,
        color="#888888",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("mcbs_prevalence_savings.png", dpi=150, bbox_inches="tight")
    print("\nChart saved to: mcbs_prevalence_savings.png")
    plt.show()


def plot_severity_curve(gap_total: float, highlight_pct: int = 10):
    pcts = np.arange(1, 101)
    savings = gap_total * pcts / 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pcts, savings, color="#aaaaaa", linewidth=2)
    ax.axhline(0, color="grey", linewidth=0.5)

    hi_savings = gap_total * highlight_pct / 100
    ax.axvline(highlight_pct, color="#2166AC", linestyle="--", linewidth=0.9)
    ax.scatter([highlight_pct], [hi_savings], color="#2166AC", s=60, zorder=5)
    ax.text(
        highlight_pct + 1.5,
        hi_savings,
        f"${hi_savings:,.0f}",
        color="#2166AC",
        fontsize=10,
        fontweight="bold",
        va="center",
    )

    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(
        "Individual Annual Savings by Hypertension Severity Reduction",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Reduction in BP severity (% toward full control)", fontsize=11)
    ax.set_ylabel("Annual savings per person (total)", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Severity proxy: self-reported BP control status (MCBS 2023, HLT_HYPECTRL). "
        "Adjusted for age, sex, race. FFS hypertensives only.",
        ha="center",
        fontsize=8,
        color="#888888",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("mcbs_severity_savings.png", dpi=150, bbox_inches="tight")
    print("Chart saved to: mcbs_severity_savings.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    fall_raw, cost_raw = load_data()
    fall = clean_fall(fall_raw)
    cost = clean_cost(cost_raw)

    # --- Ecological regression (population prevalence model) ---
    cells, p_total, p_medicare, p_oop = ecological_regression(fall, cost)
    _, _, n_hyp = compute_baseline(fall)
    prev_scenarios = build_prevalence_scenarios(n_hyp, p_total, p_medicare, p_oop)
    plot_prevalence_savings(prev_scenarios)

    # --- Severity-based individual model ---
    merged = merge_hypertensives(fall, cost)
    gap_total, gap_medicare, gap_oop = severity_regression(merged)
    sev_scenarios = build_severity_scenarios(gap_total, gap_medicare, gap_oop)
    plot_severity_curve(gap_total, highlight_pct=10)

    # savings_for_severity_reduction() is available for interactive use, e.g.:
    #   s = savings_for_severity_reduction(15, gap_total, gap_medicare, gap_oop)
    #   print(s)


if __name__ == "__main__":
    main()
