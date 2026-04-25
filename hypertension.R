# %% Set up
library(tidyverse)
library(tidymodels)
library(scales)
library(janitor)
library(survey)

# %% loading in data
.na_codes <- c("", "NA", ".", " ", "D", "R", "N")
sf_fall <- read_csv(
  "data/sfpuf2023_1_fall.csv",
  show_col_types = FALSE,
  na = .na_codes
)
cs <- read_csv("data/cspuf2023.csv", show_col_types = FALSE, na = .na_codes)
# %% Cleaning for Fall data
fall <- sf_fall |>
  select(
    # --- Identifiers & survey weights ---
    # PUF_ID uniquely identifies each beneficiary across all four files.
    # PUFFWGT is the full-sample ever-enrolled weight; use it for all
    # Fall-based prevalence estimates. The PUFF001–PUFF100 replicate
    # weights are used by svrepdesign() to compute standard errors via BRR.
    respondent_id = PUF_ID,
    survey_year = SURVEYYR,
    weight_fall = PUFFWGT,
    matches("^PUFF[0-9]{3}$"), # 100 Fall BRR replicate weights (PUFF001–PUFF100)

    # --- Medicare enrollment flags ---
    # These are annual flags: 1 = enrolled at any point during 2023.
    # A beneficiary can have both FFS and MA in the same year (plan change).
    in_ffs = ADM_FFS_FLAG_YR, # Fee-for-Service coverage
    in_ma = ADM_MA_FLAG_YR, # Medicare Advantage coverage
    dual_status = ADM_DUAL_FLAG_YR, # 1=full dual, 2=partial, 3=not dual
    has_part_d = ADM_PARTD, # Part D prescription coverage
    has_lis_subsidy = ADM_LIS_FLAG_YR, # Low-Income Subsidy (Extra Help)

    # --- Demographics ---
    # DEM_AGE is a continuous age variable in the PUF (not a category code).
    # DEM_RACE uses the MCBS 5-category scheme (see labels in section 4).
    # DEM_INCOME is an ordinal 14-point scale mapped to groups in section 4.
    # DEM_CBSA: 1 = metropolitan area, 2 = non-metropolitan.
    age = DEM_AGE,
    sex = DEM_SEX, # 1=Male, 2=Female
    race = DEM_RACE, # 1-5 coded (labeled in section 4)
    education = DEM_EDU, # highest level completed
    income = DEM_INCOME, # 14-point ordinal scale
    metro_status = DEM_CBSA, # metro vs non-metro
    marital_status = DEM_MARSTA, # 1=married, 2=widowed, etc.
    poverty_ratio = DEM_IPR_IND, # income-to-poverty ratio indicator

    # --- Hypertension: core diagnosis flag ---
    # HLT_OCHBP = 1 is the KEY outcome variable for all hypertension analysis.
    # It is self-reported: "Has a doctor ever told you that you have
    # hypertension or high blood pressure?" (MCBS community questionnaire).
    has_hypertension = HLT_OCHBP, # 1=Yes, 2=No

    # --- Hypertension: clinical management detail ---
    # These questions are asked only of beneficiaries with HLT_OCHBP == 1.
    # They capture how well hypertension is being managed and whether cost
    # is a barrier — both directly relevant to the savings model.
    bp_told_by_doctor = HLT_HYPETOLD, # doctor ever gave the diagnosis
    bp_takes_medication = HLT_HYPEMEDS, # currently on antihypertensives
    bp_monitors_at_home = HLT_HYPEHOME, # self-monitors blood pressure
    bp_self_ctrl_reported = HLT_HYPECTRL, # patient says BP is controlled
    bp_skipped_meds_cost = HLT_HYPESKIP, # skipped BP meds due to cost
    bp_trouble_paying = HLT_HYPEPAY, # trouble affording BP care
    bp_told_serious = HLT_HYPECOND, # told condition is serious
    bp_years_diagnosed = HLT_HYPEYRS, # years since initial diagnosis
    bp_years_on_meds = HLT_HYPELONG, # years taking BP medication

    # --- Cardiovascular comorbidities ---
    # Each is a binary 1/2 flag. These conditions are both caused by and
    # worsen hypertension, and are the primary drivers of downstream spending.
    # Used to construct the cv_event composite flag in section 4.
    had_heart_attack = HLT_OCMYOCAR, # myocardial infarction
    has_coronary_hd = HLT_OCCHD, # coronary heart disease
    has_heart_failure = HLT_OCCFAIL, # congestive heart failure
    has_other_heart_cond = HLT_OCHRTCND, # other specified heart condition
    had_stroke = HLT_OCSTROKE, # stroke or TIA
    has_artery_disease = HLT_OCARTERY, # peripheral artery disease

    # --- Metabolic comorbidities ---
    # Diabetes and high cholesterol frequently co-occur with hypertension
    # and confound the spending premium estimate. Included for adjustment.
    has_diabetes = HLT_OCBETES, # type 1, 2, or gestational
    has_high_cholesterol = HLT_OCCHOLES, # hyperlipidemia

    # --- Other conditions relevant to spending ---
    # Included to support multivariate adjustment in spending models.
    # Each is a binary 1/2 diagnosis flag.
    has_cancer = HLT_OCCANCER, # non-skin cancer
    has_alzheimer_dem = HLT_ALZDEM, # Alzheimer's disease or dementia
    has_depression = HLT_OCDEPRSS, # clinical depression
    has_kidney_disease = HLT_OCKIDNY, # chronic kidney disease
    has_copd = HLT_OCEMPHYS, # COPD / emphysema
    has_rheum_arthritis = HLT_OCARTHRH, # rheumatoid arthritis
    has_osteoarthritis = HLT_OCOSARTH, # osteoarthritis
    has_osteoporosis = HLT_OCOSTEOP, # osteoporosis

    # --- General health and functional status ---
    # HLT_GENHELTH is the standard 5-point self-rated health scale used
    # across MCBS, NHIS, and MEPS — allows cross-survey comparisons.
    # Functional limitation variables capture ADL/IADL impairment.
    self_rated_health = HLT_GENHELTH, # 1=Excellent to 5=Poor
    bmi_category = HLT_BMI_CAT, # 1=Underweight to 4=Obese
    has_functional_limit = HLT_FUNC_LIM, # any ADL/IADL limitation
    difficulty_decisions = HLT_DISDECSN, # cognitive/decision difficulty
    difficulty_walking = HLT_DISWALK, # mobility limitation
    difficulty_bathing = HLT_DISBATH, # self-care limitation

    # --- Preventive care ---
    # BP screening rate is a direct process measure for hypertension management.
    had_bp_screening = PRV_BPTAKEN, # blood pressure taken in past year
    had_blood_test = PRV_BLOODTST, # general blood test in past year

    # --- Access to care ---
    # These access variables are used to contextualise spending differences
    # and identify whether cost barriers affect care-seeking behaviour.
    trouble_getting_care = ACC_HCTROUBL, # any difficulty obtaining care
    delayed_care_cost = ACC_HCDELAY, # delayed/foregone care due to cost
    problems_paying_bills = ACC_PAYPROB, # difficulty with medical bills
    paying_bills_overtime = ACC_PAYOVRTM # paying bills in installments
  )

# %% Cleaning cost data
cost <- cs |>
  select(
    respondent_id = PUF_ID,
    weight_cost = CSPUFWGT, # Cost-specific analytic weight
    matches("^CSPUF[0-9]{3}$"), # 100 Cost BRR replicate weights (CSPUF001–CSPUF100)

    # Quality-control demographics — should match Fall after merge.
    # Discrepancies would indicate a join error.
    cs_age = CSP_AGE,
    cs_sex = CSP_SEX,
    cs_race = CSP_RACE,
    cs_income = CSP_INCOME,
    num_chronic_conditions = CSP_NCHRNCND, # count of chronic conditions

    # Total spending by payer — key outcome variables.
    # PAMTTOT = sum of all payer columns; use this for total spending premium.
    # Medicare FFS savings = reduction in PAMTCARE across beneficiary groups.
    spend_total = PAMTTOT, # all payers, all services
    spend_medicare_paid = PAMTCARE, # what Medicare FFS paid
    spend_medicaid_paid = PAMTCAID, # what Medicaid paid
    spend_ma_plan_paid = PAMTMADV, # what MA plan paid
    spend_out_of_pocket = PAMTOOP, # beneficiary out-of-pocket
    spend_private_insurance = PAMTALPR, # private insurance paid
    spend_discounts_writeoffs = PAMTDISC, # provider discounts
    spend_other_payers = PAMTOTH, # all other payers

    # Spending by service type — used to identify which services drive
    # the spending premium (e.g., inpatient hospital is typically the
    # largest single driver for hypertension-related spending).
    spend_inpatient_hosp = PAMTIP, # inpatient hospital stays
    spend_outpatient_hosp = PAMTOP, # outpatient hospital visits
    spend_physician_services = PAMTMP, # physician/supplier services
    spend_prescription_drugs = PAMTPM, # retail and mail Rx
    spend_dental = PAMTDU, # dental care
    spend_vision = PAMTVU, # vision services
    spend_hearing = PAMTHU, # hearing services
    spend_home_health = PAMTHH, # home health agency

    # Utilization event counts — supplement spending data.
    # High inpatient counts in hypertensive beneficiaries indicate
    # complications (stroke, MI, hypertensive crisis) driving spending.
    total_healthcare_events = PEVENTS, # all service events combined
    num_inpatient_stays = IPAEVNTS, # hospital admissions
    num_outpatient_visits = OPAEVNTS, # outpatient hospital visits
    num_physician_visits = MPAEVNTS, # physician office visits
    num_rx_fills = PMAEVNTS, # prescription fills
    num_dental_events = DUAEVNTS, # dental visits/procedures
    num_vision_events = VUAEVNTS, # vision appointments
    num_hearing_events = HUAEVNTS, # hearing appointments
    num_home_health_events = HHAEVNTS # home health visits
  )

# %% Recode fall: add binary hypertension flag
fall_c <- fall |>
  mutate(hypertension = if_else(has_hypertension == 1, 1L, 0L))

# %% Survey designs
# Fall: full sample; FFS sub-population extracted via subset() to preserve design
design_fall <- svrepdesign(
  data = fall_c,
  weights = ~weight_fall,
  repweights = "PUFF[0-9]{3}",
  type = "BRR",
  combined.weights = TRUE
)
design_fall_ffs <- subset(design_fall, in_ffs == 1)

# Cost supplement: FFS-only by construction
design_cost <- svrepdesign(
  data = cost,
  weights = ~weight_cost,
  repweights = "CSPUF[0-9]{3}",
  type = "BRR",
  combined.weights = TRUE
)

# %% Step 1 — Fall: hypertension prevalence by demographic cell
# DEM_AGE, DEM_SEX, DEM_RACE are all categorical in the PUF (codes 1–3, 1–2, 1–4)
# and use the same coding as CSP_AGE, CSP_SEX, CSP_RACE in the cost supplement.
# Cell structure: age (3) × sex (2) × race (4) = 24 cells
cells_fall <- svyby(
  ~hypertension,
  ~ age + sex + race,
  design_fall_ffs,
  svymean,
  na.rm = TRUE
) |>
  as_tibble() |>
  rename(hyp_prev = hypertension, se_hyp = se)

# %% Step 2 — Cost: mean spending by demographic cell
cells_cost <- svyby(
  ~ spend_total + spend_medicare_paid + spend_out_of_pocket,
  ~ cs_age + cs_sex + cs_race,
  design_cost,
  svymean,
  na.rm = TRUE
) |>
  as_tibble() |>
  rename(
    age = cs_age,
    sex = cs_sex,
    race = cs_race,
    se_spend_total = se1,
    se_spend_medicare = se2,
    se_spend_oop = se3
  )

# %% Step 3 — Join cells on shared demographic codes
cells <- inner_join(cells_fall, cells_cost, by = c("age", "sex", "race")) |>
  filter(!is.na(hyp_prev), !is.na(spend_total))

cat(sprintf(
  "Cells available for ecological regression: %d / 24\n",
  nrow(cells)
))

# %% Step 4 — Ecological weighted regression
# hyp_prev is on [0, 1]; coefficient = spending gap between 0% and 100% prevalence.
# Weight each cell by inverse variance of the prevalence estimate so noisier cells
# contribute less.
eco_total <- lm(spend_total ~ hyp_prev, data = cells, weights = 1 / se_hyp^2)
eco_medicare <- lm(
  spend_medicare_paid ~ hyp_prev,
  data = cells,
  weights = 1 / se_hyp^2
)
eco_oop <- lm(
  spend_out_of_pocket ~ hyp_prev,
  data = cells,
  weights = 1 / se_hyp^2
)

premium_total <- coef(eco_total)["hyp_prev"]
premium_medicare <- coef(eco_medicare)["hyp_prev"]
premium_oop <- coef(eco_oop)["hyp_prev"]

cat(
  "\n--- Spending premium per person (full hypertension → no hypertension) ---\n"
)
cat(sprintf("  Total spending:    %s\n", dollar(premium_total, accuracy = 1)))
cat(sprintf(
  "  Medicare paid:     %s\n",
  dollar(premium_medicare, accuracy = 1)
))
cat(sprintf("  Out-of-pocket:     %s\n", dollar(premium_oop, accuracy = 1)))

# %% Step 5 — Baseline: overall hypertension prevalence and FFS population size
overall_prev <- svymean(~hypertension, design_fall_ffs, na.rm = TRUE)
hyp_prev_est <- coef(overall_prev)[["hypertension"]]

n_ffs_total <- coef(svytotal(~ I(in_ffs == 1), design_fall, na.rm = TRUE))[[1]]
n_hyp <- n_ffs_total * hyp_prev_est

cat(sprintf(
  "\nBaseline: %.1f%% of FFS Medicare beneficiaries have hypertension (~%s people)\n",
  hyp_prev_est * 100,
  format(round(n_hyp, -3), big.mark = ",")
))

# %% Step 6 — Scenario table
# "If hypertension prevalence falls by X%, per-person spending falls by
#  X% × premium_total, and total FFS savings = (X% × n_hyp) × premium_total"
scenarios <- tibble(reduction_pct = c(5, 10, 15, 20, 25, 30, 50)) |>
  mutate(
    label = paste0(reduction_pct, "%"),
    persons_no_longer_hyp = round(n_hyp * reduction_pct / 100, -3),
    savings_per_person_total = premium_total * reduction_pct / 100,
    savings_per_person_medicare = premium_medicare * reduction_pct / 100,
    savings_per_person_oop = premium_oop * reduction_pct / 100,
    total_savings_billions = persons_no_longer_hyp * premium_total / 1e9
  )

scenarios |>
  transmute(
    `Reduction` = label,
    `People affected` = format(persons_no_longer_hyp, big.mark = ","),
    `Savings/person (total)` = dollar(savings_per_person_total, accuracy = 1),
    `Savings/person (Medicare)` = dollar(
      savings_per_person_medicare,
      accuracy = 1
    ),
    `Savings/person (OOP)` = dollar(savings_per_person_oop, accuracy = 1),
    `Total FFS savings` = paste0("$", round(total_savings_billions, 1), "B")
  ) |>
  print(n = Inf)

# %% Visualization
ggplot(scenarios, aes(x = reduction_pct, y = total_savings_billions)) +
  geom_col(fill = "#2166AC", width = 0.7) +
  geom_text(
    aes(label = paste0("$", round(total_savings_billions, 1), "B")),
    vjust = 1.5,
    size = 3.8,
    colour = "grey20"
  ) +
  scale_x_continuous(
    breaks = scenarios$reduction_pct,
    labels = paste0(scenarios$reduction_pct, "%")
  ) +
  scale_y_continuous(
    labels = function(x) paste0("$", x, "B"),
    expand = expansion(mult = c(0.12, 0))
  ) +
  labs(
    title = "Projected Annual Medicare FFS Savings from Reducing Hypertension",
    subtitle = paste0(
      "Ecological model: MCBS 2023 Fall (hypertension prevalence) linked to Cost Supplement (spending)\n",
      "by demographic cell (age × sex × race) · survey-weighted · FFS beneficiaries only"
    ),
    x = "Reduction in hypertension prevalence",
    y = "Annual savings (billions USD)"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey50"))

# %% Severity-based savings: merge Fall + Cost, hypertensives only
# The two files use different PUF_ID schemes: Fall = 23XXXXXX, Cost = 923XXXXX.
# The shared key is the sequential beneficiary number: fall_id - 23000000 == cost_id - 92300000.
# bp_self_ctrl_reported (HLT_HYPECTRL): 1 = BP controlled, 2 = BP not controlled.
# We use this as the severity proxy: uncontrolled BP = higher severity.
# "X% severity reduction" = moving X% of the way from uncontrolled to controlled BP.
merged <- fall_c |>
  filter(in_ffs == 1, hypertension == 1) |>
  select(respondent_id, age, sex, race, bp_self_ctrl_reported) |>
  mutate(
    .key         = respondent_id - 23000000,
    bp_uncontrolled = if_else(bp_self_ctrl_reported == 2, 1L, 0L)
  ) |>
  inner_join(
    cost |> mutate(.key = respondent_id - 92300000),
    by = ".key"
  ) |>
  select(-.key, -respondent_id.y) |>
  rename(respondent_id = respondent_id.x)

cat(sprintf("\nHypertensives matched to cost file: %d records\n", nrow(merged)))
cat(sprintf(
  "BP uncontrolled: %d (%.1f%%) | controlled: %d (%.1f%%)\n",
  sum(merged$bp_uncontrolled, na.rm = TRUE),
  100 * mean(merged$bp_uncontrolled, na.rm = TRUE),
  sum(merged$bp_uncontrolled == 0, na.rm = TRUE),
  100 * mean(merged$bp_uncontrolled == 0, na.rm = TRUE)
))

design_sev <- svrepdesign(
  data    = merged |> filter(!is.na(bp_uncontrolled)),
  weights = ~weight_cost,
  repweights = "CSPUF[0-9]{3}",
  type    = "BRR",
  combined.weights = TRUE
)

# Spending gap: uncontrolled vs controlled BP, adjusted for age/sex/race.
# Coefficient = extra annual spend for a person with uncontrolled BP.
# "Reducing severity by X%" scales this gap linearly: savings = X/100 * gap.
fit_sev_total    <- svyglm(spend_total         ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)
fit_sev_medicare <- svyglm(spend_medicare_paid ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)
fit_sev_oop      <- svyglm(spend_out_of_pocket ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)

sev_gap_total    <- coef(fit_sev_total)[["bp_uncontrolled"]]
sev_gap_medicare <- coef(fit_sev_medicare)[["bp_uncontrolled"]]
sev_gap_oop      <- coef(fit_sev_oop)[["bp_uncontrolled"]]

cat("\n--- Annual spending gap: uncontrolled vs controlled BP (adjusted) ---\n")
cat(sprintf("  Total spending:    %s\n", dollar(sev_gap_total,    accuracy = 1)))
cat(sprintf("  Medicare paid:     %s\n", dollar(sev_gap_medicare, accuracy = 1)))
cat(sprintf("  Out-of-pocket:     %s\n", dollar(sev_gap_oop,      accuracy = 1)))

# %% savings_for_severity_reduction(): individual-level savings model
# If Joe reduces his BP severity by X%, he saves X/100 * gap dollars per year.
savings_for_severity_reduction <- function(reduction_pct) {
  tibble(
    reduction_pct    = reduction_pct,
    label            = paste0(reduction_pct, "%"),
    savings_total    = sev_gap_total    * reduction_pct / 100,
    savings_medicare = sev_gap_medicare * reduction_pct / 100,
    savings_oop      = sev_gap_oop      * reduction_pct / 100
  )
}

sev_scenarios <- map(c(10, 20, 30, 40, 50, 75, 100), savings_for_severity_reduction) |>
  list_rbind()

cat("\n--- Individual severity reduction scenario table ---\n")
sev_scenarios |>
  transmute(
    `Severity reduction`        = label,
    `Savings/year (total)`      = dollar(savings_total,    accuracy = 1),
    `Savings/year (Medicare)`   = dollar(savings_medicare, accuracy = 1),
    `Savings/year (OOP)`        = dollar(savings_oop,      accuracy = 1)
  ) |>
  print(n = Inf)
