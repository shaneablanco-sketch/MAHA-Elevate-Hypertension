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
    starts_with("PUFF0"), # 100 Fall BRR replicate weights

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
    starts_with("CSPUF0"), # 100 Cost BRR replicate weights

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

# %% join Fall and Cost
fall <- fall |>
  left_join(cost, by = "respondent_id")
