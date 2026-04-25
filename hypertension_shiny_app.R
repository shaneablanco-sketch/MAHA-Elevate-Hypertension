library(shiny)
library(bslib)
library(tidyverse)
library(scales)
library(survey)

# ── One-time data load & model fit ────────────────────────────────────────────
.na_codes <- c("", "NA", ".", " ", "D", "R", "N")

sf_fall <- read_csv("data/sfpuf2023_1_fall.csv", show_col_types = FALSE, na = .na_codes)
cs      <- read_csv("data/cspuf2023.csv",         show_col_types = FALSE, na = .na_codes)

fall <- sf_fall |>
  select(
    respondent_id = PUF_ID, weight_fall = PUFFWGT,
    matches("^PUFF[0-9]{3}$"),
    in_ffs = ADM_FFS_FLAG_YR,
    age = DEM_AGE, sex = DEM_SEX, race = DEM_RACE,
    has_hypertension     = HLT_OCHBP,
    bp_self_ctrl_reported = HLT_HYPECTRL
  ) |>
  mutate(hypertension = if_else(has_hypertension == 1, 1L, 0L))

cost <- cs |>
  select(
    respondent_id = PUF_ID, weight_cost = CSPUFWGT,
    matches("^CSPUF[0-9]{3}$"),
    spend_total         = PAMTTOT,
    spend_medicare_paid = PAMTCARE,
    spend_out_of_pocket = PAMTOOP
  )

# Merge: FFS hypertensives only; join on sequential beneficiary ID
merged <- fall |>
  filter(in_ffs == 1, hypertension == 1) |>
  select(respondent_id, age, sex, race, bp_self_ctrl_reported) |>
  mutate(
    .key            = respondent_id - 23000000,
    bp_uncontrolled = if_else(bp_self_ctrl_reported == 2, 1L, 0L)
  ) |>
  inner_join(cost |> mutate(.key = respondent_id - 92300000), by = ".key") |>
  select(-.key, -respondent_id.y) |>
  rename(respondent_id = respondent_id.x) |>
  filter(!is.na(bp_uncontrolled))

design_sev <- svrepdesign(
  data       = merged,
  weights    = ~weight_cost,
  repweights = "CSPUF[0-9]{3}",
  type       = "BRR",
  combined.weights = TRUE
)

# Spending gap: uncontrolled vs controlled BP, adjusted for age/sex/race
fit_total    <- svyglm(spend_total         ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)
fit_medicare <- svyglm(spend_medicare_paid ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)
fit_oop      <- svyglm(spend_out_of_pocket ~ bp_uncontrolled + factor(age) + factor(sex) + factor(race), design_sev)

gap_total    <- coef(fit_total)[["bp_uncontrolled"]]
gap_medicare <- coef(fit_medicare)[["bp_uncontrolled"]]
gap_oop      <- coef(fit_oop)[["bp_uncontrolled"]]

pct_uncontrolled <- round(100 * mean(merged$bp_uncontrolled), 1)

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- page_sidebar(
  title = "Individual Hypertension Savings Model",
  theme = bs_theme(bootswatch = "flatly"),

  sidebar = sidebar(
    width = 320,
    h5("How much does Joe improve his BP?"),
    sliderInput(
      "severity_pct",
      "Reduction in hypertension severity",
      min = 1, max = 100, value = 10, step = 1, post = "%"
    ),
    hr(),
    p(class = "text-muted small",
      "Severity proxy: self-reported BP control status (MCBS 2023).",
      "100% = moving from fully uncontrolled to fully controlled BP.",
      "Spending gap adjusted for age, sex, and race.",
      paste0("Among matched FFS hypertensives: ", pct_uncontrolled, "% have uncontrolled BP.")
    )
  ),

  layout_columns(
    col_widths = c(4, 4, 4),
    value_box(
      title = "Annual savings (total)",
      value = textOutput("savings_total"),
      theme = "primary"
    ),
    value_box(
      title = "Annual savings (Medicare)",
      value = textOutput("savings_medicare"),
      theme = "success"
    ),
    value_box(
      title = "Annual savings (out-of-pocket)",
      value = textOutput("savings_oop"),
      theme = "info"
    )
  ),

  layout_columns(
    col_widths = c(6, 6),
    card(
      card_header("Savings by payer at selected reduction"),
      plotOutput("bar_payer", height = "280px")
    ),
    card(
      card_header("Total savings across all severity reductions"),
      plotOutput("bar_curve", height = "280px")
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  scenario <- reactive({
    raw <- as.numeric(input$severity_pct)
    if (!is.finite(raw)) raw <- 10
    pct <- min(max(raw, 1), 100) / 100   # clamp server-side regardless of client input
    list(
      pct      = raw,
      total    = gap_total    * pct,
      medicare = gap_medicare * pct,
      oop      = gap_oop      * pct
    )
  })

  fmt <- function(x) dollar(x, accuracy = 1)

  output$savings_total    <- renderText(fmt(scenario()$total))
  output$savings_medicare <- renderText(fmt(scenario()$medicare))
  output$savings_oop      <- renderText(fmt(scenario()$oop))

  output$bar_payer <- renderPlot({
    s <- scenario()
    tibble(
      payer   = c("Total", "Medicare", "Out-of-pocket"),
      savings = c(s$total, s$medicare, s$oop)
    ) |>
      mutate(payer = fct_inorder(payer)) |>
      ggplot(aes(x = payer, y = savings, fill = payer)) +
      geom_col(width = 0.55, show.legend = FALSE) +
      geom_text(
        aes(
          label = dollar(savings, accuracy = 1),
          vjust = if_else(savings >= 0, -0.4, 1.4)
        ),
        size = 4.2
      ) +
      scale_fill_manual(values = c("#2166AC", "#4DAC26", "#D7191C")) +
      scale_y_continuous(labels = dollar_format(accuracy = 1)) +
      labs(x = NULL, y = "Annual savings ($/yr)") +
      theme_minimal(base_size = 13)
  })

  output$bar_curve <- renderPlot({
    pcts <- 1:100
    curve_df <- tibble(
      pct     = pcts,
      savings = gap_total * pcts / 100
    )
    ggplot(curve_df, aes(x = pct, y = savings)) +
      geom_line(colour = "#aaaaaa", linewidth = 1) +
      geom_point(
        data = ~ filter(.x, pct == input$severity_pct),
        colour = "#2166AC", size = 4
      ) +
      geom_vline(xintercept = input$severity_pct, colour = "#2166AC", linetype = "dashed", linewidth = 0.7) +
      geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
      annotate(
        "text",
        x = input$severity_pct + 2, y = gap_total * input$severity_pct / 100,
        label = dollar(gap_total * input$severity_pct / 100, accuracy = 1),
        hjust = 0, colour = "#2166AC", size = 4, fontface = "bold"
      ) +
      scale_x_continuous(breaks = seq(0, 100, 10), labels = function(x) paste0(x, "%")) +
      scale_y_continuous(labels = dollar_format(accuracy = 1)) +
      labs(
        x = "Severity reduction",
        y = "Annual savings per person (total)"
      ) +
      theme_minimal(base_size = 13)
  })
}

shinyApp(ui, server)
