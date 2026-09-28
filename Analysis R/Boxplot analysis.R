# =============================================================================
# Boxplot LLM Performance on SSN Exams — by Year (2020-2024)
# CORREZIONE: asterischi basati su p-value Holm-adjusted (per anno, 18 test)
# =============================================================================

library(ggplot2)
library(dplyr)
library(ggtext)
library(patchwork)
library(scales)
library(readxl)

# --- Load data ---------------------------------------------------------------
setwd("C:\\Users\\edoar\\Desktop\\LLM and SSM test\\Risultati\\Analysis R")

df <- read.csv("db_LLM_SSN_accuracy_long_v2.csv", sep = ";", stringsAsFactors = FALSE)

# --- Add score column --------------------------------------------------------
df$score <- df$correct - (140 - df$correct) * 0.25

# --- Load real candidates scores ---------------------------------------------
real_stats_raw <- data.frame(
  year   = c(2020,  2021,  2022,  2023,  2024),
  n      = c(20129, 19442, 14540, 14036, 13060),
  mean   = c(76.69, 78.46, 82.57, 83.10, 81.29),
  sd     = c(20.09, 19.37, 16.34, 19.60, 15.56),
  min    = c(16,    -2,    50,    0,     50),
  q1     = c(62.25, 65.25, 70.25, 70.50, 69.75),
  median = c(77.25, 78.50, 82.50, 84.50, 81.25),
  q3     = c(91.50, 92.50, 94.25, 97.25, 92.50),
  max    = c(133.25,133.25,140,  132.50,136)
)

set.seed(42)
df_real <- do.call(rbind, lapply(seq_len(nrow(real_stats_raw)), function(i) {
  r <- real_stats_raw[i, ]
  raw <- rnorm(r$n * 5, mean = r$mean, sd = r$sd)
  raw <- raw[raw >= r$min & raw <= r$max]
  samp <- raw[seq_len(r$n)]
  med_samp <- median(samp, na.rm = TRUE)
  iqr_samp <- IQR(samp, na.rm = TRUE)
  iqr_obs  <- r$q3 - r$q1
  if (iqr_samp > 0 && iqr_obs > 0) {
    samp <- (samp - med_samp) * (iqr_obs / iqr_samp) + r$median
  }
  samp <- pmax(pmin(samp, r$max), r$min)
  data.frame(year = r$year, score = samp)
}))
df_real$year <- as.integer(df_real$year)

# --- Compute median and IQR per year for reference line and shading ----------
real_stats <- df_real %>%
  group_by(year) %>%
  summarise(median_score = median(score, na.rm = TRUE),
            q1_score     = quantile(score, 0.25, na.rm = TRUE),
            q3_score     = quantile(score, 0.75, na.rm = TRUE))

# --- Model order and category ------------------------------------------------
model_levels <- c("CLA", "DPSK", "GPT", "GROK", "MSTRL", "QWEN",
                  "MG4B", "MT8B", "mSTRL3B", "Q1_7B", "Q4B", "Q8B",
                  "MG4Bq4", "MG4Bq6", "MG4Bq8", "MT8Bq4", "MT8Bq6", "MT8Bq8")

cat_map <- c(
  CLA = "Closed", DPSK = "Closed", GPT = "Closed",
  GROK = "Closed", MSTRL = "Closed", QWEN = "Closed",
  MG4B = "Open", MT8B = "Open", mSTRL3B = "Open",
  Q1_7B = "Open", Q4B = "Open", Q8B = "Open",
  MG4Bq4 = "Quantized", MG4Bq6 = "Quantized", MG4Bq8 = "Quantized",
  MT8Bq4 = "Quantized", MT8Bq6 = "Quantized", MT8Bq8 = "Quantized"
)

df$model    <- factor(df$model, levels = model_levels)
df$category <- factor(cat_map[as.character(df$model)],
                      levels = c("Closed", "Open", "Quantized"))

# --- Colors ------------------------------------------------------------------
cat_colors <- c(Closed = "#2E86AB", Open = "#A23B72", Quantized = "#F18F01")

# --- Base theme --------------------------------------------------------------
theme_base <- theme_bw() +
  theme(
    panel.border       = element_blank(),
    axis.line.x        = element_line(color = "black", linewidth = 0.4),
    axis.line.y        = element_line(color = "black", linewidth = 0.4),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey85", linetype = "dashed", linewidth = 0.2),
    axis.text.x        = element_text(angle = 35, hjust = 1, size = 11),
    axis.text.y        = element_text(size = 11),
    axis.title         = element_text(size = 13, face = "bold"),
    plot.title         = element_markdown(size = 14, hjust = 0.5)
  )

# =============================================================================
# Pre-compute Holm-adjusted p-values for score comparisons (per anno, 18 test)
# Restituisce un named list: holm_pvals[[yr]][[model]] = p_adj
# =============================================================================
compute_holm_pvals <- function(df, df_real, years, model_levels) {
  holm_pvals <- list()
  for (yr in years) {
    real_sc <- df_real$score[df_real$year == yr]
    raw_p   <- setNames(numeric(length(model_levels)), model_levels)
    for (mdl in model_levels) {
      llm_sc      <- df$score[df$model == mdl & df$year == yr]
      if (length(llm_sc) == 0) { raw_p[mdl] <- NA; next }
      raw_p[mdl]  <- wilcox.test(llm_sc, real_sc, exact = FALSE)$p.value
    }
    adj_p <- p.adjust(raw_p, method = "holm")
    holm_pvals[[as.character(yr)]] <- as.list(adj_p)
  }
  return(holm_pvals)
}

# =============================================================================
# Helper: build one LLM panel — asterischi da p-value Holm-adjusted
# =============================================================================
make_panel <- function(df_sub, y_col, cat_name, year_val,
                       col_pos = "middle", row_pos = "middle",
                       hline_val = NULL, iqr_low = NULL, iqr_high = NULL,
                       holm_pvals_yr = NULL, star_y = NULL) {
  
  if (row_pos == "top") {
    panel_title <- paste0("<span style='color:", cat_colors[cat_name], ";'>**",
                          cat_name, "**</span>")
  } else {
    panel_title <- NULL
  }
  
  p <- ggplot(df_sub, aes(x = model, y = .data[[y_col]]))
  
  if (!is.null(iqr_low) && !is.null(iqr_high)) {
    p <- p +
      annotate("rect", xmin = -Inf, xmax = Inf,
               ymin = iqr_low, ymax = iqr_high,
               fill = "#E60000", alpha = 0.08)
  }
  
  p <- p +
    geom_boxplot(fill = cat_colors[cat_name], alpha = 0.75,
                 outlier.size = 1, outlier.alpha = 0.4,
                 color = "#333333", width = 0.8, linewidth = 0.2) +
    labs(title = panel_title, x = NULL,
         y = if (col_pos == "left") as.character(year_val) else NULL) +
    theme_base
  
  if (!is.null(hline_val)) {
    p <- p + geom_hline(yintercept = hline_val, color = "#E60000",
                        linetype = "dashed", linewidth = 0.5)
  }
  
  # Asterischi basati su p-value Holm-adjusted pre-calcolati
  if (!is.null(holm_pvals_yr) && !is.null(star_y)) {
    models_here <- levels(droplevels(df_sub$model))
    for (m_idx in seq_along(models_here)) {
      m_name <- models_here[m_idx]
      p_adj  <- holm_pvals_yr[[m_name]]
      if (is.null(p_adj) || is.na(p_adj)) next
      if (p_adj < 0.05) {
        star_label <- ifelse(p_adj < 0.001, "***",
                             ifelse(p_adj < 0.01,  "**", "*"))
        p <- p + annotate("text", x = m_idx, y = star_y,
                          label = star_label, size = 3, color = "grey30")
      }
    }
  }
  
  if (row_pos != "bottom") {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  }
  
  mt <- ifelse(row_pos == "top", 0, 0)
  mb <- ifelse(row_pos == "bottom", 0, 0)
  ml <- ifelse(col_pos == "left", 5, 15)
  mr <- 15
  p <- p + theme(plot.margin = margin(mt, mr, mb, ml, unit = "pt"))
  
  return(p)
}

# =============================================================================
# Helper: build one real candidates panel
# =============================================================================
make_real_panel <- function(df_real_sub, year_val, row_pos = "middle") {
  
  if (row_pos == "top") {
    panel_title <- "<span style='color:#E60000;'>**Candidates**</span>"
  } else {
    panel_title <- NULL
  }
  
  p <- ggplot(df_real_sub, aes(x = "SSM", y = score)) +
    geom_boxplot(fill = "#E60000", alpha = 0.75,
                 outlier.size = 0.3, outlier.alpha = 0.15,
                 color = "#333333", width = 0.4, linewidth = 0.2) +
    labs(title = panel_title, x = NULL, y = NULL) +
    theme_base
  
  if (row_pos != "bottom") {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  }
  
  p <- p + theme(plot.margin = margin(0, 5, 0, 15, unit = "pt"))
  
  return(p)
}

# =============================================================================
# Build grids
# =============================================================================
years <- c(2020, 2021, 2022, 2023, 2024)
categories <- c("Closed", "Open", "Quantized")
col_positions <- c("left", "middle", "right")

# --- Accuracy grid (invariata — nessun asterisco) ----------------------------
build_accuracy_grid <- function() {
  
  scale_acc <- scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                                  labels = percent_format(accuracy = 1))
  panels <- list()
  
  for (i in seq_along(years)) {
    yr <- years[i]
    row_pos <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
    
    for (j in seq_along(categories)) {
      cat_name <- categories[j]
      col_pos  <- col_positions[j]
      df_sub   <- df %>% filter(category == cat_name, year == yr)
      
      p <- make_panel(df_sub, "accuracy", cat_name, yr, col_pos, row_pos) + scale_acc
      
      if (j < length(categories)) {
        panels <- c(panels, list(p), list(plot_spacer()))
      } else {
        panels <- c(panels, list(p))
      }
    }
  }
  
  p_grid <- wrap_plots(panels, ncol = 5, byrow = TRUE,
                       widths = c(1, 0.01, 1, 0.01, 1)) +
    plot_annotation(
      title = element_blank(),
      theme = theme(
        plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
        panel.spacing = unit(2, "pt")
      )
    )
  
  ggsave("boxplot_accuracy_by_year.png", plot = p_grid,
         width = 9, height = 10.5, units = "in", dpi = 600, bg = "white")
  cat("Saved: boxplot_accuracy_by_year.png\n")
}

# --- Score grid (con asterischi Holm-corretti) --------------------------------
build_score_grid <- function() {
  
  # Pre-calcola tutti i p-value Holm-adjusted UNA VOLTA sola
  holm_pvals <- compute_holm_pvals(df, df_real, years, model_levels)
  
  scale_sc <- scale_y_continuous(limits = c(0, 150), breaks = seq(0, 140, 20))
  panels <- list()
  
  for (i in seq_along(years)) {
    yr <- years[i]
    row_pos  <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
    yr_stats <- real_stats %>% filter(year == yr)
    yr_holm  <- holm_pvals[[as.character(yr)]]
    
    for (j in seq_along(categories)) {
      cat_name <- categories[j]
      col_pos  <- col_positions[j]
      df_sub   <- df %>% filter(category == cat_name, year == yr)
      
      p <- make_panel(df_sub, "score", cat_name, yr, col_pos, row_pos,
                      hline_val     = yr_stats$median_score,
                      iqr_low       = yr_stats$q1_score,
                      iqr_high      = yr_stats$q3_score,
                      holm_pvals_yr = yr_holm,
                      star_y        = 148) + scale_sc
      
      panels <- c(panels, list(p), list(plot_spacer()))
    }
    
    df_real_sub <- df_real %>% filter(year == yr)
    p_real <- make_real_panel(df_real_sub, yr, row_pos) + scale_sc
    panels <- c(panels, list(p_real))
  }
  
  p_grid <- wrap_plots(panels, ncol = 7, byrow = TRUE,
                       widths = c(1, 0.01, 1, 0.01, 1, 0.01, 0.6)) +
    plot_annotation(
      title = element_blank(),
      theme = theme(
        plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
        panel.spacing = unit(2, "pt")
      )
    )
  
  ggsave("boxplot_score_by_year.png", plot = p_grid,
         width = 9, height = 10, units = "in", dpi = 600, bg = "white")
  cat("Saved: boxplot_score_by_year.png\n")
}

# =============================================================================
# Generate both plots
# =============================================================================
build_accuracy_grid()
build_score_grid()

cat("Done!\n")