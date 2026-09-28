# =============================================================================
# Boxplot LLM Performance on SSN Exams — by Year (2020-2024)
# + Export tabella Mann-Whitney (stessi p-value Holm usati negli asterischi)
# Confronti su dati grezzi reali dei candidati (punteggio prova senza CV)
# =============================================================================

library(ggplot2)
library(dplyr)
library(ggtext)
library(patchwork)
library(scales)
library(openxlsx)
library(tidyr)

# --- Load LLM data -----------------------------------------------------------
setwd("C:\\Users\\edoar\\Desktop\\LLM and SSM test\\Risultati\\Analysis R")

df <- read.csv("db_LLM_SSN_accuracy_long_v2.csv", sep = ";", stringsAsFactors = FALSE)
df$score <- df$correct - (140 - df$correct) * 0.25

# --- Load real candidates raw scores (punteggio prova senza CV) --------------
scores_raw <- read.csv("scores.csv", sep = ";", header = TRUE,
                       stringsAsFactors = FALSE, check.names = FALSE)
colnames(scores_raw) <- c("2020", "2021", "2022", "2023", "2024")

df_real <- pivot_longer(scores_raw, cols = everything(),
                        names_to = "year", values_to = "score") %>%
  mutate(year = as.integer(year)) %>%
  filter(!is.na(score))

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

years <- c(2020, 2021, 2022, 2023, 2024)

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
# Pre-compute Holm-adjusted p-values (per anno, 18 test)
# Unica funzione che alimenta sia i boxplot che la tabella Excel
# =============================================================================
compute_holm_pvals <- function(df, df_real, years, model_levels) {
  
  holm_pvals <- list()
  table_rows <- list()
  
  for (yr in years) {
    real_sc     <- df_real$score[df_real$year == yr]
    real_median <- median(real_sc, na.rm = TRUE)
    raw_p        <- setNames(numeric(length(model_levels)), model_levels)
    W_vals       <- setNames(numeric(length(model_levels)), model_levels)
    n_llm_vals   <- setNames(integer(length(model_levels)), model_levels)
    med_llm_vals <- setNames(numeric(length(model_levels)), model_levels)
    
    for (mdl in model_levels) {
      llm_sc <- df$score[df$model == mdl & df$year == yr]
      if (length(llm_sc) == 0) {
        raw_p[mdl] <- NA; W_vals[mdl] <- NA
        n_llm_vals[mdl] <- 0; med_llm_vals[mdl] <- NA
        next
      }
      wt               <- wilcox.test(llm_sc, real_sc, exact = FALSE)
      raw_p[mdl]       <- wt$p.value
      W_vals[mdl]      <- wt$statistic
      n_llm_vals[mdl]  <- length(llm_sc)
      med_llm_vals[mdl] <- median(llm_sc, na.rm = TRUE)
    }
    
    adj_p <- p.adjust(raw_p, method = "holm")
    holm_pvals[[as.character(yr)]] <- as.list(adj_p)
    
    sig_label <- function(p) {
      if (is.na(p)) return(NA)
      if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else "ns"
    }
    
    for (mdl in model_levels) {
      direction <- ifelse(is.na(med_llm_vals[mdl]), NA,
                          ifelse(med_llm_vals[mdl] > real_median, "LLM > Candidates",
                                 ifelse(med_llm_vals[mdl] < real_median, "LLM < Candidates",
                                        "LLM = Candidates")))
      table_rows[[length(table_rows) + 1]] <- data.frame(
        Year              = yr,
        Category          = cat_map[mdl],
        Model             = mdl,
        N_LLM             = n_llm_vals[mdl],
        N_Candidates      = length(real_sc),
        Median_LLM        = round(med_llm_vals[mdl], 2),
        Median_Candidates = round(real_median, 2),
        Direction         = direction,
        W                 = round(W_vals[mdl], 1),
        p_value           = raw_p[mdl],
        p_adj_Holm        = adj_p[mdl],
        Sig_raw           = sig_label(raw_p[mdl]),
        Sig_adj           = sig_label(adj_p[mdl]),
        stringsAsFactors  = FALSE
      )
    }
  }
  
  list(holm_pvals = holm_pvals, table = do.call(rbind, table_rows))
}

# =============================================================================
# Export Mann-Whitney table to Excel
# =============================================================================
export_mw_table <- function(df_res) {
  
  wb <- createWorkbook()
  
  COL_NAMES <- c("Year", "Category", "Model", "N LLM", "N Candidates",
                 "Median LLM", "Median Candidates", "Direction",
                 "W", "p-value", "p-adj (Holm)", "Sig (raw)", "Sig (adj)")
  
  HDR_FILL <- createStyle(fontName = "Arial", fontSize = 10, fontColour = "#FFFFFF",
                          fgFill = "#2F4F6E", halign = "CENTER", valign = "CENTER",
                          textDecoration = "bold", wrapText = TRUE)
  CAT_ST <- list(
    Closed    = createStyle(fontName = "Arial", fontSize = 9, fgFill = "#D6E8F5"),
    Open      = createStyle(fontName = "Arial", fontSize = 9, fgFill = "#F2DCE8"),
    Quantized = createStyle(fontName = "Arial", fontSize = 9, fgFill = "#FEF0D6")
  )
  STAR_ST <- createStyle(fontName = "Arial", fontSize = 9, fontColour = "#B22222",
                         textDecoration = "bold", halign = "CENTER")
  NS_ST   <- createStyle(fontName = "Arial", fontSize = 9, fontColour = "#888888",
                         halign = "CENTER")
  NUM_ST  <- createStyle(fontName = "Arial", fontSize = 9, numFmt = "0.00",  halign = "RIGHT")
  PVAL_ST <- createStyle(fontName = "Arial", fontSize = 9, numFmt = "0.0000", halign = "RIGHT")
  INT_ST  <- createStyle(fontName = "Arial", fontSize = 9, numFmt = "0",     halign = "RIGHT")
  CTR_ST  <- createStyle(fontName = "Arial", fontSize = 9, halign = "CENTER")
  WIDTHS  <- c(7, 11, 10, 8, 13, 11, 18, 20, 12, 12, 13, 10, 10)
  
  write_sheet <- function(ws_name, data) {
    addWorksheet(wb, ws_name)
    writeData(wb, ws_name, as.data.frame(t(COL_NAMES)), startRow = 1, colNames = FALSE)
    addStyle(wb, ws_name, HDR_FILL, rows = 1, cols = 1:13, gridExpand = TRUE)
    writeData(wb, ws_name, data, startRow = 2, colNames = FALSE)
    nr <- nrow(data)
    for (i in seq_len(nr)) {
      cn <- as.character(data$Category[i])
      addStyle(wb, ws_name, CAT_ST[[cn]],
               rows = i + 1, cols = 1:13, gridExpand = TRUE, stack = TRUE)
    }
    addStyle(wb, ws_name, INT_ST,  rows = 2:(nr+1), cols = c(4,5),      gridExpand = TRUE, stack = TRUE)
    addStyle(wb, ws_name, NUM_ST,  rows = 2:(nr+1), cols = c(6,7,9),    gridExpand = TRUE, stack = TRUE)
    addStyle(wb, ws_name, PVAL_ST, rows = 2:(nr+1), cols = c(10,11),    gridExpand = TRUE, stack = TRUE)
    addStyle(wb, ws_name, CTR_ST,  rows = 2:(nr+1), cols = c(1,3,12,13), gridExpand = TRUE, stack = TRUE)
    for (i in seq_len(nr)) {
      for (ci in c(12, 13)) {
        v   <- if (ci == 12) data$Sig_raw[i] else data$Sig_adj[i]
        sty <- if (!is.na(v) && v != "ns") STAR_ST else NS_ST
        addStyle(wb, ws_name, sty, rows = i + 1, cols = ci, stack = TRUE)
      }
    }
    setColWidths(wb, ws_name, cols = 1:13, widths = WIDTHS)
    freezePane(wb, ws_name, firstRow = TRUE)
  }
  
  write_sheet("All Years", df_res)
  for (yr in years) {
    write_sheet(paste0("SSM ", yr), df_res %>% filter(Year == yr))
  }
  
  saveWorkbook(wb, "mann_whitney_LLM_vs_candidates.xlsx", overwrite = TRUE)
  cat("Saved: mann_whitney_LLM_vs_candidates.xlsx\n")
}

# =============================================================================
# Helpers: make_panel / make_real_panel
# =============================================================================
make_panel <- function(df_sub, y_col, cat_name, year_val,
                       col_pos = "middle", row_pos = "middle",
                       hline_val = NULL, iqr_low = NULL, iqr_high = NULL,
                       holm_pvals_yr = NULL, star_y = NULL) {
  
  panel_title <- if (row_pos == "top")
    paste0("<span style='color:", cat_colors[cat_name], ";'>**", cat_name, "**</span>")
  else NULL
  
  p <- ggplot(df_sub, aes(x = model, y = .data[[y_col]]))
  
  if (!is.null(iqr_low) && !is.null(iqr_high))
    p <- p + annotate("rect", xmin = -Inf, xmax = Inf,
                      ymin = iqr_low, ymax = iqr_high, fill = "#E60000", alpha = 0.08)
  
  p <- p +
    geom_boxplot(fill = cat_colors[cat_name], alpha = 0.75,
                 outlier.size = 1, outlier.alpha = 0.4,
                 color = "#333333", width = 0.8, linewidth = 0.2) +
    labs(title = panel_title, x = NULL,
         y = if (col_pos == "left") as.character(year_val) else NULL) +
    theme_base
  
  if (!is.null(hline_val))
    p <- p + geom_hline(yintercept = hline_val, color = "#E60000",
                        linetype = "dashed", linewidth = 0.5)
  
  if (!is.null(holm_pvals_yr) && !is.null(star_y)) {
    models_here <- levels(droplevels(df_sub$model))
    for (m_idx in seq_along(models_here)) {
      m_name <- models_here[m_idx]
      p_adj  <- holm_pvals_yr[[m_name]]
      if (is.null(p_adj) || is.na(p_adj) || p_adj >= 0.05) next
      star_label <- ifelse(p_adj < 0.001, "***", ifelse(p_adj < 0.01, "**", "*"))
      p <- p + annotate("text", x = m_idx, y = star_y,
                        label = star_label, size = 3, color = "grey30")
    }
  }
  
  if (row_pos != "bottom")
    p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  
  p + theme(plot.margin = margin(0, 15, 0, ifelse(col_pos == "left", 5, 15), unit = "pt"))
}

make_real_panel <- function(df_real_sub, year_val, row_pos = "middle") {
  panel_title <- if (row_pos == "top")
    "<span style='color:#E60000;'>**Candidates**</span>" else NULL
  
  p <- ggplot(df_real_sub, aes(x = "SSM", y = score)) +
    geom_boxplot(fill = "#E60000", alpha = 0.75,
                 outlier.size = 0.3, outlier.alpha = 0.15,
                 color = "#333333", width = 0.4, linewidth = 0.2) +
    labs(title = panel_title, x = NULL, y = NULL) +
    theme_base
  
  if (row_pos != "bottom")
    p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  
  p + theme(plot.margin = margin(0, 5, 0, 15, unit = "pt"))
}

# =============================================================================
# Build grids
# =============================================================================
categories    <- c("Closed", "Open", "Quantized")
col_positions <- c("left", "middle", "right")

build_accuracy_grid <- function() {
  scale_acc <- scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                                  labels = percent_format(accuracy = 1))
  panels <- list()
  for (i in seq_along(years)) {
    yr      <- years[i]
    row_pos <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
    for (j in seq_along(categories)) {
      cat_name <- categories[j]; col_pos <- col_positions[j]
      df_sub   <- df %>% filter(category == cat_name, year == yr)
      p <- make_panel(df_sub, "accuracy", cat_name, yr, col_pos, row_pos) + scale_acc
      panels <- c(panels, list(p))
      if (j < length(categories)) panels <- c(panels, list(plot_spacer()))
    }
  }
  p_grid <- wrap_plots(panels, ncol = 5, byrow = TRUE, widths = c(1, 0.01, 1, 0.01, 1)) +
    plot_annotation(theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                                  panel.spacing = unit(2, "pt")))
  ggsave("boxplot_accuracy_by_year.png", plot = p_grid,
         width = 9, height = 10.5, units = "in", dpi = 600, bg = "white")
  cat("Saved: boxplot_accuracy_by_year.png\n")
}

build_score_grid <- function(holm_result) {
  holm_pvals <- holm_result$holm_pvals
  scale_sc   <- scale_y_continuous(limits = c(0, 150), breaks = seq(0, 140, 20))
  panels     <- list()
  for (i in seq_along(years)) {
    yr       <- years[i]
    row_pos  <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
    yr_stats <- real_stats %>% filter(year == yr)
    yr_holm  <- holm_pvals[[as.character(yr)]]
    for (j in seq_along(categories)) {
      cat_name <- categories[j]; col_pos <- col_positions[j]
      df_sub   <- df %>% filter(category == cat_name, year == yr)
      p <- make_panel(df_sub, "score", cat_name, yr, col_pos, row_pos,
                      hline_val     = yr_stats$median_score,
                      iqr_low       = yr_stats$q1_score,
                      iqr_high      = yr_stats$q3_score,
                      holm_pvals_yr = yr_holm,
                      star_y        = 148) + scale_sc
      panels <- c(panels, list(p), list(plot_spacer()))
    }
    p_real <- make_real_panel(df_real %>% filter(year == yr), yr, row_pos) + scale_sc
    panels <- c(panels, list(p_real))
  }
  p_grid <- wrap_plots(panels, ncol = 7, byrow = TRUE,
                       widths = c(1, 0.01, 1, 0.01, 1, 0.01, 0.6)) +
    plot_annotation(theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                                  panel.spacing = unit(2, "pt")))
  ggsave("boxplot_score_by_year.png", plot = p_grid,
         width = 9, height = 10, units = "in", dpi = 600, bg = "white")
  cat("Saved: boxplot_score_by_year.png\n")
}

# =============================================================================
# Run — tutto parte da un'unica chiamata a compute_holm_pvals
# =============================================================================
holm_result <- compute_holm_pvals(df, df_real, years, model_levels)

build_accuracy_grid()
build_score_grid(holm_result)
export_mw_table(holm_result$table)

cat("Done!\n")