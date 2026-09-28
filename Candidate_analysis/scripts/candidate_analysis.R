# =============================================================================
# Human-candidate reference analysis (SSM 2020-2024) and LLM comparison
#
# Run from the Candidate_analysis/ folder:
#     Rscript scripts/candidate_analysis.R
#
# Inputs  (data/):
#   candidate_scores_2020_2024.csv  anonymised national rankings: year, rank_order,
#                                   total_score, test_score, titles_score
#   llm_runs_long.csv               one row per model x year x run (correct answers,
#                                   accuracy, SSM score), built from Results/*/Consistency
# Outputs (outputs/):
#   candidate_descriptives.csv/.xlsx     N, mean, SD, min, Q1, median, Q3, max, P90, P95, P99
#   model_percentiles.csv                percentile rank of each model's median score
#   mann_whitney_LLM_vs_candidates.csv/.xlsx   Table 5 (Mann-Whitney U, Holm per year)
#   table_descriptive.csv                Tables 2-3 (median [IQR] score and accuracy)
#   Figure1_boxplot_accuracy_collapsed.png, Figure2_boxplot_accuracy_by_year.png,
#   Figure3_boxplot_score_by_year.png,  Figure4_percentile_distribution.png
#
# Definitions
#   Candidate score  = test score only ("Punteggio Prova"; in 2021 "Punteggio senza CV"),
#                      i.e. the examination score without curriculum/title points, which
#                      is the quantity comparable with the models' SSM score.
#   Quantiles        = R default (type 7, linear interpolation).
#   Percentile rank of a model in a given year = percentage of candidates whose test
#                      score is lower than or equal to the model's median score (50 runs).
#   Model SSM score  = correct - 0.25 * incorrect (forced response: no blank answers).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(ggtext)
  library(patchwork); library(scales); library(openxlsx)
})

dir.create("outputs", showWarnings = FALSE)

# --- Data --------------------------------------------------------------------
cand <- read.csv("data/candidate_scores_2020_2024.csv")
df_real <- cand %>% transmute(year = as.integer(year), score = test_score)

df <- read.csv("data/llm_runs_long.csv", stringsAsFactors = FALSE)

participants <- c(`2020` = 23756, `2021` = 19449, `2022` = 15873, `2023` = 14043, `2024` = 14125)
ranked_total <- c(`2020` = 23600, `2021` = 19442, `2022` = 15869, `2023` = 14036, `2024` = 14121)

model_levels <- c("CLA", "DPSK", "GPT", "GROK", "MSTRL", "QWEN",
                  "MG4B", "MT8B", "mSTRL3B", "Q1_7B", "Q4B", "Q8B",
                  "MG4Bq4", "MG4Bq6", "MG4Bq8", "MT8Bq4", "MT8Bq6", "MT8Bq8")
cat_map <- c(CLA = "Closed", DPSK = "Closed", GPT = "Closed", GROK = "Closed", MSTRL = "Closed", QWEN = "Closed",
             MG4B = "Open", MT8B = "Open", mSTRL3B = "Open", Q1_7B = "Open", Q4B = "Open", Q8B = "Open",
             MG4Bq4 = "Quantized", MG4Bq6 = "Quantized", MG4Bq8 = "Quantized",
             MT8Bq4 = "Quantized", MT8Bq6 = "Quantized", MT8Bq8 = "Quantized")
df$model    <- factor(df$model, levels = model_levels)
df$category <- factor(cat_map[as.character(df$model)], levels = c("Closed", "Open", "Quantized"))
years <- c(2020, 2021, 2022, 2023, 2024)

# =============================================================================
# 1. Candidate descriptives
# =============================================================================
desc <- df_real %>% group_by(year) %>% summarise(
  n_with_scores = n(), mean = mean(score), sd = sd(score), min = min(score),
  q1 = quantile(score, .25), median = median(score), q3 = quantile(score, .75),
  max = max(score), p90 = quantile(score, .90), p95 = quantile(score, .95),
  p99 = quantile(score, .99), .groups = "drop") %>%
  mutate(participants_official = participants[as.character(year)],
         ranked_total = ranked_total[as.character(year)], .after = year)
write.csv(desc, "outputs/candidate_descriptives.csv", row.names = FALSE)
write.xlsx(desc %>% mutate(across(where(is.double), ~ round(.x, 2))), "outputs/candidate_descriptives.xlsx")
print(desc %>% mutate(across(where(is.double), ~ round(.x, 2))), width = 200)

real_stats <- desc %>% transmute(year, median_score = median, q1_score = q1, q3_score = q3)

# =============================================================================
# 2. Percentile rank of each model's median score
# =============================================================================
model_medians <- df %>% group_by(year, model, category) %>%
  summarise(median_score = median(score), .groups = "drop")
model_medians$percentile <- mapply(function(yr, ms)
  100 * mean(df_real$score[df_real$year == yr] <= ms), model_medians$year, model_medians$median_score)
write.csv(model_medians %>% mutate(percentile = round(percentile, 1)),
          "outputs/model_percentiles.csv", row.names = FALSE)

# =============================================================================
# 3. Mann-Whitney U tests (Table 5): 50 runs of each model vs all candidates,
#    two-sided, normal approximation, Holm correction across the 18 models per year
# =============================================================================
sig_label <- function(p) ifelse(is.na(p), NA, ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "ns"))))
mw_rows <- list(); holm_pvals <- list()
for (yr in years) {
  real_sc <- df_real$score[df_real$year == yr]; real_med <- median(real_sc)
  res <- lapply(model_levels, function(mdl) {
    llm_sc <- df$score[df$model == mdl & df$year == yr]
    wt <- wilcox.test(llm_sc, real_sc, exact = FALSE)
    data.frame(Year = yr, Category = cat_map[mdl], Model = mdl, N_LLM = length(llm_sc),
               N_Candidates = length(real_sc), Median_LLM = median(llm_sc), Median_Candidates = real_med,
               W = unname(wt$statistic), p_value = wt$p.value, stringsAsFactors = FALSE)
  })
  res <- do.call(rbind, res)
  res$p_adj_Holm <- p.adjust(res$p_value, method = "holm")
  res$Direction  <- ifelse(res$Median_LLM > real_med, "LLM > Candidates",
                           ifelse(res$Median_LLM < real_med, "LLM < Candidates", "LLM = Candidates"))
  res$Sig_adj    <- sig_label(res$p_adj_Holm)
  holm_pvals[[as.character(yr)]] <- as.list(setNames(res$p_adj_Holm, res$Model))
  mw_rows[[length(mw_rows) + 1]] <- res
}
mw <- do.call(rbind, mw_rows); rownames(mw) <- NULL
write.csv(mw, "outputs/mann_whitney_LLM_vs_candidates.csv", row.names = FALSE)
write.xlsx(mw, "outputs/mann_whitney_LLM_vs_candidates.xlsx")

# =============================================================================
# 4. Tables 2-3: median [IQR] of score and accuracy per model and year
# =============================================================================
fmt <- function(x, pct = FALSE, d = 1) {
  q <- quantile(x, c(.5, .25, .75)); if (pct) q <- q * 100
  s <- if (pct) "%%" else ""
  sprintf(paste0("%.", d, "f", s, " [%.", d, "f", s, "\u2013%.", d, "f", s, "]"), q[1], q[2], q[3])
}
tab <- bind_rows(lapply(model_levels, function(m) {
  r <- list(Category = cat_map[[m]], Model = m)
  for (yr in years) {
    sub <- df %>% filter(model == m, year == yr)
    r[[paste0(yr, "_Score")]] <- fmt(sub$score); r[[paste0(yr, "_Accuracy")]] <- fmt(sub$accuracy, TRUE)
  }
  sub <- df %>% filter(model == m)
  r$Overall_Score <- fmt(sub$score); r$Overall_Accuracy <- fmt(sub$accuracy, TRUE)
  as.data.frame(r, check.names = FALSE)
}))
cand_row <- list(Category = "Candidates", Model = "SSM candidates")
for (yr in years) {
  s <- df_real$score[df_real$year == yr]
  cand_row[[paste0(yr, "_Score")]] <- sprintf("%.1f [%.1f\u2013%.1f]", median(s), quantile(s, .25), quantile(s, .75))
  cand_row[[paste0(yr, "_Accuracy")]] <- "—"
}
cand_row$Overall_Score <- "—"; cand_row$Overall_Accuracy <- "—"
tab <- bind_rows(tab, as.data.frame(cand_row, check.names = FALSE))
write.csv(tab, "outputs/table_descriptive.csv", row.names = FALSE, fileEncoding = "UTF-8")

# =============================================================================
# 5. Figures (same design as the original figures)
# =============================================================================
cat_colors <- c(Closed = "#2E86AB", Open = "#A23B72", Quantized = "#F18F01")
categories <- c("Closed", "Open", "Quantized"); col_positions <- c("left", "middle", "right")
theme_base <- theme_bw() + theme(
  panel.border = element_blank(),
  axis.line.x = element_line(color = "black", linewidth = 0.4),
  axis.line.y = element_line(color = "black", linewidth = 0.4),
  panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
  panel.grid.major.y = element_line(color = "grey85", linetype = "dashed", linewidth = 0.2),
  axis.text.x = element_text(angle = 35, hjust = 1, size = 11), axis.text.y = element_text(size = 11),
  axis.title = element_text(size = 13, face = "bold"), plot.title = element_markdown(size = 14, hjust = 0.5))

make_panel <- function(df_sub, y_col, cat_name, year_val, col_pos = "middle", row_pos = "middle",
                       hline_val = NULL, iqr_low = NULL, iqr_high = NULL, holm_pvals_yr = NULL, star_y = NULL,
                       y_title = NULL) {
  panel_title <- if (row_pos == "top") paste0("<span style='color:", cat_colors[cat_name], ";'>**", cat_name, "**</span>") else NULL
  p <- ggplot(df_sub, aes(x = model, y = .data[[y_col]]))
  if (!is.null(iqr_low)) p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = iqr_low, ymax = iqr_high, fill = "#E60000", alpha = 0.08)
  p <- p + geom_boxplot(fill = cat_colors[cat_name], alpha = 0.75, outlier.size = 1, outlier.alpha = 0.4,
                        color = "#333333", width = 0.8, linewidth = 0.2) +
    labs(title = panel_title, x = NULL,
         y = if (col_pos == "left") (if (is.null(y_title)) as.character(year_val) else y_title) else NULL) + theme_base
  if (!is.null(hline_val)) p <- p + geom_hline(yintercept = hline_val, color = "#E60000", linetype = "dashed", linewidth = 0.5)
  if (!is.null(holm_pvals_yr)) {
    models_here <- levels(droplevels(df_sub$model))
    for (m_idx in seq_along(models_here)) {
      p_adj <- holm_pvals_yr[[models_here[m_idx]]]
      if (is.null(p_adj) || is.na(p_adj) || p_adj >= 0.05) next
      p <- p + annotate("text", x = m_idx, y = star_y, size = 3, color = "grey30",
                        label = ifelse(p_adj < .001, "***", ifelse(p_adj < .01, "**", "*")))
    }
  }
  if (row_pos != "bottom") p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  p + theme(plot.margin = margin(0, 15, 0, ifelse(col_pos == "left", 5, 15), unit = "pt"))
}
make_real_panel <- function(df_real_sub, row_pos = "middle") {
  panel_title <- if (row_pos == "top") "<span style='color:#E60000;'>**Candidates**</span>" else NULL
  p <- ggplot(df_real_sub, aes(x = "SSM", y = score)) +
    geom_boxplot(fill = "#E60000", alpha = 0.75, outlier.size = 0.3, outlier.alpha = 0.15,
                 color = "#333333", width = 0.4, linewidth = 0.2) +
    labs(title = panel_title, x = NULL, y = NULL) + theme_base
  if (row_pos != "bottom") p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  p + theme(plot.margin = margin(0, 5, 0, 15, unit = "pt"))
}
grid_theme <- plot_annotation(theme = theme(panel.spacing = unit(2, "pt")))

# Figure 1: accuracy, all years collapsed
scale_acc <- scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), labels = percent_format(accuracy = 1))
panels <- list()
for (j in seq_along(categories)) {
  p <- make_panel(df %>% filter(category == categories[j]), "accuracy", categories[j], NA,
                  col_positions[j], row_pos = "top", y_title = "accuracy") + scale_acc +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 11), axis.ticks.x = element_line(),
          plot.margin = margin(0, 15, 12, ifelse(j == 1, 5, 15), unit = "pt"))
  panels <- c(panels, list(p)); if (j < 3) panels <- c(panels, list(plot_spacer()))
}
ggsave("outputs/Figure1_boxplot_accuracy_collapsed.png",
       wrap_plots(panels, ncol = 5, widths = c(1, .01, 1, .01, 1)) + grid_theme,
       width = 9, height = 4, units = "in", dpi = 600, bg = "white")

# Figure 2: accuracy by year
panels <- list()
for (i in seq_along(years)) {
  row_pos <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
  for (j in seq_along(categories)) {
    p <- make_panel(df %>% filter(category == categories[j], year == years[i]), "accuracy",
                    categories[j], years[i], col_positions[j], row_pos) + scale_acc
    panels <- c(panels, list(p)); if (j < 3) panels <- c(panels, list(plot_spacer()))
  }
}
ggsave("outputs/Figure2_boxplot_accuracy_by_year.png",
       wrap_plots(panels, ncol = 5, byrow = TRUE, widths = c(1, .01, 1, .01, 1)) + grid_theme,
       width = 9, height = 10.5, units = "in", dpi = 600, bg = "white")

# Figure 3: score by year, candidates median (dashed line) and IQR (band), Holm-adjusted stars
scale_sc <- scale_y_continuous(limits = c(-5, 150), breaks = seq(0, 140, 20))
panels <- list()
for (i in seq_along(years)) {
  yr <- years[i]; row_pos <- ifelse(i == 1, "top", ifelse(i == length(years), "bottom", "middle"))
  st <- real_stats %>% filter(year == yr)
  for (j in seq_along(categories)) {
    p <- make_panel(df %>% filter(category == categories[j], year == yr), "score", categories[j], yr,
                    col_positions[j], row_pos, hline_val = st$median_score, iqr_low = st$q1_score,
                    iqr_high = st$q3_score, holm_pvals_yr = holm_pvals[[as.character(yr)]], star_y = 148) + scale_sc
    panels <- c(panels, list(p), list(plot_spacer()))
  }
  panels <- c(panels, list(make_real_panel(df_real %>% filter(year == yr), row_pos) + scale_sc))
}
ggsave("outputs/Figure3_boxplot_score_by_year.png",
       wrap_plots(panels, ncol = 7, byrow = TRUE, widths = c(1, .01, 1, .01, 1, .01, .6)) + grid_theme,
       width = 9, height = 10, units = "in", dpi = 600, bg = "white")

# Figure 4: candidate score histograms with model positions (median score of the 50 runs)
model_numbers <- setNames(seq_along(model_levels), model_levels)
darken_color <- function(hex, factor = 0.6) { v <- col2rgb(hex) / 255 * factor; rgb(v[1], v[2], v[3]) }
segment_color <- function(cats) { u <- unique(cats); if (length(u) == 1) darken_color(cat_colors[u], 0.65) else "grey50" }
richtext_label <- function(models) {
  parts <- paste0("<span style='color:", cat_colors[cat_map[as.character(models)]], "'>**",
                  model_numbers[as.character(models)], "**</span>")
  paste(parts, collapse = "<span style='color:black'>/</span>")
}
assign_y_levels <- function(x_pos, min_x_gap = 4, y_levels = c(800, 950, 1100, 1250)) {
  res <- integer(length(x_pos)); last_x <- rep(-Inf, length(y_levels))
  for (i in seq_along(x_pos)) {
    ch <- which(x_pos[i] - last_x >= min_x_gap)[1]; if (is.na(ch)) ch <- which.min(last_x)
    res[i] <- ch; last_x[ch] <- x_pos[i]
  }
  y_levels[res]
}
theme_perc <- theme_bw() + theme(
  panel.border = element_blank(),
  axis.line.x = element_line(color = "black", linewidth = 0.4), axis.line.y = element_line(color = "black", linewidth = 0.4),
  panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
  panel.grid.major.y = element_line(color = "grey85", linetype = "dashed", linewidth = 0.2),
  axis.text = element_text(size = 9), axis.title = element_text(size = 11, face = "bold"))
make_year_plot <- function(yr) {
  cand_y <- df_real %>% filter(year == yr)
  mods <- model_medians %>% filter(year == yr) %>% arrange(median_score)
  p <- ggplot(cand_y, aes(x = score)) +
    geom_histogram(fill = "grey80", color = "grey70", alpha = 0.3, binwidth = 2, boundary = 0) +
    geom_vline(xintercept = median(cand_y$score), color = "#E60000", linewidth = 0.6) +
    labs(x = NULL, y = as.character(yr)) + theme_perc +
    scale_x_continuous(limits = c(-6, 142), breaks = seq(0, 140, 10), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1300), breaks = seq(0, 1200, 200))
  mods$group_id <- cumsum(c(TRUE, diff(mods$median_score) >= 0.2))
  groups <- lapply(unique(mods$group_id), function(g) {
    sub <- mods[mods$group_id == g, ]
    list(x_pos = mean(sub$median_score), seg_col = segment_color(as.character(sub$category)),
         lbl = richtext_label(sub$model))
  })
  y_nums <- assign_y_levels(sapply(groups, `[[`, "x_pos"))
  for (k in seq_along(groups))
    p <- p + annotate("segment", x = groups[[k]]$x_pos, xend = groups[[k]]$x_pos, y = 0, yend = y_nums[k] - 28,
                      color = groups[[k]]$seg_col, linewidth = 0.5, alpha = 0.85)
  for (k in seq_along(groups))
    p <- p + annotate("richtext", x = groups[[k]]$x_pos, y = y_nums[k], label = groups[[k]]$lbl,
                      hjust = 0.5, vjust = 0.5, size = 2.5, fill = "white", label.color = "grey70",
                      label.size = 0.3, label.padding = unit(0.15, "lines"))
  p
}
legend_df <- data.frame(model_num = seq_along(model_levels), model = model_levels,
                        category = cat_map[model_levels], stringsAsFactors = FALSE)
legend_df$col_idx <- ceiling(legend_df$model_num / 6); legend_df$row_idx <- ((legend_df$model_num - 1) %% 6) + 1
col_start <- c(0.75, 4.0, 7.25)
txt_df <- bind_rows(lapply(seq_len(nrow(legend_df)), function(i) {
  xb <- col_start[legend_df$col_idx[i]]; y <- -legend_df$row_idx[i]; col <- cat_colors[legend_df$category[i]]
  data.frame(x = c(xb, xb + 0.4), y = y, label = c(as.character(legend_df$model_num[i]), legend_df$model[i]),
             col = col, hjust = c(0.5, 0), bold = c(TRUE, FALSE))
}))
p_legend <- ggplot(txt_df) +
  geom_text(aes(x = x, y = y, label = label, color = I(col), hjust = hjust, fontface = ifelse(bold, "bold", "plain")), size = 3.2) +
  annotate("text", x = col_start, y = 0, hjust = 0, size = 3.4, fontface = "bold",
           label = categories, color = unname(cat_colors[categories])) +
  scale_x_continuous(limits = c(0.5, 10.5)) + scale_y_continuous(limits = c(-6.5, 0.8)) +
  theme_void() + theme(plot.margin = margin(2, 10, 5, 10))
p_all <- wrap_plots(list(wrap_plots(lapply(years, make_year_plot), ncol = 1), p_legend), ncol = 1, heights = c(5, 0.6))
ggsave("outputs/Figure4_percentile_distribution.png", p_all, width = 9, height = 9, units = "in", dpi = 600, bg = "white")

cat("Done.\n")
