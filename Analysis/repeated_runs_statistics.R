# =============================================================================
# Statistical analysis accounting for the repeated-run design (Reviewer 1, Comment 3)
#
# Run from the repository root:
#     Rscript Analysis/repeated_runs_statistics.R
#
# Inputs
#   Candidate_analysis/data/llm_runs_long.csv          one row per model x year x run
#   Candidate_analysis/data/candidate_scores_2020_2024.csv
#   Results/*/Consistency/consistency_*.xlsx           per-question outcome of every run
#                                                      (used to compute item-level accuracy)
# Outputs (Analysis/statistics_output/)
#   llm_item_accuracy.csv                one row per model x question: share of the 50 runs correct
#   kruskal_wallis_effect_sizes.csv      Table 4 (omnibus): H, df, p, Bonferroni-adjusted p, eta2_H
#   pairwise_comparisons.csv             Table S10: Dunn (Holm) + Cliff's delta (runs) +
#                                        difference in accuracy with 95% item-bootstrap CI
#   quantization_vs_full_precision.csv   RQ4: each quantized variant vs its full-precision model
#                                        (descriptive: difference, 95% item-bootstrap CI, Cliff's delta)
#   mann_whitney_effect_sizes.csv        Table 5 extension: probability of superiority (A) and
#                                        rank-biserial correlation for each model vs candidates
#
# Units of analysis
#   Kruskal-Wallis, Dunn, Mann-Whitney: one complete run of a model on one examination year
#     (accuracy or SSM score of 140 questions); 250 runs per model pooled over 2020-2024 for
#     the between-model comparisons, 50 runs per model and year for the comparison with the
#     candidates (one observation per candidate).
#   Item-bootstrap sensitivity analysis: the question (700 questions), resampled with
#     replacement and paired across models, so that the confidence interval reflects the
#     uncertainty due to the particular set of questions rather than run-to-run variability.
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(rstatix); library(readxl); library(tidyr) })
set.seed(2026)
B <- 4000
out_dir <- "Analysis/statistics_output"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

runs  <- read.csv("Candidate_analysis/data/llm_runs_long.csv")
# item-level accuracy from the Consistency files (1 = correct; letter given = incorrect; empty cell =
# no answer extracted, counted as incorrect exactly as in the run totals of llm_runs_long.csv)
key <- c(claude = "CLA", deepseek = "DPSK", `gpt-52-2025-12-11` = "GPT", grok = "GROK", mistral = "MSTRL",
         qwen = "QWEN", MedGemma_4B_IT = "MG4B", Meditron3_8B_FP16 = "MT8B", Ministral_3B_Instruct = "mSTRL3B",
         Qwen3_1_7B = "Q1_7B", Qwen3_4B_Instruct = "Q4B", Qwen3_8B = "Q8B", MedGemma_4B_Q4_K_M = "MG4Bq4",
         MedGemma_4B_Q6_K = "MG4Bq6", MedGemma_4B_Q8_0 = "MG4Bq8", Meditron3_8B_Q4_K_M = "MT8Bq4",
         Meditron3_8B_Q6_K = "MT8Bq6", Meditron3_8B_Q8_0 = "MT8Bq8")
files <- Sys.glob("Results/*/Consistency/consistency_*.xlsx")
items <- bind_rows(lapply(files, function(f) {
  m <- key[[sub("^consistency_(.*)\\.xlsx$", "\\1", basename(f))]]
  x <- suppressMessages(read_excel(f))
  x %>% transmute(year = as.integer(substr(RUN_EXAM, 1, 4)), across(starts_with("Domanda_"), ~ coalesce(as.character(.x) == "1", FALSE))) %>%
    pivot_longer(-year, names_to = "question", values_to = "correct") %>%
    mutate(question = as.integer(sub("Domanda_", "", question))) %>%
    group_by(year, question) %>% summarise(item_accuracy = mean(correct), .groups = "drop") %>% mutate(model = m)
}))
write.csv(items %>% select(model, year, question, item_accuracy), file.path(out_dir, "llm_item_accuracy.csv"), row.names = FALSE)
cand  <- read.csv("Candidate_analysis/data/candidate_scores_2020_2024.csv")

groups <- list(Proprietary = c("CLA", "DPSK", "GPT", "GROK", "MSTRL", "QWEN"),
               `Open-source` = c("MG4B", "MT8B", "mSTRL3B", "Q1_7B", "Q4B", "Q8B"),
               Quantized = c("MG4Bq4", "MG4Bq6", "MG4Bq8", "MT8Bq4", "MT8Bq6", "MT8Bq8"))

# probability of superiority A = P(X > Y) + 0.5 P(X = Y); Cliff's delta = 2A - 1
prob_sup <- function(x, y) unname(wilcox.test(x, y, exact = FALSE)$statistic) / (length(x) * length(y))

# item-level wide matrix: rows = questions (year x question), columns = models
wide <- items %>% mutate(item = paste(year, question)) %>%
  select(item, model, item_accuracy) %>% tidyr::pivot_wider(names_from = model, values_from = item_accuracy)

kw_rows <- list(); pw_rows <- list()
for (g in names(groups)) {
  d <- runs %>% filter(model %in% groups[[g]]) %>% mutate(model = factor(model, levels = groups[[g]]))
  kw <- kruskal_test(d, accuracy ~ model); es <- kruskal_effsize(d, accuracy ~ model)
  kw_rows[[g]] <- data.frame(Group = g, H = unname(kw$statistic), df = kw$df, p = kw$p,
                             p_bonferroni_3 = min(1, kw$p * 3), eta2_H = es$effsize, magnitude = as.character(es$magnitude))
  dunn <- dunn_test(d, accuracy ~ model, p.adjust.method = "holm")
  for (r in seq_len(nrow(dunn))) {
    a <- dunn$group1[r]; b <- dunn$group2[r]
    xa <- d$accuracy[d$model == a]; xb <- d$accuracy[d$model == b]
    diff_items <- wide[[a]] - wide[[b]]                       # paired by question
    boot <- replicate(B, mean(sample(diff_items, replace = TRUE)))
    pw_rows[[length(pw_rows) + 1]] <- data.frame(
      Group = g, Model_1 = a, Model_2 = b,
      Accuracy_1 = mean(xa) * 100, Accuracy_2 = mean(xb) * 100,
      Difference_pp = (mean(xa) - mean(xb)) * 100,
      CI95_items_low = quantile(boot, .025) * 100, CI95_items_high = quantile(boot, .975) * 100,
      Cliffs_delta_runs = 2 * prob_sup(xa, xb) - 1,
      Dunn_p_adj_Holm = dunn$p.adj[r])
  }
}
kw_tab <- do.call(rbind, kw_rows); pw <- do.call(rbind, pw_rows); rownames(pw) <- NULL
pw$CI_excludes_zero <- pw$CI95_items_low > 0 | pw$CI95_items_high < 0
write.csv(kw_tab, file.path(out_dir, "kruskal_wallis_effect_sizes.csv"), row.names = FALSE)
write.csv(pw, file.path(out_dir, "pairwise_comparisons.csv"), row.names = FALSE)

# Quantized variants vs their full-precision model (not part of the Kruskal-Wallis groups, so no test)
qv <- data.frame(full = rep(c("MG4B", "MT8B"), each = 3),
                 quant = c("MG4Bq4", "MG4Bq6", "MG4Bq8", "MT8Bq4", "MT8Bq6", "MT8Bq8"))
qrows <- lapply(seq_len(nrow(qv)), function(i) {
  a <- qv$quant[i]; b <- qv$full[i]
  xa <- runs$accuracy[runs$model == a]; xb <- runs$accuracy[runs$model == b]
  diff_items <- wide[[a]] - wide[[b]]
  boot <- replicate(B, mean(sample(diff_items, replace = TRUE)))
  data.frame(Quantized = a, Full_precision = b, Accuracy_quantized = mean(xa) * 100,
             Accuracy_full = mean(xb) * 100, Retention_pct = mean(xa) / mean(xb) * 100,
             Difference_pp = (mean(xa) - mean(xb)) * 100,
             CI95_items_low = quantile(boot, .025) * 100, CI95_items_high = quantile(boot, .975) * 100,
             Cliffs_delta_runs = 2 * prob_sup(xa, xb) - 1)
})
qtab <- do.call(rbind, qrows); rownames(qtab) <- NULL
write.csv(qtab, file.path(out_dir, "quantization_vs_full_precision.csv"), row.names = FALSE)

# Mann-Whitney vs candidates: effect sizes for Table 5
mw_rows <- list()
for (yr in 2020:2024) {
  cs <- cand$test_score[cand$year == yr]
  for (m in unlist(groups)) {
    s <- runs$score[runs$model == m & runs$year == yr]
    A <- prob_sup(s, cs)
    mw_rows[[length(mw_rows) + 1]] <- data.frame(Year = yr, Model = m, A = A, r_rank_biserial = 2 * A - 1)
  }
}
write.csv(do.call(rbind, mw_rows), file.path(out_dir, "mann_whitney_effect_sizes.csv"), row.names = FALSE)

print(kw_tab, digits = 4)
print(qtab, digits = 3)
print(pw %>% mutate(across(where(is.numeric), ~ round(.x, 3))), row.names = FALSE)
