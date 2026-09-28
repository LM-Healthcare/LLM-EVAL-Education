# =============================================================================
# Percentile Distribution of SSM Candidates with LLM Model Positions
# =============================================================================

library(ggplot2)
library(dplyr)
library(ggtext)
library(patchwork)
library(scales)

# --- Load data ---------------------------------------------------------------
setwd("C:\\Users\\edoar\\Desktop\\LLM and SSM test\\Risultati\\Analysis R")

df <- read.csv("db_LLM_SSN_accuracy_long_v2.csv", sep = ";", stringsAsFactors = FALSE)
df$score <- df$correct - (140 - df$correct) * 0.25

# --- Load real candidates scores ---------------------------------------------
df_real_raw <- read.csv("scores.csv", sep = ";")
colnames(df_real_raw) <- c("2020", "2021", "2022", "2023", "2024")

df_real <- tidyr::pivot_longer(df_real_raw, cols = everything(),
                               names_to = "year", values_to = "score") %>%
  mutate(year = as.integer(year)) %>%
  filter(!is.na(score))

# --- Model metadata ----------------------------------------------------------
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

model_numbers <- setNames(seq_along(model_levels), model_levels)

df$model    <- factor(df$model, levels = model_levels)
df$category <- factor(cat_map[as.character(df$model)],
                      levels = c("Closed", "Open", "Quantized"))

# --- Colors per category -----------------------------------------------------
cat_colors <- c(Closed = "#2E86AB", Open = "#A23B72", Quantized = "#F18F01")

# --- Helper: darken a hex color ----------------------------------------------
darken_color <- function(hex, factor = 0.6) {
  rgb_vals <- col2rgb(hex) / 255
  rgb_dark  <- rgb_vals * factor
  rgb(rgb_dark[1], rgb_dark[2], rgb_dark[3])
}

# --- Helper: segment color given vector of categories -----------------------
segment_color <- function(cats) {
  ucats <- unique(cats)
  if (length(ucats) == 1) darken_color(cat_colors[ucats], factor = 0.65) else "grey50"
}

# --- Helper: richtext label with each number in its own category color -------
richtext_label <- function(models) {
  nums   <- model_numbers[as.character(models)]
  cats   <- cat_map[as.character(models)]
  colors <- cat_colors[cats]
  parts  <- paste0("<span style='color:", colors, "'>**", nums, "**</span>")
  paste(parts, collapse = "<span style='color:black'>/</span>")
}

# --- Helper: assign y levels avoiding overlaps (greedy lane assignment) ------
assign_y_levels <- function(x_pos, min_x_gap = 4, y_levels = c(800, 950, 1100, 1250)) {
  n      <- length(x_pos)
  result <- integer(n)
  last_x <- rep(-Inf, length(y_levels))
  for (i in seq_len(n)) {
    chosen <- NA
    for (lv in seq_along(y_levels)) {
      if (x_pos[i] - last_x[lv] >= min_x_gap) {
        chosen <- lv
        break
      }
    }
    if (is.na(chosen)) chosen <- which.min(last_x)
    result[i] <- chosen
    last_x[chosen] <- x_pos[i]
  }
  y_levels[result]
}

# --- Base theme --------------------------------------------------------------
theme_perc <- theme_bw() +
  theme(
    panel.border       = element_blank(),
    axis.line.x        = element_line(color = "black", linewidth = 0.4),
    axis.line.y        = element_line(color = "black", linewidth = 0.4),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey85", linetype = "dashed", linewidth = 0.2),
    axis.text          = element_text(size = 9),
    axis.title         = element_text(size = 11, face = "bold"),
    plot.title         = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle      = element_text(size = 10, hjust = 0.5, color = "grey40")
  )

# =============================================================================
# Compute median score per model per year
# =============================================================================
model_medians <- df %>%
  group_by(year, model, category) %>%
  summarise(median_score = median(score, na.rm = TRUE), .groups = "drop")

# =============================================================================
# Compute percentile of each model within candidates distribution
# =============================================================================
model_medians$percentile <- NA_real_

for (i in seq_len(nrow(model_medians))) {
  yr  <- model_medians$year[i]
  ms  <- model_medians$median_score[i]
  cand_scores <- df_real$score[df_real$year == yr]
  model_medians$percentile[i] <- round(100 * mean(cand_scores <= ms), 1)
}

# =============================================================================
# Compute median score of candidates per year
# =============================================================================
real_medians <- df_real %>%
  group_by(year) %>%
  summarise(median_score = median(score, na.rm = TRUE))

# =============================================================================
# Build one plot per year
# =============================================================================
years <- c(2020, 2021, 2022, 2023, 2024)

make_year_plot <- function(yr) {
  
  cand <- df_real %>% filter(year == yr)
  mods <- model_medians %>% filter(year == yr) %>% arrange(median_score)
  cand_median <- real_medians$median_score[real_medians$year == yr]
  
  p <- ggplot(cand, aes(x = score)) +
    geom_histogram(fill = "grey80", color = "grey70", alpha = 0.3,
                   binwidth = 2, boundary = 0) +
    geom_vline(xintercept = cand_median, color = "#E60000",
               linetype = "solid", linewidth = 0.6) +
    labs(x = NULL, y = as.character(yr)) +
    theme_perc +
    scale_x_continuous(limits = c(0, 140), breaks = seq(0, 140, 10), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1300), breaks = seq(0, 1200, 200))
  
  min_gap <- 0.2
  
  # Group models with identical/near-identical scores
  mods$group_id <- NA_integer_
  g <- 1L
  mods$group_id[1] <- g
  for (k in 2:nrow(mods)) {
    if (mods$median_score[k] - mods$median_score[k - 1] < min_gap) {
      mods$group_id[k] <- g
    } else {
      g <- g + 1L
      mods$group_id[k] <- g
    }
  }
  
  # Build one entry per group
  group_ids <- unique(mods$group_id)
  groups <- lapply(seq_along(group_ids), function(gi) {
    sub <- mods[mods$group_id == group_ids[gi], ]
    list(
      x_pos    = mean(sub$median_score),
      seg_col  = segment_color(as.character(sub$category)),
      rich_lbl = if (nrow(sub) == 1)
        paste0("<span style='color:", cat_colors[as.character(sub$category[1])],
               "'>**", model_numbers[as.character(sub$model[1])], "**</span>")
      else
        richtext_label(sub$model)
    )
  })
  
  # Assign y heights with greedy lane algorithm
  x_positions <- sapply(groups, `[[`, "x_pos")
  y_nums <- assign_y_levels(x_positions, min_x_gap = 4,
                            y_levels = c(800, 950, 1100, 1250))
  
  # Pass 1: all segments
  for (k in seq_along(groups)) {
    grp   <- groups[[k]]
    y_num <- y_nums[k]
    p <- p +
      annotate("segment",
               x = grp$x_pos, xend = grp$x_pos,
               y = 0, yend = y_num - 28,
               color = grp$seg_col, linewidth = 0.5, alpha = 0.85)
  }
  
  # Pass 2: all labels on top
  for (k in seq_along(groups)) {
    grp   <- groups[[k]]
    y_num <- y_nums[k]
    p <- p +
      annotate("richtext",
               x = grp$x_pos, y = y_num,
               label = grp$rich_lbl,
               hjust = 0.5, vjust = 0.5,
               size = 2.5, fill = "white", label.color = "grey70",
               label.size = 0.3, label.padding = unit(0.15, "lines"))
  }
  
  return(p)
}

# =============================================================================
# Assemble all years in a vertical stack
# =============================================================================
plots <- lapply(years, make_year_plot)

# =============================================================================
# Legend: 3 columns of 6, number + model name only (colored by category)
# =============================================================================
legend_df <- data.frame(
  model_num = seq_along(model_levels),
  model     = model_levels,
  category  = cat_map[model_levels],
  stringsAsFactors = FALSE
)

n_per_col <- 6
legend_df$col_idx <- ceiling(legend_df$model_num / n_per_col)
legend_df$row_idx <- ((legend_df$model_num - 1) %% n_per_col) + 1

col_start <- c(0.75, 4.0, 7.25)
off_num   <- 0
off_name  <- 0.4

rows_list <- list()
for (i in seq_len(nrow(legend_df))) {
  ci    <- legend_df$col_idx[i]
  ri    <- legend_df$row_idx[i]
  xbase <- col_start[ci]
  y     <- -ri
  color <- cat_colors[legend_df$category[i]]
  
  rows_list[[length(rows_list) + 1]] <- data.frame(
    x = xbase + off_num,  y = y,
    label = as.character(legend_df$model_num[i]),
    col = color, hjust = 0.5, bold = TRUE, stringsAsFactors = FALSE)
  rows_list[[length(rows_list) + 1]] <- data.frame(
    x = xbase + off_name, y = y,
    label = legend_df$model[i],
    col = color, hjust = 0, bold = FALSE, stringsAsFactors = FALSE)
}

txt_df <- dplyr::bind_rows(rows_list)

p_legend <- ggplot(txt_df) +
  geom_text(aes(x = x, y = y, label = label, color = I(col), hjust = hjust,
                fontface = ifelse(bold, "bold", "plain")), size = 3.2) +
  annotate("text", x = col_start, y = 0, hjust = 0, size = 3.4, fontface = "bold",
           label = c("Closed", "Open", "Quantized"),
           color = unname(cat_colors[c("Closed", "Open", "Quantized")])) +
  scale_x_continuous(limits = c(0.5, 10.5)) +
  scale_y_continuous(limits = c(-(n_per_col + 0.5), 0.8)) +
  theme_void() +
  theme(plot.margin = margin(2, 10, 5, 10))

# =============================================================================
# Final assembly
# =============================================================================
p_histograms <- wrap_plots(plots, ncol = 1)

p_all <- wrap_plots(
  list(p_histograms, p_legend),
  ncol = 1,
  heights = c(5, 0.6)
) +
  plot_annotation(theme = theme(
    plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "grey40")
  ))

ggsave("percentile_distribution_numbered_v1.png", plot = p_all,
       width = 9, height = 9, units = "in", dpi = 600, bg = "white")

cat("Done!\n")