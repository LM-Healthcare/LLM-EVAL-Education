# =============================================================================
# Tabella descrittiva: mediana [IQR] score e accuracy per anno e overall
# Output: Excel (.xlsx) con openxlsx
# =============================================================================

library(dplyr)
library(tidyr)
library(openxlsx)

# --- Helper formattazione ---
fmt_score <- function(x) {
  if (all(is.na(x))) return("—")
  paste0(
    formatC(median(x, na.rm = TRUE), format = "f", digits = 1),
    " [",
    formatC(quantile(x, 0.25, na.rm = TRUE), format = "f", digits = 1),
    "–",
    formatC(quantile(x, 0.75, na.rm = TRUE), format = "f", digits = 1),
    "]"
  )
}

fmt_acc <- function(x) {
  if (all(is.na(x))) return("—")
  paste0(
    formatC(median(x, na.rm = TRUE) * 100, format = "f", digits = 1), "%",
    " [",
    formatC(quantile(x, 0.25, na.rm = TRUE) * 100, format = "f", digits = 1), "%",
    "–",
    formatC(quantile(x, 0.75, na.rm = TRUE) * 100, format = "f", digits = 1), "%",
    "]"
  )
}

# --- Dati candidati reali (da SSM2020_2024_statistiche.xlsx) ---
candidates_score <- list(
  "2020" = "77.3 [62.3–91.5]",
  "2021" = "78.5 [65.3–92.5]",
  "2022" = "82.5 [70.3–94.3]",
  "2023" = "84.5 [70.5–97.3]",
  "2024" = "81.3 [69.8–92.5]"
)
candidates_acc <- list(
  "2020" = "55.2% [44.5%–65.4%]",
  "2021" = "56.1% [46.6%–66.1%]",
  "2022" = "58.9% [50.2%–67.3%]",
  "2023" = "60.4% [50.4%–69.5%]",
  "2024" = "58.0% [49.8%–66.1%]"
)
# Accuracy calcolata come Punteggio Prova / 140 usando mediana e IQR del Punteggio Prova
# 2020: 77.25/140=55.2%, Q1=62.25/140=44.5%, Q3=91.5/140=65.4%
# Aggiusta se nel tuo df_real hai già accuracy calcolata diversamente

# --- Overall candidati ---
candidates_score_overall <- fmt_score(df_real$score)
candidates_acc_overall   <- fmt_acc(df_real$accuracy)

# --- Costruzione dataframe LLM ---
years     <- sort(unique(df$year))
cat_order <- c("Closed", "Open", "Quantized")

table_rows <- list()

for (cat_name in cat_order) {
  models_in_cat <- df %>%
    filter(category == cat_name) %>%
    pull(model) %>% unique() %>% sort()
  
  for (mod in models_in_cat) {
    row <- list(Category = cat_name, Model = mod)
    
    for (yr in years) {
      sub <- df %>% filter(model == mod, year == yr)
      row[[paste0(yr, "_Score")]]    <- fmt_score(sub$score)
      row[[paste0(yr, "_Accuracy")]] <- fmt_acc(sub$accuracy)
    }
    
    sub_all <- df %>% filter(model == mod)
    row[["Overall_Score"]]    <- fmt_score(sub_all$score)
    row[["Overall_Accuracy"]] <- fmt_acc(sub_all$accuracy)
    
    table_rows <- c(table_rows, list(row))
  }
}

# --- Riga Candidates ---
cand_row <- list(Category = "Candidates", Model = "SSM Candidates")
for (yr in as.character(years)) {
  cand_row[[paste0(yr, "_Score")]]    <- candidates_score[[yr]]
  cand_row[[paste0(yr, "_Accuracy")]] <- candidates_acc[[yr]]
}
cand_row[["Overall_Score"]]    <- candidates_score_overall
cand_row[["Overall_Accuracy"]] <- candidates_acc_overall

table_rows <- c(table_rows, list(cand_row))

final_df <- bind_rows(table_rows)

# =============================================================================
# EXCEL
# =============================================================================
wb <- createWorkbook()
addWorksheet(wb, "Descriptive", gridLines = FALSE)

# --- Stili ---
style_header_dark <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#2F2F2F", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold", wrapText = TRUE,
  border = "TopBottomLeftRight", borderColour = "#AAAAAA"
)
style_header_year <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#2E75B6", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold", wrapText = TRUE,
  border = "TopBottomLeftRight", borderColour = "#AAAAAA"
)
style_header_overall <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#555555", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold", wrapText = TRUE,
  border = "TopBottomLeftRight", borderColour = "#AAAAAA"
)
style_cat_closed <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#2E75B6", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_cat_open <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#2E8B57", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_cat_quantized <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#8B6914", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_cat_candidates <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#FFFFFF",
  fgFill = "#E60000", halign = "CENTER", valign = "CENTER",
  textDecoration = "bold",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_model <- createStyle(
  fontName = "Arial", fontSize = 9, fontColour = "#222222",
  fgFill = "#F2F2F2", halign = "LEFT", valign = "CENTER",
  textDecoration = "bold",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_data <- createStyle(
  fontName = "Arial", fontSize = 9,
  fgFill = "#EEF4FB", halign = "CENTER", valign = "CENTER",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_overall <- createStyle(
  fontName = "Arial", fontSize = 9, textDecoration = "bold",
  fgFill = "#E8E8E8", halign = "CENTER", valign = "CENTER",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_candidates_data <- createStyle(
  fontName = "Arial", fontSize = 9,
  fgFill = "#FFE8E8", halign = "CENTER", valign = "CENTER",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)
style_candidates_overall <- createStyle(
  fontName = "Arial", fontSize = 9, textDecoration = "bold",
  fgFill = "#FFCCCC", halign = "CENTER", valign = "CENTER",
  border = "TopBottomLeftRight", borderColour = "#CCCCCC"
)

n_years   <- length(years)
n_cols    <- ncol(final_df)

# =============================================================================
# RIGA 1: header gruppo (anni + overall)
# =============================================================================
writeData(wb, "Descriptive", "Category", startRow = 1, startCol = 1)
writeData(wb, "Descriptive", "Model",    startRow = 1, startCol = 2)
mergeCells(wb, "Descriptive", cols = 1, rows = 1:2)
mergeCells(wb, "Descriptive", cols = 2, rows = 1:2)
addStyle(wb, "Descriptive", style_header_dark, rows = 1:2, cols = 1:2, gridExpand = TRUE)

col_idx <- 3
for (yr in years) {
  writeData(wb, "Descriptive", as.character(yr), startRow = 1, startCol = col_idx)
  mergeCells(wb, "Descriptive", cols = col_idx:(col_idx + 1), rows = 1)
  addStyle(wb, "Descriptive", style_header_year,
           rows = 1, cols = col_idx:(col_idx + 1), gridExpand = TRUE)
  col_idx <- col_idx + 2
}
writeData(wb, "Descriptive", "Overall", startRow = 1, startCol = col_idx)
mergeCells(wb, "Descriptive", cols = col_idx:(col_idx + 1), rows = 1)
addStyle(wb, "Descriptive", style_header_overall,
         rows = 1, cols = col_idx:(col_idx + 1), gridExpand = TRUE)

# =============================================================================
# RIGA 2: Score / Accuracy
# =============================================================================
col_idx <- 3
for (yr in years) {
  writeData(wb, "Descriptive", "Score",    startRow = 2, startCol = col_idx)
  writeData(wb, "Descriptive", "Accuracy", startRow = 2, startCol = col_idx + 1)
  addStyle(wb, "Descriptive", style_header_year,
           rows = 2, cols = col_idx:(col_idx + 1), gridExpand = TRUE)
  col_idx <- col_idx + 2
}
writeData(wb, "Descriptive", "Score",    startRow = 2, startCol = col_idx)
writeData(wb, "Descriptive", "Accuracy", startRow = 2, startCol = col_idx + 1)
addStyle(wb, "Descriptive", style_header_overall,
         rows = 2, cols = col_idx:(col_idx + 1), gridExpand = TRUE)

# =============================================================================
# DATI — a partire da riga 3
# =============================================================================
writeData(wb, "Descriptive", final_df, startRow = 3, startCol = 1,
          colNames = FALSE)

n_rows        <- nrow(final_df)
year_data_cols <- 3:(2 + n_years * 2)
overall_cols   <- (2 + n_years * 2 + 1):(2 + n_years * 2 + 2)
cand_excel_row <- which(final_df$Category == "Candidates") + 2

# Stili righe LLM
llm_rows <- setdiff(3:(n_rows + 2), cand_excel_row)
addStyle(wb, "Descriptive", style_data,
         rows = llm_rows, cols = year_data_cols, gridExpand = TRUE)
addStyle(wb, "Descriptive", style_overall,
         rows = llm_rows, cols = overall_cols, gridExpand = TRUE)
addStyle(wb, "Descriptive", style_model,
         rows = llm_rows, cols = 2, gridExpand = TRUE)

# Stili riga Candidates
addStyle(wb, "Descriptive", style_candidates_data,
         rows = cand_excel_row, cols = year_data_cols, gridExpand = TRUE)
addStyle(wb, "Descriptive", style_candidates_overall,
         rows = cand_excel_row, cols = overall_cols, gridExpand = TRUE)
addStyle(wb, "Descriptive", style_cat_candidates,
         rows = cand_excel_row, cols = 1, gridExpand = TRUE)
addStyle(wb, "Descriptive", style_model,
         rows = cand_excel_row, cols = 2, gridExpand = TRUE)

# Stili colonna Category per LLM + merge
cat_styles <- list(
  Closed    = style_cat_closed,
  Open      = style_cat_open,
  Quantized = style_cat_quantized
)
for (cat_name in cat_order) {
  row_idx <- which(final_df$Category == cat_name) + 2
  addStyle(wb, "Descriptive", cat_styles[[cat_name]],
           rows = row_idx, cols = 1, gridExpand = TRUE)
  if (length(row_idx) > 1) {
    mergeCells(wb, "Descriptive", cols = 1, rows = row_idx)
  }
}

# =============================================================================
# LARGHEZZE E ALTEZZE
# =============================================================================
setColWidths(wb, "Descriptive", cols = 1, widths = 10)
setColWidths(wb, "Descriptive", cols = 2, widths = 14)
setColWidths(wb, "Descriptive", cols = 3:n_cols, widths = 20)
setRowHeights(wb, "Descriptive", rows = 1:2,            heights = 22)
setRowHeights(wb, "Descriptive", rows = 3:(n_rows + 2), heights = 18)

freezePane(wb, "Descriptive", firstActiveRow = 3, firstActiveCol = 3)

saveWorkbook(wb, "table_descriptive.xlsx", overwrite = TRUE)
cat("Saved: table_descriptive.xlsx\n")