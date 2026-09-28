# =============================================================================
# Tabella comparisons: Kruskal-Wallis globale + Dunn post-hoc (Holm-Bonferroni)
# Output: Excel unico con tutte le categorie
# =============================================================================

library(rstatix)
library(dplyr)
library(tidyr)
library(openxlsx)

build_comparison_table <- function(df, y_col, output_file) {
  
  all_rows <- list()
  
  for (cat_name in categories) {
    
    df_sub <- df %>% filter(category == cat_name)
    
    # --- Kruskal-Wallis globale ---
    kw   <- kruskal_test(df_sub, as.formula(paste(y_col, "~ model")))
    kw_p <- kw$p
    kw_label <- paste0(
      "H=", formatC(kw$statistic, format = "f", digits = 2),
      ", df=", kw$df,
      ", p=", ifelse(kw_p < 0.001, "<0.001", formatC(kw_p, format = "f", digits = 3))
    )
    
    # --- Dunn post-hoc ---
    dunn <- dunn_test(df_sub, as.formula(paste(y_col, "~ model")),
                      p.adjust.method = "holm") %>%
      mutate(
        cell_label = ifelse(p.adj < 0.001, "<0.001",
                            formatC(p.adj, format = "f", digits = 3))
      )
    
    # Matrice simmetrica modello x modello
    models_here <- sort(unique(c(dunn$group1, dunn$group2)))
    n_models    <- length(models_here)
    
    mat <- matrix("", nrow = n_models, ncol = n_models,
                  dimnames = list(models_here, models_here))
    diag(mat) <- "—"
    
    for (i in seq_len(nrow(dunn))) {
      g1 <- dunn$group1[i]
      g2 <- dunn$group2[i]
      mat[g1, g2] <- dunn$cell_label[i]
      mat[g2, g1] <- dunn$cell_label[i]
    }
    
    mat_df <- as.data.frame(mat, stringsAsFactors = FALSE)
    mat_df <- cbind(
      Category  = cat_name,
      Model     = rownames(mat_df),
      KW_global = c(kw_label, rep("", n_models - 1)),
      mat_df,
      stringsAsFactors = FALSE
    )
    rownames(mat_df) <- NULL
    
    all_rows <- c(all_rows, list(mat_df))
  }
  
  final_df <- bind_rows(all_rows)
  final_df[is.na(final_df)] <- ""
  
  # =============================================================================
  # Scrittura Excel con openxlsx
  # =============================================================================
  wb <- createWorkbook()
  addWorksheet(wb, "Comparisons")
  
  style_header <- createStyle(
    fontName = "Arial", fontSize = 10, fontColour = "#FFFFFF",
    fgFill = "#2F2F2F", halign = "CENTER", valign = "CENTER",
    textDecoration = "bold", wrapText = TRUE,
    border = "TopBottomLeftRight", borderColour = "#AAAAAA"
  )
  style_cat <- createStyle(
    fontName = "Arial", fontSize = 10, fontColour = "#FFFFFF",
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
  style_kw <- createStyle(
    fontName = "Arial", fontSize = 9, halign = "CENTER", valign = "CENTER",
    border = "TopBottomLeftRight", borderColour = "#CCCCCC"
  )
  style_cell_sig <- createStyle(
    fontName = "Arial", fontSize = 9, fontColour = "#C00000",
    halign = "CENTER", valign = "CENTER", textDecoration = "bold",
    border = "TopBottomLeftRight", borderColour = "#CCCCCC"
  )
  style_cell_ns <- createStyle(
    fontName = "Arial", fontSize = 9, fontColour = "#888888",
    halign = "CENTER", valign = "CENTER",
    border = "TopBottomLeftRight", borderColour = "#CCCCCC"
  )
  style_diag <- createStyle(
    fontName = "Arial", fontSize = 9, fontColour = "#AAAAAA",
    fgFill = "#E8E8E8", halign = "CENTER", valign = "CENTER",
    border = "TopBottomLeftRight", borderColour = "#CCCCCC"
  )
  
  writeData(wb, "Comparisons", final_df, startRow = 1, startCol = 1,
            headerStyle = style_header, borders = "all",
            borderColour = "#AAAAAA")
  
  n_rows         <- nrow(final_df)
  n_cols         <- ncol(final_df)
  model_cols_idx <- 4:n_cols
  
  addStyle(wb, "Comparisons", style_cat,
           rows = 2:(n_rows + 1), cols = 1, gridExpand = TRUE)
  addStyle(wb, "Comparisons", style_model,
           rows = 2:(n_rows + 1), cols = 2, gridExpand = TRUE)
  addStyle(wb, "Comparisons", style_kw,
           rows = 2:(n_rows + 1), cols = 3, gridExpand = TRUE)
  
  for (r in 2:(n_rows + 1)) {
    for (c in model_cols_idx) {
      val <- as.character(final_df[r - 1, c])
      if (val == "—") {
        addStyle(wb, "Comparisons", style_diag, rows = r, cols = c)
      } else if (val == "") {
        # cella vuota, nessuno stile
      } else if (val == "<0.001" || (!is.na(suppressWarnings(as.numeric(val))) && as.numeric(val) < 0.05)) {
        addStyle(wb, "Comparisons", style_cell_sig, rows = r, cols = c)
      } else {
        addStyle(wb, "Comparisons", style_cell_ns, rows = r, cols = c)
      }
    }
  }
  
  setColWidths(wb, "Comparisons", cols = 1, widths = 14)
  setColWidths(wb, "Comparisons", cols = 2, widths = 20)
  setColWidths(wb, "Comparisons", cols = 3, widths = 30)   # più larga per la label completa
  setColWidths(wb, "Comparisons", cols = model_cols_idx,
               widths = rep(18, length(model_cols_idx)))
  setRowHeights(wb, "Comparisons", rows = 1, heights = 30)
  setRowHeights(wb, "Comparisons", rows = 2:(n_rows + 1), heights = 18)
  
  freezePane(wb, "Comparisons", firstActiveRow = 2, firstActiveCol = 4)
  
  saveWorkbook(wb, output_file, overwrite = TRUE)
  cat("Saved:", output_file, "\n")
}

# =============================================================================
# Esegui per accuracy e score
# =============================================================================
build_comparison_table(df, "accuracy", "comparisons_accuracy.xlsx")
build_comparison_table(df, "score",    "comparisons_score.xlsx")

cat("Done!\n")