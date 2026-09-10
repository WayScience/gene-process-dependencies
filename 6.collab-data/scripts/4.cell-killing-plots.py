suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(tidyr)
  library(ggplot2)
  library(MESS)
  library(drc)
  library(ggrepel)
  library(colorspace)
  library(patchwork)
  library(cowplot)
})

select <- dplyr::select
filter <- dplyr::filter

cell_killing_file <- file.path("./data/cell-killing.csv")
raw <- read_csv(cell_killing_file, col_names = FALSE, show_col_types = FALSE)
colnames(raw) <- c("V1","V2","V3","V4","V5")
 
results <- list()
current_sample <- NA_character_
current_drug   <- NA_character_
 
for (i in seq_len(nrow(raw))) {
 
  if (!is.na(raw$V1[i])) {
    current_sample <- raw$V1[i]
    current_drug   <- raw$V2[i]
    next
  }
 
  if (!is.na(raw$V2[i]) && grepl("nM", raw$V2[i], ignore.case = TRUE)) {
    current_drug <- raw$V2[i]
    next
  }
 
  conc <- suppressWarnings(as.numeric(raw$V2[i]))
 
  if (!is.na(conc)) {
    results[[length(results) + 1]] <- tibble(
      sample = current_sample,
      drug = current_drug,
      concentration_nM = conc,
      surviving_fraction = mean(
        c(
          suppressWarnings(as.numeric(raw$V3[i])),
          suppressWarnings(as.numeric(raw$V4[i])),
          suppressWarnings(as.numeric(raw$V5[i]))
        ),
        na.rm = TRUE
      )
    )
  }
}
 
cell_killing_long <- bind_rows(results)
 
# Sanity check: every (sample, drug) should have ~10 dose points
stopifnot(nrow(cell_killing_long) > 0)
message("Parsed ", nrow(cell_killing_long), " concentration/SF rows across ",
        n_distinct(cell_killing_long$sample), " samples and ",
        n_distinct(cell_killing_long$drug), " raw drug labels.")
print(cell_killing_long %>% count(sample, drug))

final_drug_file <- file.path("./results/final_drug_w_paths.csv")
final_drug_w_paths <- read_csv(final_drug_file)
final_drug_w_paths <- final_drug_w_paths %>%
  mutate(ModelID = if_else(ModelID == "GSM7305243", "MAF-868", ModelID))

cell_killing_df <- cell_killing_long %>%
  arrange(sample, drug, concentration_nM) %>%
  group_by(sample, drug) %>%
  mutate(
    x = log10(concentration_nM + 1),
    x_scaled = (x - min(x)) / (max(x) - min(x))
  ) %>%
  summarize(
    auc = MESS::auc(x = x_scaled, y = surviving_fraction),
    .groups = "drop"
  )


clean_sample <- function(x) {
  x <- str_remove(x, "\\s*\\(.*\\)")     # remove "(WT)", "(M)"
  x <- str_remove(x, "_SHC\\d+")          # remove "_SHC202" etc.
  x <- str_trim(x)
  x <- str_replace(x, "^MAF868$", "MAF-868")  # explicit fix for the hyphen
  x
}
 
clean_drug <- function(x) {
  x_lower <- str_to_lower(x)
  case_when(
    str_detect(x_lower, "deazaneplanocin") ~ "3-deazaneplanocin-a",
    str_detect(x_lower, "axitinib")        ~ "axitinib",
    str_detect(x_lower, "cladribine")      ~ "cladribine",
    TRUE ~ str_squish(str_remove(x_lower, "\\s*\\(.*\\)"))
  )
}
 
cell_killing_df <- cell_killing_df %>%
  mutate(
    sample = clean_sample(sample),
    drug   = clean_drug(drug)
  ) %>%
  rename(ModelID = sample, name = drug)
 
final_drug_w_paths <- final_drug_w_paths %>%
  mutate(
    ModelID = clean_sample(ModelID),
    name    = clean_drug(name)
  )

message("\nDistinct (ModelID, name) in cell_killing_df not found in final_drug_w_paths:")
unmatched <- anti_join(cell_killing_df, final_drug_w_paths, by = c("ModelID", "name")) %>%
  filter(name %in% c("3-deazaneplanocin-a", "cladribine", "axitinib")) %>%
  select(ModelID, name)
print(unmatched)

plot_df <- cell_killing_df %>%
  left_join(final_drug_w_paths, by = c("ModelID", "name")) %>%
  filter(name %in% c("3-deazaneplanocin-a", "cladribine", "axitinib")) %>%
  mutate(
    one_minus_auc = 1 - auc,
    drug = factor(name,
      levels = c("3-deazaneplanocin-a", "cladribine", "axitinib"),
      labels = c("3-Deazaneplanocin-A", "Cladribine", "Axitinib")
    ),
    has_latent = !is.na(latent_score)
  )
 
message("\nFinal plot_df row counts per drug (non-NA latent_score):")
print(plot_df %>% filter(!is.na(latent_score)) %>% count(drug))

plot_df$ModelID <- factor(plot_df$ModelID)
samples <- levels(plot_df$ModelID)
sample_colors <- setNames(
  qualitative_hcl(length(samples), palette = "Pastel 1"),
  samples
)

make_panel <- function(data, y_var, y_label, is_log = FALSE) {
  d <- data %>% filter(!is.na(latent_score), !is.na(.data[[y_var]]))
 
  if (nrow(d) < 2) {
    # Not enough points to fit a line -- return an empty/annotated panel
    # instead of letting lm() error out
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5,
                 label = paste0("Insufficient data\n(n=", nrow(d), ")"),
                 size = 3, color = "#999999") +
        theme_void()
    )
  }
 
  fit  <- lm(as.formula(paste(y_var, "~ latent_score")), data = d)
  r2   <- summary(fit)$r.squared
  
  annot <- paste0("R\u00b2 = ", round(r2, 2))
 
  p <- ggplot(d, aes(x = latent_score, y = .data[[y_var]], fill = ModelID)) +
    geom_smooth(aes(group = 1), method = "lm", se = FALSE,
                color = "#272bf3", linewidth = 1.2, linetype = "dashed") +
    geom_point(shape = 21, size = 6, stroke = 0.8, color = "black") +
    geom_text_repel(
        aes(label = ModelID),
        size = 5,
        color = "#333333",

        #  key spacing controls
        box.padding = 1,
        point.padding = 0.8,

        # stronger separation
        force = 1.5,
        force_pull = 0.3,

        # allow more movement before stopping
        max.overlaps = Inf,

        # cleaner aesthetics
        segment.color = "#BBBBBB",
        segment.size = 0.4,

        # helps spread labels more evenly
        min.segment.length = 0
      ) +
    annotate("label", x = Inf, y = Inf, label = annot,
             hjust = 1.1, vjust = 13, size = 5,
             color = "#1a1717", fill = "white",
             label.size = 0.3, label.padding = unit(0.3, "lines")) +
    scale_fill_manual(values = sample_colors) +
    labs(x = "Latent Score", y = y_label, fill = "Sample") +
    theme_minimal(base_size = 18) +
    theme(
      panel.background = element_rect(fill = "white", color = "#DDDDDD"),
      plot.background  = element_rect(fill = "#F8F9FA", color = NA),
      panel.grid.major = element_line(color = "#CCCCCC", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 18, color = "#555555"),
      axis.text  = element_text(size = 15, color = "#444444"),
      legend.position = "none"
    )
 
  if (is_log) {
    p <- p + scale_y_continuous(labels = function(v) paste0("1/", round(10^(-v), 0)))
  }
 
  p
}


drugs_display <- c("3-Deazaneplanocin-A", "Cladribine", "Axitinib")
 
auc_panels <- map(drugs_display, \(d) {
  make_panel(filter(plot_df, drug == d), "one_minus_auc",
             "1 \u2013 AUC (higher = more sensitive)") +
    ggtitle(d) +
    theme(plot.title = element_text(face = "bold", size = 20, color = "#1A1A2E"))
})
 


auc_row <- wrap_plots(auc_panels, ncol = 3)

final_plot <- auc_row +
  plot_annotation(
    title = "Drug Latent Score vs. Cell-Killing Activity",
    subtitle = "Higher values = greater sensitivity",
    theme = theme(
      plot.title = element_text(face = "bold", size = 25, color = "#1A1A2E"),
      plot.subtitle = element_text(size = 18, color = "#666666"),
      plot.background = element_rect(fill = "#F8F9FA", color = NA)
    )
  )
 
ggsave("./visualize/drug_latent_vs_killing.png", final_plot,
       width = 15, height = 7, dpi = 180, bg = "#F8F9FA")
 
message("Saved: drug_latent_vs_killing.png")