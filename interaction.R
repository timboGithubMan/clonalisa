needed <- c("ggplot2","dplyr","stringr","forcats","tidyr",
            "tibble","nlme","emmeans","broom.mixed","gridExtra", "codetools")

to_get <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_get))
    install.packages(to_get, repos = "https://cloud.r-project.org")
	
library(emmeans)
library(dplyr)
library(stringr)
library(forcats)
library(ggplot2)
library(nlme)
library(broom.mixed)
library(tibble)
library(gridExtra)
library(tidyr)

input_csv <- commandArgs(trailingOnly=TRUE)

data <- read.csv(input_csv) %>%
  mutate(well      = PlateWell,
         genotype    = factor(Group.Line),
         treatment = factor(Group.Treatment),
         treatment = relevel(treatment, "DMSO"),
         subgroup = well
  ) %>%
  rename(
    time     = "Relative.Time..hrs.",
    value    = cell_density
  )
  # filter(time < 80 | time > 100)
output_dir <- file.path(dirname(input_csv), "model_treatment_genotype", "logistic_growth_by_genotype")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

pdf(
  file      = file.path(output_dir, "growth_summary.pdf"),  # goes in output_dir
  width     = 11,            # 11 x 8.5 in  ⇒ landscape US-Letter
  height    = 8.5,
  onefile   = TRUE,          # keep adding pages
  paper     = "special"      # let width/height rule (alt: paper = "USr" or "a4r")
)
on.exit(dev.off(), add = TRUE)   # closes the device even if the script errors

data <- data %>% 
  arrange(subgroup, treatment, genotype, well, time) %>%          # make “first” reliable
  group_by(subgroup, treatment, genotype, well) %>% 
  mutate(
    Tref         = first(time),                # earliest absolute time in the well
    shifted_time = time - Tref
  ) %>% 
  # ── PER-WELL AVERAGE AT EACH TIME POINT ──────────────────────────────
  group_by(subgroup, treatment, genotype, well, shifted_time, time, Tref) %>% 
  summarise(
    value = mean(value, na.rm = TRUE),         # mean per well/time point
    .groups = "drop"
  ) %>% 
  # ── N0 = MEAN AT THE FIRST TIME POINT ────────────────────────────────
  arrange(subgroup, treatment, genotype, well, shifted_time, time, Tref) %>%  # ensure first row is earliest
  group_by(subgroup, treatment, genotype, well) %>%               
  mutate(N0 = first(value)) %>%                # first *mean* value in each well
  ungroup()

## BASIC QC ####################################################
# ---------------------------------------------------------------
# 1. Distribution of all N0 values (histogram + density overlay)
# ---------------------------------------------------------------
starting_density_filter_min <- 10000
starting_density_filter_max <- 200000

p_dist <- ggplot(data, aes(x = N0)) +
  geom_histogram(aes(y = after_stat(count)), bins = 50,
                 fill = "steelblue", colour = "white", alpha = 0.7) +
  # geom_density(aes(y = ..count..), linewidth = 1) +
  theme_minimal() +
  
  scale_x_log10() +
  geom_vline(
    xintercept = starting_density_filter_min,          # raw‐data value (before log10 transform)
    colour     = "red",
    linewidth  = 1,             # thickness (optional)
    linetype   = "solid"        # or "dashed", "dotdash", etc.
  ) +
  geom_vline(
    xintercept = starting_density_filter_max,          # raw‐data value (before log10 transform)
    colour     = "red",
    linewidth  = 1,             # thickness (optional)
    linetype   = "solid"        # or "dashed", "dotdash", etc.
  ) +
  labs(title = "QC filter for starting density (N0) - distribution across all wells",
       x = "N0",
       y = "Count")

p_dist          # print first plot to the active device
ggsave(file.path(output_dir, "qc_N0_distribution_histogram.pdf"),
       plot = p_dist, width = 8, height = 6)

data_unique_N0 <- data %>%           # 1. keep only shifted_time == 0
  filter(shifted_time == 0) %>%     
  mutate(                           # 2. shuffle the colour order for each subgroup
    subgroup = factor(subgroup,             #    (changes the order that ggplot assigns hues)
                  levels = sample(unique(subgroup)))
  )

library(scales)   # make sure this is loaded for label_number()

p_dist_facet <- ggplot(
  data_unique_N0,
  aes(x = N0, y = 0, colour = subgroup)
) +
  geom_jitter(
    width  = 0,
    height = 0.25,
    alpha  = 1,
    size   = 3
  ) +
  scale_x_log10(
    minor_breaks = rep(1:9, 20) * 10^rep(-9:10, each = 9),
    labels = label_number(accuracy = 1),   # <-- plain-number ticks,
  ) +
  facet_wrap(~treatment, ncol = 1, strip.position = "right") +
  guides(colour = "none") +
  labs(
    title  = "QC filter for starting density (N0) - Facet by treatment, Color by subgroup",
    x      = "Starting Density (N0)",
    y      = NULL,
    colour = "subgroup"
  ) +
  geom_vline(
    xintercept = starting_density_filter_min,          # raw‐data value (before log10 transform)
    colour     = "red",
    linewidth  = 1,             # thickness (optional)
    linetype   = "solid"        # or "dashed", "dotdash", etc.
  ) +
  geom_vline(
    xintercept = starting_density_filter_max,          # raw‐data value (before log10 transform)
    colour     = "red",
    linewidth  = 1,             # thickness (optional)
    linetype   = "solid"        # or "dashed", "dotdash", etc.
  ) +
  theme_minimal() +
  theme(
    strip.text.y.right = element_text(angle = 0),
    strip.placement    = "outside",
    # black gridlines on the x-axis
    panel.grid.major.x = element_line(linewidth = 0.5, colour = "darkgray"),
    panel.grid.minor.x = element_line(linewidth = 0.5, colour = "darkgray"),
    
    # keep y-axis grids/ticks/labels off
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank()
  )

p_dist_facet   # draw the plot
ggsave(file.path(output_dir, "qc_N0_distribution_facet_treatment.pdf"),
       plot = p_dist_facet, width = 8, height = 12)

data <- data %>% filter(N0 > starting_density_filter_min)
data <- data %>% filter(N0 < starting_density_filter_max)


logistic_growth <- function(time, K, r, N0) {
  K / (1 + ((K - N0) / N0) * exp(-r * time))
}

n_fix <-
  nlevels(data$genotype) * nlevels(data$treatment)  # start-value length
init_K <- max(data[["value"]], na.rm = TRUE) * 1.1
init_r <- 0.05

start_vals <- c(K = init_K, r = rep(init_r, n_fix))
full_fit <- try(
  nlme(
    value ~ logistic_growth(shifted_time, K, r, N0),
    data    = data,
    fixed   = list(K ~ 1, r ~ genotype * treatment),
    random  = list(subgroup = pdDiag(r ~ 1)),
    start   = start_vals,
    na.action = na.omit,
    control = nlmeControl(
      maxIter   = 1e3,
      msMaxIter = 1e3,
      msVerbose = TRUE
    ),
  ),
  silent = TRUE          
)

if (inherits(full_fit, "try-error")) {
  K_fixed <- 3700000
  message("Full nlme() fit failed:\n  ",
          attr(full_fit, "condition")$message,
          "\n…refitting with K fixed at ",
          K_fixed)
  
  logistic_growth_fixed <- function(time, r, N0)
    logistic_growth(time, K_fixed, r, N0)
  
  nlme_mod <- nlme(
    value  ~ logistic_growth_fixed(shifted_time, r, N0),
    data   = data,
    fixed  = r ~ genotype * treatment,           
    random = list(subgroup = pdDiag(r ~ 1)),
    start  = rep(init_r, n_fix),
    na.action = na.omit,
  )
} else {
  nlme_mod <- full_fit 
}

if ("K" %in% names(fixef(nlme_mod))) {
  K_est <- fixef(nlme_mod)["K"]   # free-estimated K
} else {
  K_est <- K_fixed                     # fixed K
}

combined_residuals <- data %>%
  mutate(
    fitted  = fitted(nlme_mod, level = 0),
    residual   = resid (nlme_mod, level = 0, type = "normalized"),
    treatment_genotype  = interaction(treatment, genotype,   # <-- treatment × genotype
                               sep = ":", drop = TRUE)
  ) %>%
  filter(shifted_time > 0)

# ── 1.  Residuals vs Fitted with slope -------------------------------------
lm_fit <- lm(residual ~ fitted, data = combined_residuals)
slope     <- coef(lm_fit)[2]
intercept <- coef(lm_fit)[1]

plot_resid_fitted <- ggplot(combined_residuals,
                            aes(fitted, residual)) +
  geom_point(alpha = .5, colour = "steelblue") +
  geom_abline(slope = slope, intercept = intercept,
              colour = "black", linewidth = .8) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  theme_minimal() +
  labs(title     = "Residuals vs fitted",
       subtitle  = sprintf("lm slope = %.4f", slope),
       x = "Fitted", y = "Residual")

# ── 2.  Residuals vs Time  --------------------------------------------------
plot_resid_time <- ggplot(combined_residuals,
                          aes(shifted_time, residual)) +
  geom_point(alpha = .5, colour = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  theme_minimal() +
  labs(title = "Residuals vs time", x = "Time (h)", y = "Residual")

# ── 3.  Q–Q plot  -----------------------------------------------------------
plot_resid_qq <- ggplot(combined_residuals,
                        aes(sample = residual)) +
  stat_qq(alpha = .4, size = .6) +
  stat_qq_line() +
  theme_minimal() +
  labs(title = "Residual Q–Q plot")

# ── 4.  Residuals vs Interaction (updated) --------------------------------
plot_resid_subgroup <- ggplot(combined_residuals,
                          aes(subgroup, residual)) +
  geom_jitter(alpha = .3, width = .25, colour = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  theme_minimal() +
  theme(axis.text.x  = element_blank(),      # remove labels
        axis.ticks.x = element_blank()) +    # remove tick marks
  labs(title = "Residuals vs subgroup", x = "subgroup", y = "Residual")

# ── 5.  Residuals vs treatment (updated) ---------------------------------------
plot_resid_treatment <- ggplot(combined_residuals,
                          aes(treatment, residual)) +
  geom_jitter(alpha = .3, width = .25, colour = "orange") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  theme_minimal() +
  facet_wrap(~genotype, ncol = 1) +                       # keep facets
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # 45° labels
  labs(title = "Residuals vs treatment", x = "treatment", y = "Residual")

# ── 6.  BLUPs for random effects -------------------------------------------
blup_df <- ranef(nlme_mod) %>%              # list element “subgroup”
  rownames_to_column("subgroup") %>% 
  rename(blup_r = `r.(Intercept)`) %>%
  mutate(subgroup = factor(subgroup, levels = subgroup))

plot_blup_subgroup <- ggplot(blup_df, aes(subgroup, blup_r)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random effect (subgroup)",
       x = "Subgroup", y = "BLUP for r")

# ── 6.  Arrange & export ----------------------------------------------------
grid_3x2 <- grid.arrange(plot_resid_fitted, plot_resid_time,
                         plot_resid_qq, plot_resid_subgroup,
                         plot_resid_treatment, plot_blup_subgroup,
                         ncol = 3)

ggsave(file.path(output_dir, "residual_diagnostics_grid.pdf"),
       grid_3x2, width = 13, height = 9)

# -------------------------------------------------------------
# 1. Estimated marginal means for r (one per genotype × treatment)
# -------------------------------------------------------------
emm <- emmeans(nlme_mod, ~ treatment | genotype,    # <- interaction model
               param = "r")
emm_df <- as.data.frame(emm)

# -------------------------------------------------------------
# 2. Identify the 0.00uM DMSO row in every genotype (= control)
# -------------------------------------------------------------
ctrl_df <- emm_df %>% 
  filter(treatment == "DMSO") %>%                 # the control level
  transmute(genotype,
            ctrl_emmean = emmean,
            ctrl_SE     = SE)

# -------------------------------------------------------------
# 3. Join controls back, do %‑diff, SE, p‑value, adjust
# -------------------------------------------------------------
emm_df_pct <- emm_df %>% 
  left_join(ctrl_df, by = "genotype") %>% 
  mutate(
    pct_diff    = 100 * (emmean / ctrl_emmean - 1),
    pct_diff_se = 100 * SE      / ctrl_emmean
  )

## ----- contrasts & multiplicity correction  --------------------
cts <- contrast(emm, "trt.vs.ctrl",          # treatment – 0.00uM DMSO inside genotype
                ref = "DMSO", by = "genotype")  # grouping factor

cts_df <- summary(cts, infer = TRUE,         # adds CI and p.value
                  adjust = "bonferroni") |>
  as.data.frame() |>
  mutate(
    treatment = str_remove(contrast, " - DMSO$"),   # "ANKRD12 - 0.00uM DMSO" → "ANKRD12"
    signif_label = case_when(
      p.value < 0.0001 ~ "****",
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE            ~ "ns"
    )
  ) |>
  select(genotype, treatment, signif_label)

## merge the labels back
emm_df_pct <- emm_df_pct |>
  left_join(cts_df, by = c("genotype", "treatment")) |>
  mutate(signif_label = if_else(treatment == "DMSO", "", signif_label))

# -------------------------------------------------------------
# 5. (optional) order treatments within each genotype by effect size
# -------------------------------------------------------------
emm_df_pct <- emm_df_pct %>% 
  group_by(genotype) %>% 
  mutate(treatment = fct_reorder(treatment, pct_diff)) %>% 
  ungroup()

##############################################################################
##  5)  Assemble per-well r components                ────────────────────────
##############################################################################
library(tibble)
library(dplyr)

## --- 5·a  Fixed part  -------------------------------------------------------
fe <- fixef(nlme_mod)
names(fe) <- sub("^r\\.", "", names(fe))      # drop the “r.” prefix

well_tbl <- data %>% 
  select(genotype, treatment, subgroup, well) %>% 
  distinct()

mm <- model.matrix(~ genotype * treatment, data = well_tbl)

# keep only the columns that survived in the fitted model
keep_cols <- intersect(colnames(mm), names(fe))
mm        <- mm[, keep_cols, drop = FALSE]
fe_vec    <- fe [ keep_cols ]

well_tbl$r_fixed <- as.numeric(mm %*% fe_vec)

## --- 5·b  Random intercepts (single “subgroup” grouping) -----------------------
re_subgroup_df <- ranef(nlme_mod) %>%          # <- now a data frame
  as.data.frame() %>%                      # columns: group, term, level, estimate
  filter(term == "r.(Intercept)") %>%      # keep the intercepts
  transmute(subgroup      = level,             # rename for the join
            r_subgroup_re = estimate)

## --- 5·c  Combine -----------------------------------------------------------
well_components <- well_tbl %>% 
  left_join(re_subgroup_df, by = "subgroup") %>% 
  mutate(
    r_subgroup_re      = coalesce(r_subgroup_re, 0),
    total_r_manual = r_fixed + r_subgroup_re
  )

##  Genotype main-effect ---------------------------------------------------
emm_genotype <- emmeans(nlme_mod, ~ genotype, param = "r")
emm_genotype_df <- as.data.frame(emm_genotype)

plot_genotype_effect <- ggplot(emm_genotype_df,
                               aes(x = genotype, y = emmean*2400)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = emmean*2400 - SE*2400, ymax = emmean*2400 + SE*2400),
                width = .25, linewidth = .9) +
  theme_classic(base_size = 12) +
  labs(title = "Growth Rate by Genotype",
       x = "genotype", y = "Percent Growth Per Day") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot_genotype_effect
ggsave(file.path(output_dir, "genotype_main_effect.pdf"),
       plot = plot_genotype_effect, width = 7, height = 5)

##  Treatment main-effect ---------------------------------------------------
emm_treatment <- emmeans(nlme_mod, ~ treatment, param = "r")
emm_treatment_df <- as.data.frame(emm_treatment)

plot_treatment_effect <- ggplot(emm_treatment_df,
                               aes(x = treatment, y = emmean*2400)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = emmean*2400 - SE*2400, ymax = emmean*2400 + SE*2400),
                width = .25, linewidth = .9) +
  theme_classic(base_size = 12) +
  labs(title = "Growth Rate by Treatment",
       x = "genotype", y = "Percent Growth Per Day") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot_treatment_effect
ggsave(file.path(output_dir, "treatment_main_effect.pdf"),
       plot = plot_treatment_effect, width = 7, height = 5)

##############################################################################
##  6)  Bar-plot: genotype-specific %Δr vs 0.00uM DMSO (fixed-effects model) ───────────
##############################################################################
# rebuild the factor with alphabetically-sorted levels
emm_df_pct <- emm_df_pct %>% 
  mutate(treatment = factor(treatment,               # keep the same data
                            levels = sort(levels(treatment))))  # new order
plot_treatment_pct <- ggplot(emm_df_pct,
                             aes(x   = treatment,
                                 y   = pct_diff,
                                 fill = genotype)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_errorbar(aes(ymin = pct_diff - pct_diff_se,
                    ymax = pct_diff + pct_diff_se),
                width = .3, size = .8,
                position = position_dodge(width = 0.9)) +
  geom_text(aes(label = signif_label,
                y = pct_diff + ifelse(pct_diff >= 0,
                                      pct_diff_se, -pct_diff_se) +
                  ifelse(pct_diff >= 0, 1, -2)),
            size = 8,
            position = position_dodge(width = 0.9),
            vjust = 0) +
  theme_classic(base_size = 12) +
  labs(title = "Growth-rate vs matched DMSO",
       x     = "treatment",
       y     = "Percent difference from matched DMSO (%)") +
  theme(axis.text.x  = element_text(angle = 45, hjust = 1),
        legend.position = "none")

plot_treatment_pct
ggsave(file.path(output_dir, "treatment_pct.pdf"),
       plot = plot_genotype_effect, width = 7, height = 5)

# ## ‣ 6·a   Growth-rate, r̂, for every subgroup (clone) ---------------------------
# subgroup_r_tbl <- well_components %>%                       # 1 row / well
#   group_by(subgroup, genotype, treatment) %>%                 # collapse
#   summarise(r_subgroup = mean(total_r_manual), .groups = "drop")
# 
# ## ‣ 6·b   genotype-specific 0.00uM DMSO reference  (one row per genotype) ---------------
# ctrl_r_tbl <- subgroup_r_tbl %>%                 # ← starts from the per-subgroup table
#   filter(treatment == "0.00uM DMSO") %>%             # keep the 0.00uM DMSO subgroups
#   group_by(genotype) %>%                       # one genotype at a time
#   summarise(r_ctrl = mean(r_subgroup),           # mean of all its 0.00uM DMSO clones
#             .groups = "drop")
# 
# ## ‣ 6·c   Percent-difference vs that 0.00uM DMSO ------------------------------------
# subgroup_pct_tbl <- subgroup_r_tbl %>%
#   left_join(ctrl_r_tbl, by = "genotype") %>%
#   mutate(pct_diff = 100 * (r_subgroup / r_ctrl - 1))
# 
# ## ‣ 6·d   Overlay the jitter-dodged points ----------------------------------
# plot_treatment_pct <- plot_treatment_pct +
#   geom_point(data = subgroup_pct_tbl,
#              aes(x      = treatment,
#                  y      = pct_diff),         # colour matches the bars
#              position = position_jitterdodge(
#                dodge.width  = 0.9,         # align with bars
#                jitter.width = 0.25),       # horizontal spread
#              size   = 2.2,
#              alpha  = 0.85,
#              shape  = 21,
#              stroke = 0.25)

##############################################################################
##  Treatment × Genotype interaction – annotate significance vs 0.00 µM DMSO
##############################################################################

## 1.  Get the estimated marginal means that emmip() would plot
##     (plotit = FALSE returns the data instead of the ggplot object)
emmip_df <- emmip(
  nlme_mod,
  genotype ~ treatment,
  param  = "r",
  CIs    = TRUE,
  plotit = FALSE               # <-- just give me the data frame
) |>
  as_tibble() |>
  mutate(
    emmean = yvar*2400,             # make names match earlier data frames
    SE     = SE*2400,
    LCL    = LCL*2400,
    UCL    = UCL*2400
  )

## 2.  Attach the “*, **, ***” labels that you already computed
label_df <- emmip_df |>
  left_join(cts_df, by = c("genotype", "treatment")) |>
  mutate(signif_label = replace_na(signif_label, ""))     # baseline → ""

## 3.  Build a fresh interaction plot with the labels
interaction_plot <- ggplot(
  label_df,
  aes(treatment, emmean,
      colour = genotype, group = genotype)
) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = LCL, ymax = UCL), width = .1, linewidth = 1) +
  geom_text(
    aes(y = UCL, label = signif_label),
    vjust = -0.4,            # move ~0.4 line-heights above the bar
    size  = 6,
    fontface = "bold",
    colour = "black"
  ) +
  theme_classic(base_size = 12) +
  labs(
    title = "Treatment Effect across Genotypes",
    x = "treatment",
    y = "Percent Growth per Day"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave(
  file.path(output_dir, "interaction_treatment_by_genotype_sig.pdf"),
  plot  = interaction_plot,
  width = 9, height = 6
)
interaction_plot

##############################################################################
##  7)  Predicted growth curves vs observed (per-well r̂)  ───────────────────
##############################################################################

# ── aggregate observed means per well/time ───────────────────────────────────
fit_data <- data %>% 
  group_by(subgroup, well, time, shifted_time, Tref) %>% 
  summarise(
    value = mean(value, na.rm = TRUE),
    N0    = mean(N0),
    .groups = "drop"
  ) %>% 
  left_join(
    well_components %>% select(subgroup, well, total_r_manual),
    by = c("subgroup", "well")
  )

# ── prediction grid per well ────────────────────────────────────────────────
pred_grid_well <- fit_data %>% 
  group_by(subgroup, well) %>% 
  summarise(
    min_time = 0,
    max_time = max(shifted_time),
    N0_well  = first(N0),
    r        = first(total_r_manual),
    .groups = "drop"
  ) %>% 
  rowwise() %>% 
  do({
    t_seq <- seq(.$min_time, .$max_time, length.out = 100)
    tibble(
      subgroup = .$subgroup,
      well = .$well,
      shifted_time = t_seq,
      pred = logistic_growth(t_seq, K_est, .$r, .$N0_well)
    )
  }) %>% 
  ungroup()

pred_grid_well <- pred_grid_well %>% 
  left_join(fit_data %>% distinct(subgroup, well, Tref),
            by = c("subgroup", "well")) %>% 
  mutate(time = shifted_time + Tref)

##############################################################################
##  7)  Predicted growth curves vs observed (per-well r̂)  ───────────────────
##############################################################################

# ── aggregate observed means per well/time ──────────────────────────────────
fit_data <- data %>% 
  group_by(                    # keep the meta-columns!
    treatment, genotype,       #  ← NEW
    subgroup, well,
    time, shifted_time, Tref
  ) %>% 
  summarise(
    value = mean(value, na.rm = TRUE),
    N0    = mean(N0),
    .groups = "drop"
  ) %>% 
  left_join(
    well_components %>% 
      select(treatment, genotype, subgroup, well, total_r_manual),  # ← NEW
    by = c("treatment", "genotype", "subgroup", "well")             # ← NEW
  )

# ── prediction grid per well ────────────────────────────────────────────────
pred_grid_well <- fit_data %>% 
  group_by(treatment, genotype, subgroup, well) %>%   # ← keep meta-cols
  summarise(
    min_time = 0,
    max_time = max(shifted_time),
    N0_well  = first(N0),
    r        = first(total_r_manual),
    .groups = "drop"
  ) %>% 
  rowwise() %>% 
  do({
    t_seq <- seq(.$min_time, .$max_time, length.out = 100)
    tibble(
      treatment = .$treatment,          # ← copy down
      genotype  = .$genotype,           # ← copy down
      subgroup  = .$subgroup,
      well      = .$well,
      shifted_time = t_seq,
      pred = logistic_growth(t_seq, K_est, .$r, .$N0_well)
    )
  }) %>% 
  ungroup() %>% 
  left_join(
    fit_data %>% 
      distinct(treatment, genotype, subgroup, well, Tref),
    by = c("treatment", "genotype", "subgroup", "well")
  ) %>% 
  mutate(time = shifted_time + Tref)

##############################################################################
##  8·b  Plot: linear & log scales  ──────────────────────────────────────────
##############################################################################
library(ggplot2)

facet_end_medians <- fit_data %>%                       # one row per facet
  group_by(genotype, treatment) %>%                   # ⬅ facets
  filter(time == max(time, na.rm = TRUE)) %>%         # latest time-point
  summarise(median_latest = median(value, na.rm = TRUE),  # mean across wells
            .groups = "drop")

hline_vals <- c(                                      # two values to plot
  min(facet_end_medians$median_latest, na.rm = TRUE),     # “lowest” facet
  max(facet_end_medians$median_latest, na.rm = TRUE)      # “greatest” facet
)

## 1. find a global y-range -------------------------------------------------
y_range <- range(
  c(pred_grid_well$pred,
    fit_data$value,
    fit_data$N0,
    hline_vals),            # include the new lines so they’re never clipped
  na.rm = TRUE
)

## 2. linear scale ----------------------------------------------------------
curves_linear <- ggplot() +
  geom_line(
    data  = pred_grid_well,
    aes(time, pred, group = well),
    alpha = 0.5
  ) +
  geom_point(
    data  = fit_data,
    aes(time, value),
    size  = 1, alpha = 0.6, colour = "red"
  ) +
  geom_point(
    data  = fit_data,
    aes(Tref, N0),
    size  = 1, alpha = 0.6
  ) +
  geom_hline(                              # <-- both reference lines
    data = data.frame(yint = hline_vals),
    aes(yintercept = yint),
    linetype = "dashed", colour = "red"
  ) +
  facet_grid(
    rows = vars(genotype),
    cols = vars(treatment)
  ) +
  coord_cartesian(ylim = y_range) +
  theme_minimal() +
  labs(
    title = "Predicted vs Observed Growth Curves per Well",
    x = "Time (hrs)",
    y = "Cell Density"
  )

## 3. log scale -------------------------------------------------------------
log_range <- y_range
log_range[1] <- ifelse(log_range[1] <= 0,
                       min(pred_grid_well$pred[pred_grid_well$pred > 0]),
                       log_range[1])

curves_log <- curves_linear +
  scale_y_log10(limits = log_range) +
  labs(title = "Predicted vs Observed Growth Curves per Well (log-scale)")

## 4. save & print ----------------------------------------------------------
ggsave(file.path(output_dir, "predicted_vs_observed_linear.pdf"),
       plot = curves_linear, width = 12, height = 9)

ggsave(file.path(output_dir, "predicted_vs_observed_log.pdf"),
       plot = curves_log, width = 12, height = 9)

print(curves_linear)
print(curves_log)