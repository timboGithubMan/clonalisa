#!/usr/bin/env Rscript
###############################################################################
##  Generic logistic-growth analysis – full script with corrected ggsave()
##
##  USAGE
##    Rscript growth_generic.R  <input_csv>  <fe1>  [fe2]  [interaction_flag]  [ref_level]
##
##      input_csv         : full path to *.csv
##      fe1               : column name for the first fixed-effect factor
##      fe2   (optional)  : column name for the second fixed-effect factor
##      interaction_flag  : 1 / 0  → include the fe1*fe2 interaction (default 0)
##      ref_level         : reference level inside fe2 (e.g. “DMSO”)
###############################################################################

## ─────────────────────────────────────────────────────────────────────────────
## 0)  Parse command-line arguments
## ─────────────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)

input_csv        <- args[1]
fe1_name         <- if (length(args) >= 2) args[2] else
                      stop("Supply a column name for fe1.")
fe2_name         <- if (length(args) >= 3) args[3] else ""
interaction_flag <- if (length(args) >= 4) as.logical(as.integer(args[4])) else FALSE
ref_level        <- if (length(args) >= 5) args[5] else ""

cat("\n###  ARGUMENT SUMMARY  ##############################################\n",
    "CSV file          : ", input_csv, "\n",
    "Fixed effect 1    : ", fe1_name,  "\n",
    "Fixed effect 2    : ", ifelse(fe2_name == "", "(none)", fe2_name), "\n",
    "Interaction flag  : ", interaction_flag, "\n",
    "Reference level   : ", ifelse(ref_level == "", "(none)", ref_level), "\n",
    "#######################################################################\n\n")

## ─────────────────────────────────────────────────────────────────────────────
## 1)  Load / install required packages
## ─────────────────────────────────────────────────────────────────────────────
needed <- c("ggplot2","dplyr","stringr","forcats","tidyr","tibble","nlme",
            "emmeans","broom.mixed","gridExtra","codetools","scales")

to_get <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_get))
    install.packages(to_get, repos = "https://cloud.r-project.org")
invisible(lapply(needed, library, character.only = TRUE))

## ─────────────────────────────────────────────────────────────────────────────
## 1·a)  Helper – alphanumeric levels (letters before numbers)
## ─────────────────────────────────────────────────────────────────────────────
alpha_levels <- function(x) {
  lv <- unique(as.character(x))
  letters <- lv[stringr::str_detect(lv, "^[A-Za-z]")]
  digits  <- lv[stringr::str_detect(lv, "^[0-9]")]
  others  <- setdiff(lv, c(letters, digits))
  c(stringr::str_sort(letters, numeric = TRUE),
    stringr::str_sort(digits,  numeric = TRUE),
    others)
}

## ─────────────────────────────────────────────────────────────────────────────
## 2)  Read CSV  ▸ strip Group.* prefixes
## ─────────────────────────────────────────────────────────────────────────────
data <- read.csv(input_csv, check.names = FALSE)
names(data) <- sub("^Group[.-]", "", names(data))

cat("Columns in the CSV:\n"); print(colnames(data)); cat("\n")

required_base <- c("PlateWell", "Relative Time (hrs)", "cell_density")
missing_base  <- setdiff(required_base, colnames(data))
if (length(missing_base))
    stop("Missing required columns: ", paste(missing_base, collapse = ", "))

for (nm in c(fe1_name, fe2_name)) {
  if (nzchar(nm) && !nm %in% colnames(data))
      stop("Column \"", nm, "\" not found.")
}

## ─────────────────────────────────────────────────────────────────────────────
## 3)  Canonical columns & factors
## ─────────────────────────────────────────────────────────────────────────────
data <- data %>%
  mutate(
    well     = PlateWell,
    fe1      = factor(.data[[fe1_name]], levels = alpha_levels(.data[[fe1_name]])),
    subgroup = well
  )

if (fe2_name != "") {
  data <- data %>%
  mutate(fe2 = factor(.data[[fe2_name]],
                       levels = alpha_levels(.data[[fe2_name]])))
  if (ref_level != "" && ref_level %in% levels(data$fe2))
      data$fe2 <- relevel(data$fe2, ref = ref_level)
} else {
  data$fe2 <- factor("ALL")
  interaction_flag <- FALSE
}

data <- data %>%
  rename(time = "Relative Time (hrs)",
         value = cell_density)

## ─────────────────────────────────────────────────────────────────────────────
## 4)  Output directory & PDF sink
## ─────────────────────────────────────────────────────────────────────────────
tag <- paste0("logistic_", fe1_name,
              if (fe2_name != "") paste0("_", fe2_name))
output_dir <- file.path(dirname(input_csv), "model_outputs", tag)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

pdf(file = file.path(output_dir, "growth_summary.pdf"),
    width = 11, height = 8.5, onefile = TRUE, paper = "special")
on.exit(dev.off(), add = TRUE)

## ─────────────────────────────────────────────────────────────────────────────
## 5)  Pre-processing
## ─────────────────────────────────────────────────────────────────────────────
data <- data %>%
  arrange(subgroup, fe2, fe1, well, time) %>%
  group_by(subgroup, fe2, fe1, well) %>%
  mutate(Tref = first(time),
         shifted_time = time - Tref) %>%
  group_by(subgroup, fe2, fe1, well,
           shifted_time, time, Tref) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  arrange(subgroup, fe2, fe1, well, shifted_time) %>%
  group_by(subgroup, fe2, fe1, well) %>%
  mutate(N0 = first(value)) %>%
  ungroup()

## ── QC plots ────────────────────────────────────────────────────────────────
N0_min <- 1e4; N0_max <- 2e5

p_hist <- ggplot(data, aes(N0)) +
  geom_histogram(bins = 50, fill = "steelblue", colour = "white", alpha = .7) +
  scale_x_log10() +
  geom_vline(xintercept = c(N0_min, N0_max), colour = "red") +
  theme_minimal() +
  labs(title = "Starting density (N0) histogram")
print(p_hist)
ggsave(file.path(output_dir, "qc_N0_histogram.pdf"),
       plot = p_hist, width = 8, height = 6)

data_unique_N0 <- data %>%
  filter(shifted_time == 0) %>%
  mutate(subgroup = factor(subgroup, levels = sample(unique(subgroup))))

p_facet <- ggplot(data_unique_N0,
                  aes(N0, 0, colour = subgroup)) +
  geom_jitter(height = 0.25, size = 3) +
  scale_x_log10(minor_breaks = rep(1:9, 20) * 10^rep(-9:10, each = 9),
                labels = label_number(accuracy = 1)) +
  facet_wrap(~fe2, ncol = 1, strip.position = "right") +
  guides(colour = "none") +
  geom_vline(xintercept = c(N0_min, N0_max), colour = "red") +
  theme_minimal() +
  labs(title = "QC filter for starting density (N0)",
       x = "N0") +
  theme(strip.text.y.right = element_text(angle = 0),
        panel.grid.major.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
print(p_facet)
ggsave(file.path(output_dir, "qc_N0_facet.pdf"),
       plot = p_facet, width = 8, height = 12)

data <- data %>% filter(N0 > N0_min, N0 < N0_max)

## ─────────────────────────────────────────────────────────────────────────────
## 6)  NLME logistic-growth model
## ─────────────────────────────────────────────────────────────────────────────
logistic_growth <- function(t, K, r, N0)
  K / (1 + ((K - N0)/N0)*exp(-r*t))

rhs <- if (interaction_flag) "fe1*fe2" else
         paste(c("fe1", if (fe2_name != "") "fe2"), collapse = " + ")
design_formula <- as.formula(paste("~", rhs))
r_formula      <- as.formula(paste("r ~", rhs))

init_K  <- max(data$value, na.rm = TRUE) * 1.1
init_r  <- 0.05
n_fix   <- ncol(model.matrix(design_formula, data))
start_vals <- c(K = init_K, r = rep(init_r, n_fix))

try_full <- try(nlme(
  value ~ logistic_growth(shifted_time, K, r, N0),
  data    = data,
  fixed   = list(K ~ 1, r_formula),
  random  = list(subgroup = pdDiag(r ~ 1)),
  start   = start_vals,
  na.action = na.omit,
  control = nlmeControl(maxIter = 1e3, msMaxIter = 1e3, msVerbose = TRUE)
), silent = TRUE)

if (inherits(try_full, "try-error")) {
  K_fixed <- 3.7e6
  cat("Full nlme() failed – refitting with fixed K =", K_fixed, "\n")
  logistic_growth_fixed <- function(t, r, N0)
      logistic_growth(t, K_fixed, r, N0)
  nlme_mod <- nlme(
    value  ~ logistic_growth_fixed(shifted_time, r, N0),
    data   = data,
    fixed  = r_formula,
    random = list(subgroup = pdDiag(r ~ 1)),
    start  = rep(init_r, n_fix),
    na.action = na.omit
  )
  K_est <- K_fixed
} else {
  nlme_mod <- try_full
  K_est    <- fixef(nlme_mod)["K"]
}

## ─────────────────────────────────────────────────────────────────────────────
## 7)  Residual-diagnostic grid
## ─────────────────────────────────────────────────────────────────────────────
comb <- data %>%
  mutate(fitted = fitted(nlme_mod, level = 0),
         residual = resid(nlme_mod, level = 0, type = "normalized"),
         fe2 = fe2) %>%
  filter(shifted_time > 0)

lm_fit <- lm(residual ~ fitted, data = comb)

plot_resid_fit <- ggplot(comb, aes(fitted, residual)) +
  geom_point(alpha = .5, colour = "steelblue") +
  geom_abline(slope = lm_fit$coefficients[2],
              intercept = lm_fit$coefficients[1]) +
  geom_hline(yintercept = 0, colour = "red", linetype = "dashed") +
  theme_minimal() + labs(title = "Residuals vs fitted")

plot_resid_time <- ggplot(comb, aes(shifted_time, residual)) +
  geom_point(alpha = .5, colour = "darkgreen") +
  geom_hline(yintercept = 0, colour = "red", linetype = "dashed") +
  theme_minimal() + labs(title = "Residuals vs time")

plot_resid_qq <- ggplot(comb, aes(sample = residual)) +
  stat_qq(alpha = .4) + stat_qq_line() +
  theme_minimal() + labs(title = "Residual Q–Q")

plot_resid_subgroup <- ggplot(comb, aes(subgroup, residual)) +
  geom_jitter(alpha = .3, width = .25, colour = "purple") +
  geom_hline(yintercept = 0, colour = "red", linetype = "dashed") +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  labs(title = "Residuals vs subgroup")

plot_resid_fe2 <- ggplot(comb, aes(fe2, residual)) +
  geom_jitter(alpha = .3, width = .25, colour = "orange") +
  geom_hline(yintercept = 0, colour = "red", linetype = "dashed") +
  theme_minimal() + facet_wrap(~fe1, ncol = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Residuals vs second factor")

blup_df <- ranef(nlme_mod) %>% rownames_to_column("subgroup") %>%
  rename(blup_r = `r.(Intercept)`)

plot_blup <- ggplot(blup_df, aes(subgroup, blup_r)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random effect (subgroup)", y = "BLUP for r")

grid_3x2 <- grid.arrange(plot_resid_fit, plot_resid_time, plot_resid_qq,
                         plot_resid_subgroup, plot_resid_fe2, plot_blup, ncol = 3)
ggsave(file.path(output_dir, "residual_diagnostics_grid.pdf"),
       plot = grid_3x2, width = 13, height = 9)

## ─────────────────────────────────────────────────────────────────────────────
## 8)  EMMs, contrasts, %-difference vs reference
## ─────────────────────────────────────────────────────────────────────────────
if (fe2_name != "" && ref_level != "") {
  emm <- emmeans(nlme_mod, as.formula(paste("~ fe2 | fe1")), param = "r")
  emm_df <- as.data.frame(emm)

  ctrl_df <- emm_df %>% filter(fe2 == ref_level) %>%
    transmute(fe1, ctrl_emmean = emmean, ctrl_SE = SE)

  emm_df_pct <- emm_df %>% left_join(ctrl_df, by = "fe1") %>%
    mutate(pct_diff = 100 * (emmean / ctrl_emmean - 1),
           pct_diff_se = 100 * SE / ctrl_emmean)

  cts_df <- contrast(emm, "trt.vs.ctrl",
                     ref = ref_level, by = "fe1") %>%
    summary(infer = TRUE, adjust = "bonferroni") %>%
    as.data.frame() %>%
    mutate(fe2 = stringr::str_remove(contrast, paste0(" - ", ref_level,"$")),
           signif_label = dplyr::case_when(
             p.value < 0.0001 ~ "****",
             p.value < 0.001  ~ "***",
             p.value < 0.01   ~ "**",
             p.value < 0.05   ~ "*",
             TRUE             ~ "ns"
           ))

  emm_df_pct <- emm_df_pct %>%
    left_join(cts_df %>% select(fe1, fe2, signif_label),
              by = c("fe1", "fe2")) %>%
    mutate(signif_label = dplyr::coalesce(signif_label, "")) %>%
    ungroup()

  plot_pct <- ggplot(emm_df_pct,
                     aes(fe2, pct_diff, fill = fe1)) +
    geom_col(position = position_dodge(.9)) +
    geom_errorbar(aes(ymin = pct_diff - pct_diff_se,
                      ymax = pct_diff + pct_diff_se),
                  width = .3,
                  position = position_dodge(.9)) +
    geom_text(aes(label = signif_label,
                  y = pct_diff + ifelse(pct_diff >= 0,
                                        pct_diff_se, -pct_diff_se) +
                        ifelse(pct_diff >= 0, 1, -2)),
              position = position_dodge(.9), vjust = 0, size = 5) +
    theme_classic(base_size = 12) +
    labs(title = paste("Growth-rate vs", ref_level),
         x = fe2_name,
         y = paste("Percent difference vs", ref_level, "(%)")) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")
  print(plot_pct)
  ggsave(file.path(output_dir, "percent_diff_vs_ref.pdf"),
         plot = plot_pct, width = 7, height = 5)
}

## ─────────────────────────────────────────────────────────────────────────────
## 9)  Main-effect bar plots
## ─────────────────────────────────────────────────────────────────────────────
emm_fe1 <- emmeans(nlme_mod, ~ fe1, param = "r")
plot_fe1 <- ggplot(as.data.frame(emm_fe1),
                   aes(fe1, emmean*2400)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = emmean*2400 - SE*2400,
                    ymax = emmean*2400 + SE*2400),
                width = .25) +
  theme_classic(base_size = 12) +
  labs(title = paste("Growth rate by", fe1_name),
       x = fe1_name, y = "Percent growth per day") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(plot_fe1)
ggsave(file.path(output_dir, paste0(fe1_name, "_main.pdf")),
       plot = plot_fe1, width = 7, height = 5)

if (fe2_name != "") {
  emm_fe2 <- emmeans(nlme_mod, ~ fe2, param = "r")
  plot_fe2 <- ggplot(as.data.frame(emm_fe2),
                     aes(fe2, emmean*2400)) +
    geom_col(fill = "steelblue") +
    geom_errorbar(aes(ymin = emmean*2400 - SE*2400,
                      ymax = emmean*2400 + SE*2400),
                  width = .25) +
    theme_classic(base_size = 12) +
    labs(title = paste("Growth rate by", fe2_name),
         x = fe2_name, y = "Percent growth per day") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(plot_fe2)
  ggsave(file.path(output_dir, paste0(fe2_name, "_main.pdf")),
         plot = plot_fe2, width = 7, height = 5)
}

## ─────────────────────────────────────────────────────────────────────────────
## 10)  Interaction plot with labels
## ─────────────────────────────────────────────────────────────────────────────
if (fe2_name != "" && ref_level != "") {
  emmip_df <- emmip(nlme_mod, fe1 ~ fe2, param = "r",
                    CIs = TRUE, plotit = FALSE) %>% as_tibble() %>%
    mutate(emmean = yvar*2400, SE = SE*2400, LCL = LCL*2400, UCL = UCL*2400)

  label_df <- emmip_df %>%
    left_join(cts_df, by = c("fe1", "fe2")) %>%
    mutate(signif_label = replace_na(signif_label, ""))

  interaction_plot <- ggplot(label_df,
                             aes(fe2, emmean, colour = fe1, group = fe1)) +
    geom_line(linewidth = 2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = LCL, ymax = UCL),
                  width = .1, linewidth = 1) +
    geom_text(aes(y = UCL, label = signif_label),
              vjust = -0.4, size = 6, colour = "black") +
    theme_classic(base_size = 12) +
    labs(title = paste(fe2_name, "effect across", fe1_name),
         x = fe2_name, y = "Percent growth per day",
		 colour = fe1_name) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(interaction_plot)
  ggsave(file.path(output_dir, "interaction_plot.pdf"),
         plot = interaction_plot, width = 9, height = 6)
}

## ─────────────────────────────────────────────────────────────────────────────
## 11)  Per-well r̂ components
## ─────────────────────────────────────────────────────────────────────────────
fe <- fixef(nlme_mod); names(fe) <- sub("^r\\.", "", names(fe))

well_tbl <- data %>% select(fe1, fe2, subgroup, well) %>% distinct()
mm <- model.matrix(~ fe1 * fe2, data = well_tbl)
keep_cols <- intersect(colnames(mm), names(fe))
well_tbl$r_fixed <- as.numeric(mm[, keep_cols, drop = FALSE] %*% fe[keep_cols])

re_subgroup_df <- ranef(nlme_mod) %>% as.data.frame() %>%
  filter(term == "r.(Intercept)") %>%
  transmute(subgroup = level, r_subgroup_re = estimate)

well_components <- well_tbl %>%
  left_join(re_subgroup_df, by = "subgroup") %>%
  mutate(r_subgroup_re = coalesce(r_subgroup_re, 0),
         total_r_manual = r_fixed + r_subgroup_re)

## ─────────────────────────────────────────────────────────────────────────────
## 12)  Predicted vs observed curves
## ─────────────────────────────────────────────────────────────────────────────
fit_data <- data %>%
  group_by(fe1, fe2, subgroup, well, time, shifted_time, Tref) %>%
  summarise(value = mean(value, na.rm = TRUE),
            N0 = mean(N0), .groups = "drop") %>%
  left_join(well_components %>% select(fe1, fe2, subgroup, well, total_r_manual),
            by = c("fe1","fe2","subgroup","well"))

pred_grid_well <- fit_data %>%
  group_by(fe1, fe2, subgroup, well) %>%
  summarise(min_time = 0,
            max_time = max(shifted_time),
            N0_well = first(N0),
            r = first(total_r_manual), .groups = "drop") %>%
  rowwise() %>%
  do({
    ts <- seq(.$min_time, .$max_time, length.out = 100)
    tibble(fe1 = .$fe1, fe2 = .$fe2, subgroup = .$subgroup, well = .$well,
           shifted_time = ts,
           pred = logistic_growth(ts, K_est, .$r, .$N0_well))
  }) %>% ungroup() %>%
  left_join(fit_data %>% distinct(fe1, fe2, subgroup, well, Tref),
            by = c("fe1","fe2","subgroup","well")) %>%
  mutate(time = shifted_time + Tref)

facet_end_medians <- fit_data %>%
  group_by(fe1, fe2) %>%
  filter(time == max(time)) %>%
  summarise(median_latest = median(value), .groups = "drop")
hline_vals <- range(facet_end_medians$median_latest, na.rm = TRUE)

y_range <- range(c(pred_grid_well$pred, fit_data$value,
                   fit_data$N0, hline_vals), na.rm = TRUE)

curves_linear <- ggplot() +
  geom_line(data = pred_grid_well, aes(time, pred, group = well), alpha = .5) +
  geom_point(data = fit_data, aes(time, value), size = 1,
             colour = "red", alpha = .6) +
  geom_point(data = fit_data, aes(Tref, N0), size = 1, alpha = .6) +
  geom_hline(data = data.frame(yint = hline_vals),
             aes(yintercept = yint), colour = "red", linetype = "dashed") +
  facet_grid(rows = vars(fe1), cols = vars(fe2)) +
  coord_cartesian(ylim = y_range) +
  theme_minimal() +
  labs(title = "Predicted vs Observed Growth Curves per Well",
       x = "Time (hrs)", y = "Cell Density")
print(curves_linear)

log_range <- y_range
log_range[1] <- ifelse(log_range[1] <= 0,
                       min(pred_grid_well$pred[pred_grid_well$pred > 0]),
                       log_range[1])

curves_log <- curves_linear +
  scale_y_log10(limits = log_range) +
  labs(title = "Predicted vs Observed Growth Curves per Well (log-scale)")
print(curves_log)

ggsave(file.path(output_dir, "predicted_vs_observed_linear.pdf"),
       plot = curves_linear, width = 12, height = 9)
ggsave(file.path(output_dir, "predicted_vs_observed_log.pdf"),
       plot = curves_log,    width = 12, height = 9)

###############################################################################
cat("\nAnalysis complete — results in:\n   ", output_dir, "\n")
