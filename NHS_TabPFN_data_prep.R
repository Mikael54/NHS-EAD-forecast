# NHS TabPFN Data Preparation
# Extends NHS_RNN_data_prep.R with additional features for the TabPFN pipeline.
# Output: data/processed_tabpfn.csv + data/feature_documentation.csv

library(tidyverse)
library(lubridate)
library(e1071)
library(imputeTS)
library(zoo)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Raw challenge CSV. It ships in this repo as a Git-LFS zip; run this script from the repo root
# after unzipping it (paths below are relative to the repo root):
#     git lfs pull
#     unzip data/turingAI_forecasting_challenge_dataset.csv.zip -d data/
# For the assessment run (from 6 Jun 2026), make sure this file is the UPDATED dataset that
# contains real values for 1 Oct 2025 - 31 Mar 2026 (the dev file has -9999 dummies there). No
# other edits are needed - the filter below already spans the assessment window and -9999 dummies
# are handled automatically.
DATA_PATH   <- "data/turingAI_forecasting_challenge_dataset.csv"
OUT_DIR     <- "data"
dir.create(OUT_DIR, showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

cat("\n--- Beginning Data Preparation (TabPFN variant) ---\n")

# ============================================================================
# 1. LOAD + CLEAN (from NHS_RNN_data_prep.R)
# ============================================================================

raw_data <- read.csv(DATA_PATH)

raw_data <- raw_data %>%
  mutate(
    dt = parse_date_time(dt, orders = c("Ymd HMS", "Ymd")),
    dt = force_tz(dt, tzone = "UTC"),
    date = as.Date(dt),
    time = format(dt, "%H:%M:%S"),
    # -9999 marks missing/dummy data (incl. the assessment period in the dev file).
    # Convert to NA so predictors get imputed and dummy-target rows drop out via na.omit.
    # This makes the script work BOTH before and after the assessment data is released.
    value = ifelse(value <= -9990, NA_real_, value)
  ) %>%
  # Include the full assessment window (1 Oct 2025 - 31 Mar 2026). While the assessment
  # target is still a dummy it becomes NA and na.omit removes those rows (so today this
  # behaves like the old <= 2025-09-30 filter); once real targets arrive they are kept.
  filter(dt < as.POSIXct("2026-04-01 00:00:00", tz = "UTC"))

forecasting_df <- raw_data %>%
  mutate(midday_day = if_else(
    format(dt, "%H:%M:%S") <= "12:00:00", date, date + 1
  )) %>%
  select(-any_of(c("coverage", "coverage_label", "variable_type", "dt", "date", "time"))) %>%
  group_by(midday_day, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    id_cols = midday_day,
    names_from = metric_name,
    values_from = value,
    names_sep = "_"
  ) %>%
  arrange(midday_day)

cat("Total columns:", ncol(forecasting_df), "\n")
cat("Total days:", nrow(forecasting_df), "\n")

# ============================================================================
# 2. COLUMN NAMING (from NHS_RNN_data_prep.R)
# ============================================================================

target_candidates <- names(forecasting_df)[grepl("estimated_avoidable_deaths", names(forecasting_df), ignore.case = TRUE)]
if (length(target_candidates) != 1) stop("Expected exactly one target column.")
target_col_raw <- target_candidates[1]
target_idx <- which(names(forecasting_df) == target_col_raw)

cols_to_abbrev <- setdiff(names(forecasting_df), c("midday_day", target_col_raw))
abbrev_names <- make.names(abbreviate(cols_to_abbrev, minlength = 8), unique = TRUE)
names(forecasting_df)[names(forecasting_df) %in% cols_to_abbrev] <- abbrev_names

clean_names <- colnames(forecasting_df) %>%
  gsub("[0-9]", "", .) %>%
  gsub("[()]", "", .) %>%
  gsub("[ -]", "_", .) %>%
  gsub("%", "pct", .) %>%
  gsub("[^[:alnum:]_]", "", .) %>%
  tolower()

colnames(forecasting_df) <- make.names(clean_names, unique = TRUE)
colnames(forecasting_df)[target_idx] <- "estimated_avoidable_deaths"

cols_to_use <- names(forecasting_df)[!names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths")]

abbrev_df <- rbind(
  data.frame(original_name = target_col_raw, new_name = "estimated_avoidable_deaths", stringsAsFactors = FALSE),
  data.frame(original_name = cols_to_abbrev, new_name = cols_to_use, stringsAsFactors = FALSE)
)
write.csv(abbrev_df, file.path(OUT_DIR, "column_name_map.csv"), row.names = FALSE)

# ============================================================================
# 3. CALENDAR FEATURES (from NHS_RNN_data_prep.R + extended)
# ============================================================================

# England bank holidays 2023-2026 (hard-coded; no external data allowed)
england_bank_holidays <- as.Date(c(
  # 2023
  "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-08",
  "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26",
  # 2024
  "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
  "2024-08-26", "2024-12-25", "2024-12-26",
  # 2025
  "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
  "2025-08-25", "2025-12-25", "2025-12-26",
  # 2026 (assessment period)
  "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25",
  "2026-08-31", "2026-12-25", "2026-12-28"
))

forecasting_df <- forecasting_df %>%
  mutate(
    # existing calendar features
    dow             = lubridate::wday(midday_day, week_start = 1),
    is_weekend      = if_else(dow %in% c(6, 7), 1L, 0L),
    month           = lubridate::month(midday_day),
    quarter         = lubridate::quarter(midday_day),
    winter_indicator = if_else(month %in% c(12, 1, 2), 1L, 0L),
    sin_month       = sin(2 * pi * month / 12),
    cos_month       = cos(2 * pi * month / 12),
    # new calendar features
    day_of_month    = lubridate::mday(midday_day),
    days_since_start = as.integer(midday_day - min(midday_day)),
    is_bank_holiday = if_else(midday_day %in% england_bank_holidays, 1L, 0L),
    sin_dow         = sin(2 * pi * dow / 7),
    cos_dow         = cos(2 * pi * dow / 7)
  )

# ============================================================================
# 4. MISSING VALUE IMPUTATION (from NHS_RNN_data_prep.R)
# ============================================================================

numeric_cols <- names(forecasting_df)[sapply(forecasting_df, is.numeric)]
impute_cols  <- setdiff(numeric_cols, c("estimated_avoidable_deaths"))

safe_kalman <- function(x) {
  out <- tryCatch(suppressWarnings(na_kalman(x)), error = function(e) x)
  if (anyNA(out)) out <- suppressWarnings(na_interpolation(x, option = "linear"))
  out
}

forecasting_df <- forecasting_df %>%
  mutate(across(all_of(impute_cols), ~ safe_kalman(.)))

forecasting_df <- na.omit(forecasting_df)
cat("Rows after imputation and NA filtering:", nrow(forecasting_df), "\n")

# ============================================================================
# 5. ROLLING FEATURES ON PREDICTORS (from NHS_RNN_data_prep.R)
# ============================================================================

predictors <- setdiff(colnames(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))
rolling_windows <- c(3, 7, 14, 30)

create_rolling_features <- function(data, vars, windows) {
  result <- data
  for (var in vars) {
    for (w in windows) {
      result[[paste0(var, "_roll_mean_", w)]] <-
        zoo::rollmean(data[[var]], k = w, fill = NA, align = "right")
      result[[paste0(var, "_roll_sd_", w)]] <-
        zoo::rollapply(data[[var]], width = w, FUN = sd, fill = NA, align = "right")
      result[[paste0(var, "_roll_max_", w)]] <-
        zoo::rollapply(data[[var]], width = w, FUN = max, fill = NA, align = "right")
      result[[paste0(var, "_change_", w)]] <-
        data[[var]] - dplyr::lag(data[[var]], w)
    }
  }
  result
}

forecasting_df <- create_rolling_features(forecasting_df, predictors, rolling_windows)

# ============================================================================
# 6. TARGET LAGS AND ROLLING TARGET FEATURES (extended)
# ============================================================================

# Lags 3-14 days (lag < 3 are invalid due to 3-day reporting lag)
for (lag_day in c(3, 4, 5, 6, 7, 8, 9, 10, 14)) {
  forecasting_df[[paste0("ead_lag", lag_day)]] <-
    dplyr::lag(forecasting_df$estimated_avoidable_deaths, lag_day)
}

# Lag 1 retained for training rows where the lag is fully in the past
# (disabled during inference: only lags >= 3 valid at forecast time)
for (lag_day in c(1)) {
  forecasting_df[[paste0("ead_lag", lag_day)]] <-
    dplyr::lag(forecasting_df$estimated_avoidable_deaths, lag_day)
}

# Rolling statistics of the target (applied to lag-3 shifted series to avoid leakage)
ead_lagged3 <- dplyr::lag(forecasting_df$estimated_avoidable_deaths, 3)
forecasting_df$ead_roll_mean_7  <- zoo::rollmean(ead_lagged3, k = 7,  fill = NA, align = "right")
forecasting_df$ead_roll_sd_7    <- zoo::rollapply(ead_lagged3, width = 7,  FUN = sd, fill = NA, align = "right")
forecasting_df$ead_roll_mean_14 <- zoo::rollmean(ead_lagged3, k = 14, fill = NA, align = "right")
forecasting_df$ead_roll_sd_14   <- zoo::rollapply(ead_lagged3, width = 14, FUN = sd, fill = NA, align = "right")
forecasting_df$ead_roll_mean_30 <- zoo::rollmean(ead_lagged3, k = 30, fill = NA, align = "right")
forecasting_df$ead_roll_sd_30   <- zoo::rollapply(ead_lagged3, width = 30, FUN = sd, fill = NA, align = "right")
forecasting_df$ead_7day_trend   <- ead_lagged3 - dplyr::lag(ead_lagged3, 7)

# --- Fix B: NHS pressure regime features ---
# Rolling max over 7 and 14 days of lagged target (peak pressure indicator)
forecasting_df$ead_roll_max_7  <- zoo::rollapply(ead_lagged3, width = 7,  FUN = max, fill = NA, align = "right")
forecasting_df$ead_roll_max_14 <- zoo::rollapply(ead_lagged3, width = 14, FUN = max, fill = NA, align = "right")

# Rolling 75th and 90th percentile — signals whether we are in a high-tail regime
forecasting_df$ead_roll_q75_30 <- zoo::rollapply(
  ead_lagged3, width = 30,
  FUN = function(x) quantile(x, 0.75, na.rm = TRUE),
  fill = NA, align = "right"
)
forecasting_df$ead_roll_q90_30 <- zoo::rollapply(
  ead_lagged3, width = 30,
  FUN = function(x) quantile(x, 0.90, na.rm = TRUE),
  fill = NA, align = "right"
)

# Ratio of recent (3-day) mean to 30-day mean — >1 means we are elevated above baseline
forecasting_df$ead_recent_vs_baseline <- forecasting_df$ead_roll_mean_7 /
  (forecasting_df$ead_roll_mean_30 + 0.01)

# High-pressure binary indicator: 7-day mean of lagged target above 0.9
forecasting_df$ead_high_pressure <- as.integer(
  !is.na(forecasting_df$ead_roll_mean_7) & forecasting_df$ead_roll_mean_7 > 0.9
)

# --- Fix C: Deviation / z-score features ---
# How far the most recent known value sits above/below the 30-day baseline
forecasting_df$ead_deviation_from_mean <- ead_lagged3 - forecasting_df$ead_roll_mean_30
forecasting_df$ead_z_score_30 <- forecasting_df$ead_deviation_from_mean /
  (forecasting_df$ead_roll_sd_30 + 0.01)

# Same relative to 14-day mean (faster-moving baseline)
forecasting_df$ead_deviation_from_mean_14 <- ead_lagged3 - forecasting_df$ead_roll_mean_14
forecasting_df$ead_z_score_14 <- forecasting_df$ead_deviation_from_mean_14 /
  (forecasting_df$ead_roll_sd_14 + 0.01)

forecasting_df <- na.omit(forecasting_df)

predictors <- setdiff(colnames(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))
cat("Total predictors after feature engineering:", length(predictors), "\n")

# ============================================================================
# 7. SKEWNESS TRANSFORMATIONS (from NHS_RNN_data_prep.R)
# ============================================================================

skewness_results <- data.frame(variable = character(), original_skewness = numeric(),
                                transformation = character(), stringsAsFactors = FALSE)

for (col in predictors) {
  x <- forecasting_df[[col]]
  if (!is.numeric(x)) next
  skew_val <- e1071::skewness(x, na.rm = TRUE)
  transformation <- "none"
  if (is.na(skew_val)) {
    skewness_results <- rbind(skewness_results, data.frame(
      variable = col, original_skewness = NA_real_, transformation = "none"))
    next
  }
  if (abs(skew_val) > 1) {
    if (skew_val > 1 && all(x > 0, na.rm = TRUE)) {
      forecasting_df[[col]] <- log1p(x); transformation <- "log1p"
    } else if (skew_val > 1) {
      forecasting_df[[col]] <- sqrt(x - min(x, na.rm = TRUE) + 1); transformation <- "sqrt"
    } else if (skew_val < -1) {
      forecasting_df[[col]] <- x^2; transformation <- "squared"
    }
  }
  skewness_results <- rbind(skewness_results, data.frame(
    variable = col, original_skewness = skew_val, transformation = transformation))
}

write.csv(skewness_results, "results/skewness_transformations_tabpfn.csv", row.names = FALSE)

# ============================================================================
# 8. FEATURE DOCUMENTATION
# ============================================================================

all_features <- setdiff(colnames(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))

feature_type <- case_when(
  grepl("^(dow|is_weekend|month|quarter|winter|sin_month|cos_month|day_of_month|days_since_start|is_bank_holiday|sin_dow|cos_dow)$", all_features) ~ "calendar",
  grepl("^ead_(lag|roll|7day)", all_features) ~ "target_lag_or_rolling",
  grepl("_(roll_mean|roll_sd|roll_max|change)_", all_features) ~ "rolling",
  TRUE ~ "raw_predictor"
)

feature_doc <- data.frame(
  feature_name = all_features,
  type = feature_type,
  description = case_when(
    grepl("^dow$", all_features) ~ "Day of week (1=Mon, 7=Sun)",
    grepl("^is_weekend$", all_features) ~ "Weekend indicator (1=Sat/Sun)",
    grepl("^month$", all_features) ~ "Calendar month (1-12)",
    grepl("^quarter$", all_features) ~ "Calendar quarter (1-4)",
    grepl("^winter_indicator$", all_features) ~ "Winter months indicator (Dec/Jan/Feb)",
    grepl("^sin_month$", all_features) ~ "Sine cyclical encoding of month",
    grepl("^cos_month$", all_features) ~ "Cosine cyclical encoding of month",
    grepl("^day_of_month$", all_features) ~ "Day of month (1-31)",
    grepl("^days_since_start$", all_features) ~ "Integer days since study start (linear trend)",
    grepl("^is_bank_holiday$", all_features) ~ "England bank holiday indicator",
    grepl("^sin_dow$", all_features) ~ "Sine cyclical encoding of day-of-week",
    grepl("^cos_dow$", all_features) ~ "Cosine cyclical encoding of day-of-week",
    grepl("^ead_lag[0-9]+$", all_features) ~ paste0("Target lag: estimated_avoidable_deaths at t-", gsub("ead_lag", "", all_features)),
    grepl("^ead_roll_mean_", all_features) ~ paste0("Rolling mean of lagged target (window=", gsub(".*_(\\d+)$", "\\1", all_features), " days)"),
    grepl("^ead_roll_sd_", all_features) ~ paste0("Rolling SD of lagged target (window=", gsub(".*_(\\d+)$", "\\1", all_features), " days)"),
    grepl("^ead_7day_trend$", all_features) ~ "7-day directional trend of lagged target",
    grepl("_roll_mean_", all_features) ~ paste0("Rolling mean of predictor (window=", gsub(".*_(\\d+)$", "\\1", all_features), " days)"),
    grepl("_roll_sd_", all_features) ~ paste0("Rolling SD of predictor (window=", gsub(".*_(\\d+)$", "\\1", all_features), " days)"),
    grepl("_roll_max_", all_features) ~ paste0("Rolling max of predictor (window=", gsub(".*_(\\d+)$", "\\1", all_features), " days)"),
    grepl("_change_", all_features) ~ paste0("Change over ", gsub(".*_(\\d+)$", "\\1", all_features), " days"),
    TRUE ~ "Raw predictor (daily mean)"
  ),
  stringsAsFactors = FALSE
)

write.csv(feature_doc, file.path(OUT_DIR, "feature_documentation.csv"), row.names = FALSE)
cat("Feature documentation written:", nrow(feature_doc), "features documented.\n")

# ============================================================================
# 9. SAVE
# ============================================================================

output_path <- file.path(OUT_DIR, "processed_tabpfn.csv")
write.csv(forecasting_df, output_path, row.names = FALSE)
cat("Processed data saved to:", output_path, "\n")
cat("Final dimensions:", nrow(forecasting_df), "rows x", ncol(forecasting_df), "columns\n")
