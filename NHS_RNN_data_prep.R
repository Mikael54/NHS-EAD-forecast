# NHS System Pressure Forecast Data Preparation

# Required Libraries
library(tidyverse)
library(lubridate)
library(e1071)
library(imputeTS)
library(zoo)

# ============================================================================
# DATA PREPARATION
# ============================================================================

cat("\n--- Beginning Data Preparation ---\n")

raw_data <- read.csv("data/turingAI_forecasting_challenge_dataset.csv")

raw_data <- raw_data %>%
  mutate(
    dt = parse_date_time(dt, orders = c("Ymd HMS", "Ymd")),
    dt = force_tz(dt, tzone = "UTC"),
    date = as.Date(dt),
    time = format(dt, "%H:%M:%S")
  ) %>%
  filter(dt <= as.POSIXct("2025-09-30 00:00:00", tz = "UTC"))

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

# Identify target column before renaming
target_candidates <- names(forecasting_df)[grepl("estimated_avoidable_deaths", names(forecasting_df), ignore.case = TRUE)]
if (length(target_candidates) != 1) {
  stop("Expected exactly one target column matching 'estimated_avoidable_deaths'.")
}
target_col_raw <- target_candidates[1]
target_idx <- which(names(forecasting_df) == target_col_raw)

# Clean and abbreviate column names
cols_to_abbrev <- setdiff(names(forecasting_df), c("midday_day", target_col_raw))

abbrev_names <- make.names(abbreviate(cols_to_abbrev, minlength = 8), unique = TRUE)
names(forecasting_df)[names(forecasting_df) %in% cols_to_abbrev] <- abbrev_names

clean_names <- colnames(forecasting_df) %>%
  gsub("[0-9]", "", .) %>%
  gsub("[()]", "", .) %>%
  gsub("[ -]", "_", .) %>%
  gsub("%", "pct", .) %>%
  gsub("[^[:alnum:]_]", "", .)

clean_names <- tolower(clean_names)
colnames(forecasting_df) <- make.names(clean_names, unique = TRUE)

# Ensure target column naming stays consistent
colnames(forecasting_df)[target_idx] <- "estimated_avoidable_deaths"

# Store mapping for reference
cols_to_use <- names(forecasting_df)[!names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths")]

abbrev_df <- data.frame(
  original_name = cols_to_abbrev,
  new_name = cols_to_use,
  stringsAsFactors = FALSE
)

target_map <- data.frame(
  original_name = target_col_raw,
  new_name = "estimated_avoidable_deaths",
  stringsAsFactors = FALSE
)

abbrev_df <- rbind(target_map, abbrev_df)

write.csv(abbrev_df, "data/column_name_map.csv", row.names = FALSE)

# ==========================================================================
# CALENDAR FEATURES
# ==========================================================================

forecasting_df <- forecasting_df %>%
  mutate(
    dow = lubridate::wday(midday_day, week_start = 1),
    is_weekend = if_else(dow %in% c(6, 7), 1L, 0L),
    month = lubridate::month(midday_day),
    quarter = lubridate::quarter(midday_day),
    winter_indicator = if_else(month %in% c(12, 1, 2), 1L, 0L),
    sin_month = sin(2 * pi * month / 12),
    cos_month = cos(2 * pi * month / 12)
  )

# ==========================================================================
# HANDLE MISSING VALUES
# ==========================================================================

numeric_cols <- names(forecasting_df)[sapply(forecasting_df, is.numeric)]
impute_cols <- setdiff(numeric_cols, c("estimated_avoidable_deaths", "midday_day"))

safe_kalman <- function(x) {
  out <- tryCatch(
    suppressWarnings(na_kalman(x)),
    error = function(e) x
  )
  if (anyNA(out)) {
    out <- suppressWarnings(na_interpolation(x, option = "linear"))
  }
  out
}

forecasting_df <- forecasting_df %>%
  mutate(across(all_of(impute_cols), ~ safe_kalman(.)))

forecasting_df <- na.omit(forecasting_df)

cat("Rows after Kalman imputation and NA filtering:", nrow(forecasting_df), "\n")

# ==========================================================================
# FEATURE ENGINEERING
# ==========================================================================

predictors <- setdiff(colnames(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))

rolling_windows <- c(3, 7, 14, 30)

create_rolling_features <- function(data, vars, windows = rolling_windows) {
  result <- data
  for (var in vars) {
    for (window in windows) {
      result[[paste0(var, "_roll_mean_", window)]] <-
        zoo::rollmean(data[[var]], k = window, fill = NA, align = "right")
      result[[paste0(var, "_roll_sd_", window)]] <-
        zoo::rollapply(data[[var]], width = window, FUN = sd, fill = NA, align = "right")
      result[[paste0(var, "_roll_max_", window)]] <-
        zoo::rollapply(data[[var]], width = window, FUN = max, fill = NA, align = "right")
      result[[paste0(var, "_change_", window)]] <-
        data[[var]] - dplyr::lag(data[[var]], window)
    }
  }
  result
}

forecasting_df <- create_rolling_features(forecasting_df, predictors, windows = rolling_windows)

lag_days <- c(1, 3, 7, 14)
for (lag_day in lag_days) {
  forecasting_df[[paste0("estimated_avoidable_deaths_lag", lag_day)]] <-
    dplyr::lag(forecasting_df$estimated_avoidable_deaths, lag_day)
}

forecasting_df <- forecasting_df %>%
  na.omit()

predictors <- setdiff(colnames(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))

# ==========================================================================
# HANDLE SKEWNESS WITH TRANSFORMATIONS
# ==========================================================================

skewness_results <- data.frame(
  variable = character(),
  original_skewness = numeric(),
  transformation = character(),
  stringsAsFactors = FALSE
)

for (col in predictors) {
  x <- forecasting_df[[col]]

  if (is.numeric(x)) {
    skew_val <- e1071::skewness(x, na.rm = TRUE)
    transformation <- "none"

    if (is.na(skew_val)) {
      skewness_results <- rbind(skewness_results, data.frame(
        variable = col,
        original_skewness = NA_real_,
        transformation = transformation
      ))
      next
    }

    if (abs(skew_val) > 1) {
      if (skew_val > 1 && all(x > 0, na.rm = TRUE)) {
        forecasting_df[[col]] <- log1p(x)
        transformation <- "log1p"
      } else if (skew_val > 1) {
        forecasting_df[[col]] <- sqrt(x - min(x, na.rm = TRUE) + 1)
        transformation <- "sqrt"
      } else if (skew_val < -1) {
        forecasting_df[[col]] <- x^2
        transformation <- "squared"
      }
    }

    skewness_results <- rbind(skewness_results, data.frame(
      variable = col,
      original_skewness = skew_val,
      transformation = transformation
    ))
  }
}

dir.create("results", showWarnings = FALSE)
write.csv(skewness_results, "results/skewness_transformations.csv", row.names = FALSE)

# ==========================================================================
# SAVE PROCESSED DATA
# ==========================================================================

output_path <- "data/processed_daily_metrics.csv"
write.csv(forecasting_df, output_path, row.names = FALSE)

cat("Prepared data saved to:", output_path, "\n")
