
input_file <- "D:/Research/OECD Research/ICIO2025econX.RData"

env <- new.env()
load(input_file, envir = env)

X <- env$ICIO2025econX

targets <- c(
  "BRN_29",
  "CHL_29",
  "SGP_02",
  "SGP_05",
  "SGP_06",
  "SGP_07",
  "SGP_08",
  "SGP_09"
)

cat("\n=== MATRIX STRUCTURE ===\n")
cat("Class:", class(X), "\n")
cat("Dimensions:", paste(dim(X), collapse = " x "), "\n")
cat("2022 row exists:", "2022" %in% rownames(X), "\n")

cat("\n=== AFFECTED SOURCE CELLS ===\n")

for (key in targets) {
  cat(
    key,
    "| column exists:", key %in% colnames(X),
    "| 2022 value:",
    if (key %in% colnames(X)) X["2022", key] else NA_real_,
    "\n"
  )
}

cat("\n=== OTHER YEARS FOR AFFECTED INDUSTRIES ===\n")

for (key in targets) {
  if (key %in% colnames(X)) {
    cat("\n", key, "\n")
    print(X[, key, drop = FALSE])
  }
}

cat("\n=== ALL YEARS: SELECTED COMPARISON INDUSTRIES ===\n")

comparison_keys <- c(
  "BRN_19", "BRN_20",
  "CHL_07", "CHL_08", "CHL_29",
  "SGP_02", "SGP_19", "SGP_20", "SGP_29"
)

comparison_keys <- intersect(comparison_keys, colnames(X))

print(X[, comparison_keys, drop = FALSE])
