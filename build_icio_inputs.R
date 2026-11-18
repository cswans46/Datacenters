
# ---------------------------------------------------------------------
# Build OECD ICIO input files for the data-centre emissions model
#
# Source:
#   ICIO2025econX.RData
#
# Extracts:
#   - 2022 gross output by ICIO industry
#   - Industry-to-country and industry-to-sector mapping
#
# Outputs:
#   - sector_map.csv
#   - icio_output.csv
# ---------------------------------------------------------------------

input_file <- "D:/Research/OECD Research/ICIO2025econX.RData"
output_dir <- "D:/Research/Datacenters"

sector_map_file <- file.path(output_dir, "sector_map.csv")
icio_output_file <- file.path(output_dir, "icio_output.csv")

target_year <- "2022"

# ---------------------------------------------------------------------
# 1. Validate paths
# ---------------------------------------------------------------------

if (!file.exists(input_file)) {
  stop("ICIO gross-output file not found: ", input_file)
}

if (!dir.exists(output_dir)) {
  stop("Output directory does not exist: ", output_dir)
}

# ---------------------------------------------------------------------
# 2. Load OECD ICIO gross output
# ---------------------------------------------------------------------

cat("Loading OECD ICIO gross output...\n")

env <- new.env()
load(input_file, envir = env)

if (!exists("ICIO2025econX", envir = env)) {
  stop("Object ICIO2025econX was not found in the RData file.")
}

X <- env$ICIO2025econX

if (!is.matrix(X)) {
  stop("ICIO2025econX is not a matrix.")
}

cat("Matrix dimensions:", paste(dim(X), collapse = " x "), "\n")

# ---------------------------------------------------------------------
# 3. Validate year and industry dimensions
# ---------------------------------------------------------------------

if (is.null(rownames(X)) || is.null(colnames(X))) {
  stop("ICIO matrix is missing year or industry identifiers.")
}

if (!target_year %in% rownames(X)) {
  stop("Year ", target_year, " is not present in the ICIO matrix.")
}

io_keys <- colnames(X)

if (length(io_keys) != 4250) {
  stop(
    "Unexpected number of ICIO industries: ",
    length(io_keys),
    ". Expected 4250."
  )
}

if (anyDuplicated(io_keys)) {
  stop("Duplicate ICIO industry keys detected.")
}

# ---------------------------------------------------------------------
# 4. Extract 2022 gross output
# ---------------------------------------------------------------------

output_values <- as.numeric(X[target_year, ])

if (length(output_values) != length(io_keys)) {
  stop("Output vector length does not match the industry-key count.")
}

if (any(!is.finite(output_values))) {
  stop("Non-finite gross-output values detected.")
}

if (any(output_values < 0)) {
  stop("Negative gross-output values detected.")
}

icio_output <- data.frame(
  io_key = io_keys,
  output_usd_mn = output_values,
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------
# 5. Parse country and sector identifiers
#
# Standard key:
#   AGO_01
#
# Special aggregate key:
#   EU27_2020_01
# ---------------------------------------------------------------------

is_eu27 <- startsWith(io_keys, "EU27_2020_")

country <- ifelse(
  is_eu27,
  "EU27_2020",
  substr(io_keys, 1, 3)
)

sector_code <- ifelse(
  is_eu27,
  substring(io_keys, nchar("EU27_2020_") + 1),
  substring(io_keys, 5)
)

# Validate that the keys follow the expected structure.
expected_keys <- ifelse(
  is_eu27,
  paste0(country, "_", sector_code),
  paste0(country, "_", sector_code)
)

if (any(io_keys != expected_keys)) {
  bad <- io_keys[io_keys != expected_keys]
  stop(
    "Could not parse the following ICIO keys:\n",
    paste(head(bad, 20), collapse = "\n")
  )
}

if (any(is.na(country) | country == "")) {
  stop("Missing country codes detected.")
}

if (any(is.na(sector_code) | sector_code == "")) {
  stop("Missing sector codes detected.")
}

sector_map <- data.frame(
  io_key = io_keys,
  country = country,
  sector_code = sector_code,
  sector_label = sector_code,
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------
# 6. Validate generated tables
# ---------------------------------------------------------------------

if (nrow(sector_map) != 4250) {
  stop("Unexpected sector-map row count: ", nrow(sector_map))
}

if (nrow(icio_output) != 4250) {
  stop("Unexpected output row count: ", nrow(icio_output))
}

if (!identical(sector_map$io_key, icio_output$io_key)) {
  stop("Industry keys do not match between generated files.")
}

# ---------------------------------------------------------------------
# 7. Write CSV files
# ---------------------------------------------------------------------

write.csv(
  sector_map,
  sector_map_file,
  row.names = FALSE,
  na = ""
)

write.csv(
  icio_output,
  icio_output_file,
  row.names = FALSE,
  na = ""
)

# ---------------------------------------------------------------------
# 8. Read-back validation
# ---------------------------------------------------------------------

sector_check <- read.csv(sector_map_file, stringsAsFactors = FALSE)
output_check <- read.csv(icio_output_file, stringsAsFactors = FALSE)

if (nrow(sector_check) != 4250 || nrow(output_check) != 4250) {
  stop("CSV read-back validation failed.")
}

if (!identical(sector_check$io_key, output_check$io_key)) {
  stop("CSV read-back industry-key validation failed.")
}

# ---------------------------------------------------------------------
# 9. Diagnostics
# ---------------------------------------------------------------------

cat("\n============================================\n")
cat("OECD ICIO INPUT EXTRACTION SUCCESSFUL\n")
cat("============================================\n")

cat("\nYear:", target_year, "\n")
cat("Number of industries:", nrow(icio_output), "\n")
cat("Number of countries:", length(unique(country)), "\n")

cat(
  "Total gross output (USD million):",
  format(sum(icio_output$output_usd_mn), big.mark = ","),
  "\n"
)

cat("\nCountry codes:\n")
print(sort(unique(country)))

cat("\nFirst 10 industry records:\n")
print(head(sector_map, 10))

cat("\nFirst 10 output records:\n")
print(head(icio_output, 10))

cat("\nFiles written:\n")
cat(sector_map_file, "\n")
cat(icio_output_file, "\n")

cat("\nBoth files passed read-back validation.\n")
