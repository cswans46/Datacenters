
source_dir <- "D:/Research/OECD Research"

cat("Loading OECD transaction array...\n")
env <- new.env()
load(file.path(source_dir, "ICIO2025econZ.RData"), envir = env)
Z <- env$ICIO2025econZ

year <- "2022"
year_idx <- match(year, dimnames(Z)[[1]])

targets <- c(
  "SGP_02", "SGP_05", "SGP_06", "SGP_07",
  "SGP_08", "SGP_09"
)

comparators <- c(
  "SGP_01", "SGP_03", "SGP_19",
  "SGP_20", "SGP_29"
)

industries <- c(targets, comparators)
industry_idx <- match(industries, dimnames(Z)[[2]])

if (anyNA(industry_idx) || is.na(year_idx)) {
  stop("A requested year or industry code was not found.")
}

cat("Calculating 2022 transaction margins...\n")

# Dimension 2 = supplying industry (intermediate sales)
sales_matrix <- Z[year_idx, industry_idx, , drop = FALSE]
sales_matrix <- matrix(
  sales_matrix,
  nrow = length(industries),
  dimnames = list(industries, dimnames(Z)[[3]])
)

# Dimension 3 = purchasing industry (intermediate purchases)
purchase_matrix <- Z[year_idx, , industry_idx, drop = FALSE]
purchase_matrix <- matrix(
  purchase_matrix,
  nrow = length(dimnames(Z)[[2]]),
  dimnames = list(dimnames(Z)[[2]], industries)
)

sales <- rowSums(sales_matrix, na.rm = TRUE)
purchases <- colSums(purchase_matrix, na.rm = TRUE)

# OECD gross output values from the companion X matrix
xenv <- new.env()
load(file.path(source_dir, "ICIO2025econX.RData"), envir = xenv)
X <- xenv$ICIO2025econX

output <- as.numeric(X[year, industries])

result <- data.frame(
  industry = industries,
  gross_output_X = output,
  intermediate_sales = as.numeric(sales),
  intermediate_purchases = as.numeric(purchases),
  nonzero_sales_cells = rowSums(sales_matrix != 0, na.rm = TRUE),
  nonzero_purchase_cells = colSums(purchase_matrix != 0, na.rm = TRUE)
)

cat("\n=== 2022 SINGAPORE TRANSACTION MARGINS ===\n")
print(result, row.names = FALSE, digits = 8)

cat("\n=== TARGET INDUSTRIES: NONZERO TRANSACTION COUNTS ===\n")
print(
  result[result$industry %in% targets,
         c("industry", "nonzero_sales_cells", "nonzero_purchase_cells")],
  row.names = FALSE
)
