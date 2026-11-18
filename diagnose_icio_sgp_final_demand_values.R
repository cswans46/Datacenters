
source_dir <- "D:/Research/OECD Research"

env <- new.env()
load(file.path(source_dir, "ICIO2025econFD.RData"), envir = env)
FD <- env$ICIO2025econFD

year <- "2022"
year_idx <- match(year, dimnames(FD)[[1]])

targets <- c(
  "SGP_02", "SGP_05", "SGP_06",
  "SGP_07", "SGP_08", "SGP_09"
)

industry_idx <- match(targets, dimnames(FD)[[2]])

if (is.na(year_idx) || anyNA(industry_idx)) {
  stop("Year or industry code not found.")
}

# Sum final demand across all destination countries.
fd_matrix <- FD[year_idx, industry_idx, , drop = FALSE]
fd_matrix <- matrix(
  fd_matrix,
  nrow = length(targets),
  dimnames = list(targets, dimnames(FD)[[3]])
)

result <- data.frame(
  industry = targets,
  final_demand_sales = rowSums(fd_matrix, na.rm = TRUE),
  nonzero_final_demand_cells =
    rowSums(fd_matrix != 0, na.rm = TRUE)
)

cat("\n=== SINGAPORE FINAL DEMAND, 2022 ===\n")
print(result, row.names = FALSE, digits = 10)

cat("\n=== NONZERO DESTINATION ENTRIES ===\n")
for (i in seq_along(targets)) {
  nz <- which(fd_matrix[i, ] != 0 & !is.na(fd_matrix[i, ]))
  if (length(nz) > 0) {
    cat("\n", targets[i], "\n")
    print(data.frame(
      destination = colnames(fd_matrix)[nz],
      value = as.numeric(fd_matrix[i, nz])
    ), row.names = FALSE)
  }
}
