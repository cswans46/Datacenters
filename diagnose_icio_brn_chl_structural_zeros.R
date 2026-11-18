
# Diagnose Brunei and Chile automotive-sector structural zeros
# in the OECD ICIO 2025 source matrices.

data_dir <- "D:/Research/OECD Research"

load_icio_object <- function(filename, object_name) {
  env <- new.env()
  load(file.path(data_dir, filename), envir = env)

  if (!exists(object_name, envir = env)) {
    stop("Object not found: ", object_name)
  }

  get(object_name, envir = env)
}

X <- load_icio_object(
  "ICIO2025econX.RData",
  "ICIO2025econX"
)

Z <- load_icio_object(
  "ICIO2025econZ.RData",
  "ICIO2025econZ"
)

FD <- load_icio_object(
  "ICIO2025econFD.RData",
  "ICIO2025econFD"
)

targets <- c(
  "BRN_29",
  "CHL_29",
  paste0("CHN_", c(
    "01", "02", "03", "05", "06", "07", "08", "09",
    "10T12", "13T15", "16", "17T18", "19", "20", "21",
    "22", "23", "24A", "24B", "25", "26", "27", "28",
    "29", "301", "302T309", "31T33", "35", "36T39",
    "41T43", "45T47", "49", "50", "51", "52", "53",
    "55T56", "58T60", "61", "62T63", "64T66", "68",
    "69T75", "77T82", "84", "85", "86T88", "90T93",
    "94T96", "97T98"
  )),
  paste0("MEX_", c(
    "01", "02", "03", "05", "06", "07", "08", "09",
    "10T12", "13T15", "16", "17T18", "19", "20", "21",
    "22", "23", "24A", "24B", "25", "26", "27", "28",
    "29", "301", "302T309", "31T33", "35", "36T39",
    "41T43", "45T47", "49", "50", "51", "52", "53",
    "55T56", "58T60", "61", "62T63", "64T66", "68",
    "69T75", "77T82", "84", "85", "86T88", "90T93",
    "94T96", "97T98"
  ))
)

years <- dimnames(X)[[1]]
industries <- dimnames(X)[[2]]

results <- list()

for (target in targets) {

  i <- match(target, industries)

  if (is.na(i)) {
    warning("Industry not found: ", target)
    next
  }

  for (year in years) {

    y <- match(year, dimnames(Z)[[1]])

    # X: gross output
    gross_output <- X[year, i]

    # Z: intermediate sales and purchases
    intermediate_sales <- sum(Z[y, i, ], na.rm = TRUE)
    intermediate_purchases <- sum(Z[y, , i], na.rm = TRUE)

    nonzero_sales_cells <- sum(Z[y, i, ] != 0, na.rm = TRUE)
    nonzero_purchase_cells <- sum(Z[y, , i] != 0, na.rm = TRUE)

    # FD: final-demand sales across all destinations
    fd_year <- match(year, dimnames(FD)[[1]])
    fd_industry <- match(target, dimnames(FD)[[2]])

    final_demand_sales <- sum(
      FD[fd_year, fd_industry, ],
      na.rm = TRUE
    )

    nonzero_fd_cells <- sum(
      FD[fd_year, fd_industry, ] != 0,
      na.rm = TRUE
    )

    results[[length(results) + 1]] <- data.frame(
      industry = target,
      year = as.integer(year),
      gross_output = as.numeric(gross_output),
      intermediate_sales = intermediate_sales,
      intermediate_purchases = intermediate_purchases,
      final_demand_sales = final_demand_sales,
      nonzero_sales_cells = nonzero_sales_cells,
      nonzero_purchase_cells = nonzero_purchase_cells,
      nonzero_fd_cells = nonzero_fd_cells
    )
  }
}

diagnostics <- do.call(rbind, results)

write.csv(
  diagnostics,
  "icio_brn_chl_structural_zero_diagnostics.csv",
  row.names = FALSE
)

cat("\n=== 2022 RESULTS ===\n")

print(
  diagnostics[diagnostics$year == 2022, ],
  row.names = FALSE
)

cat("\n=== TIME-SERIES TRANSITIONS ===\n")

for (target in targets) {

  cat("\n---", target, "---\n")

  d <- diagnostics[diagnostics$industry == target, ]

  print(
    d[
      d$gross_output != 0 |
      d$intermediate_sales != 0 |
      d$intermediate_purchases != 0 |
      d$final_demand_sales != 0,
    ],
    row.names = FALSE
  )
}

cat("\nSaved: icio_brn_chl_structural_zero_diagnostics.csv\n")
