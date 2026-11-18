
input_file <- "D:/Research/OECD Research/ICIO2025econX.RData"

env <- new.env()
load(input_file, envir = env)
X <- env$ICIO2025econX

cat("\n=== OBJECT ATTRIBUTES ===\n")
print(attributes(X)[setdiff(names(attributes(X)), c("dim", "dimnames"))])

cat("\n=== ROW NAMES ===\n")
print(rownames(X))

cat("\n=== COLUMN NAME EXAMPLES ===\n")
cat("First 15:\n")
print(head(colnames(X), 15))
cat("Last 15:\n")
print(tail(colnames(X), 15))

cat("\n=== INDUSTRY COVERAGE ===\n")
for (country in c("BRN", "CHL", "SGP")) {
  keys <- grep(paste0("^", country, "_"), colnames(X), value = TRUE)
  vals <- X["2022", keys]

  cat("\n", country,
      "| industries:", length(keys),
      "| positive:", sum(vals > 0),
      "| zero:", sum(vals == 0), "\n")

  if (country == "SGP") {
    print(data.frame(
      industry = keys,
      value_2022 = as.numeric(vals)
    ), row.names = FALSE)
  }
}

cat("\n=== OTHER OECD OBJECTS IN SAME DIRECTORY ===\n")
files <- list.files(
  "D:/Research/OECD Research",
  pattern = "\\.(RData|zip)$",
  full.names = FALSE,
  ignore.case = TRUE
)
print(files)
