
input_file <- "D:/Research/OECD Research/ICIO2025econZ.RData"

cat("Loading OECD intermediate transaction data...\n")
env <- new.env()
loaded <- load(input_file, envir = env)

cat("\n=== LOADED OBJECTS ===\n")
print(loaded)

for (nm in loaded) {
  obj <- env[[nm]]

  cat("\n=== OBJECT:", nm, "===\n")
  cat("Class:", paste(class(obj), collapse = ", "), "\n")
  cat("Dimensions:", paste(dim(obj), collapse = " x "), "\n")
  cat("Dimension names:\n")

  dn <- dimnames(obj)
  if (is.null(dn)) {
    cat("No dimension names\n")Rscript diagnose_icio_sgp_transactions.R
  } else {
    for (i in seq_along(dn)) {
      cat("Dimension", i,
          "| length:", length(dn[[i]]),
          "| first labels:", paste(head(dn[[i]], 4), collapse = ", "),
          "\n")
    }
  }

  cat("Object size (GB):",
      round(as.numeric(object.size(obj)) / 1024^3, 3), "\n")
}
