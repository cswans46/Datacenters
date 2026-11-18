
input_file <- "D:/Research/OECD Research/ICIO2025econX.RData"
source_dir <- "D:/Research/OECD Research"

cat("\n=== LOADED OBJECTS ===\n")
env <- new.env()
loaded_objects <- load(input_file, envir = env)
print(loaded_objects)

X <- env$ICIO2025econX

cat("\n=== MATRIX METADATA ===\n")
cat("Class:", paste(class(X), collapse = ", "), "\n")
cat("Dimensions:", paste(dim(X), collapse = " x "), "\n")
cat("Storage mode:", storage.mode(X), "\n")
cat("Object size (MB):", round(as.numeric(object.size(X)) / 1024^2, 2), "\n")
cat("Attributes:\n")
print(setdiff(names(attributes(X)), c("dim", "dimnames")))

cat("\n=== SINGAPORE SOURCE MATRIX VALUES (2022) ===\n")
sgp <- grep("^SGP_", colnames(X), value = TRUE)

sgp_summary <- data.frame(
  industry = sgp,
  value_2022 = as.numeric(X["2022", sgp]),
  stringsAsFactors = FALSE
)

cat("Singapore industries:", length(sgp), "\n")
cat("Zero industries:", sum(sgp_summary$value_2022 == 0), "\n")
cat("Positive industries:", sum(sgp_summary$value_2022 > 0), "\n")
cat("Non-finite values:", sum(!is.finite(sgp_summary$value_2022)), "\n")

cat("\nSelected Singapore industries:\n")
selected <- c("SGP_02", "SGP_05", "SGP_06", "SGP_07",
              "SGP_08", "SGP_09", "SGP_19", "SGP_20", "SGP_29")
print(sgp_summary[sgp_summary$industry %in% selected, ], row.names = FALSE)

cat("\n=== POTENTIAL ORIGINAL SOURCE FILES ===\n")
if (dir.exists(source_dir)) {
  files <- list.files(
    source_dir,
    full.names = TRUE,
    recursive = FALSE,
    ignore.case = TRUE
  )

  relevant <- files[
    grepl("\\.(zip|rdata|rda|csv|xlsx|xls)$", files, ignore.case = TRUE) &
    grepl("icio|2016.?2022|2017.?2022|2021.?2022|readme",
          basename(files), ignore.case = TRUE)
  ]

  if (length(relevant) == 0) {
    cat("No matching files found in:", source_dir, "\n")
  } else {
    info <- file.info(relevant)
    result <- data.frame(
      filename = basename(relevant),
      size_MB = round(info$size / 1024^2, 2),
      modified = as.character(info$mtime),
      row.names = NULL
    )
    print(result, row.names = FALSE)
  }
} else {
  cat("Source directory not found:", source_dir, "\n")
}
