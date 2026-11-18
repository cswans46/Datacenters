source_dir <- "D:/Research/OECD Research"

archives <- c(
  "ICIO2025econA.zip",
  "ICIO2025econB.zip",
  "ICIO2025econZ.zip",
  "ICIO2025econOTHER.zip"
)

for (filename in archives) {
  path <- file.path(source_dir, filename)

  cat("\n===", filename, "===\n")

  if (!file.exists(path)) {
    cat("File not found\n")
    next
  }

  contents <- tryCatch(
    unzip(path, list = TRUE),
    error = function(e) {
      cat("Could not list archive:", conditionMessage(e), "\n")
      NULL
    }
  )

  if (is.null(contents)) next

  cat("Files in archive:", nrow(contents), "\n")
  cat("Total uncompressed size (GB):",
      round(sum(contents$Length, na.rm = TRUE) / 1024^3, 2), "\n")

  cat("First 8 entries:\n")
  print(head(contents[, c("Name", "Length")], 8), row.names = FALSE)

  cat("Entries with CSV, RData, or XLS extensions:\n")
  relevant <- contents[
    grepl("\\.(csv|rdata|rda|xls|xlsx)$",
          contents$Name, ignore.case = TRUE),
  ]

  if (nrow(relevant) == 0) {
    cat("None identified\n")
  } else {
    print(head(relevant[, c("Name", "Length")], 12), row.names = FALSE)
  }
}