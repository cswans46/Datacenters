
source_dir <- "D:/Research/OECD Research"

env <- new.env()
load(file.path(source_dir, "ICIO2025econFD.RData"), envir = env)

cat("=== LOADED OBJECTS ===\n")
print(ls(env))

for (nm in ls(env)) {
  obj <- env[[nm]]
  cat("\nObject:", nm, "\n")
  cat("Class:", paste(class(obj), collapse = ", "), "\n")
  cat("Dimensions:", paste(dim(obj), collapse = " x "), "\n")
  if (!is.null(dimnames(obj))) {
    for (i in seq_along(dimnames(obj))) {
      cat("Dimension", i, "first labels:",
          paste(head(dimnames(obj)[[i]], 5), collapse = ", "), "\n")
    }
  }
}
