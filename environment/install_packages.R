# Read package list
packages <- read.csv("package_R.csv")

# Check and install BiocManager
if (any(packages$Source == "Bioconductor")) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
}

# Sub-source installation packages
for (pkg in packages$Package) {
  source <- packages$Source[packages$Package == pkg]
  if (source == "CRAN") {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
  } else if (source == "Bioconductor") {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      BiocManager::install(pkg, update = FALSE) # Disable update of installed packages
    }
  } else {
    message(paste("Package", pkg, "from unknown source. Install manually."))
  }
}