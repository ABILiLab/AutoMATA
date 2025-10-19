# Read package list
# packages <- read.csv("package_R.csv")

tryCatch({
  packages <- read.csv("/tmp/package_R.csv", header = TRUE, fileEncoding="UTF-8")
}, error = function(e) {

  print(list.files("/tmp"))  # 打印目录内容
  stop(e)
})

# 添加清华源
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
# Check and install BiocManager
if (any(packages$Source == "Bioconductor")) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
}
BiocManager::install(repos = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor")



 


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
