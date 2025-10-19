rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(limma)
library(dplyr)
library(pheatmap)
library(ggplot2)
library(optparse) 

option_list <- list(
  make_option(c("-i", "--expression_file"), type="character", default="../../data/analysis_example/expression_fpkm.txt", action="store", help="This argument is expression file path"),
  make_option(c("-k", "--info_file"), type="character", default="../../data/analysis_example/group_info_fpkm.txt", action="store", help="This argument is group info information file path"),
  make_option(c("-c", "--fc_thr"), type="double", action="store", default="2", help="This argument decides log2FC threshold for differential expression analysis"),
  make_option(c("-d", "--padj_thr"), type="double", action="store", default="2", help="This argument decides padj threshold for differential expression analysis"),
  make_option(c("-e", "--correction"), type="character", action="store", default="BH", help="This argument defines hypothesis correction method, including none, BH, BY, holm, hochberg, hommel, bonferroni")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to conduct differential expression analysis and generate volcano and cluster plots!", add_help_option=FALSE))

# set threshold
fc_thr <- opt$fc_thr
padj_thr <- opt$padj_thr

# read data
fpkm <- read.table(opt$expression_file, row.names = 1, header = TRUE, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE, fill = TRUE, comment.char = "",quote = "")
group_info <- read.table(opt$info_file, header = TRUE, sep = "\t", check.names = FALSE, fill = TRUE, comment.char = "")

# Delete this line if the line name is empty
if(any(is.na(fpkm[,1]))) {
  fpkm <- fpkm[!is.na(fpkm[,1]), ]
  cat("The row with the gene name NA has been deleted\n")
}

# If there are duplicate line names
if(any(duplicated(fpkm[,1]))) {
  rownames(fpkm) <- make.unique(as.character(fpkm[,1])) 
  fpkm <- fpkm[, -1]
}

# Delete rows with too low an expression
fpkm <- fpkm[which(rowSums(fpkm)!=0),]

# Empty value processing KNN imputation
if(any(is.na(fpkm))) {
  # KNN imputation
  cat("begin KNN imputation\n")
  library(impute)
  fpkm <- impute.knn(as.matrix(fpkm))$data
}

log_fpkm <- log2(fpkm + 1) #log process
log_fpkm[log_fpkm == -Inf] = 0 # Replace the logged negative infinity value with 0


# construct design matrix
groups <- factor(group_info$Group, levels = c("Control", "Treatment"))
design <- model.matrix(~0 + groups)
colnames(design) <- levels(groups)

# run linear model
fit <- lmFit(log_fpkm, design)
contrasts <- makeContrasts(Treatment - Control, levels = design)
fit2 <- contrasts.fit(fit, contrasts)
fit2 <- eBayes(fit2, trend=TRUE)

# Extract the results of the differential expression analysis
diff_results <- topTable(fit2, coef = 1, number = Inf, adjust.method = opt$correction)


# rename column names
diff_results <- dplyr::rename(diff_results, pvalue = P.Value, padj = adj.P.Val)  # P.Value -> pvalue, adj.P.Val -> padj
gene <- rownames(diff_results)
diff_results <- cbind(gene,diff_results)
# Sort the table, ascending by padj value, continuing in descending log2FC order for the same padj value.
diff_results <- diff_results[order(diff_results$padj, diff_results$logFC, decreasing = c(FALSE, TRUE)), ]


# # Save results to file
# write.table(diff_results, filename, row.names = FALSE, sep='\t')


# Screening for differential genes
# log2FC≥1 & padj<0.01, signal is up, which represents a significantly upregulated gene
# log2FC≤-1 & padj<0.01, signal is down, which represents a significantly downregulated gene
# The rest, signal is none, which represents non-distinct genes
diff_results[which(diff_results$logFC >= fc_thr & diff_results$padj < padj_thr),'sig'] <- 'up'
diff_results[which(diff_results$logFC <= -fc_thr & diff_results$padj < padj_thr),'sig'] <- 'down'
diff_results[which(abs(diff_results$logFC) <= fc_thr | diff_results$padj >= padj_thr),'sig'] <- 'none'

# save all up and down genes
res1_select <- subset(diff_results, sig %in% c('up', 'down'))
filename <- paste(getwd(),"/result/select_all.txt", sep="")
write.table(res1_select, file = filename, row.names = FALSE, sep='\t', quote = FALSE)

# save all up genes
res1_up <- subset(diff_results, sig == 'up')
filename <- paste(getwd(),"/result/select_up.txt", sep="")
write.table(res1_up, file = filename, row.names = FALSE, sep='\t', quote = FALSE)

# save all down genes
res1_down <- subset(diff_results, sig == 'down')
filename <- paste(getwd(),"/result/select_down.txt", sep="")
write.table(res1_down, file = filename,  row.names = FALSE, sep='\t', quote = FALSE) 


# draw volcano plot
library(ggplot2)
print("Begin Drawing Volcano Plot")
p1 <- ggplot(diff_results, aes(x = logFC, y = -log10(padj))) + 
  annotate("rect", xmin = sort(res1_down$logFC)[1], xmax = max(res1_down$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj))), fill = "#CCDFF1") + #文章颜色#E6E7FC DOWN
  annotate("rect", xmin = min(res1_up$logFC), xmax = max(res1_up$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj))), fill = "#cfc6fe") + #文章颜色#FDE7E9 #DCEABB  #ffe7ff(浅粉)  #cfc6fe(浅紫) UP
  
  annotate("text", x = (sort(res1_down$logFC)[1] + max(res1_down$logFC)) / 2, y = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj)))+0.1, label = "DOWN", color = "#5EA7D3", size = 5, lineheight = 0.8, vjust = 0) + #文章颜色#857CD9  # DOWN label
  annotate("text", x = (min(res1_up$logFC) + max(res1_up$logFC)) / 2, y = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj)))+0.1, label = "UP", color = "#b285b2", size = 5, lineheight = 0.8, vjust = 0) + #文章颜色#FF7D81  # UP label
  annotate("text", x = max(res1_up$logFC)+0.1, y = -log10(0.05), label = "α = 0.05", color = "#b285b2", size = 4.5) +

  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = -log10(0.05), linetype = "dotted", color = "#b285b2", linewidth = 0.6) +

  geom_point(data = diff_results, shape = 21, color = "black", alpha = 0.1, size = 1.2, stroke = 0.7, fill = "grey60") +
  scale_x_continuous(limits = c(floor(sort(res1_down$logFC)[1]), ceiling(max(res1_up$logFC)))) +
  
  labs(x = expression(paste(log[2],"FC",sep="")), y = expression(paste(-log[10]," adj.p-value",sep=""))) + 
  theme_classic(base_size = 15) + 
  theme (legend.position = "none")

result_path <- paste(getwd(),"/result/volcano", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 7.5, height = 6)
}


# draw df_cluster_heatmap
library(ComplexHeatmap)
library(ggplotify)
print("Begin Drawing Cluster Heatmap")
df <- counts[intersect(rownames(counts),rownames(res1_select)),]
df2<- as.matrix(df)  
col_annotation <- group_info                                               
rownames(col_annotation) <- col_annotation[,1]
col_annotation <- col_annotation[,-1,drop=FALSE]

p1 <- pheatmap(df2,
                column_split = as.factor(group_info$Group),
                color = colorRampPalette(c("purple", "white", "yellow"))(255),
                clustering_distance_rows = "euclidean",
                clustering_distance_cols = "euclidean",
                show_colnames = T,
                show_rownames = T,
                annotation_col = col_annotation,
                annotation_colors = list(Group=c(Control='#cfc6fe',Treatment='#CCDFF1')),
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                            legend_width = 0.2),
                scale = "row")

result_path <- paste(getwd(),"/result/df_cluster_heatmap", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width=20, height=20)
}



