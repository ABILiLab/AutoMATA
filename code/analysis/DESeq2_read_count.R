# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(DESeq2)
library(dplyr)
library(optparse)

option_list <- list(
  make_option(c("-i", "--expression_file"), type="character", default="../../data/analysis_example/expression_read_count.txt", action="store", help="This argument is expression file path"),
  make_option(c("-k", "--info_file"), type="character", default="../../data/analysis_example/group_info_read_count.txt", action="store", help="This argument is group information file path"),
  make_option(c("-c", "--fc_thr"), type="double", action="store", default="1", help="This argument decides log2FC threshold for differential expression analysis"),
  make_option(c("-d", "--padj_thr"), type="double", action="store", default="1", help="This argument decides padj threshold for differential expression analysis")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to conduct differential expression analysis and generate volcano and cluster plots!", add_help_option=FALSE))

# Screening differential gene threshold
fc_thr <- opt$fc_thr
padj_thr <- opt$padj_thr

# Load data
# NOTE: 1. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here
#       2. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2
#       3. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
counts <- read.table(opt$expression_file, row.names = 1, header = TRUE, sep = "\t", check.names = FALSE)
group_info <- read.table(opt$info_file, header = TRUE, sep = "\t", check.names = FALSE)
groups <- factor(group_info$Group, levels = c("Control", "Treatment"))


# Delete rows with low expression (less than 1 read)
counts <- counts[rowMeans(counts)>1,]

# Create  DESeqDataSet object 
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = data.frame(group = groups),
  design = ~ group
)

# Filter low expression genes (at least 10 reads summing)
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]

# Differential expression analysis
dds <- DESeq(dds, minReplicatesForReplace=5, parallel = FALSE)  

# Extract results
DESeq2_results <- results(dds, contrast = c("group", "Treatment", "Control"))
DESeq2_results <- as.data.frame(DESeq2_results)


# Change column names: log2FoldChange -> logFC
DESeq2_results <- dplyr::rename(DESeq2_results, logFC = log2FoldChange)
gene <- rownames(DESeq2_results)
DESeq2_results <- cbind(gene,DESeq2_results)
# Sort the table, ascending by padj value, continuing with log2FC for the same padj value.
DESeq2_results <- DESeq2_results[order(DESeq2_results$padj, DESeq2_results$logFC, decreasing = c(FALSE, TRUE)), ]

# # Save results to file
# write.table(DESeq2_results, filename, row.names = FALSE, sep='\t')

# Screening for differential genes
# log2FC≥1 & padj<0.01, signal is up, which represents a significantly upregulated gene
# log2FC≤-1 & padj<0.01, signal is down, which represents a significantly downregulated gene
# The rest, signal is none, which represents non-distinct genes
DESeq2_results[which(DESeq2_results$logFC >= fc_thr & DESeq2_results$padj < padj_thr),'sig'] <- 'up'
DESeq2_results[which(DESeq2_results$logFC <= -fc_thr & DESeq2_results$padj < padj_thr),'sig'] <- 'down'
DESeq2_results[which(abs(DESeq2_results$logFC) <= fc_thr | DESeq2_results$padj >= padj_thr),'sig'] <- 'none'

# save all up and down genes
res1_select <- subset(DESeq2_results, sig %in% c('up', 'down'))
filename <- paste(getwd(),"/result/select_all.txt", sep="")
write.table(res1_select, file = filename, row.names = FALSE, sep='\t', quote = FALSE)


# save all up genes
res1_up <- subset(DESeq2_results, sig == 'up')
filename <- paste(getwd(),"/result/select_up.txt", sep="")
write.table(res1_up, file = filename, row.names = FALSE, sep='\t', quote = FALSE)

# save all down genes
res1_down <- subset(DESeq2_results, sig == 'down')
filename <- paste(getwd(),"/result/select_down.txt", sep="")
write.table(res1_down, file = filename,  row.names = FALSE, sep='\t', quote = FALSE)



# draw volcano plot
library(ggplot2)
print("Begin Drawing Volcano Plot")
p1 <- ggplot(DESeq2_results, aes(x = logFC, y = -log10(padj))) + 
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

  geom_point(data = DESeq2_results, shape = 21, color = "black", alpha = 0.1, size = 1.2, stroke = 0.7, fill = "grey60") +
  scale_x_continuous(limits = c(floor(sort(res1_down$logFC)[1]), ceiling(max(res1_up$logFC)))) +
  
  labs(x = expression(paste(log[2],"FC",sep="")), y = expression(paste(-log[10]," adj.p-value",sep=""))) +
  theme_classic(base_size = 15) + 
  theme (legend.position = "none")

result_path <- paste(getwd(),"/result/volcano", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 7.5, height = 6)
}

print("End Drawing Volcano Plot")


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
                heatmap_legend_param = list(legend_height = unit(4, "cm"),  # Set the legend height
                                            legend_width = 0.2),  # Set the legend width
                scale = "row")  # "row", "column" and "none"
p1 <- as.ggplot(p1)
result_path <- paste(getwd(),"/result/df_cluster_heatmap", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width=20, height=20)
}
print("End Drawing Cluster Heatmap")

