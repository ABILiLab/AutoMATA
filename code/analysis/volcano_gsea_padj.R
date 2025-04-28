rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(ggplot2)
library(dplyr)
library(ggrepel)
library(optparse)


# Note: Keep logFC, padj, gene columns of dataset consistent


option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/volcano_example.txt", action="store", help="This argument is input path"),
  make_option(c("-g", "--gmt"), type="character", default="../../data/analysis_example/volcano_gsea_example.gmt", action="store", help="This argument is gmt file path. Options are none or gmt file path"),
  make_option(c("-a", "--fc_thr"), type="double", default=0.5, action="store", help="This argument is the threshold of logFC"),  # 0.5  1   1.5   0.1  0.1
  make_option(c("-b", "--padj_thr"), type="double", default=0.05, action="store", help="This argument is the threshold of padj (adjusted p-value)"),  # 0.05  0.05  0.2  0.2
  make_option(c("-c", "--top"), type="integer", default=200, action="store", help="This parameter is the number of higher-order genes to be emphasized"),  # 200  200  30  30
  make_option(c("-d", "--top_fc_thr"), type="double", default=0., action="store", help="This parameter is the threshold of logFC for the higher-order gene"),  # 0  2  0.09  0.09
  make_option(c("-e", "--top_padj_thr"), type="double", default=0.01, action="store", help="This parameter is the threshold of padj for the higher-order gene"),  # 0.01  0.01 0.15  0.15
  make_option(c("-f", "--gene_sig"), type="character", default="KRAS,FOSL1,MYC",action="store", help="This argument is showing marked genes")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw volcano with or without GSEA plot!"))

# read data
data <- read.table(opt$input,header = TRUE, sep = "\t", check.names = FALSE)  # "\t"   ,
data <- na.omit(data)


#  process marked gene
if (opt$gene_sig != ""){
  gene_sig <- unlist(strsplit(opt$gene_sig, ","))
}else{
  gene_sig <- opt$gene_sig
}

# the threshold of logFC and padj
fc_thr <- opt$fc_thr
padj_thr <- opt$padj_thr
# Thresholds for filtering the first 200 genes: top's thresholds top_fc_thr and top_padj_thr
top <- opt$top
if (top != 0){
  top_fc_thr <- opt$top_fc_thr
  top_padj_thr <- opt$top_padj_thr
}



# Extract gene sets
if (opt$gmt != "none"){

  gmt <- readLines(opt$gmt)
  gene_set1 <- strsplit(gmt[1], "\t")[[1]][-c(1,2)]  # Take out the element where the line is and delete the first and second elements
  gene_set2 <- strsplit(gmt[2], "\t")[[1]][-c(1,2)]
  # select genes in gmt file and marked genes
  data_gene_set <- data %>%
  filter(gene %in% c(gene_set1, gene_set2, gene_sig)) %>%
  mutate(gene_set = case_when(
    gene %in% gene_set1 ~ "gene_set1",
    gene %in% gene_set2 ~ "gene_set2",
    gene %in% gene_sig ~ "gene_sig"
  ))
  # other genes
  data_other_genes <- data %>%
    filter(!gene %in% c(gene_set1, gene_set2, gene_sig))
} else{

  # select genes in marked genes
  data_gene_set <- data %>%
  filter(gene %in% c(gene_sig)) %>%
  mutate(gene_set = case_when(
      gene %in% gene_sig ~ "gene_sig"
  ))
  # other genes
  data_other_genes <- data %>%
  filter(!gene %in% c(gene_sig))

}


# Filter DOWN/UP DEGs （Blue/Purple）
down_genes <- data %>% filter(padj < padj_thr, logFC < -fc_thr)
up_genes <- data %>% filter(padj < padj_thr, logFC > fc_thr)

# Filter DOWN/UP && top200 DEGs  (dotted frame)
# TOP thresholds should be tighter than normal thresholds
if (top != 0){
  top_down200_genes <- data %>% 
  filter(padj < top_padj_thr, logFC < -top_fc_thr) %>% 
  arrange(logFC) %>% 
  head(top)

  top_up200_genes <- data %>% 
    filter(padj < top_padj_thr, logFC > top_fc_thr) %>% 
    arrange(desc(logFC)) %>% 
    head(top)
}



# draw volcano
p1 <- ggplot(data, aes(x = logFC, y = -log10(padj))) + 
  annotate("rect", xmin = sort(down_genes$logFC)[1], xmax = max(down_genes$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(down_genes$padj)) >= -log10(min(up_genes$padj)), -log10(min(down_genes$padj)), -log10(min(up_genes$padj))), fill = "#CCDFF1") +
  annotate("rect", xmin = min(up_genes$logFC), xmax = max(up_genes$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(down_genes$padj)) >= -log10(min(up_genes$padj)), -log10(min(down_genes$padj)), -log10(min(up_genes$padj))), fill = "#cfc6fe") +
  
  annotate("text", x = (sort(down_genes$logFC)[1] + max(down_genes$logFC)) / 2, y = ifelse(-log10(min(down_genes$padj)) >= -log10(min(up_genes$padj)), -log10(min(down_genes$padj)), -log10(min(up_genes$padj)))+0.1, label = "DOWN", color = "#5EA7D3", size = 5, lineheight = 0.8, vjust = 0) + # DOWN label
  annotate("text", x = (min(up_genes$logFC) + max(up_genes$logFC)) / 2, y = ifelse(-log10(min(down_genes$padj)) >= -log10(min(up_genes$padj)), -log10(min(down_genes$padj)), -log10(min(up_genes$padj)))+0.1, label = "UP", color = "#b285b2", size = 5, lineheight = 0.8, vjust = 0) + # UP label
  annotate("text", x = max(up_genes$logFC)+0.1, y = -log10(0.05), label = "α = 0.05", color = "#b285b2", size = 4.5) +


  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = -log10(0.05), linetype = "dotted", color = "#b285b2", linewidth = 0.6) +  # α的线
  geom_point(data = data_other_genes, shape = 21, color = "black", alpha = 0.1, size = 1.2, stroke = 0.7) +
  geom_point(data = data_gene_set, aes(fill = gene_set), shape = 21, color = "black", size = 1.8, stroke = 0.8) +
  geom_label_repel(data = filter(data_gene_set, gene_set == "gene_sig"), aes(label = gene), size = 5, box.padding=unit(0.35, "lines"), segment.colour = "grey30") +
  scale_fill_manual(values = c("gene_sig" = "#07F1F9")) +
  
  annotate("rect", xmin = min(top_up200_genes$logFC), xmax = max(top_up200_genes$logFC), ymin = -log10(top_padj_thr), ymax = -log10(min(top_up200_genes$padj)), fill = "transparent", linetype = ifelse(top==0, "blank", "dotted"), color = "black", linewidth = 0.6) +
  annotate("rect", xmin = sort(top_down200_genes$logFC)[1], xmax = max(top_down200_genes$logFC), ymin = -log10(top_padj_thr), ymax = -log10(min(top_down200_genes$padj)), fill = "transparent", linetype = ifelse(top==0, "blank", "dotted"), color = "black", linewidth = 0.6) +
  scale_x_continuous(limits = c(floor(sort(down_genes$logFC)[1]), ceiling(max(up_genes$logFC)))) +
  
  labs(x = expression(paste(log[2],"FC",sep="")), y = expression(paste(-log[10]," adj.p-value",sep=""))) + # y轴：-log10(genes$padj)
  theme_classic(base_size = 15) + 
  theme (legend.position = "none")


if (opt$gmt == "none"){
  # save picture without GSEA
  result_path <- paste(getwd(),"/result/volcano", sep="")
  for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
    ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 7.5, height = 6)
  }
} else {
  # ——GSEA——
  # add signature enrichment bar
  p2 <- p1 + 
    scale_y_continuous(limits = c(-2, -log10(min(up_genes$padj))), expand = c(0, 0)) +
    geom_linerange(data = filter(data_gene_set, gene_set == "gene_set2"), aes(x = logFC, ymin = -1, ymax = -0.35), color = "#FFBF00", size = 0.5, linewidth = 0.1) +
    geom_linerange(data = filter(data_gene_set, gene_set == "gene_set1"), aes(x = logFC, ymin = -1.75, ymax = -1.1), color = "#C6C6C6", size = 0.5, linewidth = 0.1) + 
    scale_fill_manual(values = c("gene_set2" = "#FFBF00", "gene_set1" = "#C6C6C6", "gene_sig" = "#07F1F9")) +
    annotate("text", x = -2.5, y = -0.7, label = "SIGNALING_UP", color = "#FFBF00", size = 4.5) +
    annotate("text", x = -2.5, y = -1.4, label = "SIGNALING_DN", color = "#C6C6C6", size = 4.5)


  # save picture with GSEA
  result_path <- paste(getwd(),"/result/volcano", sep="")
  for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
    ggsave(paste(result_path, dev, sep = "."), p2, device = dev, width = 7.5, height = 6)
  }

}



