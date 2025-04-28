rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(clusterProfiler)
library(dplyr)
library(GOplot)
library(DOSE)
library(ggplot2)
library(tidyr)
library(yulab.utils)
library(optparse)


# NOTE: Display images sorted by GeneRatio. 
# If you want to use chord/cluster, make sure that the uploaded file has numeric data with the column name logFC to sort by the size of the value, and make sure that the first column is the gene name (symbol).
# Images of type bubble only need the gene name (symbol).

option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/kegg_enrichment_example.txt", action="store", help="This argument is input path"),
  make_option(c("-a", "--type"), type="character", default="bubble", action="store", help="This argument is the type of figure. Options are bubble, chord, or cluster."),
  make_option(c("-b", "--organism"), type="character", default="hsa", action="store", help="This argument decides organism. Options are hsa, mmu, bos, or dme."),
  make_option(c("-c", "--pvalue"), type="double", default=0.05, action="store", help="This argument decides pvalue threshold for KEGG enrichment analysis"),
  make_option(c("-d", "--qvalue"), type="double", default=0.05, action="store", help="This argument decides qvalue threshold for KEGG enrichment analysis"),
  make_option(c("-e", "--termNum"), type="integer", default=5, action="store", help="This argument is the number of terms for each ontology to be displayed")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw KEGG Enrichment!", add_help_option=FALSE))

# Parameters of enrichment analysis (significance level)
pvalue <- opt$pvalue
qvalue <- opt$qvalue

termNum <- opt$termNum
type <- opt$type
org <- opt$organism 

# read data
table_data <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)



# Extract the first column of the table, the gene name column
gene_column <- table_data[[1]]
# Store EntrezID for genes
entrez_ids <- c()

# Get the EntrezID of the gene
for (gene in gene_column){
    id <- tryCatch({
        if (org == "hsa"){
            library(org.Hs.eg.db)
            mget(gene, org.Hs.egSYMBOL2EG)
        }else if(org == "mmu"){
            library(org.Mm.eg.db)
            mget(gene, org.Mm.egSYMBOL2EG)
        }else if (org == "bos"){
            library(org.Bt.eg.db)
            mget(gene, org.Bt.egSYMBOL2EG)
        }else if (org == "dme"){
            library(org.Dm.eg.db)
            mget(gene, org.Dm.egSYMBOL2EG)
        }
    }, error = function(e) {
        NA
    })
    entrez_ids <- c(entrez_ids, id)
}



# add entrez_ids column to the table_data
entrez_ids <- as.character(entrez_ids)
table_data$entrezID <- entrez_ids
table_data <- table_data[!is.na(table_data$entrezID), ]



# KEGG enrichment analysis
gene <- table_data$entrezID
KEGG <- enrichKEGG(gene = gene, 
                    organism = org, 
                    pvalueCutoff = pvalue, 
                    qvalueCutoff = qvalue)

KEGG <- as.data.frame(KEGG)


KEGG$geneID <- as.character(sapply(KEGG$geneID, function(x) {
    ids = strsplit(x, split = "/")[[1]]
    idx = match(ids, as.character(table_data$entrezID))
    symbols = table_data$gene[idx]
    paste(symbols, collapse = "/")
}))

# # Filter enrichment analysis results
# pval <- 0.05
# adj <- 0.05
# KEGG <- KEGG[KEGG$pvalue < pval & KEGG$p.adjust < adj, ]

# save analysis results
result_path <- paste(getwd(),"/result/kegg_enrichment", sep="")
filename <- paste(result_path, "_result.txt", sep="")
write.table(KEGG, file = filename, sep = "\t", row.names = FALSE, quote = FALSE)



KEGG <- KEGG[order(KEGG$Count, decreasing = TRUE), ] # Sort by Count descending
# create dataframe
kegg <- data.frame(
    Category ="ALL",
    ID = KEGG$ID,
    Term = KEGG$Description,
    Genes = gsub("/", ", ", KEGG$geneID),
    adj_pval = KEGG$p.adjust
)
# Extract gene and logFC from raw data, create dataframe genelist
genelist <- data.frame(
    ID = table_data$gene,
    logFC = table_data$logFC
)
# Use the circle_dat function to create data for a circular layout for subsequent circular plots
KEGGcirc <- circle_dat(kegg, genelist)
termNum <- ifelse(nrow(kegg) < termNum, nrow(kegg), termNum)
geneNum <- nrow(genelist)


# chord plot
if (type == "chord"){
    # kegg$Term[1:termNum] is the GO entry that needs to be displayed
    chord <- chord_dat(KEGGcirc, genelist[1:geneNum, ], kegg$Term[1:termNum])
    if (!is.null(chord)){
        gochord <- GOChord(chord,
                space=0.02,
                gene.order="logFC",
                gene.space=0.25,
                gene.size = 5,
                process.label = 10
                ) 
    }else{
        print("No created chord object")
    }
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), gochord, device = dev, width = 15, height = 15)
    }

}

# cluster plot
if (type == "cluster"){
    chord <- chord_dat(KEGGcirc, genelist[1:geneNum, ], kegg$Term[1:termNum])
    if (!is.null(KEGG) && nrow(KEGG) > 0){
        keggcluster <- GOCluster(KEGGcirc, as.character(KEGG[1:termNum, 3]))
    }else{
        print("No KEGG enrichment data found, we can not draw KEGG cluster plot")
    }
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), keggcluster, device = dev, width = 15, height = 8)
    }
}


# bubble plot
if (type == "bubble"){
    KEGG.top <- head(KEGG, n = termNum)
    KEGG.top$Description <- factor(KEGG.top$Description, levels = c(KEGG.top$Description %>% as.data.frame() %>% pull()))
    # The GeneRatio column in the KEGG.top dataframe is rounded (to three decimal places)
    ev = function(x) {eval(parse(text=x))}
    KEGG.top$GeneRatio <- round(sapply(KEGG.top$GeneRatio, ev), 3)
    # drawing bubble plot
    bubble <- ggplot(KEGG.top, aes(x=GeneRatio, y=Description, color = -log10(pvalue))) +
        geom_point(aes(size=Count)) +
        theme_bw() +
        scale_y_discrete(labels = function(y) str_wrap(y, width = 50)) + 
        labs(size = "Counts", x = "GeneRatio", y = NULL, title=NULL) +
        scale_color_gradient(low = "#a392fa", high = "#fee153", guide = "colorbar", name = "log10(p-value)")+ # Color gradient Settings
        theme(axis.text = element_text(size = 16, color = "black"),
              axis.title = element_text(size = 16, color = "black"),
              title = element_text(size = 18, color = "black")) +
        guides(color = guide_colorbar(reverse = TRUE))

    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), bubble, device = dev, width = 15, height = 8)
    }
}




