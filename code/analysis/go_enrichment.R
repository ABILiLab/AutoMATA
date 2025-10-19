rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(clusterProfiler)
library(GOplot)
library(ggplot2)
library(optparse)
library(tidyr)
library(dplyr)
library(DOSE) 


# NOTE: Display images sorted by GeneRatio. 
# If you want to use chord/cluster/circle, make sure that the uploaded file has numeric data with the column name logFC to sort by the size of the value, and make sure that the first column is the gene name (symbol).
# Images of type bubble and bar only need the gene name (symbol).

option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/go_enrichment_example.txt", action="store", help="This argument is input path"),
  make_option(c("-a", "--type"), type="character", action="store",default="bubble", help="This argument is the type of figure. Options are bubble, bar, chord, cluster, circle"),
  make_option(c("-b", "--organism"), type="character", default="Homo_sapiens", action="store", help="This argument decides organism. Options are Homo_sapiens, Mus_musculus, Bovine, Homo_sapiens, Drosophila_melanogaster"),
  make_option(c("-c", "--pvalue"), type="double", action="store", default="0.05", help="This argument decides pvalue threshold for GO enrichment analysis"),
  make_option(c("-d", "--qvalue"), type="double", action="store", default="0.05", help="This argument decides qvalue threshold for GO enrichment analysis"),
  make_option(c("-e", "--termNum"), type="integer", default="5", action="store", help="This argument is the number of terms for each ontology to be displayed"),
  make_option(c("-g", "--adjust"), type="character", default="BH", action="store", help="This argument is the pvalue adjustment method for GO enrichment analysis, one of holm, hochberg, hommel, bonferroni, BH, BY, fdr, none")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw GO Enrichment!", add_help_option=FALSE))

organism <- opt$organism
if (organism == "Homo_sapiens"){
    library(org.Hs.eg.db)
    db <- "org.Hs.eg.db"
}else if(organism == "Mus_musculus"){
    library(org.Mm.eg.db)
    db <- "org.Mm.eg.db"
}else if (organism == "Bovine"){
    library(org.Bt.eg.db)
    db <- "org.Bt.eg.db"
}else if (organism == "Drosophila_melanogaster"){
    library(org.Dm.eg.db)
    db <- "org.Dm.eg.db" 
}

# Parameters of enrichment analysis (significance level)
pvalue <- opt$pvalue 
qvalue <- opt$qvalue 
type <- opt$type
termNum <- opt$termNum
adjust <- opt$adjust

# read data
table_data <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)

# Extract the first column of the table, the gene name column
gene_column <- table_data[[1]]
# Store EntrezID for genes
entrez_ids <- c()

# Get the EntrezID of the gene
for (gene in gene_column){
    id <- tryCatch({
        if (organism == "Homo_sapiens"){
            mget(gene, org.Hs.egSYMBOL2EG)[[1]][1]
        }else if(organism == "Mus_musculus"){
            mget(gene, org.Mm.egSYMBOL2EG)[[1]][1]
        }else if (organism == "Bovine"){
            mget(gene, org.Bt.egSYMBOL2EG)[[1]][1]
        }else if (organism == "Fly"){
            mget(gene, org.Dm.egSYMBOL2EG)[[1]][1]
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


# GO enrichment analysis
gene <- table_data$entrezID
GO <- enrichGO(gene = gene, 
               OrgDb = db, 
               pvalueCutoff = pvalue, 
               qvalueCutoff = qvalue, 
               ont = "ALL",  # "ALL", "BP", "MF", and "CC" 
               readable  = TRUE,
               pAdjustMethod = adjust)


# # Filter enrichment analysis results
# pval <- 0.05
# adj <- 0.05
# GO <- GO[(GO$p.adjust < adj) & (GO$pvalue < pval), ]

# save analysis results
result_path <- paste(getwd(),"/result/go_enrichment", sep="")
filename <- paste(result_path, "_result.txt", sep="")
write.table(GO, file = filename, sep = "\t", row.names = FALSE)


# Plotting the results of GO enrichment analysis
GO <- GO[order(GO$Count, decreasing = TRUE), ]  # Sort by Count descending
# bubble plot
if (type == "bubble"){
    # Determine whether the GO result is null
    if (length(GO) > 0){
        # Sorted by GeneRatio
        bubble <- dotplot(GO, showCategory = termNum, split="ONTOLOGY", orderBy="GeneRatio") +   
            facet_grid(ONTOLOGY ~ ., scale = "free")  # Create faceted meshes, one for each ontology
    }else{
        print("No GO enrichment data found")
    }

    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), bubble, device = dev, width = 15, height = 10)
    }
}


# bar plot
if (type == "bar"){
    if (length(GO) > 0){
        bar <- barplot(GO, drop=TRUE, showCategory = termNum, split="ONTOLOGY", orderBy="GeneRatio") +
            facet_grid(ONTOLOGY ~ ., scale = "free")
    }else{
        print("No GO enrichment data found")
    }
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), bar, device = dev, width = 15, height = 10)
    }
}


# prepare data for chord/cluster/circle plot
if (type == "chord" || type == "cluster" || type == "circle"){
    # Convert filtered GO results to dataframe (ONTOLOGY, ID, Description, geneID, p.adjust)
    go <- data.frame(
        Category = GO$ONTOLOGY,
        ID = GO$ID,
        Term = GO$Description,
        Genes = gsub("/", ", ", GO$geneID), 
        adj_pval = GO$p.adjust
    )
    # Extract gene and logFC from raw data, create dataframe genelist
    genelist <- data.frame(
        ID = table_data$gene,
        logFC = table_data$logFC
    )
    row.names(genelist) <- genelist$ID

    # Use the circle_dat function to create data for a circular layout for subsequent circular plots
    circ <- circle_dat(go, genelist)

    # Set termNum based on the number of terms in the go data box, if it is less than termNum then use the actual number of terms
    termNum <- ifelse(nrow(go) < termNum, nrow(go), termNum)

    # Get the number of genes in genelist
    geneNum = nrow(genelist)
}

# chord plot
if (type == "chord"){
    # go$Term[1:termNum] is the GO entry that needs to be displayed
    chord <- chord_dat(circ, genelist[1:geneNum, ], go$Term[1:termNum])
    if (!is.null(chord)){
        print("Drawing GO chord plot")
        gochord <- GOChord(chord,
                space=0.02,
                gene.order="logFC",  # "logFC", "alphabetical", "none"
                gene.space=0.25,
                gene.size = 5,  # gene name size
                process.label = 10
                ) 
    }else{
        print("No created chord object")
    }
    for(dev in c("png", "pdf", "jpeg", "tiff", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), gochord, device = dev, width = 15, height = 15)
    }

}

# cluter plot
if (type == "cluster"){
    chord <- chord_dat(circ, genelist[1:geneNum, ], go$Term[1:termNum])
    if (!is.null(GO) && nrow(GO) > 0){
        gocluster <- GOCluster(circ, as.character(GO[1:termNum, 3]))
    }else{
        print("No GO enrichment data found, we can not draw GO cluster plot")
    }
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), gocluster, device = dev, width = 15, height = 8)
    }

}

# circle plot
if (type == "circle"){
    go_circ <- GOCircle(circ, zsc.col=c("purple", "black", "cyan"), label.size=4)
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), go_circ, device = dev, width = 15, height = 8)
    }

}

