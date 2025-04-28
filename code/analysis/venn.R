rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(VennDetail)
library(ggplot2)
library(dplyr)
library(VennDiagram) 
library(ggupset)
library(optparse)



option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/venn_example.txt", action="store", help="This argument is input path"),
  make_option(c("-t", "--type"), type="character", default="Venn", action="store", help="This argument is type of plot. Options are Venn, Vennpie, Barplot.")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw Venn Plot!"))

type <- opt$type
gene_dataset <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)

# Delete empty data
final_list <- list()
for (i in 1:ncol(gene_dataset)){
    print(i)
    gene_list <- gene_dataset[,i]
    new_list <- character()
    for (i in (1:length(gene_list))){
        if (gene_list[i] != ""){
            append_gene <- as.character(gene_list[i])
            new_list <- append(new_list,append_gene)
        }
    }
    print(new_list)
    final_list <- append(final_list,list(new_list))
}

result_path <- paste(getwd(),"/result/venn", sep="")

# draw venn plot
ven <- venndetail(final_list)
if (type=="Venn"){
    # classic venn
    pdf(paste(result_path, ".pdf",sep=""), width = 8, height = 8)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()  # Turn off the device to save the file
    
    jpeg(paste(result_path, ".jpeg",sep=""), width = 800, height = 800)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()
    
    tiff(paste(result_path,".tiff",sep=""), width = 800, height = 800)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()
    
    png(paste(result_path,".png",sep=""), width = 800, height = 800)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()
    
    bmp(paste(result_path,".bmp",sep=""), width = 800, height = 800)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()
    
    svg(paste(result_path,".svg",sep=""), width = 8, height = 8)
    plot(ven, type="venn", col="black", cat.cex=ncol(gene_dataset), alpha=0.5,cex=3)
    dev.off()
}else if (type=="Vennpie"){
    # Vennpie
    pdf(paste(result_path,".pdf",sep=""), width = 8, height = 8)
    plot(ven, type="vennpie")
    dev.off()

    jpeg(paste(result_path,".jpeg",sep=""), width = 800, height = 800)
    plot(ven, type="vennpie")
    dev.off()

    tiff(paste(result_path,".tiff",sep=""), width = 800, height = 800)
    plot(ven, type="vennpie")
    dev.off()

    png(paste(result_path,".png",sep=""), width = 800, height = 800)
    plot(ven, type="vennpie")
    dev.off()

    bmp(paste(result_path,".bmp",sep=""), width = 800, height = 800)
    plot(ven, type="vennpie")
    dev.off()

    svg(paste(result_path,".svg",sep=""), width = 8, height = 8)
    plot(ven, type="vennpie")
    dev.off()
}else if (type=="Barplot") {
    # barplot
    p1 <- dplot(ven, order = TRUE, textsize = 4)
    for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
        ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 10, height = 8)
    }

}else{
    # upset plot
    plot(ven, type="upset",cat.cex=ncol(gene_dataset),cex=3)
    dev.off()
}


