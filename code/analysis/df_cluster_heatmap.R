rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(ComplexHeatmap) 
library(ggplot2)
library(ggplotify)
library(optparse) 

# command line parameter
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/df_gene_cluster_example.txt", action="store", help="This argument is input path"),
  make_option(c("-a", "--type"), type="character", default="data_with_row_col", action="store", help="This argument is the type of figure. The optional options data_with_row_col, data_with_col_annotation, data_with_row_annotation or only_data"),
  make_option(c("-b", "--show_row_name"), type="logical", default=FALSE, action="store", help="This argument decides whether to show row name. The optional options are TRUE or FALSE"),
  make_option(c("-c", "--show_col_name"), type="logical", default=TRUE, action="store", help="This argument decides whether to show column name. The optional options are TRUE or FALSE"),
  make_option(c("-d", "--clustering_dis_row"), type="character", action="store", default="euclidean", help="This argument decides the method for clustering distance for rows. The optional options are correlation, euclidean, maximum, manhattan, canberra, binary or minkowski"),
  make_option(c("-e", "--clustering_dis_col"), type="character", default="euclidean", action="store", help="This argument decides the method for clustering distance for columns. The optional options are correlation, euclidean, maximum, manhattan, canberra, binary or minkowski"),
  make_option(c("-g", "--annotation_col_file"), default="../../data/analysis_example/df_gene_cluster_annotation_col.txt", type="character", action="store", help="This argument is the path of column annotation file"),
  make_option(c("-f", "--annotation_row_file"), default="../../data/analysis_example/df_gene_cluster_annotation_row.txt", type="character", action="store", help="This argument is the path of row annotation file"),
  make_option(c("-h", "--scal"), type="character", default="none", action="store", help="This argument decides whether to center and scale data. The optional options are row, column or none"),
  make_option(c("-k", "--group"), type="logical", default=TRUE, action="store", help="This argument decides whether to display data by group. Requires the column annotation file to have a group column name of group")
)

opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw Differential Gene Cluster Heatmap!", add_help_option=FALSE))
# load parameters
type <- opt$type
show_col_name <- opt$show_col_name
show_row_name <- opt$show_row_name
clustering_dis_row <- opt$clustering_dis_row
clustering_dis_col <- opt$clustering_dis_col
scal <- opt$scal
group <- opt$group

if (type == "only_data") {
    # Read expression matrix
    z_score_selected_expression <- read.table(opt$input, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)  # \t
}else if (type == "data_with_col_annotation") {
    # read column annotation file
    annotation_col <- read.table(opt$annotation_col_file, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
    z_score_selected_expression <- read.table(opt$input, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
}else if (type == "data_with_row_annotation") {
    # read row annotation file
    annotation_row <- read.table(opt$annotation_row_file , header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
    z_score_selected_expression <- read.table(opt$input, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
}else {
    annotation_row <- read.table(opt$annotation_row_file , header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
    annotation_col <- read.table(opt$annotation_col_file, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)
    z_score_selected_expression <- read.table(opt$input, header = TRUE, sep = "\t", quote="\"", fill=TRUE, comment.char="", row.names=1)

}


# draw heatmap
if (type == "only_data") {
    p1 <- pheatmap(z_score_selected_expression,
            color = colorRampPalette(c("purple", "white", "yellow"))(255),  # color
            clustering_distance_rows = clustering_dis_row,
            clustering_distance_cols = clustering_dis_col,
            show_colnames = show_col_name,
            show_rownames = show_row_name,
            fontsize = 20,
            fontsize_col =20,  
            heatmap_legend_param = list(legend_height = unit(4, "cm"),  # Set the legend height
                                        legend_width = 0.2),  # Set the legend width
            scale = scal)
    p1 <- as.ggplot(p1)
    
}


if (type == "data_with_col_annotation") {
    # Add column annotation (Add a heatmap above)
    if (group == TRUE){
        # Separated by group
        p1 <- pheatmap(z_score_selected_expression,
                column_split = as.factor(annotation_col$group),
                color = colorRampPalette(c("purple", "white", "yellow"))(255), 
                clustering_distance_rows = clustering_dis_row,
                clustering_distance_cols = clustering_dis_col,
                show_colnames = show_col_name,
                show_rownames = show_row_name,
                annotation_col = annotation_col,
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                            legend_width = 0.2),
                scale = scal)

    }else{
        # No separation by groups
        p1 <- pheatmap(z_score_selected_expression,

                color = colorRampPalette(c("purple", "white", "yellow"))(255),
                clustering_distance_rows = clustering_dis_row,
                clustering_distance_cols = clustering_dis_col,
                show_colnames = show_col_name,
                show_rownames = show_row_name,
                annotation_col = annotation_col,
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                            legend_width = 0.2),
                scale = scal)
    }
    p1 <- as.ggplot(p1)
}


# Add row annotation
if (type == "data_with_row_annotation") {
    p1 <- pheatmap(z_score_selected_expression,
            color = colorRampPalette(c("purple", "white", "yellow"))(255), 
            clustering_distance_rows = clustering_dis_row,
            clustering_distance_cols = clustering_dis_col,
            show_colnames = show_col_name,
            show_rownames = show_row_name,
            annotation_row = annotation_row,
            fontsize = 20,
            fontsize_col =20,  
            heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                        legend_width = 0.2),
            scale = scal)
    p1 <- as.ggplot(p1)
 }

# Add row and column annotation
if (type == "data_with_row_col") {
    if (group == TRUE){
        # Separated by group
        p1 <- pheatmap(z_score_selected_expression,
                column_split = as.factor(annotation_col$group),
                color = colorRampPalette(c("purple", "white", "yellow"))(255), 
                clustering_distance_rows = clustering_dis_row,
                clustering_distance_cols = clustering_dis_col,
                show_colnames = show_col_name,
                show_rownames = show_row_name,
                annotation_col = annotation_col,
                annotation_row = annotation_row,
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                            legend_width = 0.2),
                scale = scal)

    }else{
        # No separated by group
        p1 <- pheatmap(z_score_selected_expression,
                color = colorRampPalette(c("purple", "white", "yellow"))(255),
                clustering_distance_rows = clustering_dis_row,
                clustering_distance_cols = clustering_dis_col,
                show_colnames = show_col_name,
                show_rownames = show_row_name,
                annotation_col = annotation_col,
                annotation_row = annotation_row,
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),
                                            legend_width = 0.2),
                scale = scal) 
    }
    p1 <- as.ggplot(p1)
}

# Enter p1 in the terminal/console to view the picture
# save the picture
result_path <- paste(getwd(),"/result/df_cluster_heatmap", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width=20, height=20)  # 20, 20

}
