rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")

library(pacman)
pacman::p_unload(pacman::p_loaded(), character.only = TRUE)

library(ggplot2) 
library(vegan) 
library(dplyr)  
library(ggrepel) 
library(ggpubr) 
library(patchwork) 
library(optparse)

option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/pca_example.txt", action="store", help="This argument is input path"),
  make_option(c("-c", "--confidence_level"), type="double", default=0.95, action="store", help="This argument is confidence level"),
  make_option(c("-b", "--boundary"), type="logical", default=TRUE, action="store", help="This argument decides whether to add boundary plot"),
  make_option(c("-p", "--permanova"), type="logical", default=TRUE, action="store", help="This argument decides whether to add PERMANOVA analysis"),
  make_option(c("-m", "--method"), type="character", default="bray", action="store", help="If permanova is TRUE, this argument is PERMANOVA method, can be 'manhattan', 'euclidean', 'canberra', 'clark', 'bray', 'kulczynski', 'jaccard', 'gower', 'altGower', 'morisita', 'horn', 'mountford', 'raup', 'binomial', 'chao', 'cao', 'mahalanobis', 'chisq', 'chord', 'hellinger', 'aitchison', or 'robust.aitchison'.")

)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw PCA!"))

# Significance star
get_stars <- function(p){
    if (p <= 0.001) {
    return("***")
  } else if (p <= 0.01) {
    return("**")
  } else if (p <= 0.05) {
    return("*")
  } else {
    return("")
  }
}

confidence_level <- opt$confidence_level
method <- opt$method 
permanova <- opt$permanova 
boundary <- opt$boundary

# Read the data. The first column of the data is the group information, the column name is Group.
table_data <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)

# PCA was performed and standardized, and pca was performed by removing group information.
pca <- summary(rda(dplyr::select(table_data, colnames(table_data)[-1]), scale=T))

# Record the proportion explained of PC1 and PC2
pc1_Explained <- round(pca$cont$importance[2, 1]*100, 2)
pc2_Explained <- round(pca$cont$importance[2, 2]*100, 2)

# Extract the coordinates of each sample on the PC axis and information on the contribution of each variable to the PC axis
coords <- data.frame(pca$sites) %>% mutate(group = c(table_data[,1]))
coords$group <- factor(coords$group, levels = unique(table_data[,1]))
var <- data.frame(pca$species) %>% mutate(func = rownames(pca$species))
var$func <- factor(var$func, levels = rownames(pca$species), labels = rownames(pca$species))

# PERMANOVA analysis
if (permanova){
    nova <- adonis2(vegdist(dplyr::select(table_data, colnames(table_data)[-1]), method=method) ~ Group, data = table_data)
    R2 <- round(nova$R2[1], 3)
    Pr <- round(nova$`Pr(>F)`[1], 4)
    significance_stars <- get_stars(Pr)  # star
}


# determine the size and shape of the confidence ellipse.
group_unique <- unique(coords$group)
oval_data <- lapply(group_unique, function(g){
    subset <- dplyr::filter(coords, group == g)
    mean_data <- colMeans(subset[, c("PC1", "PC2")])
    cov_data <- cov(subset[, c("PC1", "PC2")])
    oval_point <- ellipse::ellipse(cov_data, centre = mean_data, level = confidence_level)
    data.frame(Group = g, oval_point)
})
oval_data <- do.call(rbind, oval_data)



# drawing the PCA plot
p1 <- ggplot()+
  geom_point(data = coords, aes(x = PC1, y = PC2, fill = group), size = 3, color = "transparent", shape = 21)+
  geom_segment(data = var, aes(x = 0, y = 0, xend = -1.25 * PC1, yend = 1.25 * PC2),
                arrow = arrow(angle = 22.5, length = unit(0.25, "cm"), type = "closed")) + 
  geom_text_repel(data = var, aes(x = -1.275 * PC1, y = 1.275 * PC2, label = func), size = 3.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey") +
  geom_path(data = oval_data, aes(x = PC1, y = PC2, group = Group, color = Group), show.legend = FALSE, linetype = "dashed") +
  geom_polygon(data = oval_data, aes(x = PC1, y = PC2, group = Group, fill = Group), alpha = 0.2) + 
  scale_color_manual(values = c("#1F77B4FF", "#FF7F0EFF", "#2CA02CFF")) +
  scale_fill_manual(values = c("#1F77B4FF", "#FF7F0EFF", "#2CA02CFF")) +
  labs(x = paste("PC1 (", pc1_Explained, "%)",sep=""), y = paste("PC2 (", pc2_Explained, "%)",sep="")) +
  scale_x_continuous(limits = c(min(coords$PC1)-1.5, max(coords$PC1)+1.5)) + 
  scale_y_continuous(limits = c(min(coords$PC2)-1.5, max(coords$PC2)+1.9)) + 
  theme_classic2() + 
  theme(legend.title = element_blank(),
        legend.key.size = unit(35, "pt"),
        axis.line = element_line(color = "black"),
        axis.ticks = element_blank())

if (permanova){
    p1 <- p1 + annotate("text", label = paste0("PERMANOVA", "\n", "R^2", " = ", R2, "\n","p = ", Pr, significance_stars), x = -1.7, y = 2.5)  # 添加PERMANOVA结果
}

if (boundary){
    legend <- get_legend(p1) # extract legend
    legend <- as_ggplot(legend)
    p1 <- p1 + theme(legend.position = "none")

    # boundary plot of PC1 axis 
    p2 <- ggplot(data = coords) +
    geom_density(aes(x = PC1, fill=group), alpha = 0.2, 
                color = 'black', position = 'identity', show.legend = FALSE) +
    scale_fill_manual(values=c("#1F77B4FF","#FF7F0EFF","#2CA02CFF")) +
    scale_x_continuous(limits = c(min(coords$PC1)-1.5, max(coords$PC1)+1.5)) +
    theme_classic() +
    theme(legend.title = element_blank(),
            axis.title = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank())


    # # boundary plot of PC2 axis 
    p3 <- ggplot(data = coords) + 
    geom_density(aes(x = PC2, fill=group), alpha = 0.2, 
                color = 'black', position = 'identity', show.legend = FALSE) +
    scale_fill_manual(values=c("#1F77B4FF","#FF7F0EFF","#2CA02CFF")) + 
    scale_x_continuous(limits = c(min(coords$PC2)-1.5, max(coords$PC2)+1.9)) +
    theme_classic() +
    theme(legend.title = element_blank(),
            axis.title = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank()) +
    coord_flip() 


    # Customized Layout Structure
    design <- "224
               113
               113"
    p1 <- p1 + p2 + p3 + legend + plot_layout(design = design) 

}


# save
result_path <- paste(getwd(),"/result/pca", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 8.8, height = 6)

}
