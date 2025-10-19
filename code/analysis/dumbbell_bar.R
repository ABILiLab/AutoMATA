rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(ggplot2)
library(dplyr)
library(patchwork)
library(optparse) 

# NOTE: Keep the contents and name of the first column in both files similar and the same, respectively.


# command line parameter
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/dumbbell_example.txt", action="store", help="This argument is the file path of dumbbell plot"),
  make_option(c("-c", "--data_bar"), type="character", default="../../data/analysis_example/dumbbell_barplot_example.txt", action="store", help="This argument is the file path of bar plot"),
  make_option(c("-a", "--y_label"), type="character", action="store", default="Number of Transitions", help="This argument is the y label of dumbbell plot"),
  make_option(c("-b", "--mark_fams"), type="character", default="Notothenioid,Sebastidae,Liparidae,Zoarcidae,Pleuronectidae", action="store", help="This argument is terms that will be marked")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw dumbbell with bar plot!"))


y_label <- opt$y_label  # label

if (opt$mark_fams != ""){  # Split the marked terms
  mark_fams <- unlist(strsplit(opt$mark_fams, ","))
}else{
  mark_fams <- opt$mark_fams
}

# read data
data_dumbbell <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)
data_bar <- read.table(opt$data_bar, header = TRUE, sep = "\t", check.names = FALSE)

# rename column names
colnames(data_bar) <- c("family", "tridepth", "nsp")
colnames(data_dumbbell) <- c("family", "observed_num", "expected_median", "expected_quartile0.05", "expected_quartile0.95")

# Sort by the number of (observed_num-expected_median)
data_dumbbell_sorted <- data_dumbbell %>%
  arrange(observed_num-expected_median) %>%
  mutate(col = case_when(
          observed_num >= expected_quartile0.05 & observed_num <= expected_quartile0.95 ~ "Within expectation",
          observed_num > expected_quartile0.95 ~ "Above expectation",
          observed_num < expected_quartile0.05 ~ "Below expectation"))

order <- data_dumbbell_sorted$family

# all colors
barcolors <- c("#DDF7FF","#C0DBFF","#93AFF8" ,"#5E7DC2", "#2C5494")

# Sort by family order
data_bar$family <- factor(data_bar$family, levels = order)
data_bar_sorted <- data_bar[order(data_bar$family), ]

# set text colors
famcol <- ifelse(levels(as.factor(data_bar$family)) %in% mark_fams, "#F8766D", "grey30")

# drawing
# stacked bar plot
p1 <- ggplot(data_bar_sorted, 
    aes(x = family, y = nsp, fill = factor(tridepth, levels = unique(data_bar[, 2])))) +
    geom_bar(position = "fill", stat = "identity", width = 0.7) +
    scale_fill_manual(values = barcolors[1:length(unique(data_bar[, 2]))]) +  # Customize, use as many colors as there are different values in a column
    coord_flip() +
    labs(x = NULL, y = NULL) + 
    theme_classic() +
    theme(axis.text.y = element_text(colour = famcol), 
            axis.text.x = element_blank(),
            axis.ticks = element_blank(),
            axis.line = element_blank(),
            legend.margin = margin(1, -8, 1, 1),
            legend.title = element_blank(),
            legend.text = element_text(size = 9, colour = 'grey30', margin = margin(r = 10)),
            legend.key.size = unit(0.45,'cm'),
            legend.key.spacing.x = unit(-1, 'cm'),
            legend.key.spacing.y = unit(0, 'cm'),
            # legend.position = c(-0.8, -0.03), 
            legend.position = c(0, -0.03), 
            legend.background = element_rect(fill = 'white', colour = 'grey50',
                                            linewidth = 0.3)) + 
    guides(fill = guide_legend(ncol = 2))


# dumbbell plot
data_dumbbell_sorted$family <- factor(data_dumbbell_sorted$family, levels = order)
p2 <- ggplot(data_dumbbell_sorted, aes(x = family, y = observed_num)) +
              geom_segment(aes(x = family, xend = family,   # Thick grey line
                              y = expected_quartile0.05, yend = expected_quartile0.95),
                          col = "grey90", lwd = 3, lineend = "round") +
              geom_segment(aes(x = family, xend = family, # Black line
                              y = expected_median, yend = observed_num),
                          col = "grey50", lwd = 0.6) +
              geom_point(aes(x = family, y = expected_median),  # white circle
                          pch = 21, fill = "white", col = "grey50", size = 3, stroke = 1) +
              geom_point(aes(col = factor(col, levels = c("Above expectation", "Within expectation", "Below expectation"))), size = 3.3) +  # Colored dot
              coord_flip() +
              ylab(y_label) +
              scale_color_manual(values = c("#00A08A", "grey40","#F2AD00")) +
              theme_classic() +
              theme(plot.margin = unit(c(0, 0.4, 0, 0), "cm"),
                      panel.grid.major.x = element_blank(),
                      panel.border = element_blank(),
                      panel.grid.major.y = element_line(size = 0.4),
                      axis.text.y = element_blank(),
                      axis.line.x = element_line(size = 0.2),
                      axis.line.y = element_blank(),
                      axis.ticks.x = element_line(size = 0.1),
                      axis.ticks.y = element_blank(),
                      axis.title = element_text(size = 12),
                      axis.title.y = element_blank(),
                      legend.margin = margin(1, 5, 1, 1),
                      legend.title = element_blank(),
                      legend.text = element_text(size = 9, color = "grey30"),
                      legend.key.spacing.y = unit(-0.15, 'cm'),
                      legend.position = c(0.75, 0.15), 
                      legend.background = element_rect(fill = 'white', color = "grey50",
                                                      linewidth = 0.3))

p1 <- p1 + p2 + plot_layout(widths = c(1, 9))


# Enter p1 in the terminal/console to view the picture
# save picture
result_path <- paste(getwd(),"/result/dumbbell_bar", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 5, height = 6.2)

}
