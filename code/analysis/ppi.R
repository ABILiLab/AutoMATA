rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(tidyverse)
library(clusterProfiler)
library(STRINGdb)
library(igraph)
library(ggraph)
getOption('timeout')
options(timeout=100000)
library(optparse)


option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/ppi_example.txt", action="store", help="This argument is input path"),
#   make_option(c("-j", "--jobID"), type="character", default="protein_case", action="store", help="This argument is jobID"),  # 默认ppi_example
  make_option(c("-a", "--type"), type="character", default="SYMBOL", action="store", help="This argument is the type of data, inluding SYMBOL, ENTREZID, and ENSEMBL"), 
  make_option(c("-b", "--org"), type="character", default="Homo_sapiens", action="store", help="This argument is the organism, inluding Mus_musculus, Homo_sapiens, Drosophila_melanogaster, and Bos_taurus"),   # 默认Mus_musculus
  make_option(c("-c", "--score_threshold"), type="double", default=400, action="store", help="The interaction results were filtered according to the protein interaction scores."), # 默认400
  make_option(c("-d", "--plot_type"), type="character", default="linear", action="store", help="This parameter is the type of plot, including linear, kk, and stress"), # 默认linear
  make_option(c("-e", "--show_num"), type="integer", default=5, action="store", help="Only gene names with more than <show_num> nodes are shown in the plot") # 默认5
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw PPI!"))
print("opt done")


type = opt$type
score_threshold <- opt$score_threshold
plot_type <- opt$plot_type 
show_num <- opt$show_num 

if (opt$org == "Homo_sapiens"){
    org <- "org.Hs.eg.db"
    species <- 9606
    library(org.Hs.eg.db)
}else if(opt$org == "Mus_musculus"){
    org <- "org.Mm.eg.db"
    species <- 10090
    library(org.Mm.eg.db)
}else if (opt$org == "Bos_taurus"){
    species <- 9913
    org <- "org.Bt.eg.db"
    library(org.Bt.eg.db)
}else if (opt$org == "Drosophila_melanogaster"){
    species <- 7227
    org <- "org.Dm.eg.db" 
    library(org.Dm.eg.db)
}

data <- read.table(opt$input,header = TRUE, check.names = FALSE)
print("read data done")
data <- na.omit(data)

string_db <- STRINGdb$new( version="12", species=species,
                           score_threshold=score_threshold, input_directory="")


if(type == "ENSEMBL"){
    data[,1] <- gsub("\\.\\d+", "", data[,1])
}

if(type != "ENTREZID") {
    data <- data[,1] %>% bitr(fromType = type, toType = "ENTREZID", OrgDb = org, drop = T)  # drop NA or not
}

if(type == "ENTREZID"){
    colnames(data)[1] <- "ENTREZID"
}


data_mapped <- data %>% string_db$map(my_data_frame_id_col_names = "ENTREZID", 
                removeUnmappedRows = TRUE)

hit<-data_mapped$STRING_id
info <- string_db$get_interactions(hit)
links <- info %>%
  mutate(from = data_mapped[match(from, data_mapped$STRING_id), "SYMBOL"]) %>% 
  mutate(to = data_mapped[match(to, data_mapped$STRING_id), "SYMBOL"]) %>%  
  dplyr::select(from, to , last_col()) %>% 
  dplyr::rename(weight = combined_score)
nodes <- links %>% { data.frame(data = c(.$from, .$to)) } %>% distinct()
net <- igraph::graph_from_data_frame(d=links,vertices=nodes,directed = F)
igraph::V(net)$deg <- igraph::degree(net)
igraph::V(net)$size <- igraph::degree(net)/5 
igraph::E(net)$width <- igraph::E(net)$weight/10



# The parameter setting here is related to the size of the node and the number of lines it is connected to. 
# The more lines there are, the larger the node will be. 
# The width of the line is related to the score of its protein-protein interaction; 
# the higher the score, the wider it is.
cir <- FALSE
if(plot_type == "linear"){
    cir <- TRUE
}
p1 <- ggraph(net, layout = plot_type, circular = cir)+
  geom_edge_arc(aes(edge_width=width), color = "lightblue", show.legend = F)+
  geom_node_point(aes(size=size), color="orange", alpha=0.7)+
  geom_node_text(aes(filter=deg>show_num, label=name), size = 1.5, repel = F)+
  scale_edge_width(range = c(0.2,1))+
  scale_size_continuous(range = c(1,10) )+
  guides(size=F)+
  theme_graph()

# save picture
  result_path <- paste(getwd(),"/result/ppi", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
    ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 7.5, height = 6)
}
