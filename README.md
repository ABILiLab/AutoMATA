# AutoMATA

**AutoMATA: a deep learning-enhanced bioinformatics platform for multi-omics data processing, exploration and modelling**

## Code Usage

### 1. Environment

```bash
# Installation of the environment required to train/apply the model
conda env create -f environment.yaml

# Installation of the environment required to analyze data
conda env create -f environment_R.yaml
Rscript install_packages.R
```

### 2. Model training

```bash
# Note:
# About cmd parameters
# 1. kfold: a number represents kfold. 0 means not use kfold, 3 means use 3-fold stratified cross validation
# 2. ratio: a string represents split ratio. "0" means not use split, "8:1:1" means split a dataset into train, validation and test datasets by "8:1:1", and the seperator is colon(:)
# 3. epochs: the number of training epochs
# 4. es: the number of early stopping patience. 'es' is usually smaller than epochs, you can set it to the value same as 'epochs' to disable 'es'
# 5. lr: learning rate
# 6. bs: batch size
# 7. loss_function: please input one of crossentropy, nllloss and focalloss
# 8. optimizer_function: please input one of adam, rmsprop and sgd
# 9. output_size: The number of labels. The range is [2, 7], 2 for binary classification, 4 for 4-class classification.
# 10. random_seed: Keep the same random seed to ensure reproducibility.

# About model parameters
# User can change hidden_size_1 to fit the distribution of data, dropout_rate to prevent overfitting, etc.

# About data
# The first column is the name of samples, the last column is the label of samples (label must be a integer number in order from 0 to 5), the first row is the column name (the last column must be 'Label'). The seperater is '\t'.

# Next is the command line examples to train the model
conda activate automata
cd code/train
# train AutoEncoder (using default parameters)
python autoencoder.py --kfold 0 --ratio 0 --epochs 50 --es 10 --lr 0.01 --bs 32 --loss_function crossentropy --optimizer_function adam --output_size 2 -–random_seed 42
# train CNN
python cnn.py --kfold 0 --ratio 8:1:1 --epochs 50 --es 10 --lr 0.01 --bs 32 --loss_function crossentropy --optimizer_function adam --output_size 4 -–random_seed 42
# train LSTM
python lstm.py --kfold 4 --ratio 0 --epochs 50 --es 10 --lr 0.005 --bs 32 --loss_function nllloss --optimizer_function rmsprop --output_size 2 -–random_seed 42
# train MLP (using default parameters)
python mlp.py
```



### 3. Model application

```bash
# Note:
# About cmd parameters
# 1. bs: batch size
# 2. model_type: please input one of CNN, AutoEncoder, LSTM, MLP, RBFN, RNN and Transformer
# 3. model_path: the path of model you want to use.
# 4. model_autoencoder_path: when 'model_type' is AutoEncoder, the parameter should be specialized to the path of genertated model_autoencoder.pth

# About data
# The first column is the name of samples, the last column is the label of samples (label must be a integer number in order from 0 to 5), the first row is the column name (the last column must be 'Label'). The seperater is '\t'.

# Next is the command line examples to apply the model
conda activate automata
cd code/application
# apply CNN (using default parameters)
python general.py
# apply Transformer
python general.py --model_type Transformer --model_path ./model/model.pth --bs 16
# apply AutoEncoder
python general.py --model_type AutoEncoder --model_path ./model/model_cls.pth --model_autoencoder_path ./model/model_autoencoder.pth --bs 8
# apply SOM
python som.py
```



### 4. Data analysis

```bash
conda activate R_442
cd code/analysis
```

#### **correlation heatmap**

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row of input data file is the column name. The seperater is '\t'.

# cmd examples
Rscript cor_heatmap.r
Rscript cor_heatmap.r -i cor_heatmap_test.txt 
```

#### differential gene cluster heatmap

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row of input data file is the column name. The first column of input data file is the row name. The seperater is '\t'.
# -a (type): the type of output figure, decide whether need to output row/column annotations. Please input one of data_with_row_col, data_with_col_annotation, data_with_row_annotation and only_data. 'data_with_row_col' indicates that annotations for rows and columns need to be output. 'data_with_col_annotation' indicates that annotations for columns need to be output (The group/Age/Grade/Stage/Sex in the example figure). 'data_with_row_annotation' indicates that annotations for rows need to be output (The Path/Celltype in the example figure). 'only_data' indicates that no annotations are output.
# -b (show_row_name): decides whether to show row name. Please TRUE or FALSE.
# -c (show_col_name): decides whether to show column name. Please TRUE or FALSE.
# -d (clustering_dis_row): decides the method for clustering distance for rows. Please input one of correlation, euclidean, maximum, manhattan, canberra, binary and minkowski.
# -e (clustering_dis_col): decides the method for clustering distance for columns. Please input one of correlation, euclidean, maximum, manhattan, canberra, binary and minkowski.
# -g (annotation_col_file): the path of column annotation file. If 'type' parameter is 'data_with_row_col' or 'data_with_col_annotation', then 'annotation_col_file' should be specialized.
# -f (annotation_row_file): the path of row annotation file. If 'type' parameter is 'data_with_row_col' or 'data_with_row_annotation', then 'annotation_row_file' should be specialized.
# -h (scal): decides whether to center and scale data. Please input one of row, column or none. 'row' and 'column' denote scale data to row and column respectively, 'none' represent do not scale data.
# -k (group): decides whether to display data by group. Requires the column annotation file to have a group column name of group. Please input TRUE or FALSE.

# cmd examples
Rscript df_cluster_heatmap.R
Rscript df_cluster_heatmap.R -i df_gene_cluster_test.txt -a data_with_col_annotation -g df_gene_cluster_annotation_col.txt -c FALSE -b TRUE -d correlation -e correlation -h row -k TRUE
```

#### PCA

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column of the data is the group information. The seperater is '\t'.
# -c (confidence_level): confidence level
# -b (boundary): decides whether to add boundary plots, which are the top and right sub-plots of the figure. Please input TRUE or FALSE.
# -p (permanova): decides whether to add PERMANOVA analysis. Please input TRUE or FALSE.
# -m (method): the method for permanova analysis. Please input one of 'manhattan', 'euclidean', 'canberra', 'clark', 'bray', 'kulczynski', 'jaccard', 'gower', 'altGower', 'morisita', 'horn', 'mountford', 'raup', 'binomial', 'chao', 'cao', 'mahalanobis', 'chisq', 'chord', 'hellinger', 'aitchison', and 'robust.aitchison'

# cmd examples
Rscript pca.R
Rscript pca.R -i pca_test.txt -c 0.92 -b FALSE -p FALSE
```

#### VENN

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row is the column names, and each column is the information for each group. The seperater is '\t'.
# -t (type): the type of plot. Please input one of Venn, Vennpie and Barplot.

# cmd examples
Rscript venn.R
Rscript venn.R -i venn_test.txt -t Vennpie
```

#### volcano with GSEA

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. Keep logFC, padj, gene column names of dataset consistent. The seperater is '\t'.
# -g (gmt): the path of gmt data file. If you want to conduct GSEA analysis, then specialize it to the path gmt data file, otherwise set it to 'none'.
# -a (fc_thr): the threshold of logFC. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -b (padj_thr): the threshold of padj (adjusted p-value). If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -c (top): the number of higher-order genes to be emphasized. The emphasized genes are in the dashed box.
# -d (top_fc_thr): the threshold of logFC for the higher-order gene. It should be more strict than 'fc_thr'
# -e (top_padj_thr): the threshold of padj for the higher-order gene. It should be more strict than 'padj_thr'
# -f (gene_sig): the list of marked genes, the separator must be comma ','

# cmd examples
Rscript volcano_gsea_padj.R
Rscript volcano_gsea_padj.R -i volcano_test.txt -g none -a 1 -b 0.05 -c 30 -d 2 -e 0.01 -f KRAS,FOSL1
```

#### GO Enrichment

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column is gene symbol, and the column name is gene. The data file only needs the gene symbol if 'type' is bubble or bar, otherwise needs to make sure that the data file has numeric data with the column name logFC to sort by the size of the value.
# -a (type): the type of figure. Please input one of bubble, bar, chord, cluster and circle.
# -b (organism): Please input one of Homo_sapiens, Mus_musculus, Bovine, Homo_sapiens and Drosophila_melanogaster.
# -c (pvalue): pvalue threshold for GO enrichment analysis. The strict pvalue threshold is 0.01 (pvalue<0.01), and lenient threshold is 0.05 (pvalue<0.05).
# -d (qvalue): qvalue threshold for GO enrichment analysis. The gold standard is 0.05.
# -e (termNum): the number of terms for each ontology to be displayed
# -g (adjust): the pvalue adjustment method for GO enrichment analysis. Please input one of holm, hochberg, hommel, bonferroni, BH, BY, fdr, none

# cmd examples
Rscript go_enrichment.R
Rscript go_enrichment.R -i go_enrichment_test.txt -a bubble -b Mus_musculus -c 0.01 -e 10 -g BH
```

#### KEGG Enrichment

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column is gene symbol, and the column name is gene. The data file only needs the gene symbol if 'type' is bubble, otherwise needs to make sure that the data file has numeric data with the column name logFC to sort by the size of the value.
# -a (type): the type of figure. Please input one of bubble, chord, and cluster.
# -b (organism): Please input one of hsa, mmu, bos, and dme.
# -c (pvalue): pvalue threshold for KEGG enrichment analysis. The strict pvalue threshold is 0.01 (pvalue<0.01), and lenient threshold is 0.05 (pvalue<0.05).
# -d (qvalue): qvalue threshold for KEGG enrichment analysis. The gold standard is 0.05.
# -e (termNum): the number of terms for each ontology to be displayed
# -g (adjust): the pvalue adjustment method for KEGG enrichment analysis. Please input one of holm, hochberg, hommel, bonferroni, BH, BY, fdr, none

# cmd examples
Rscript kegg_enrichment.R
Rscript kegg_enrichment.R -i kegg_enrichment_test.txt -a bubble -b dme -d 0.01 -e 10 -g BH
```

#### Dumbbell_Bar

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. This file is used to draw dumbbell diagrams
# -c (data_bar): the file path of bar plot. This file is used to draw barplot diagrams. Keep the contents and name of the first column in both two files similar and same respectively.
# -a (y_label): the y label of dumbbell_bar plot
# -b (mark_fams): the terms that will be marked, which are factors in the first column of data files, the separator must be comma ','

# cmd examples
Rscript dumbbell_bar.R
Rscript dumbbell_bar.R -i dumbbell_test.txt -c dumbbell_barplot_test.txt -a number -b Notothenioid

```

#### DESeq2 for read count

```bash
# Note:
# About cmd parameters
# -i (expression_file): the path of expression file. The first column is row names, the first row is column names.
# -k (info_file): the path of group information file. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
# -c (fc_thr): the log2FC threshold for differential expression analysis. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -d (padj_thr): the padj threshold for differential expression analysis. If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -e (correction): This argument defines hypothesis correction method. Please input one of none, BH, BY, holm, hochberg, hommel, or bonferroni

# cmd examples
Rscript DESeq2_read_count.R
Rscript DESeq2_read_count.R -i expression.txt -k info.txt -c 2 -d 0.05 -e BH
```

#### limma for fpkm

```bash
# Note:
# About cmd parameters
# -i (expression_file): the path of expression file. The first column is row names, the first row is column names.
# -k (info_file): the path of group information file. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
# -c (fc_thr): the log2FC threshold for differential expression analysis. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -d (padj_thr): the padj threshold for differential expression analysis. If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -e (correction): This argument defines hypothesis correction method. Please input one of none, BH, BY, holm, hochberg, hommel, bonferroni

# cmd examples
Rscript limma_fpkm_df.R
Rscript limma_fpkm_df.R -i expression.txt -k info.txt -c 2 -d 0.05 -e BH
```

#### PPI Network

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row is column name, and the first column is gene. The file can be consist of gene symbol, ENTREZID, or ENSEMBL
# -a (type): this argument is the type of data, please input one of SYMBOL, ENTREZID, and ENSEMBL
# -b (organism): this argument is the organism, please input one of Mus_musculus, Homo_sapiens, Drosophila_melanogaster, and Bos_taurus
# -c (score_threshold): the interaction results were filtered according to the protein interaction scores
# -d (plot_type): this parameter is the type of plot, please input one of linear, kk, and stress
# -e (show_num): only genes with more than <show_num> nodes are shown in the plot

# cmd examples
Rscript ppi.R
Rscript ppi.R -i ppi_example.txt -a SYMBOL -b Homo_sapiens -c 400 -d linear -e 5
```



### 5. **Hyperparameter optimization**

1. kfold: Generally start by trying 3, 5, or 10; 5-fold cross-validation is commonly used.

2. Dataset split ratio: Common train:validation:test set split ratios are 7:1.5:1.5 or 8:1:1. More training data is generally better.
3. Batch size: Typically start by trying 32, 64, or 128. Smaller batch sizes usually provide better generalization but train slower. Larger batch sizes train faster but may overfit.
4. Learning rate: Usually start training with 0.1, 0.01, 0.001, or 0.0001. Adjust the learning rate based on the Acc-loss curves. If the training loss curve does not decrease, try increasing the learning rate; if both training and validation loss curves plateau, it might be due to too small a learning rate or model convergence.
5.  Optimizer: Adam has strong adaptability and is the default optimizer; SGD requires more fine-tuned learning rate adjustment; RMSprop performs well in RNNs and certain tasks. We recommend trying Adam first, then SGD if higher accuracy is needed.
6. Loss Function: CrossEntropy is the standard loss for multi-class problems; NLLLoss is used with LogSoftmax; Focal Loss is used to address class imbalance issues. Users can choose the appropriate loss function based on the class distribution of their dataset.
7. Epochs: Number of training rounds. Users can initially set a relatively large value (e.g., 200) to observe the loss and accuracy during training and determine when overfitting or underfitting starts. Reduce epochs if overfitting occurs, and increase epochs if underfitting occurs. Rely on early stopping mechanisms to prevent overfitting.
8. Early stopping patience: Start by trying 5, 8, or 10. If the validation loss fluctuates significantly, increase the patience value.



## Docker Usage

### 1. Unzip the project

The docker version can be downloaded at Zenodo (DOI: 10.5281/zenodo.15294581) and HuggingFace (https://huggingface.co/spaces/CongWang33/AutoMATA).

**AutoMATA.zip** is the docker version of AutoMATA webserver.

```bash
unzip AutoMATA.zip -d AutoMATA
cd AutoMATA
```

### 2. Build image

```bash
docker-compose build
```

### 3. Start service

```bash
docker-compose up -d
```


## NOTE
AutoMATA to be free to academia, charge for industry.
