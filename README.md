# scGAD

Graph attention autoencoder model with dual decoder for clustering single-cell RNA sequencing data


## Requirements

Python --- 3.6

Numpy --- 1.16.4

Pandas --- 1.1.5

Scipy --- 1.5.4

Sklearn --- 0.24.2

Keras --- 2.2.4


##Datasets

Ting is obtained from https://github.com/shaoqiangzhang/scRNAseq_Datasets

Darmanis is obtained from https://sourceforge.net/projects/transcriptomeassembly/files/SC3-e/Data/

Buettner, Pollen and Kolodziejczyk are obtained from https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/data

Muraro and Baron-mouse are obtained from https://sourceforge.net/projects/transcriptomeassembly/files/SD-h/Data/

Mouse_ES, Mouse_bladder_cell and PBMC4k are obtained from https://github.com/ttgump/scDeepCluster/tree/master/scRNA-seq%20data

Young and Quake_10X_Spleen are obtained from https://github.com/xuebaliang/scziDesk/tree/master/dataset

PBMC3k is obtained from https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k


##Example

The scGAD.py is an example of the dataset Buettner.

Input: data.csv( Rows represent cells, columns represent genes ), n_clusters( the number of cluster ), true_labels.csv

Run: python scGAD.py Buettner 3 --subtype_path dataset/Buettner_true_labs.csv

Output: pre_labels, NMI, ARI.
