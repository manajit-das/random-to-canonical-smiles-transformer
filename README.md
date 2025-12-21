# random-to-canonical-smiles-transformer
## Scalable Seq2Seq Transformer for Molecular SMILES Translation

This repository implements a **distributed, multi-GPU Seq2Seq Transformer**
for molecular SMILES-to-SMILES translation, designed to scale from
single-GPU experiments to multi-GPU DGX systems.

Key features:
- Regex-based chemically valid SMILES tokenization
- Shared src–tgt vocabulary with BOS/EOS/PAD handling
- Offline tokenization for large-scale datasets
- DistributedDataParallel (DDP) training on Slurm clusters
- Batch greedy decoding for inference

Usage
Step 1: Dataset Preparation

Download the dataset consisting of three CSV files: train.csv, val.csv, and test.csv.
Each CSV file must contain two columns:

src_smiles: randomized (non-canonical) SMILES

tgt_smiles: corresponding canonical SMILES

Place all files inside the data/ directory.

data/
 ├── train.csv
 ├── val.csv
 └── test.csv

Step 2: Preprocessing and Tokenization

Navigate to the preprocess/ directory and run the preprocessing script to tokenize SMILES and build a shared vocabulary:

Step 2: Preprocessing and Tokenization

Navigate to the preprocess/ directory and run the preprocessing script to tokenize SMILES and build a shared vocabulary:
