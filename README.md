# random-to-canonical-smiles-transformer

## Scalable Seq2Seq Transformer for Molecular SMILES Translation

This repository implements a distributed, multi-GPU Seq2Seq Transformer for molecular SMILES-to-SMILES translation.  
The system is designed to scale from single-GPU experiments to large multi-GPU DGX systems using PyTorch Distributed Data Parallel (DDP).

---

## Key Features

- Regex-based chemically valid SMILES tokenization
- Shared source–target vocabulary with BOS / EOS / PAD handling
- Offline tokenization for large-scale datasets
- DistributedDataParallel (DDP) training on Slurm clusters
- Batch greedy decoding for inference

---

## Usage

### 1. Dataset Preparation

Prepare three CSV files:

- `train.csv`
- `val.csv`
- `test.csv`

Each CSV file must contain the following columns:

- `src_smiles`: Randomized (non-canonical) SMILES  
- `tgt_smiles`: Canonical SMILES  

Place the files inside the `data/` directory:


---

### 2. Preprocessing and Tokenization

Navigate to the `preprocess/` directory and run the preprocessing script:

```bash
srun python preprocess_smiles.py \
  --train_csv ./../data/lotus_train42.csv \
  --val_csv ./../data/lotus_val42.csv \
  --test_csv ./../data/lotus_test42.csv \
  --out_dir ./../data/

This step performs the following operations:

Builds a shared vocabulary from both source and target SMILES

Tokenizes SMILES using a regex-based tokenizer

Stores tokenized sequences and sequence lengths for efficient training

The generated output files are:

data/
 ├── train.pt
 ├── val.pt
 └── test.pt

These .pt files are ready for multi-GPU Transformer training.


### 2. Preprocessing and Tokenization

Navigate to the `preprocess/` directory and run the preprocessing script:

```bash
srun python preprocess_smiles.py \
  --train_csv ./../data/lotus_train42.csv \
  --val_csv ./../data/lotus_val42.csv \
  --test_csv ./../data/lotus_test42.csv \
  --out_dir ./../data/

This step performs the following operations:

Builds a shared vocabulary from both source and target SMILES

Tokenizes SMILES using a regex-based tokenizer

Stores tokenized sequences and sequence lengths for efficient training

The generated output files are:

data/
 ├── train.pt
 ├── val.pt
 └── test.pt

These .pt files are ready for multi-GPU Transformer training.





