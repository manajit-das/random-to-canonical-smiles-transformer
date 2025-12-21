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
Each file must contain two columns:

src_smiles: randomized (non-canonical) SMILES

tgt_smiles: corresponding canonical SMILES

Place all files inside the data/ directory:

data/
 ├── train.csv
 ├── val.csv
 └── test.csv

Step 2: Preprocessing and Tokenization

Navigate to the preprocess/ directory and run the preprocessing script to tokenize SMILES and build a shared vocabulary:

srun python preprocess_smiles.py \
  --train_csv ./../data/lotus_train42.csv \
  --val_csv ./../data/lotus_val42.csv \
  --test_csv ./../data/lotus_test42.csv \
  --out_dir ./../data/


This step:

Builds a single vocabulary shared across source and target SMILES

Tokenizes SMILES using a regex-based tokenizer

Saves preprocessed datasets for efficient training

Generated files:

data/
 ├── train.pt
 ├── val.pt
 └── test.pt


These .pt files contain tokenized sequences and sequence lengths and are ready for multi-GPU training.

Step 3: Multi-GPU Training

From the project root directory, train the Transformer model using PyTorch Distributed Data Parallel (DDP):

srun torchrun --nproc_per_node=4 train_mgpu.py \
  --data_path ./data \
  --ckpt_path ./checkpoints \
  --num_epochs 100 \
  --batch_size 64


Important notes:

Request the same number of GPUs in your Slurm script:

#SBATCH --gres=gpu:4


A sample Slurm script (run.sh) is provided.

Training uses NCCL backend, DistributedSampler, and synchronized gradient updates.

Model checkpoints are saved to:

checkpoints/

Step 4: Evaluation

Evaluate the trained model on the test set using:

python test.py


This script:

Randomly samples 1000 molecules from the test set

Performs greedy decoding

Reports exact match accuracy and SMILES validity
