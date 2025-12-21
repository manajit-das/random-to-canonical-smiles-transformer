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
```

This step performs the following operations:

1. Builds a shared vocabulary from both source and target SMILES

2. Tokenizes SMILES using a regex-based tokenizer

3. Stores tokenized sequences and sequence lengths for efficient training

The generated output files are:

```text
data/
 ├── train.pt
 ├── val.pt
 └── test.pt
```
These `.pt` files are ready for multi-GPU Transformer training.

### 3. Multi-GPU Training

From the project root directory, train the Transformer model using Distributed Data Parallel (DDP):

```bash
srun torchrun --nproc_per_node=4 train_mgpu.py \
  --data_path ./data \
  --ckpt_path ./checkpoints \
  --num_epochs 100 \
  --batch_size 64
```
Important notes:

Request the same number of GPUs in your Slurm script:

```bash
#SBATCH --grus=gpu:4
```
A sample Slurm script (run.sh) is provided for reference.

Training uses PyTorch DDP, DistributedSampler, and NCCL backend.

Model checkpoints are saved to: `checkpoints`

### 4. Evaluation

Run inference and evaluation on the test set using:


```bash
python test.py
```

The script:

Samples 1000 random test molecules

Performs greedy decoding

Reports exact match accuracy and SMILES validity
```

```
## Project Structure

```text
random-to-canonical-smiles-transformer/
├── data/
│   ├── lotus_train.csv
│   ├── lotus_val.csv
│   └── lotus_test.csv
├── preprocess/
│   └── preprocess_smiles.py
├── model/
│   └── transformer.py
├── train_mgpu.py
├── test.py
├── run.sh
├── requirements.txt
└── README.md
```








