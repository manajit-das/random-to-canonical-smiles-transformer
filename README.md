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

🚀 Usage
1️⃣ Dataset Preparation

Download the dataset containing three CSV files:

train.csv

val.csv

test.csv

Each CSV must contain the following columns:

src_smiles → Randomized (non-canonical) SMILES

tgt_smiles → Canonical SMILES

Place the files inside the data/ directory:

```text
data/
 ├── train.pt
 ├── val.pt
 └── test.pt
```


2️⃣ Preprocessing & Tokenization

Navigate to the preprocess/ directory and run the preprocessing script:

srun python preprocess_smiles.py \
  --train_csv ./../data/lotus_train42.csv \
  --val_csv ./../data/lotus_val42.csv \
  --test_csv ./../data/lotus_test42.csv \
  --out_dir ./../data/


This step performs the following:

✅ Builds a shared vocabulary from both source and target SMILES

✅ Tokenizes SMILES using a regex-based tokenizer

✅ Stores tokenized sequences and lengths for fast training

Output files:

data/
 ├── train.pt
 ├── val.pt
 └── test.pt


These .pt files are ready for multi-GPU Transformer training.

3️⃣ Multi-GPU Training (DDP)

From the project root, train the model using PyTorch Distributed Data Parallel (DDP):

srun torchrun --nproc_per_node=4 train_mgpu.py \
  --data_path ./data \
  --ckpt_path ./checkpoints \
  --num_epochs 100 \
  --batch_size 64


⚠️ Make sure your Slurm script requests the same number of GPUs:

#SBATCH --gres=gpu:4


A sample Slurm launcher (run.sh) is provided.

📦 Trained model checkpoints are saved to:

checkpoints/

4️⃣ Evaluation

Evaluate the trained model on the test set:

python test.py


The evaluation script:

🔹 Randomly samples 1000 molecules from the test set

🔹 Performs greedy decoding

🔹 Reports:

Exact match accuracy

SMILES validity
