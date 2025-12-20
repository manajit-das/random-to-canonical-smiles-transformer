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
