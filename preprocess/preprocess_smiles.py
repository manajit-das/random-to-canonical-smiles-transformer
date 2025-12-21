import argparse
import time
from pathlib import Path

import torch
import pandas as pd

from data_utils import create_vocab, SmilesTokenizer


SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "<UNK>"]


def load_smiles(csv_path):
    df = pd.read_csv(csv_path)
    src = df["src_smiles"].astype(str).tolist()
    tgt = df["tgt_smiles"].astype(str).tolist()
    return src, tgt

def save_dataset(
    out_path,
    src_tokens,
    tgt_tokens,
    src_lengths,
    tgt_lengths,
    vocab_list,
):
    torch.save(
        {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "vocab": vocab_list,
        },
        out_path,
    )


def process_split(csv_path, out_path, vocab_list, tokenizer):
    src, tgt = load_smiles(csv_path)
    # tokenize
    src_tok, src_len = tokenizer.batch_encode(src, add_bos_eos=False)
    tgt_tok, tgt_len = tokenizer.batch_encode(tgt, add_bos_eos=True)

    save_dataset(
        out_path,
        src_tok,
        tgt_tok,
        src_len,
        tgt_len,
        vocab_list,
    )

    print(f"Saved {len(src_tok)} samples -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize SMILES train/val/test datasets"
    )

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)

    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument(
        "--vocab_out",
        default="vocab.pt",
        help="Where to save the vocabulary",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Build vocab from TRAIN ONLY (best practice)
    # --------------------------------------------------
    train_src, train_tgt = load_smiles(args.train_csv)
    vocab = create_vocab(train_src + train_tgt)
    vocab_list = SPECIAL_TOKENS  + sorted(vocab)
    tokenizer = SmilesTokenizer(vocab_list)
     
    torch.save({
        "vocab": tokenizer.vocab,
        },
        "vocab.pt",)


    # --------------------------------------------------
    # Process splits
    # --------------------------------------------------
    process_split(
        args.train_csv,
        out_dir / "train.pt",
        vocab_list,
        tokenizer
    )

    process_split(
        args.val_csv,
        out_dir / "val.pt",
        vocab_list,
        tokenizer
    )

    process_split(
        args.test_csv,
        out_dir / "test.pt",
        vocab_list,
        tokenizer
    )

    elapsed_min = (time.time() - start_time) / 60
    print(f"Total preprocessing time: {elapsed_min:.2f} minutes")


if __name__ == "__main__":
    main()
