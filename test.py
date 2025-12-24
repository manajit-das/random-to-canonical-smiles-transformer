
import torch
from models import MyModel
import re
from rdkit import Chem
import time
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from preprocess.data_utils import TokenizedSeq2SeqDataset, my_collate_fn, SmilesTokenizer

st_time = time.time()

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def safe_canonicalize(smiles):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def pad_and_mask(tokens, seq_len):
      max_len = max(seq_len)
      batch_size = len(tokens)
      padded = torch.zeros(batch_size, max_len, dtype=torch.long)
      att_masks = torch.zeros(batch_size, max_len, dtype=torch.float)
      for i, seq in enumerate(tokens):
          L = len(seq)
          padded[i, :L] = torch.tensor(seq)
          att_masks[i, :L] = 1.0
      seq_len = torch.tensor(seq_len, dtype=torch.long)
      return padded, att_masks, seq_len



def test_batch_greedy_decode(model, test_tok_path, tokenizer, device, max_len=50, batch_size=64):
    """
    Batch greedy decoding for SMILES canonicalization.
    src_smiles_list: list[str]
    returns: list[str]
    """
    # ---------- Source preparation ----------
    test_data = TokenizedSeq2SeqDataset(test_tok_path)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=my_collate_fn)
    
    batch_src_smiles =[]
    batch_pred_smiles = []

    for batch in test_loader:
        src_tok_batch, src_att_mask_batch, src_seq_len, tgt_padded, tgt_att_masks, tgt_seq_len = batch
        src_smiles = [tokenizer.decode(i) for i in src_tok_batch]
        batch_src_smiles.extend(src_smiles)

        src_tok_batch = src_tok_batch.to(device)
        src_att_mask_batch = src_att_mask_batch.to(device)
        
        B = len(src_tok_batch)
        
        # ---------- Target initialization ----------
        tgt_sequences = [['BOS'] for _ in range(B)]
        Finished = torch.zeros(B, dtype=torch.bool)  # stays persistent

        PAD_ID = tokenizer.char_to_int['PAD']
        EOS_ID = tokenizer.char_to_int['EOS']

        model.eval()
        with torch.no_grad():

            for _ in range(max_len):

                # ---- build decoder input (B, t) ----
                tgt_toks = []
                tgt_toks_len = []

                for seq in tgt_sequences:
                    ids = [tokenizer.char_to_int[t] for t in seq]
                    tgt_toks.append(ids)
                    tgt_toks_len.append(len(ids))

                tgt_tok_batch, tgt_att_mask_batch, _ = pad_and_mask(
                    tgt_toks, tgt_toks_len
                )

                tgt_tok_batch = tgt_tok_batch.to(device)
                tgt_att_mask_batch = tgt_att_mask_batch.to(device)

                # ---- forward pass ----
                logits = model(
                    src_tok_batch,
                    src_att_mask_batch,
                    tgt_tok_batch,
                    tgt_att_mask_batch,
                )

                # ---- greedy next-token selection ----
                next_tokens = torch.argmax(logits[:, -1, :], dim=1)  # (B,)

                # ---- update sequences ----
                for i, tok_id in enumerate(next_tokens):

                    if Finished[i]:
                        tgt_sequences[i].append('PAD')
                        continue

                    if tok_id.item() == EOS_ID:
                        Finished[i] = True
                        tgt_sequences[i].append('PAD')
                    else:
                        tgt_sequences[i].append(tokenizer.int_to_char[tok_id.item()])

                # ---- stopping condition ----
                if Finished.all():
                    break

        # ---------- Post-processing ----------
        final_smiles_list = []

        for seq in tgt_sequences:
            # remove BOS and everything after PAD
            cleaned = []
            for tok in seq[1:]:
                if tok == 'PAD':
                    break
                cleaned.append(tok)
            final_smiles_list.append(''.join(cleaned))
        
        batch_pred_smiles.extend(final_smiles_list)

    return batch_src_smiles, batch_pred_smiles

'''
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='batch wise prediction')
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--bs', type=int, defualt=64)
    parser.add_argument('--max_len', type=int, default=600)
    return parser.parse_args()


def main():
    args = arg_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"./{args.model_path}/aug_seq2seq_model.pt", map_location=device)
    vocab_list = checkpoint["vocab"]
    tokenizer = SmilesTokenizer(vocab_list)
    model = MyModel(len(tokenizer.vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    true_smiles_list, out = test_batch_greedy_decode(model, f"./{args.test_path}/test.pt", max_len=args.max_len, batch_size=args.bs)

    num_correct = 0
    num_valid = 0
    N = len(true_smiles_list)

    for pred, tgt in tqdm(zip(out, true_smiles_list), total=len(true_smiles_list)):
        if is_valid_smiles(pred):
            num_valid += 1

        tgt = safe_canonicalize(tgt) #sometime tgt itself contain "<UNK>"
        if tgt is None:
            continue

        if pred == tgt:
            num_correct += 1

    acc = 100.0 * num_correct / N
    validity = 100.0 * num_valid / N

    print(f"Exact-match accuracy: {acc:.2f}%")
    print(f"Validity: {validity:.2f}%")
    en_time = time.time()
    print(f"Time required: {(en_time-st_time)/60}")
    if args.save_pred:
        df = pd.DataFrame(zip(true_smiles_list, out))
        df.to_csv(f"./{args.test_path}/prediction.csv", index=False)
'''

import argparse
import os
import time
import torch
import pandas as pd
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description="Batch inference for SMILES canonicalization")
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=600)
    return parser.parse_args()


def main(args):
    # --- paths ---
    test_pt = os.path.join(args.test_path, "test.pt")
    ckpt_file = os.path.join(args.model_path, "aug_seq2seq_model.pt")

    # --- load checkpoint ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_file, map_location=device)

    vocab_list = checkpoint["vocab"]
    tokenizer = SmilesTokenizer(vocab_list)

    model = MyModel(len(tokenizer.vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- run decoding ---
    true_smiles_list, out = test_batch_greedy_decode(
        model,
        test_pt,
        tokenizer,
        device,
        max_len=args.max_len,
        batch_size=args.bs
    )

    num_valid = 0
    num_correct = 0
    valid_targets = 0
    N = len(true_smiles_list)

    for pred, tgt in tqdm(zip(out, true_smiles_list), total=N):

        if is_valid_smiles(pred):
            num_valid += 1

        tgt_canon = safe_canonicalize(tgt)
        if tgt_canon is None:     # skip bad targets containing <UNK>
            continue

        valid_targets += 1

        if pred == tgt_canon:
            num_correct += 1

    acc = 100.0 * num_correct / valid_targets if valid_targets > 0 else 0
    validity = 100.0 * num_valid / N

    print(f"Exact-match accuracy: {acc:.2f}%")
    print(f"Validity: {validity:.2f}%")


    if args.save_pred:
        df = pd.DataFrame({
            "true_smiles": true_smiles_list,
            "pred_smiles": out
        })
        df.to_csv(os.path.join(args.test_path, "prediction.csv"), index=False)
        print("Prediction saved to prediction.csv")


if __name__ == "__main__":
    start = time.time()
    args = arg_parse()
    main(args)
    print(f"Time required: {(time.time() - start)/60:.2f} minutes")


