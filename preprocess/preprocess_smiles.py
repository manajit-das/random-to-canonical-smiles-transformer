# preprocess_smiles.py
import re
import torch
import pandas as pd
import time


st_time = time.time()


SMI_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|"
    r"b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|\+|\\/|:|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9])"
)

csv_path = "lotus_train42.csv"
out_path = "lotus_train42.pt"

df = pd.read_csv(csv_path)
src_smiles = df["src_smiles"].astype(str).tolist()
tgt_smiles = df["tgt_smiles"].astype(str).tolist()

# --------------------------------------------------
# 1. Build shared vocabulary
# --------------------------------------------------
def create_vocab(smiles_list):
    vocab = set()
    for smiles in smiles_list:
        tokens = re.findall(SMI_REGEX_PATTERN, smiles)
        vocab.update(tokens)
    return vocab

vocab = create_vocab(src_smiles + tgt_smiles)

special_tokens = ["PAD", "BOS", "EOS", "<UNK>"]
vocab_list = special_tokens + sorted(vocab)

char_to_int = {tok: i for i, tok in enumerate(vocab_list)}
int_to_char = {i: tok for i, tok in enumerate(vocab_list)}

PAD_IDX = char_to_int["PAD"]
BOS_IDX = char_to_int["BOS"]
EOS_IDX = char_to_int["EOS"]
UNK_IDX = char_to_int["<UNK>"]

# --------------------------------------------------
# 2. Tokenization
# --------------------------------------------------
def tokenize_smiles(smiles):
    tokens = re.findall(SMI_REGEX_PATTERN, smiles)
    return [char_to_int.get(t, UNK_IDX) for t in tokens]

def tokenize_smiles_list(smiles_list, add_bos_eos=False):
    tokenized = []
    lengths = []

    for s in smiles_list:
        ids = tokenize_smiles(s)
        if add_bos_eos:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        tokenized.append(torch.tensor(ids, dtype=torch.long))
        lengths.append(len(ids))

    return tokenized, lengths

src_tokenized, src_lengths = tokenize_smiles_list(src_smiles, add_bos_eos=False)
tgt_tokenized, tgt_lengths = tokenize_smiles_list(tgt_smiles, add_bos_eos=True)

# --------------------------------------------------
# 3. Save
# --------------------------------------------------
torch.save(
    {
        "src_tokens": src_tokenized,
        "tgt_tokens": tgt_tokenized,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
        "vocab": vocab_list,
    },
    out_path
)

print(f"Saved {len(src_tokenized)} samples to {out_path}")
print(f"Vocab size: {len(vocab_list)}")

# --------------------------------------------------
# 4. tokenize and save the validation dataset as well
# --------------------------------------------------
val_df = pd.read_csv('lotus_val42.csv')
val_src = val_df['src_smiles'].tolist()
val_tgt = val_df['tgt_smiles'].tolist()

val_src_tokenized, val_src_lengths = tokenize_smiles_list(val_src, add_bos_eos=False)
val_tgt_tokenized, val_tgt_lengths = tokenize_smiles_list(val_tgt, add_bos_eos=True)

torch.save(
    {
        "src_tokens": val_src_tokenized,
        "tgt_tokens": val_tgt_tokenized,
        "src_lengths": val_src_lengths,
        "tgt_lengths": val_tgt_lengths,
        "vocab": vocab_list,
    },
    "lotus_val42.pt"
)

# --------------------------------------------------
# 5. tokenize and save the test dataset as well
# --------------------------------------------------
test_df = pd.read_csv('lotus_test42.csv')
test_src = test_df['src_smiles'].tolist()
test_tgt = test_df['tgt_smiles'].tolist()

test_src_tokenized, test_src_lengths = tokenize_smiles_list(test_src, add_bos_eos=False)
test_tgt_tokenized, test_tgt_lengths = tokenize_smiles_list(test_tgt, add_bos_eos=True)

torch.save(
    {
        "src_tokens": test_src_tokenized,
        "tgt_tokens": test_tgt_tokenized,
        "src_lengths": test_src_lengths,
        "tgt_lengths": test_tgt_lengths,
        "vocab": vocab_list,
    },
    "lotus_test42.pt"
)

en_time = time.time()
minutes = (en_time - st_time) / 60
print(f"Time required for data processing: {minutes:.2f} minutes.")


