
import torch
from models import MyModel
import re
from rdkit import Chem
import time

st_time = time.time()

SMI_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|"
    r"b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|\+|\\/|:|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9])")

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def smiles_matcher(pred_smiles, tgt_smiles):
  tgt_smi = Chem.MolToSmiles(Chem.MolFromSmiles(tgt_smiles))
  return pred_smiles == tgt_smi

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


def batch_greedy_decode(model, src_smiles_list, max_len=50):
    """
    Batch greedy decoding for SMILES canonicalization.
    src_smiles_list: list[str]
    returns: list[str]
    """

    B = len(src_smiles_list)

    # ---------- Source preparation ----------
    src_toks = []
    src_toks_len = []

    for smi in src_smiles_list:
        toks = re.findall(SMI_REGEX_PATTERN, smi)
        #ids = [char_to_int[t] for t in toks]
        ids = [char_to_int.get(t, char_to_int['<UNK>']) for t in toks]
        src_toks.append(ids)
        src_toks_len.append(len(ids))

    src_tok_batch, src_att_mask_batch, _ = pad_and_mask(src_toks, src_toks_len)

    src_tok_batch = src_tok_batch.to(device)
    src_att_mask_batch = src_att_mask_batch.to(device)

    # ---------- Target initialization ----------
    tgt_sequences = [['BOS'] for _ in range(B)]
    Finished = torch.zeros(B, dtype=torch.bool)  # stays persistent

    PAD_ID = char_to_int['PAD']
    EOS_ID = char_to_int['EOS']

    model.eval()
    with torch.no_grad():

        for _ in range(max_len):

            # ---- build decoder input (B, t) ----
            tgt_toks = []
            tgt_toks_len = []

            for seq in tgt_sequences:
                ids = [char_to_int[t] for t in seq]
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
                    tgt_sequences[i].append(int_to_char[tok_id.item()])

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

    return final_smiles_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("dgx_lotus_seq2seq_model.pt", map_location=device)
vocab_list = checkpoint["vocab"]
print('The vocab list:', vocab_list)
vocab_size = len(vocab_list)

model = MyModel(vocab_size)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
#model.eval()
#vocab_list = checkpoint["vocab"]
char_to_int = {tok: i for i, tok in enumerate(vocab_list)}
int_to_char = {i: tok for i, tok in enumerate(vocab_list)}

#-------------------------------------------------------
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("lotus_test42.csv")
df = df.sample(1000)

true_smiles_list = df['tgt_smiles'].tolist()
input_smiles_list = df['src_smiles'].tolist()
pred_smiles_list = batch_greedy_decode(model, input_smiles_list,max_len=600)

num_correct = 0
num_valid = 0
N = len(true_smiles_list)

#for pred, tgt in zip(pred_smiles_list, true_smiles_list):
for pred, tgt in tqdm(zip(out, true_smiles_list), total=len(true_smiles_list)):
    if is_valid_smiles(pred):
        num_valid += 1
    if pred == Chem.MolToSmiles(Chem.MolFromSmiles(tgt)):
        num_correct += 1

acc = 100.0 * num_correct / N
validity = 100.0 * num_valid / N

print(f"Exact-match accuracy: {acc:.2f}%")
print(f"Validity: {validity:.2f}%")
en_time = time.time()
print(f"Time required: {(en_time-st_time)/60}")


