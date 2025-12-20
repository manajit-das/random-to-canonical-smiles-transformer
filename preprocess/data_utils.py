

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import re

SMI_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|"
    r"b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|\+|\\/|:|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9])"
)

class SmilesTokenizer:
    def __init__(self, vocab_list):
        self.vocab = vocab_list
        self.char_to_int = {tok: i for i, tok in enumerate(vocab_list)}
        self.int_to_char = {i: tok for i, tok in enumerate(vocab_list)}

        self.pad_idx = self.char_to_int["PAD"]
        self.bos_idx = self.char_to_int["BOS"]
        self.eos_idx = self.char_to_int["EOS"]
        self.unk_idx = self.char_to_int["<UNK>"]

        self.regex = re.compile(SMI_REGEX_PATTERN)

    def tokenize(self, smiles):
        tokens = self.regex.findall(smiles)
        return [self.char_to_int.get(t, self.unk_idx) for t in tokens]

    def encode(self, smiles, add_bos_eos=False):
        ids = self.tokenize(smiles)

        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]

        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, smiles_list, add_bos_eos=False):
        tokenized = []
        lengths = []

        for s in smiles_list:
            ids = self.encode(s, add_bos_eos)
            tokenized.append(ids)
            lengths.append(len(ids))

        return tokenized, lengths

    def decode(self, ids, remove_special_tokens=True):
        """
        ids: list[int] | torch.Tensor
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        tokens = []
        for i in ids:
            if remove_special_tokens and i in {
                self.pad_idx,
                self.bos_idx,
                self.eos_idx,
            }:
                continue
            tokens.append(self.int_to_char.get(i, "<UNK>"))

        return "".join(tokens)

    def batch_decode(self, batch_ids, remove_special_tokens=True):
        return [
                self.decode(ids, remove_special_tokens)
                for ids in batch_ids
                ]


def create_vocab(smiles_list):
    vocab = set()
    for smiles in smiles_list:
        tokens = re.findall(SMI_REGEX_PATTERN, smiles)
        vocab.update(tokens)
    return vocab

'''
ckpt = torch.load("../data/vocab.pt")
tokenizer = SmilesTokenizer(ckpt["vocab"])
smi = "CC(=O)O"

ids = tokenizer.encode(smi, add_bos_eos=True)
print(ids)
# tensor([BOS, C, C, (, =, O, ), O, EOS])

decoded = tokenizer.decode(ids)
print(decoded)
# "CC(=O)O"

exit()
'''



class TokenizedSeq2SeqDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu")

        self.src = data["src_tokens"]
        self.tgt = data["tgt_tokens"]
        self.src_lengths = data["src_lengths"]
        self.tgt_lengths = data["tgt_lengths"]
        self.vocab_size= len(data["vocab"])
        self.vocab_list = data["vocab"]

        assert len(self.src) == len(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            self.src[idx],
            self.tgt[idx],
            self.src_lengths[idx],
            self.tgt_lengths[idx],
        )
        

def my_collate_fn(batch):
    src_tokens, tgt_tokens, src_seq_len, tgt_seq_len = zip(*batch)


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

    src_padded, src_att_masks, src_seq_len = pad_and_mask(src_tokens, src_seq_len)
    tgt_padded, tgt_att_masks, tgt_seq_len = pad_and_mask(tgt_tokens, tgt_seq_len)

    return src_padded, src_att_masks, src_seq_len, tgt_padded, tgt_att_masks, tgt_seq_len



