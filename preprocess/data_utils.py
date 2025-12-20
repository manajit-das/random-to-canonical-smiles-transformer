

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


class mydataset(Dataset):
  def __init__(self, input_csv_path):
    df = pd.read_csv(input_csv_path)
    self.src_smiles = df['src_smiles'].tolist()
    self.tgt_smiles = df['tgt_smiles'].tolist()
    self.vocab = self.create_vocab(self.src_smiles+self.tgt_smiles)
    self.vocab_list = sorted(self.vocab)
    #add special chracters
    self.vocab_list = ['PAD', 'BOS', 'EOS', '<UNK>'] + self.vocab_list

    self.char_to_int = {tok: i for i, tok in enumerate(self.vocab_list)}
    self.int_to_char = {i: tok for i, tok in enumerate(self.vocab_list)}
    self.vocab_size = len(self.vocab_list)


  @staticmethod
  def create_vocab(smiles_list):
    vocab = set()
    for smiles in smiles_list:
        tokens = re.findall(SMI_REGEX_PATTERN, smiles)
        vocab.update(tokens)        # set.update works with a list
    return vocab
  
  def tokenize_smiles(self, smiles):
    tokens = re.findall(SMI_REGEX_PATTERN, smiles)
    token_id = [self.char_to_int.get(t, self.char_to_int['<UNK>']) for t in tokens]
    return token_id

  def __len__(self):
    return len(self.src_smiles)

  def __getitem__(self, idx):
    src_smiles = self.src_smiles[idx]
    tgt_smiles = self.tgt_smiles[idx]
    #tokenize the smiles
    src_tokens = self.tokenize_smiles(src_smiles)
    tgt_tokens = [self.char_to_int['BOS']] + self.tokenize_smiles(tgt_smiles) + [self.char_to_int['EOS']]
    src_seq_len = len(src_tokens)
    tgt_seq_len = len(tgt_tokens)
    return src_tokens, tgt_tokens, src_seq_len, tgt_seq_len


import torch

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



