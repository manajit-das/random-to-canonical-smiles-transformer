

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=600,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        src_padded,
        src_att_masks,
        tgt_padded,
        tgt_att_masks,
    ):
        """
        src_padded: (B, S)
        tgt_padded: (B, T)
        """

        B, S = src_padded.shape
        _, T = tgt_padded.shape
        device = src_padded.device

        # --- embeddings ---
        src_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        tgt_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        src_emb = self.token_embedding(src_padded) + self.pos_embedding(src_pos)
        tgt_emb = self.token_embedding(tgt_padded) + self.pos_embedding(tgt_pos)

        # (S, B, E)
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        # --- masks ---
        src_key_padding_mask = (src_att_masks == 0)
        tgt_key_padding_mask = (tgt_att_masks == 0)

        # causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # --- transformer ---
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # (T, B, E) → (B, T, E)
        out = out.permute(1, 0, 2)

        logits = self.fc_out(out)
        return logits




