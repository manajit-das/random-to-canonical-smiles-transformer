

import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from preprocess.data_utils import TokenizedSeq2SeqDataset, my_collate_fn
from model.models import MyModel
import argparse

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
print(f"Rank {rank}/{world_size} running on GPU {local_rank}",flush=True)
dist.init_process_group(backend='nccl',init_method='env://')

def run_validation(model, val_loader, criterion, device):
    PAD_IDX = 0
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in val_loader:
            src_padded, src_att_masks, _, tgt_padded, tgt_att_masks, _ = batch
            src_padded = src_padded.to(device)
            src_att_masks = src_att_masks.to(device)
            tgt_padded = tgt_padded.to(device)
            tgt_att_masks = tgt_att_masks.to(device)

            tgt_input = tgt_padded[:, :-1]
            tgt_output = tgt_padded[:, 1:]
            tgt_att_input = tgt_att_masks[:, :-1]

            logits = model(src_padded, src_att_masks, tgt_input, tgt_att_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            num_tokens = (tgt_output != PAD_IDX).sum()

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    # 🔥 aggregate across GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    return (total_loss / total_tokens).item()

def parse_args():
    parser = argparse.ArgumentParser(description="train/val inputs")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def main(local_rank, world_size, rank, data_path, model_save_path, num_epochs=2, batch_size=64):
    mydata = TokenizedSeq2SeqDataset(f"./{data_path}/train.pt")
    sampler = DistributedSampler(mydata,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,)
    myloader = DataLoader(mydata, batch_size=batch_size, collate_fn=my_collate_fn, sampler=sampler)
    
    #prepare the validation dataset
    val_data = TokenizedSeq2SeqDataset(f"./{data_path}/val.pt")
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=my_collate_fn, sampler=val_sampler)


    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    model = MyModel(mydata.vocab_size) 
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
 

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
            )


    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    MAX_GRAD_NORM = 1.0

    num_epochs = num_epochs

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for batch in myloader:
            src_padded, src_att_masks, src_seq_len, tgt_padded, tgt_att_masks, tgt_seq_len = batch

            src_padded = src_padded.to(device)
            src_att_masks = src_att_masks.to(device)
            tgt_padded = tgt_padded.to(device)
            tgt_att_masks = tgt_att_masks.to(device)

            # ---- teacher forcing ----
            tgt_input = tgt_padded[:, :-1]
            tgt_output = tgt_padded[:, 1:]
            tgt_att_input = tgt_att_masks[:, :-1]

            # ---- forward ----
            logits = model(
                src_padded,
                src_att_masks,
                tgt_input,
                tgt_att_input,
            )
            # logits: (B, T-1, vocab)

            # ---- loss ----
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            # ---- backward ----
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(myloader)
       
        #---------------validation-------------
        val_sampler.set_epoch(epoch)
        val_loss = run_validation(model, val_loader, criterion, device)
        if rank == 0:
            print(
                    f"Epoch {epoch+1:03d} | "
                    f"train loss = {avg_loss:.4f} | "
                    f"val loss = {val_loss:.4f}"
                    ) 

        best_val_loss = float('inf')

        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            os.makedirs(f"./{model_save_path}", exist_ok=True)

            torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),
                        "vocab": mydata.vocab_list,   # optional but VERY useful
                        },
                    f"./{model_save_path}/seq2seq_model.pt",)



if __name__ == "__main__":
    import time
    st_time = time.time()
    args = parse_args()
    main(local_rank, world_size, rank, data_path=args.data_path, model_save_path=args.ckpt_path,
            num_epochs = args.num_epochs, batch_size = args.batch_size)
    en_time = time.time()
    print('Time required:', (en_time-st_time)/60)



