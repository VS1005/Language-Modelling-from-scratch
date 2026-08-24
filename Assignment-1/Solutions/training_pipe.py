import torch
import os
import typing
import numpy as np

from .adamw import AdamW
from .CosineLRS import lr_cos_sch
from .GradClip import gradclip
from .checkpointing import save_checkpoint
from .cross_entropy import cross_ent
from .data_loader import get_batch
from .transformer_lm import tranformer_lm


train_data=np.load(train_path=..., mmap_mode="r")
valid_data=np.load(valid_path=..., mmap_mode="r")

model=tranformer_lm(
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=2048,
    attn_pdrop=...,
    residual_pdrop=...
).to(device=...)

optimizer=AdamW(
    model.parameters(),
    lr=...,
    betas=...,
    eps=...,
    weight_decay=...
)

learning_rate=lr_cos_sch(
    t=...,
    a_max=...,
    a_min=...,
    tw=...,
    tc=...
)

for grp in optimizer.param_groups:
    grp["lr"]=learning_rate

inputs, targets=get_batch(
    data=...,
    batch_size=...,
    cntxt_len=...,
    device=...
)

optimizer.zero_grad()

logits=model(inputs)

batch_sz_actual, context_len_actual, vocab_size_actual=logits.shape

loss=cross_ent(
    logits.reshape(-1, logits.shape[-1]),
    targets.reshape(-1)
)

loss.backward()

gradclip(model.parameters(), max_l2=...)

optimizer.step()

model.eval()

with torch.no_grad():
    valid_inputs, valid_targets=get_batch(
        valid_data,
        batch_size=...,
        cntxt_len=...,
        device=...
    )

    valid_logits=model(valid_inputs)
    valid_loss=cross_ent(
        valid_logits.reshape(-1, valid_logits.shape[-1]),
        valid_targets.reshape(-1)
    )
model.train()


save_checkpoint(model=model,
                optimizer=optimizer,
                iteration=...,
                out=...)