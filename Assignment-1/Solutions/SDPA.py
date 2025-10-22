import torch
import math
import torch.nn.functional as f
from typing import IO, BinaryIO, Iterable, Optional, Type

def scaled_dot_product_attention(K:torch.FloatTensor, Q:torch.FloatTensor, V:torch.FloatTensor, mask:Optional[torch.BoolTensor] = None, dropoutp: Optional[float] = None) -> torch.FloatTensor:
    dk=K.shape[-1]
    score=(Q @ K.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        if mask.dim()<score.dim():
            mask=mask.unsqueeze(0)
            while mask.dim()<score.dim():
               mask=mask.unsqueeze(0) 
        score = torch.where(mask, torch.tensor(float('-inf'), dtype=score.dtype, device=score.device), score)
    attn_wts=f.softmax(score, dim=-1)
    attn_wts=torch.nan_to_num(attn_wts,nan=0.0)

    if dropoutp>0:
        attn_wts=f.dropout(score,p=dropoutp,training=True)
    return attn_wts@V