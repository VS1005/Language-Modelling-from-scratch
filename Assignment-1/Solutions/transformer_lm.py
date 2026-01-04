import torch
import torch.nn as nn
from .Transformer_block import Tranformer_block
from .RMSNorm import RMSNorm
from .softmax import softmax

class tranformer_lm(torch.nn.Module):
    def __init__(self, vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float
    ):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size, d_model)
        self.pos_emb=nn.Embedding(context_length, d_model)
        self.dropout=nn.Dropout(residual_pdrop)
        self.layers=nn.ModuleList([
            Tranformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attn_pdrop=attn_pdrop,
                residual_pdrop=residual_pdrop
            )
            for _ in range(num_layers)
        ])
        self.norm=RMSNorm(d_model=d_model)
        self.lm_head=nn.Linear(d_model, vocab_size, bias=False)
        # self.softmax=softmax()
        
    
    def forward(self, idx:torch.LongTensor)->torch.FloatTensor:
        B,T=idx.shape
        dev=idx.device
        pos=torch.arange(T, device=dev)
        x=self.tok_emb(idx)+self.pos_emb(pos)
        x=self.dropout(x)
        for layer in self.layers:
            x=layer(x)
        x=self.norm(x)
        x=self.lm_head(x)
        return x

        

