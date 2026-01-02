import torch
from .RMSNorm import RMSNorm
from .POFF import positionwise_feedforward
from .MulHeadSelfAttn import multihead_self_attention

class Tranformer_block(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_pdrop: float | None = None, residual_pdrop: float | None = None):
        super().__init__()
        self.norm1=RMSNorm(d_model)
        self.norm2=RMSNorm(d_model)
        self.attn=multihead_self_attention(d_model,num_heads,attn_pdrop)
        self.poff=positionwise_feedforward(d_model,d_ff)
        self.dropout1=torch.nn.Dropout(residual_pdrop)
        self.dropout2=torch.nn.Dropout(residual_pdrop)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        attn_out=self.attn(self.norm1(x))
        x=x+self.dropout1(attn_out)
        poff_out=self.poff(self.norm2(x))
        x=x+self.dropout2(poff_out)
        return x
    
        
        
