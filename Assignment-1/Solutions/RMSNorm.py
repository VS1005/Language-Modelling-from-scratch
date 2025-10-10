import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model))

    # x -> activation 
    # input : (batch_sz, seq_lenght, d_model)
    def forward(self,x):
        rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        return (x/rms)*self.weight
    
        