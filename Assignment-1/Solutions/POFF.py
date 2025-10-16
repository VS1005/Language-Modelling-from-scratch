import torch
import torch.nn as nn
import math
sqrt2=math.sqrt(2)
def gelu(x:torch.FloatTensor)->torch.FloatTensor:
    return 0.5*x*(1+torch.erf(x/sqrt2))

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(positionwise_feedforward, self).__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)

    def forward(self,x):
        hidden=self.w1(x)
        hidden=gelu(hidden)
        output=self.w2(hidden)

        return output

