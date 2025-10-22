import torch

def softmax(x:torch.FloatTensor, dim:int)->torch.FloatTensor:
    max_val=torch.max(x, dim=dim,keepdim=True)[0]
    shift=x-max_val
    exp_val=torch.exp(shift)
    sum_exp=torch.sum(exp_val, dim=dim, keepdim=True)

    return exp_val/sum_exp

