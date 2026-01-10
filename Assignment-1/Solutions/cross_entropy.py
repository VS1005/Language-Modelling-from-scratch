import torch
from .softmax import softmax
def cross_ent(inp:torch.FloatTensor, tar:torch.LongTensor):
    max_per=inp.max(dim=-1, keepdim=True).values
    shift=inp-max_per
    sum_exp=torch.exp(shift).sum(dim=1)
    logsumexp=max_per.squeeze(-1)+torch.log(sum_exp)
    tar_logit=inp.gather(dim=1, index=tar.unsqueeze(-1)).squeeze(-1)
    loss=logsumexp-tar_logit
    return loss.mean()
