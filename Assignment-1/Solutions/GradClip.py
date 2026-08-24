import torch
from typing import Iterable
def gradclip(params:Iterable[torch.nn.Parameter], max_l2:float):
    eps=1e-6
    grads=[]
    for p in params:
        # Assuming backward is called
        if p.grad is not None:
            grads.append(p.grad) # p.grad is a tensor created by autograd to store gradients (not part of the forward graph)
    if not grads: # If loss.backward() not called
        return
    total_norm_sq=0.0
    for g in grads:
        total_norm_sq+=g.norm(2).item()**2 # .item() converts the tensor norm to a Python float and prevents autograd tracking
    total_norm=total_norm_sq**0.5
    if total_norm>max_l2:
        scale=max_l2/(total_norm+eps)
        for g in grads:
            g.mul_(scale) # g.mul_(scale) is safe because it modifies gradients after backward, not forward tensors