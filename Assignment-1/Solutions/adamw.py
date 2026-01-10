import math
import torch
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0):
        defaults=dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss=None
        if closure is not None:
            loss=closure()
        for grp in self.param_groups:
            lr=grp['lr']
            b1,b2=grp['betas']
            eps=grp['eps']
            weight_decay=grp['weight_decay']

            for p in grp['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)
                m=state['m']
                v=state['v']
                state['step']+=1
                t=state['step']
                m.mul_(b1).add_(grad, alpha=(1-b1))
                v.mul_(b2).addcmul_(grad, grad, value=(1-b2))
                bias_cr1=1-b1**t
                bias_cr2=1-b2**t
                adap_lr=lr*math.sqrt(bias_cr2)/bias_cr1
                denom=v.sqrt().add_(eps)
                p.data.addcdiv_(m,denom,value=-adap_lr)
                if weight_decay!=0:
                    p.data.add_(p.data, alpha=-lr*weight_decay)
        return loss

            