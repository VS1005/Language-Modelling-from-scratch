import numpy
import torch

def get_batch(data:numpy.array, batch_size:int, cntxt_len:int, device:str)->tuple[torch.Tensor, torch.Tensor]:
    start_inds=numpy.random.randint(low=0, high=len(data)-cntxt_len, size=batch_size)
    inputs=numpy.stack(
        [
            data[i:i+cntxt_len] for i in start_inds
        ]
    )
    targets=numpy.stack(
        [
            data[i+1:i+cntxt_len+1] for i in start_inds
        ]
    )
    inputs=torch.from_numpy(inputs).long().to(device)
    targets=torch.from_numpy(targets).long().to(device)

    return inputs, targets
