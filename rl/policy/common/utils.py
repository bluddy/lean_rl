import torch as th
from typing import Iterable

def polyak_update(params: Iterable[th.nn.Parameter], target_params: Iterable[th.nn.Parameter], tau: float) -> None:
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
