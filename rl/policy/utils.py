import torch as th

def polyak_update(params: Iterable[th.nn.Parameter], target_params: Iterable[th.nn.Parameter], tau: float) -> None:
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)
