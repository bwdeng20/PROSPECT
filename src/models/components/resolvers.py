from typing import Any, List, Union

import torch
import torch_geometric.nn.norm as pyg_norm


def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolver(classes: List[Any], query: Union[Any, str], only_return_cls=False, *args, **kwargs):
    if query is None:
        return query

    if not isinstance(query, str):
        raise f"Resolver only parse `class.__name__` str, e.g., query=`ReLU`, but got {type(query)}: {query}"

    query = normalize_string(query)
    for cls in classes:
        if query == normalize_string(cls.__name__):
            return cls(*args, **kwargs) if not only_return_cls else cls

    return ValueError(
        f"Could not resolve '{query}' among the choices "
        f"{set(normalize_string(cls.__name__) for cls in classes)}"
    )


def activation_resolver(query: Union[Any, str] = "relu", only_return_cls=False, *args, **kwargs):
    if query == "identity":
        return torch.nn.Identity()
    acts = [
        act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    ]
    return resolver(acts, query, only_return_cls, *args, **kwargs)


def norm_resolver(query: str = "batchnorm", only_return_cls=False, *args, **kwargs):
    norms = [
        norm
        for norm in vars(pyg_norm).values()
        if isinstance(norm, type) and issubclass(norm, torch.nn.Module)
    ]
    return resolver(norms, query, only_return_cls, *args, **kwargs)


def distill_resolver(query: Union[Any, str] = "SoftTarget", only_return_cls=False, *args, **kwargs):
    from . import kd_losses
    if isinstance(query, str):
        distillers = [
            distiller
            for distiller in vars(kd_losses).values()
            if isinstance(distiller, type) and issubclass(distiller, torch.nn.Module)
        ]
        return resolver(distillers, query, only_return_cls, *args, **kwargs)
    else:
        return pre_resolver_class_instance(query, kd_losses.KnowledgeDistiller)


def loss_resolver(query: Union[Any, str] = "CrossEntropy", only_return_cls=False, *args, **kwargs):
    import torch.nn.modules.loss as th_losses
    if isinstance(query, str):
        losses = [
            loss
            for loss in vars(th_losses).values()
            if isinstance(loss, type) and issubclass(loss, torch.nn.Module)
        ]
        query = query + "loss" if isinstance(query, str) else query
        return resolver(losses, query, only_return_cls, *args, **kwargs)
    else:
        return pre_resolver_class_instance(query, th_losses._Loss, *args, **kwargs)


def optimizer_resolver(query: Union[Any, str] = "adam", only_return_cls=False, *args, **kwargs):
    import torch.optim as th_optimizers
    if isinstance(query, str):
        optimizers = [
            opt
            for opt in vars(th_optimizers).values()
            if isinstance(opt, type) and issubclass(opt, th_optimizers.Optimizer)
        ]
        return resolver(optimizers, query, only_return_cls, *args, **kwargs)
    else:
        return pre_resolver_class_instance(query, th_optimizers.Optimizer)


def lr_scheduler_resolver(
        query: Union[Any, str] = "ReduceLROnPlateau", only_return_cls=False, *args, **kwargs
):
    import torch.optim.lr_scheduler as th_lr_scheduler
    if isinstance(query, str):
        schedulers = [
            sch
            for sch in vars(th_lr_scheduler).values()
            if isinstance(sch, type)
               and (
                       issubclass(sch, th_lr_scheduler.LRScheduler)
                       or issubclass(sch, th_lr_scheduler.ReduceLROnPlateau)
               )
        ]
        return resolver(schedulers, query, only_return_cls, *args, **kwargs)
    else:
        return pre_resolver_class_instance(query, th_lr_scheduler.LRScheduler)


def pre_resolver_class_instance(query: Union[Any, str], target_class, *args, **kwargs):
    if query is None:
        return query
    if isinstance(query, target_class):
        return query
    elif issubclass(query, target_class):
        return query(*args, **kwargs)
    else:
        raise f"Only parse instance or class"


if __name__ == "__main__":
    # act = activation_resolver("leaky Relu", negative_slope=0.01)
    print(norm_resolver("LayerNorm", in_channels=3))
    norm = norm_resolver("BatchNorm", in_channels=3)
    act = activation_resolver(torch.nn.LeakyReLU(), negative_slope=0.02)
    print(norm)
    print(act)
    act = activation_resolver(None, negative_slope=0.02)
    print(act)

    vanilla_distiller = distill_resolver(temperature=2.0)
    print(vanilla_distiller)

    loss = loss_resolver("CrossEntropy")
    print(loss)

    print(loss_resolver(torch.nn.MSELoss()))
