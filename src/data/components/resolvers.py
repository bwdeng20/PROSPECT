import torch
from typing import Callable, Optional, Union, List, Any

import torch_geometric.transforms as pyg_transform


def normalize_string(s: str) -> str:
    return s.lower().replace("_", "-").replace(" ", "")


def parse_transform(
    transform_str: Optional[Union[str, Callable]] = None,
    sep: str = "-",
    return_identity: bool = False,
):
    """Parse a string joining different transform names that come from
    :module:`torch_geometric.transforms`. For example, 'ToSparseTensor,ToUndirected'
    means a transform composed by :class:`torch_geometric.transforms.ToSparseTensor`
    and :class:`torch_geometric.transforms.ToUndirected`

    Args:
        transform_str (string, Callable, optional): a string determining a transform
        sep (string): the separator symbol of :obj:`transform_str`
        return_identity (bool): If True, return f(x)=x else None
    Returns:
        composed_transform (torch_geometric.transforms.Compose): the
            transform
    """

    def identity(x):
        return x

    if transform_str is None:
        return identity if return_identity else None
    if isinstance(transform_str, Callable):
        return transform_str
    transform_classes = [
        getattr(pyg_transform, one_trans)()
        for one_trans in transform_str.split(sep)
        if one_trans != ""
    ]
    composed_transform = pyg_transform.Compose(transform_classes)
    return composed_transform
