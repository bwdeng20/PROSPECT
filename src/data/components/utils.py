import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple, Optional, List
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
from sklearn.preprocessing import LabelEncoder


def are3masks_disjoint(mask1, mask2, mask3):
    assert are_masks_disjoint(mask1, mask2)
    assert are_masks_disjoint(mask1, mask3)
    assert are_masks_disjoint(mask2, mask3)
    return True


def are_masks_disjoint(mask1: Tensor, mask2: Tensor):
    assert mask1.dtype == mask2.dtype == torch.bool
    m1_or_m2 = torch.logical_or(mask1, mask2)
    return torch.all(torch.logical_xor(mask1[m1_or_m2], mask2[m1_or_m2]))


def index2mask(index, num_nodes):
    if isinstance(index, Tensor):
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=index.device)
    else:
        mask = np.zeros(num_nodes, dtype=bool)
    mask[index] = 1
    return mask


def mask2index(mask):
    return mask.nonzero().ravel()


def is_spmatrix_symmetric(spm):
    return (abs(spm - spm.T) > 1e-12).nnz == 0


def is_dict_value_overlapped(dictionary):
    all_entries = np.concatenate(list(dictionary.values()))
    unique_entries = np.unique(all_entries)
    return len(unique_entries) < len(all_entries)


def group_node_with_degree(degrees, fr=0.0, return_dict=False):
    assert 0 <= fr < 1
    nnode = len(degrees)
    sort_idx = np.argsort(degrees)

    # filter out fr% nodes with the lowest degrees and fr% nodes with the highest degrees
    num2filter = int(nnode * fr)
    if fr > 0:
        assert num2filter != 0, "Please choose smaller ratio of nodes to filter out"
    filtered_sort_idx = sort_idx[num2filter:-num2filter] if num2filter > 0 else sort_idx
    chunked_sort_idx = np.array_split(filtered_sort_idx, 3)  # three groups
    # -1, 0, 1, 2, 3 means:
    # -1: filtered-out and low-degree,
    #  0: low-degree, 1: medium degree, 2: high degree,
    #  3: filtered-out and high-degree
    if return_dict:
        degree_group = {-1: sort_idx[:num2filter], 3: sort_idx[-num2filter:]}
        for i, chunk in enumerate(chunked_sort_idx):
            degree_group[i] = chunk
    else:
        degree_group = np.ones(nnode) * -2
        for i, chunk in enumerate(chunked_sort_idx):
            degree_group[chunk] = i
        degree_group[sort_idx[:num2filter]] = -1
        degree_group[sort_idx[-num2filter:]] = 3
    return degree_group


def remove_self_loops(
        edge_index: Tensor, edge_attr: Union[Optional[Tensor], List[Tensor]] = None
) -> Tuple[Tensor, OptTensor]:
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if edge_attr is None:
        return edge_index

    if isinstance(edge_attr, Tensor):
        edge_attr = edge_attr[mask]
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [e[mask] for e in edge_attr]
    return edge_index, edge_attr


@functional_transform("remove_self_loops")
class RemoveSelfLoops(BaseTransform):
    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue
            if isinstance(data, Data):
                keys, values = [], []
                for key, value in store.items():
                    if key == "edge_index":
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store._edge_index, values = remove_self_loops(store.edge_index, values)
                for key, value in zip(keys, values):
                    store[key] = value
        return data


@functional_transform("normalize_label")
class NormalizeLabel(BaseTransform):
    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        old_y = data.y.cpu().numpy()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(old_y)
        new_y = self.label_encoder.transform(old_y)
        data.y = torch.as_tensor(new_y, device=data.y.device)
        data.num_classes = data.y.max() + 1
        return data


@functional_transform("onehot_node_fea")
class OneHotNodeFeature(BaseTransform):
    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        if data.get("x") is not None:
            raise RuntimeError(f"The input Data or HeteroData has attribute `x`.")
        data.x = torch.eye(data.num_nodes, dtype=torch.float32, device=data.y.device)
        return data
