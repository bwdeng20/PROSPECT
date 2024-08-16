from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree

from .utils import (
    group_node_with_degree,
    index2mask,
    mask2index,
    is_dict_value_overlapped,
    are3masks_disjoint,
)


def split_grb(
    degrees,
    train=0.6,
    val=0.1,
    test_per_partition=0.1,
    fr=0.05,
    mode="F",
    seed=None,
    return_mask=False,
):
    """`"Graph robustness benchmark: Benchmarking the adversarial robustness of graph
    machine learning".

    <https://arxiv.org/abs/1902.08412>`_ paper (NeurIPS2021)

    Step1: Rank all nodes in terms of the degrees and filter out  fr% nodes with the lowest degrees
            and fr% nodes with the highest degree (totally 2*fr% of ORIGINAL nodes).
    Step2: The rest nodes are divided into three equal partitions without overlapping, and randomly sample
            test_per_partition% nodes (without repetition) from each partition
            (totally 3*test_per_partition% of the REST nodes).
    Step3:  The test subsets with different levels of degree are marked as Small/Medium/Large/Full (‘S/M/L/F’)
            with ‘F’ containing all test nodes.
    Args:
        degrees: np.ndarray
        train:  float
        val:    float
        test_per_partition: float
        fr: float
        mode: str
            ("F", "S", "M", "L")
        seed: optional[int]
        return_mask: bool
            If True, return mask instead of indices

    Returns: List[np.ndarray]
        The train/val/test indices
    """
    assert mode in ("F", "S", "M", "L")
    np_rng = np.random.default_rng(seed)
    nnodes = len(degrees)

    degree_group_dict = group_node_with_degree(degrees, fr=fr, return_dict=True)
    assert len(degree_group_dict) == 5, "The nodes are not grouped into five partitions as GRB"

    assert not is_dict_value_overlapped(degree_group_dict)

    num_filter_node = nnodes - len(degree_group_dict[-1]) - len(degree_group_dict[3])
    num_test_per_part = int(num_filter_node * test_per_partition)

    test_node_group = []
    for i in range(3):
        nodes = degree_group_dict[i]
        test4part = np_rng.choice(nodes, num_test_per_part, replace=False)
        test_node_group.append(test4part)

    all_test_nodes = np.concatenate(test_node_group)
    all_non_test_nodes = np.setdiff1d(np.arange(nnodes), all_test_nodes)
    # remove nodes from filter-out group as GRB
    all_train_val_nodes = np.setdiff1d(all_non_test_nodes, degree_group_dict[-1])
    all_train_val_nodes = np.setdiff1d(all_train_val_nodes, degree_group_dict[3])

    idx_val = np_rng.choice(all_train_val_nodes, int(num_filter_node * val), replace=False)
    idx_train = np.setdiff1d(all_train_val_nodes, idx_val)
    if train is not None and 0 < train < 0.6:  # select from the rest nodes
        # train, val, testS, testM, testL = ?, 0.1, 0.1, 0.1, 0.1
        idx_train = np_rng.choice(idx_train, int(num_filter_node * train), replace=False)

    if mode == "F":
        test_nodes = all_test_nodes
    elif mode == "S":
        test_nodes = test_node_group[0]
    elif mode == "M":
        test_nodes = test_node_group[1]
    elif mode == "L":
        test_nodes = test_node_group[2]
    else:
        raise ValueError(f"{mode} is not valid")
    if return_mask:
        return (
            index2mask(idx_train, nnodes),
            index2mask(idx_val, nnodes),
            index2mask(test_nodes, nnodes),
        )
    return idx_train, idx_val, test_nodes


class Splitter:
    def __init__(self, train_val_test: Sequence = (10, 10, 80), seed: int = None):
        train_val_test = np.array(train_val_test)
        self._train_val_test = self.pre_process_ratio(train_val_test)
        self._min_ratio = min(self._train_val_test)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed

    @staticmethod
    def pre_process_ratio(train_val_test):
        if train_val_test[0] > 1:  # e.g., 60, 10, 30
            train_val_test = train_val_test / 100
        # else e.g., 0.6, 0.1, 0.3
        assert train_val_test.sum() <= 1, f"The split ratio sum({train_val_test})>1 is invalid"
        return train_val_test

    @staticmethod
    def _check_split(train_val_test):
        if sum(train_val_test) not in (100, 1):
            raise ValueError(f"the split rate {train_val_test} is wrong")

    def _check_ratio(self, num_total):
        if int(self._min_ratio * num_total) < 1:
            raise ValueError(
                f"The split ratio {self._min_ratio} is too small to get"
                f"one sample out of {num_total} samples"
            )

    @property
    def train_val_test(self):
        return self._train_val_test

    @train_val_test.setter
    def train_val_test(self, new_split):
        self._check_split(new_split)
        self._train_val_test = new_split

    @property
    def train_ratio(self):
        return self._train_val_test[0]

    @property
    def val_ratio(self):
        return self._train_val_test[1]

    @property
    def test_ratio(self):
        return self._train_val_test[2]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class NullSplitter(Splitter):
    def __call__(self, pyg_data):  # the public split is fixed and loaded from pyg data
        return pyg_data


class GCNRandomSplitter(Splitter):
    def __init__(self, train_val_test: Sequence = (20, 500, 1000), seed: int = None):
        super().__init__(train_val_test, seed)

    @staticmethod
    def pre_process_ratio(train_val_test):  # no ratio-like assert
        assert (
            train_val_test[0] > 1 and train_val_test[1] > 1 and train_val_test[2] > 1
        ), f"The split ratio sum({train_val_test})>1 is invalid"
        return train_val_test

    def get_gcn_split(
        self,
        label,
        num_classes,
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        return_mask=True,
    ):
        train_mask = torch.zeros(label.shape, dtype=torch.bool)
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
        for c in range(num_classes):
            idx = (label == c).nonzero(as_tuple=False).view(-1)
            randperm = self.np_rng.permutation(idx.size(0))
            randperm = torch.as_tensor(randperm, dtype=torch.long)
            idx = idx[randperm[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)

        randperm = self.np_rng.permutation(remaining.size(0))
        randperm = torch.as_tensor(randperm, dtype=torch.long)
        remaining = remaining[randperm]

        val_mask.fill_(False)
        val_mask[remaining[:num_val]] = True

        test_mask.fill_(False)
        test_mask[remaining[num_val : num_val + num_test]] = True

        are3masks_disjoint(train_mask, val_mask, test_mask)
        if return_mask:
            return train_mask, val_mask, test_mask
        else:
            return mask2index(train_mask), mask2index(val_mask), mask2index(test_mask)

    def __call__(self, pyg_data):
        self._check_ratio(pyg_data.num_nodes)
        train_marker, val_marker, test_marker = self.get_gcn_split(
            pyg_data.y,
            pyg_data.num_classes,
            num_train_per_class=self.train_ratio,
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            return_mask=True,
        )
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data


class RandomSplitter(Splitter):
    def __init__(self, train_val_test: Sequence = (10, 10, 80), seed: int = None):
        super().__init__(train_val_test, seed)

    def __call__(self, pyg_data):
        self._check_ratio(pyg_data.num_nodes)
        train_marker, val_marker, test_marker = self.get_random_split(pyg_data.num_nodes, return_mask=True)
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data

    def get_random_split(self, num_total: int, return_mask=True):
        shuffled_idx = torch.as_tensor(self.np_rng.permutation(num_total), dtype=torch.long)
        n1 = int(num_total * self.train_ratio)
        n2 = int(num_total * self.val_ratio)
        train_idx = shuffled_idx[:n1]
        val_idx = shuffled_idx[-n2:]

        if return_mask:
            train_marker = torch.zeros(num_total, dtype=torch.bool)
            val_marker = torch.zeros_like(train_marker)
            test_marker = torch.ones_like(train_marker)
            train_marker[train_idx] = True
            val_marker[val_idx] = True
            test_marker[train_marker] = False
            test_marker[val_marker] = False
            assert are3masks_disjoint(train_marker, val_marker, test_marker)
        else:
            train_marker = train_idx
            val_marker = val_idx
            test_marker = shuffled_idx[n1:n2]
        return train_marker, val_marker, test_marker


class GrbSplitter(Splitter):
    """`"Graph robustness benchmark: Benchmarking the adversarial robustness of graph
    machine learning".

    <https://arxiv.org/abs/1902.08412>`_ paper (NeurIPS2021)
    """

    def __init__(
        self,
        train_val_test: Sequence = (0.6, 0.1, 0.1),
        fr: float = 0.05,
        mode="F",
        seed: int = None,
    ):
        super().__init__(train_val_test, seed)
        self.fr = fr
        self.mode = mode

    def __call__(self, pyg_data):
        self._check_ratio(pyg_data.num_nodes * (1 - self.fr * 2))
        degrees = degree(pyg_data.edge_index).cpu().numpy()
        train_marker, val_marker, test_marker = self.get_grb_split(degrees, return_mask=True)
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data

    def get_grb_split(self, degrees, return_mask=True):
        return split_grb(
            degrees,
            train=self.train_ratio,
            val=self.val_ratio,
            test_per_partition=self.test_ratio,
            fr=self.fr,
            mode=self.mode,
            seed=self.seed,
            return_mask=return_mask,
        )

    @staticmethod
    def parse_grb_str(setting):
        require_llc = "llc" in setting
        if "grbs" in setting:
            tmode = "S"
        elif "grbm" in setting:
            tmode = "M"
        elif "grbl" in setting:
            tmode = "L"
        else:
            tmode = "F"
        return tmode, require_llc


class NettackSplit(Splitter):
    def __init__(self, train_val_test: Sequence = (10, 10, 80), seed: int = None):
        super().__init__(train_val_test, seed)

    def __call__(self, pyg_data):
        self._check_ratio(pyg_data.num_nodes)
        train_marker, val_marker, test_marker = self.get_nettack_split(
            pyg_data.num_nodes, pyg_data.y, return_mask=True
        )
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data

    def get_nettack_split(self, num_total: int, stratify, return_mask=True):
        train_mask, val_mask, test_mask = split_net_meta(
            num_total,
            self.val_ratio,
            self.test_ratio,
            stratify.cpu().numpy(),
            seed=self.seed,
            return_mask=return_mask,
        )

        return train_mask, val_mask, test_mask


def split_net_meta(num_total, val_size=0.1, test_size=0.8, stratify=None, seed=None, return_mask=False):
    """This setting follows nettack/mettack, where we split the nodes into 10% training,
    10% validation and 80% testing data_tools according to the labels of all nodes.

    Parameters
    ----------
    num_total : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data_tools is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    return_mask: bool
        If True, return mask

    Args:
        return_mask:
    """
    idx = np.arange(num_total)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=seed,
        train_size=train_size + val_size,
        test_size=test_size,
        stratify=stratify,
    )

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(
        idx_train_and_val,
        random_state=seed,
        train_size=train_size / (train_size + val_size),
        test_size=val_size / (train_size + val_size),
        stratify=stratify,
    )

    if return_mask:
        return (
            index2mask(idx_train, num_total),
            index2mask(idx_val, num_total),
            index2mask(idx_test, num_total),
        )
    return idx_train, idx_val, idx_test
