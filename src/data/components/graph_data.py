import logging
import os.path as osp
from pathlib import Path
from typing import Callable, Optional, Union
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (
    Amazon,
    Planetoid,
    Reddit,
    Reddit2,
    WebKB,
    WikipediaNetwork,
    PolBlogs
)
from torch_geometric.data import InMemoryDataset, Data

from .resolvers import normalize_string, parse_transform
from .utils import index2mask, is_spmatrix_symmetric

logger = logging.getLogger(__file__)


class UnifiedReader:
    PlanetoidDataNames = ["cora", "citeseer", "pubmed"]
    OgbnDataNames = ["ogbn-arxiv", "ogbn-products", "ogbn-proteins"]
    OtherDataNames = ["reddit", "reddit2", "polblogs"]
    AmazonDataNames = ["a-computers", "a-photo", "amazon-computers", "amazon-photo"]
    WikiDataNames = ["chameleon", "squirrel"]
    WebKBDataNames = ["cornell", "texas", "wisconsin"]

    SupportedNodeDatasetNames = (
            PlanetoidDataNames + OtherDataNames + OgbnDataNames + AmazonDataNames + WikiDataNames + WebKBDataNames
    )

    def __init__(
            self,
            name: str,
            root: str = "data",
            transform: Optional[Union[str, Callable]] = None,
            pre_transform: Optional[Union[str, Callable]] = None,
    ):
        self.name = normalize_string(name)
        self.root = root
        self.transform = parse_transform(transform, return_identity=True)
        self.pre_transform = parse_transform(pre_transform)
        self.read_dataset = None

    def get_pyg_data(self):  # for node-level task
        # The "Data" of PyG built-in dataset has "train_mask", "val_mask",
        # "test_mask" that can be converted to indices by "LightningDataModule".
        if self.name in self.PlanetoidDataNames:
            self.read_dataset = (
                Planetoid(
                    self.root,
                    self.name,
                    "public",
                    transform=self.transform,
                    pre_transform=self.pre_transform,
                )
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.OgbnDataNames:
            self.read_dataset = (
                PygNodePropPredDataset(self.name, self.root, self.transform, self.pre_transform)
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]
            pyg_data = self.ogb_idx_split2mask(self.read_dataset.get_idx_split(), pyg_data)

        elif self.name in self.OtherDataNames:
            if self.name == "reddit":
                self.read_dataset = (
                    Reddit(
                        osp.join(self.root, "reddit"),
                        self.transform,
                        self.pre_transform,
                    )
                    if self.read_dataset is None
                    else self.read_dataset
                )
            elif self.name == "reddit2":  # a sparser version of Reddit
                self.read_dataset = (
                    Reddit2(
                        osp.join(self.root, "reddit2"),
                        self.transform,
                        self.pre_transform,
                    )
                    if self.read_dataset is None
                    else self.read_dataset
                )
            elif self.name == "polblogs":
                self.read_dataset = (
                    PolBlogs(osp.join(self.root, "polblogs"),
                             self.transform,
                             self.pre_transform,
                             )
                    if self.read_dataset is None else self.read_dataset
                )
            else:
                raise ValueError
            pyg_data = self.read_dataset[0]
        elif self.name in self.AmazonDataNames:
            self.read_dataset = Amazon(
                osp.join(self.root, "amazon"),
                self.name.split("-")[-1],  # "photo" pr "computers"
                self.transform,
                self.pre_transform,
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.WikiDataNames:
            self.read_dataset = (
                WikipediaNetwork(
                    self.root,
                    self.name,
                    self.transform,
                    self.pre_transform,
                )
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.WebKBDataNames:
            self.read_dataset = (
                WebKB(
                    self.root,
                    self.name,
                    self.transform,
                    self.pre_transform,
                )
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        else:
            raise ValueError
        pyg_data["name"] = self.name
        pyg_data["num_classes"] = self.read_dataset.num_classes
        # some datasets, e.g., ogbn-arxiv has labels of shape (N,1)
        pyg_data.y.squeeze_(-1)
        return pyg_data

    def get_pyg_dataset(self):  # TODO cope with graph-level
        raise NotImplementedError

    @staticmethod
    def ogb_idx_split2mask(ogb_idx_split, pyg_data):
        """
        TODO: Support Heterogenerous
        """

        def normalize_valid_name(k):
            return k if k != "valid" else "val"

        num_nodes = pyg_data.num_nodes
        for k, idx in ogb_idx_split.items():
            nk = normalize_valid_name(k) + "_mask"
            pyg_data[nk] = index2mask(idx, num_nodes)
        return pyg_data

    @staticmethod
    def dump_pyg_data2npz(data, dump_root, dump_name):
        dump_root = Path(dump_root)
        dump_root.mkdir(exist_ok=True, parents=True)
        file_path = dump_root / f"{dump_name}.npz"

        edge_index = data.edge_index.cpu().numpy()
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        train_mask = data.train_mask.cpu().numpy() if hasattr(data, "train_mask") else None
        val_mask = data.val_mask.cpu().numpy() if hasattr(data, "val_mask") else None
        test_mask = data.test_mask.cpu().numpy() if hasattr(data, "test_mask") else None
        edge_attr = data.edge_attr.cpu().numpy() if hasattr(data, "edge_attr") else None

        np.savez(
            file_path,
            edge_index=edge_index,
            node_attr=x,
            node_label=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_attr=edge_attr,
        )


class PreAtk(InMemoryDataset):
    def __init__(
            self, root, name, setting, ptb_budget, seed, transform=None, pre_transform=None, pre_filter=None
    ):
        self.name = name
        self.setting = setting
        self.ptb_budget = ptb_budget
        self.seed = seed
        sub_root = "_".join([self.head_fn, self.raw_file_base, self.tail_fn]).strip("_")
        super(PreAtk, self).__init__(osp.join(root, sub_root), transform, pre_transform, pre_filter)
        self.global_info = None
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_base(self):
        name_seed_set_ptb = f"{self.name}_seed={self.seed}_set={self.setting}_ptb={self.ptb_budget}"
        return name_seed_set_ptb

    @property
    def tail_fn(self):
        return ""

    @property
    def head_fn(self):
        return ""

    @property
    def raw_file_names(self):
        raw_fn = "_".join([self.head_fn, self.raw_file_base, self.tail_fn]).strip("_")
        raw_fn = raw_fn + ".npz"
        return [raw_fn]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass
        # raise "Downloading not supported at now"

    def load_npz(self, path):
        loaded = np.load(path, allow_pickle=True)
        logger.info(loaded.files)
        global_info = loaded["global_info"]

        ori_adj = loaded["ori_adj"].item()
        ori_node_fea = loaded["ori_node_fea"]
        modified_adj = loaded["modified_adj"].item()
        modified_fea = loaded["modified_node_fea"]

        node_label = loaded["node_label"]
        idx_train = loaded["idx_train"]
        idx_val = loaded["idx_val"]
        idx_test = loaded["idx_test"]

        assert is_spmatrix_symmetric(ori_adj)
        assert is_spmatrix_symmetric(modified_adj)
        num_node = ori_adj.shape[-1]

        x = torch.from_numpy(ori_node_fea).to(torch.float32)
        ori_adj = ori_adj.tocoo()
        ori_row = torch.from_numpy(ori_adj.row).to(torch.long)
        oir_col = torch.from_numpy(ori_adj.col).to(torch.long)
        ori_edge_index = torch.stack([ori_row, oir_col], dim=0)

        y = torch.from_numpy(node_label).to(torch.long)
        train_mask = torch.from_numpy(index2mask(idx_train, num_node))
        val_mask = torch.from_numpy(index2mask(idx_val, num_node))
        test_mask = torch.from_numpy(index2mask(idx_test, num_node))

        modified_adj = modified_adj.tocoo()
        md_row = torch.from_numpy(modified_adj.row).to(torch.long)
        md_col = torch.from_numpy(modified_adj.col).to(torch.long)
        md_edge_index = torch.stack([md_row, md_col], dim=0)
        md_x = torch.from_numpy(modified_fea).to(torch.float32)

        targets = loaded.get("targets", None)
        targets = torch.from_numpy(targets).to(torch.long) if targets is not None else None
        target_mask = test_mask if targets is None else index2mask(targets, num_node)

        oir_dt = Data(
            edge_index=ori_edge_index,
            x=x,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            target_mask=target_mask,
            attacked=False,
            attack_type=self.head_fn,
        )
        atk_dt = Data(
            edge_index=md_edge_index,
            x=md_x,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            target_mask=target_mask,
            attacked=True,
            attack_type=self.head_fn,
        )

        return [atk_dt, oir_dt], global_info

    def process(self):
        data_list, self.global_info = self.load_npz(self.raw_paths[0])
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_pyg_data(self, load_clean=False):
        """dataset[0]: ptb graph, dataset[1]: clean graph"""
        if not load_clean:
            assert self[0].attacked
            return self[0]
        else:
            assert not self[1].attacked
            return self[1]

    def which_data(self):
        data_desc = self.raw_file_names[0]
        return data_desc.replace(".npz", "")


class PreMetaAtk(PreAtk):
    def __init__(
            self,
            root,
            name,
            setting,
            ptb_budget,
            seed,
            lambda_=0.0,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.lambda_ = float(lambda_)
        super(PreMetaAtk, self).__init__(
            root, name, setting, ptb_budget, seed, transform, pre_transform, pre_filter
        )

    @property
    def tail_fn(self):
        return f"lambda={self.lambda_}"

    @property
    def head_fn(self):
        return "metattack"


class PreNetAtk(PreAtk):
    def __init__(
            self,
            root,
            name,
            setting,
            ptb_budget,
            seed,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        super(PreNetAtk, self).__init__(
            root, name, setting, ptb_budget, seed, transform, pre_transform, pre_filter
        )

    @property
    def head_fn(self):
        return "nettack"
