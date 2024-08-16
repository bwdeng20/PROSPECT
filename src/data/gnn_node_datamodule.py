import copy
import warnings
from typing import Callable, Optional, Sequence, Union
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import get_input_nodes
from torch_geometric.sampler import NumNeighbors

from torch_geometric.data.lightning.datamodule import (
    DataLoader,
    NodeLoader,
    NeighborSampler,
    infer_input_nodes,
    kwargs_repr,
    InputNodes
)

from lightning import LightningDataModule as PLLightningDataModule
import torch_geometric.transforms as T

from .components.graph_data import UnifiedReader, PreMetaAtk, PreNetAtk
from .components.resolvers import parse_transform
from .components.split import RandomSplitter, GrbSplitter, NettackSplit, NullSplitter, GCNRandomSplitter
from .components.utils import RemoveSelfLoops, NormalizeLabel
from lightning.pytorch.utilities import rank_zero_warn


def seq2list(seq):
    if seq is None:
        return None
    else:
        return list(seq)


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs):
        super().__init__()

        if not has_val:
            self.val_dataloader = None

        if not has_test:
            self.test_dataloader = None

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers',
                          kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            rank_zero_warn(f"The 'shuffle={kwargs['shuffle']}' option is "
                           f"ignored in '{self.__class__.__name__}'. Remove it "
                           f"from the argument list to disable this warning")
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kwargs_repr(**self.kwargs)})'


class ModifiedLightningNodeData(LightningDataModule):
    r"""Converts a :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`data` for multi-GPU node-level
    training via `PyTorch Lightning <https://www.pytorchlightning.ai>`_.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.NeighborLoader`.


    .. note::

        - This is a MODIFIED VERSION of PyG LightningNodeData to support inductive
            data split and different neighbor sizes for `train`, `val`, and `test`
            dataloaders.

        - Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:
    """

    def __init__(
            self,
            data: Union[Data, HeteroData],
            input_train_nodes: InputNodes = None,
            input_val_nodes: InputNodes = None,
            input_test_nodes: InputNodes = None,
            loader: str = "neighbor",
            num_train_neighbors: NumNeighbors = None,
            num_val_neighbors: NumNeighbors = None,
            num_test_neighbors: NumNeighbors = None,
            num_inductive_neighbors: NumNeighbors = None,
            batch_size: int = 1,
            num_workers: int = 0,
            inductive_test_rate: float = 0.0,
            transform: Optional[Union[str, Callable]] = None,
            keep_original_id: bool = False,
            distill_train: bool = False,
            device: Optional[Union[str, torch.device]] = None,
            **kwargs,
    ):
        assert loader in ["full", "neighbor"]
        assert (
                0 <= inductive_test_rate <= 1
        ), "the ratio of inductive test node to all test nodes  is within [0,1]"

        input_train_nodes, input_val_nodes, input_test_nodes = self.infer_nodes(
            data, input_train_nodes, input_val_nodes, input_test_nodes
        )

        if loader not in ['full', 'neighbor']:
            raise ValueError(f"Undefined 'loader' option (got '{loader}')")

        if loader == 'full' and batch_size != 1:
            rank_zero_warn(f"Re-setting 'batch_size' to 1 in "
                           f"'{self.__class__.__name__}' for loader='full' "
                           f"(got '{batch_size}')")
            batch_size = 1

        if loader == 'full' and num_workers != 0:
            rank_zero_warn(f"Re-setting 'num_workers' to 0 in "
                           f"'{self.__class__.__name__}' for loader='full' "
                           f"(got '{num_workers}')")
            num_workers = 0

        num_train_neighbors = seq2list(num_train_neighbors)
        num_val_neighbors = seq2list(num_val_neighbors)
        num_test_neighbors = seq2list(num_test_neighbors)
        num_inductive_neighbors = seq2list(num_inductive_neighbors)

        super().__init__(
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.transform = parse_transform(transform)
        self.distill_train = distill_train

        # return node_type when hetero, test_nodes is LongTensor instead of bool mask
        test_node_type, test_nodes = get_input_nodes(data, input_test_nodes)

        self.data = data.to(device) if device is not None else data
        # keep the original node idx
        self.keep_original_id = keep_original_id
        if keep_original_id:
            self.data.n_id = torch.arange(self.data.num_nodes, device=self.device_dt)
        self.loader = loader
        self.inductive_test_rate = inductive_test_rate

        if inductive_test_rate > 0:  # if inductive; TODO HeteroData
            num_test_nodes = len(test_nodes)
            num_inductive_test = int(num_test_nodes * inductive_test_rate)
            inductive_test_nodes = test_nodes[:num_inductive_test]

            self.obs_nodes: InputNodes = torch.ones(data.num_nodes, dtype=torch.bool, device=self.device_dt)
            self.obs_nodes[inductive_test_nodes] = 0
            # train/val/test/pred BOOL masks are also sampled
            self.obs_data = self.data.subgraph(self.obs_nodes)
            # infer the new masks in the observed subgraph
            self.train_nodes = infer_input_nodes(self.obs_data, "train")
            self.val_nodes = infer_input_nodes(self.obs_data, "val")
            # transductive test_nodes in the observed graph
            self.test_nodes = infer_input_nodes(self.obs_data, "test")
            # inductive test nodes in the original ENTIRE graph self.data
            self.inductive_test_nodes = inductive_test_nodes
        else:
            self.obs_data = self.data  # by default transductive
            self.train_nodes = input_train_nodes
            self.val_nodes = input_val_nodes
            self.test_nodes = input_test_nodes
            # pseudo 1-size batch for pred
            self.inductive_test_nodes = torch.LongTensor([0])

        if loader == "full":
            if kwargs.get("pin_memory", False):
                warnings.warn(
                    f"Re-setting 'pin_memory' to 'False' in "
                    f"'{self.__class__.__name__}' for loader='full' "
                    f"(got 'True')"
                )
            self.kwargs["pin_memory"] = False

        if loader == "neighbor":
            assert num_train_neighbors is not None, "Require num_train_neighbors"
            # Training Sampler is defined on the observed graph, which is the entire
            # graph under the transductive setting, or the observed graph under the
            # inductive setting.
            self.train_neighbor_sampler = NeighborSampler(
                data=self.obs_data,
                num_neighbors=num_train_neighbors,
                replace=kwargs.get("replace", False),
                directed=kwargs.get("directed", True),
                time_attr=kwargs.get("time_attr", None),
                is_sorted=kwargs.get("is_sorted", False),
                share_memory=num_workers > 0,
            )

            if num_val_neighbors is not None:
                self.val_neighbor_sampler = NeighborSampler(
                    data=copy.copy(self.obs_data),
                    num_neighbors=num_val_neighbors,
                    replace=kwargs.get("replace", False),
                    directed=kwargs.get("directed", True),
                    time_attr=kwargs.get("time_attr", None),
                    is_sorted=kwargs.get("is_sorted", False),
                    share_memory=num_workers > 0,
                )
            else:
                self.val_neighbor_sampler = self.train_neighbor_sampler

            if num_test_neighbors is not None:
                self.test_neighbor_sampler = NeighborSampler(
                    data=copy.copy(self.obs_data),
                    num_neighbors=num_test_neighbors,
                    replace=kwargs.get("replace", False),
                    directed=kwargs.get("directed", True),
                    time_attr=kwargs.get("time_attr", None),
                    is_sorted=kwargs.get("is_sorted", False),
                    share_memory=num_workers > 0,
                )
            else:
                self.test_neighbor_sampler = self.val_neighbor_sampler
            # pred sampler is for inductive nodes
            num_inductive_neighbors = (
                num_inductive_neighbors if num_inductive_neighbors is not None else num_val_neighbors
            )
            self.inductive_test_neighbor_sampler = NeighborSampler(
                data=copy.copy(self.data),
                num_neighbors=num_inductive_neighbors,
                replace=kwargs.get("replace", False),
                directed=kwargs.get("directed", True),
                time_attr=kwargs.get("time_attr", None),
                is_sorted=kwargs.get("is_sorted", False),
                share_memory=num_workers > 0,
            )

    @property
    def device_dt(self):
        return self.data.x.device

    @staticmethod
    def infer_nodes(data, train_nodes=None, val_nodes=None, test_nodes=None):
        """get the train/val/test split when either split is specified or builtin."""
        if train_nodes is None:
            train_nodes = infer_input_nodes(data, split="train")
        if val_nodes is None:
            val_nodes = infer_input_nodes(data, split="val")
            if val_nodes is None:
                val_nodes = infer_input_nodes(data, split="valid")

        if test_nodes is None:
            test_nodes = infer_input_nodes(data, split="test")

        return torch.tensor(train_nodes), torch.tensor(val_nodes), torch.tensor(test_nodes)

    def prepare_data(self):
        if self.loader == 'full':
            try:
                num_devices = self.trainer.num_devices
            except AttributeError:
                # PyTorch Lightning < 1.6 backward compatibility:
                num_devices = self.trainer.num_processes
                num_devices = max(num_devices, self.trainer.num_gpus)

            if num_devices > 1:
                raise ValueError(
                    f"'{self.__class__.__name__}' with loader='full' requires "
                    f"training on a single device")
        super().prepare_data()

    def dataloader(self, input_nodes: InputNodes, shuffle: bool = False, split: str = "train") -> DataLoader:
        data = self.data if split == "inductive_test" else self.obs_data
        if self.loader == "full":
            warnings.filterwarnings("ignore", ".*does not have many workers.*")
            warnings.filterwarnings("ignore", ".*data loading bottlenecks.*")
            data = self.transform(data) if self.transform is not None else data
            return torch.utils.data.DataLoader(
                [data],
                shuffle=False,
                collate_fn=lambda xs: xs[0],
                **self.kwargs,
            )

        if self.loader == "neighbor":
            nbr_sampler = getattr(self, split + "_neighbor_sampler")
            return NodeLoader(
                data=data,
                input_nodes=input_nodes,
                node_sampler=nbr_sampler,
                shuffle=shuffle,
                transform=self.transform,
                **self.kwargs,
            )

        raise NotImplementedError

    def distill_dataloader(self) -> DataLoader:
        # All observed nodes are used to distill knowledge, "val" split is only to get
        # `self.obs_data` and the `val_neighbor_sampler` suitable for SAGE inference.
        # Note that the student model won't backward the classification loss on nodes other
        # than those marked by "Data.train_mask", meaning that "split=val" won't leak information.
        return self.dataloader(None, shuffle=True, split="val")

    def train_dataloader(self, distill: bool = False) -> DataLoader:
        """"""
        if distill or self.distill_train:
            return self.distill_dataloader()
        return self.dataloader(self.train_nodes, shuffle=True, split="train")

    def val_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.val_nodes, shuffle=False, split="val")

    def test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.test_nodes, shuffle=False, split="test")

    def inductive_test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.inductive_test_nodes, shuffle=False, split="inductive_test")

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f"{self.__class__.__name__}({kwargs})"


class UnifiedNodeReader(UnifiedReader):
    WoLabelDataNames = UnifiedReader.AmazonDataNames
    SupportedSetting = [
        "gcn",
        "net_meta",
        "nettack",
        "grb",
        "llc-grb",
        "grbs",
        "grbm",
        "grbl",
        "llc-grbs",
        "llc-grbm",
        "llc-grbl",
    ]

    def __init__(
            self,
            name: str,
            root: str = "data",
            setting: str = "random",
            undirected: bool = True,
            remove_self_loop: bool = True,
            transform: Optional[Union[str, Callable]] = None,
            pre_transform: Optional[Union[str, Callable]] = None,
            train_val_test: Sequence = None,
            seed: int = None,
    ):
        super().__init__(name, root, transform, pre_transform)
        self.setting = setting
        self.seed = seed
        self.undirected = undirected
        self.remove_self_loop = remove_self_loop
        if train_val_test is None:
            self.train_val_test = [10, 10, 80]
        else:
            self.train_val_test = train_val_test

        self.require_llc = self.is_llc_required(setting)
        self.splitter = self.get_split_transform()

        trans = list()
        if self.remove_self_loop:
            trans.append(RemoveSelfLoops())
        if self.undirected:
            trans.append(T.ToUndirected(reduce="add"))
        if self.require_llc:
            trans.append(T.LargestConnectedComponents())
        trans.append(NormalizeLabel())
        trans.append(self.splitter)

        self.transform = T.Compose([*trans, self.transform])

    def get_split_transform(self):
        if self.setting == "nettack":
            return NettackSplit(self.train_val_test, self.seed)
        elif "grb" in self.setting:
            tmode, require_llc = GrbSplitter.parse_grb_str(self.setting)
            return GrbSplitter(self.train_val_test, mode=tmode, seed=self.seed)
        elif self.setting == "gcn":
            self.train_val_test = [20, 500, 1000]
            return GCNRandomSplitter(self.train_val_test, seed=self.seed)
        elif self.setting == "random":
            return RandomSplitter(self.train_val_test, seed=self.seed)
        elif self.setting == "gcn_public":
            return NullSplitter()
        else:
            raise ValueError(f"{self.setting} not a valid setting")

    @staticmethod
    def is_llc_required(setting):
        if setting in ("net_meta", "nettack") or "llc" in setting:
            return True
        else:
            return False


class UnifiedNodeData(ModifiedLightningNodeData):
    def __init__(
            self,
            name: str,
            data_dir: str = "data",
            undirected: bool = True,
            remove_self_loop: bool = True,
            transform: Optional[Union[str, Callable]] = None,
            mid_transform: Optional[Union[str, Callable]] = None,
            pre_transform: Optional[Union[str, Callable]] = None,
            input_train_nodes: InputNodes = None,
            input_val_nodes: InputNodes = None,
            input_test_nodes: InputNodes = None,
            loader: str = "neighbor",
            num_train_neighbors: NumNeighbors = None,
            num_val_neighbors: NumNeighbors = None,
            num_test_neighbors: NumNeighbors = None,
            num_inductive_neighbors: NumNeighbors = None,
            batch_size: int = 1,
            num_workers: int = 0,
            inductive_test_rate: float = 0.0,
            train_val_test: Sequence = None,
            setting: str = "random",
            keep_original_id: bool = False,
            distill_train: bool = False,
            seed: int = 15,
            device: Union[str, torch.device, int] = None,
            **kwargs,
    ):
        """PL pyg customized interfaces for Node-level tasks.

        Args:
            name: the dataset name, e.g., cora, pubmed, a-photo, reddit
            data_dir: the root dir of the dataset
            undirected: If True, convert a graph to undirected one
            remove_self_loop: If True, remove the self-loops in a graph
            transform: callables or class names (str) of :`torch_geometric.transforms `
            pre_transform: like transform
            input_train_nodes:
            input_val_nodes:
            input_test_nodes:
            loader: If `neighbor`, load graph data as GraphSAGE; if "full", load the
                entire graph
            num_train_neighbors:
            num_val_neighbors:
            num_test_neighbors:
            num_inductive_neighbors:
            batch_size:
            num_workers:
            inductive_test_rate: the inductive node ratio in the test set
            train_val_test: If assigned, re-split randomly the whole dataset into
                three parts in accordance with the given ratios in `train_val_test`
            keep_original_id: If True, the loaded subgraph contains the original node
                indices of all nodes in `data.n_id`
            **kwargs: Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
        """
        self.reader = UnifiedNodeReader(
            name,
            data_dir,
            setting=setting,
            undirected=undirected,
            remove_self_loop=remove_self_loop,
            pre_transform=pre_transform,
            transform=mid_transform,
            train_val_test=train_val_test,
            seed=seed,
        )
        pyg_data = self.reader.get_pyg_data()
        super().__init__(
            data=pyg_data,
            input_train_nodes=input_train_nodes,
            input_val_nodes=input_val_nodes,
            input_test_nodes=input_test_nodes,
            loader=loader,
            num_train_neighbors=num_train_neighbors,
            num_val_neighbors=num_val_neighbors,
            num_test_neighbors=num_test_neighbors,
            num_inductive_neighbors=num_inductive_neighbors,
            batch_size=batch_size,
            num_workers=num_workers,
            inductive_test_rate=inductive_test_rate,
            transform=transform,
            keep_original_id=keep_original_id,
            distill_train=distill_train,
            device=device,
            **kwargs,
        )


class UnifiedAtkNodeData(ModifiedLightningNodeData):
    def __init__(
            self,
            name: str,
            data_dir: str = "atk_data",
            attack_method: str = "metattack",
            load_clean: bool = False,
            setting: str = "nettack",
            ptb_budget: Union[float, int] = None,
            lambda_: float = 0.0,
            seed: int = 15,
            loader: str = "neighbor",
            input_train_nodes: InputNodes = None,
            input_val_nodes: InputNodes = None,
            input_test_nodes: InputNodes = None,
            transform: Optional[Union[str, Callable]] = None,
            mid_transform: Optional[Union[str, Callable]] = None,
            pre_transform: Optional[Union[str, Callable]] = None,
            num_train_neighbors: NumNeighbors = None,
            num_val_neighbors: NumNeighbors = None,
            num_test_neighbors: NumNeighbors = None,
            num_inductive_neighbors: NumNeighbors = None,
            batch_size: int = 1,
            num_workers: int = 0,
            inductive_test_rate: float = 0.0,
            keep_original_id: bool = False,
            distill_train: bool = False,
            device: Union[str, torch.device, int] = None,
            **kwargs,
    ):
        self._load_clean = bool(load_clean)
        attack_method = attack_method.lower()
        if attack_method == "metattack":
            self.reader = PreMetaAtk(
                root=data_dir,
                name=name,
                setting=setting,
                ptb_budget=ptb_budget,
                seed=seed,
                lambda_=lambda_,
                transform=mid_transform,
                pre_transform=pre_transform,
            )
        elif attack_method == "nettack":
            self.reader = PreNetAtk(
                root=data_dir,
                name=name,
                setting=setting,
                ptb_budget=ptb_budget,
                seed=seed,
                transform=mid_transform,
                pre_transform=pre_transform,
            )
        else:
            raise ValueError

        pyg_data = self.reader.get_pyg_data(self._load_clean)
        super().__init__(
            data=pyg_data,
            input_train_nodes=input_train_nodes,
            input_val_nodes=input_val_nodes,
            input_test_nodes=input_test_nodes,
            loader=loader,
            num_train_neighbors=num_train_neighbors,
            num_val_neighbors=num_val_neighbors,
            num_test_neighbors=num_test_neighbors,
            num_inductive_neighbors=num_inductive_neighbors,
            batch_size=batch_size,
            num_workers=num_workers,
            inductive_test_rate=inductive_test_rate,
            transform=transform,
            keep_original_id=keep_original_id,
            distill_train=distill_train,
            device=device,
            **kwargs,
        )

    def which_data(self):
        data_desc = self.reader.raw_file_names[0]
        return data_desc.replace(".npz", f"_clean={self._load_clean}")
