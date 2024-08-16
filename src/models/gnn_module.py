from typing import Any, List, Mapping, Union

from lightning import LightningModule
import torch
import torch.nn.functional as func
from torchmetrics import Accuracy, MaxMetric, MeanMetric
from .components.resolvers import optimizer_resolver, lr_scheduler_resolver


class GNNLitModule(LightningModule):
    def __init__(
            self,
            gnn: Any,
            optimizer: Union[str, torch.optim.Optimizer] = None,
            optimizer_args: Union[List[Mapping[str, Any]], Mapping[str, Any]] = None,
            scheduler: Union[str, Any] = None,
            scheduler_args: Mapping[str, Any] = None,
    ):
        """
        Args:
            gnn: An initialized PyG-based GNN model
            optimizer:
                1)The name of a pytorch optimizer;
                2)A pytorch optimizer class (not instance!);
                3)A partial pytorch optimizer, e.g., partial(Adam,lr=0.001). The partial
                    initialized arguments WILL be overriden by "optimizer_args"
                4)If None, return Adam(self.parameters(),lr=0.001) by default
            optimizer_args:
                1) Mapping[str,Any], a pytorch optimizer param group like
                    {"lr":0.001,"wd":0.0001}, where the missing parameter key-value
                    pair "param":x will be merged after invoking "configure_optimizers"
                    to make up {"params":x,"lr":0.001,"wd":0.0001}
                2) List[Mapping[str, Any], A list of Dicts in 1).The i-th Dict is the
                    optimizer arguments for the i-th group of parameters
            scheduler:
                1) str: A pytorch learning rate scheduler name
                2) A pytorch scheduler class (not instance!)
                3) A partial pytorch scheduler, e.g., partial(StepLR,step_size=10). The
                    partial initialized arguments WON'T be overriden by "scheduler_args"

            scheduler_args:
                A dict of scheduler arguments, e.g., {"step_size":20, "gamma":0.2} for
                StepLR
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["gnn"])
        self.gnn = gnn
        self.tape = getattr(self.gnn, "tape", False)

        nclass = gnn.out_channels

        self.train_acc = Accuracy(task="multiclass", num_classes=nclass)
        self.val_acc = Accuracy(task="multiclass", num_classes=nclass)
        self.test_accs = torch.nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=nclass) for _ in range(2)]
        )
        self.test_all_acc = Accuracy(task="multiclass", num_classes=nclass)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x, edge_index, edge_weight=None):
        out = self.gnn(x, edge_index, edge_weight)
        return out[-1] if self.gnn.tape else out

    def step(self, data, split):
        bs = data.get("batch_size", None)
        topo = data.get("adj_t", data.get("edge_index", None))
        wgt = data.get("edge_weight", None)
        if bs is None:  # SAGE with Full Graph
            split_mask = data.get(split + "_mask")
            logit = self.forward(data.x, topo, wgt)[split_mask]
            y = data.y[split_mask]
        else:  # SAGE with Neighbor Sampler
            logit = self.forward(data.x, topo, wgt)[:bs]
            y = data.y[:bs]
        return logit, y, len(y)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()
        self.val_acc_best.reset()

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     # will be used for the user to aggregate the outputs from training_step
    #     # https://github.com/Lightning-AI/lightning/issues/5550
    #     self.train_acc.reset()

    def training_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target, batch_size = self.step(data, "train")
        loss = func.cross_entropy(logit, target)
        pred = logit.argmax(dim=-1)
        self.train_acc(pred, target)
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return {"loss": loss}

    def validation_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target, batch_size = self.step(data, "val")
        loss = func.cross_entropy(logit, target)
        pred = logit.argmax(dim=-1)
        self.val_acc(pred, target)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a
        #  metric object otherwise metric would be reset by lightning after each epoch
        self.log("val_acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def test_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target, batch_size = self.step(data, "test")
        pred = logit.argmax(dim=-1)
        self.test_accs[dataloader_idx](pred, target)
        self.test_all_acc.update(pred, target)
        self.log(
            "test_acc",
            self.test_accs[dataloader_idx],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            add_dataloader_idx=True,  # automatically add dl_idx suffix to `test_acc`
        )

    def on_test_epoch_end(self) -> None:
        [acc.reset() for acc in self.test_accs]
        self.log("test_all_acc", self.test_all_acc.compute())
        self.test_all_acc.reset()

    def configure_optimizers(self):
        param_groups = self.gnn.default_optimizer_scheme()
        if param_groups is None:
            param_groups = [{"params": self.parameters()}]

        optim_args = self.hparams.optimizer_args
        if optim_args is not None:
            full_param_groups = []
            if isinstance(optim_args, list):  # multiple groups of params [{},...,{}]
                for i, group_arg in enumerate(optim_args):
                    # constitute the i-th param group by merging
                    # {"params": xxx} in 'param_groups' and
                    # {"weight_decay": wd_i, "lr_i":lr_i} in 'optim_args'
                    param_group_with_arg = param_groups[i] | group_arg
                    full_param_groups.append(param_group_with_arg)
            else:  # only one param group args in "optim_args"
                full_param_groups.append(param_groups[0] | optim_args)
        else:
            full_param_groups: List = param_groups

        init_opt = self.hparams.optimizer
        if init_opt is None:  # default optimizer
            opt = torch.optim.Adam(full_param_groups, lr=0.001)
        elif isinstance(init_opt, str):
            opt = optimizer_resolver(init_opt, full_param_groups)
        else:
            opt = init_opt(full_param_groups)
        init_sch = self.hparams.scheduler

        if init_sch is None:
            return [opt]
        elif isinstance(init_sch, str):
            scheduler_args = self.hparams.scheduler_args or {}
            sch = lr_scheduler_resolver(init_sch, optimizer=opt, **scheduler_args)
        else:
            sch = init_sch(optimizer=opt)

        lr_scheduler_config = {
            "scheduler": sch,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [opt], [lr_scheduler_config]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        # When any of torch.nn.utils.parametrizations, e.g., spectral_norm, is applied to
        # "weight" of a "submodule", the stata_dict contains both a weight named
        # "submodule.weight" and the renamed original learnable parameters handled by
        # "submodule.parametrization.weight.original". Non-strict state_dict loader is
        # required to support loading state_dict to both the original model and the model
        # with parametrization.
        super().load_state_dict(state_dict, strict)

    @torch.no_grad()
    def inference(self):
        pass
