from typing import Any, Callable, List, Optional, Union, Sequence, Tuple
import copy
import torch
from functools import cached_property, lru_cache
from pathlib import Path
from torch import Tensor
from torch.nn import ModuleList, ModuleDict
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torchmetrics import Accuracy, MaxMetric
from torch_geometric.typing import OptTensor
from .components.resolvers import distill_resolver, loss_resolver, lr_scheduler_resolver
from .components.utils import load_th_or_pl_ckpt2md
from .components.gnns import NewBasicGNN
from .components.mlps import MLP

ChoirOutputElm = Tuple[OptTensor, OptTensor, Tensor, List[OptTensor]]
ChoirOutput = Sequence[ChoirOutputElm]

ZippedChoirOutput = Tuple[Tuple[OptTensor], Tuple[OptTensor], Tuple[Tensor], Tuple[List[OptTensor]]]


def build_models(model, model_ckpt=None, num_model=None, map_location="cpu") -> ModuleList:
    """return 1d ModuleList"""
    if not isinstance(model, Sequence):
        num_model = 1 if num_model is None else num_model
        models = torch.nn.ModuleList([copy.deepcopy(model) for _ in range(num_model)])
    else:
        models = torch.nn.ModuleList(model)

    if model_ckpt is not None:
        ckpts = [model_ckpt] if not isinstance(model_ckpt, (list, tuple)) else model_ckpt
        for i in range(len(ckpts)):  # load ckpt for the first len(model_ckpt) models
            models[i] = load_th_or_pl_ckpt2md(models[i], ckpts[i], map_location=map_location)
    return models


def build_2d_metrics(metric_cls, num_model, num_dataloader=1, **kwargs) -> ModuleList:
    """return 2d nested ModuleList[num_model, num_dataloaders]"""
    metric_2dlist = [ModuleList([metric_cls(**kwargs) for _ in range(num_dataloader)])
                     for _ in range(num_model)]
    return ModuleList(metric_2dlist)


def build_2d_metric_dict(metric_cls, md_group_names, num_model, num_dataloader=1, **kwargs) -> ModuleDict:
    """return ModuleDict[<md_group, ModuleList[num_model, num_dataloaders]>, ...]"""
    dic = {md_group: build_2d_metrics(metric_cls, num_model[i], num_dataloader, **kwargs)
           for i, md_group in enumerate(md_group_names)}
    return ModuleDict(dic)


def reset_2d_metrics(metric2dlist):
    for metric1model in metric2dlist:
        for metric1model1loader in metric1model:
            metric1model1loader.reset()


def backward_iter(iter2op):
    for each in iter2op:
        each.backward()


def zero_grad_iter(iter2op):
    for each in iter2op:
        each.zero_grad()


def step_iter(iter2op, debug=False):
    for each in iter2op:
        if debug:
            assert isinstance(each, (LightningOptimizer, LRScheduler)), \
                f"Should have wrapped `torch.optimizer` with `LightningOptimizer` or `torch.LRScheduler`, " \
                f"but got {type(each)}"
        each.step()


class GDMLPLitModule(LightningModule):
    def __init__(
            self,
            teacher: Union[torch.nn.Module, Sequence[torch.nn.Module]],
            student: Union[torch.nn.Module, Sequence[torch.nn.Module]],
            optimizer_s: Any,
            optimizer_t: Optional[Any] = None,
            student_ckpt: Union[str, List[str], Path, List[Path]] = None,
            teacher_ckpt: Union[str, List[str], Path, List[Path]] = None,
            num_student: int = None,
            num_teacher: int = None,
            scheduler_s: Optional[Union[str, Any]] = None,
            scheduler_t: Optional[Union[str, Any]] = None,
            scheduler_s_kwargs: Optional[dict] = None,
            scheduler_t_kwargs: Optional[dict] = None,
            lambda_aim: Union[float, Sequence[float]] = 1.0,
            lambda_kd_fea: Union[float, Sequence[float]] = 1.0,
            lambda_kd_logit: Union[float, Sequence[float]] = 1.0,
            feature_distiller: Optional[Union[Callable, str]] = None,
            logit_distiller: Union[Callable, str] = "SoftTarget",
            fd_kwargs: Optional[dict] = None,
            ld_kwargs: Optional[dict] = None,
            off_wd_kwargs: Optional[dict] = None,
            on_wd_kwargs: Optional[dict] = None,
            aim_criterion: Union[Callable, str] = "CrossEntropy",
            student_forward: bool = True,
            log_verbose: int = 1,
            log_distill: bool = False,
            map_location: str = "cpu",
    ):
        super().__init__()
        # Check input args
        if isinstance(teacher, LightningModule):
            raise TypeError(
                "Please use native torch models instead of pytorch-lightning wrapped ones as teachers"
            )

        if isinstance(student, LightningModule):
            raise TypeError(
                "Please use native torch models instead of pytorch-lightning wrapped ones as students"
            )

        if optimizer_t is None:
            assert teacher_ckpt, (
                f"{self.__class__}(teacher={repr(teacher)[:50]},"
                f"student={repr(student)[:50]}) is now for off-line distillation, "
                f"pre-trained teacher is needed"
            )

        if feature_distiller is None and logit_distiller is None:
            raise ValueError("Either one feature-based or logit-based distiller " "must be specified")

        # set weights for different distillation loss functions
        if isinstance(lambda_aim, Sequence):
            self.lambda_aim_s, self.lambda_aim_t = lambda_aim
        else:
            self.lambda_aim_s = self.lambda_aim_t = lambda_aim
        if isinstance(lambda_kd_fea, Sequence):
            self.lambda_kd_fea_s, self.lambda_kd_fea_t = lambda_kd_fea
        else:
            self.lambda_kd_fea_s = self.lambda_kd_fea_t = lambda_kd_fea
        if isinstance(lambda_kd_logit, Sequence):
            self.lambda_kd_logit_s, self.lambda_kd_logit_t = lambda_kd_logit
        else:
            self.lambda_kd_logit_s = self.lambda_kd_logit_t = lambda_kd_logit

        if self.lambda_aim_s == self.lambda_kd_fea_s == self.lambda_kd_logit_s == 0:
            raise ValueError(
                "The loss weights for the target task, feature distillation,"
                "and logit distillation shouldn't have been zeros at the same time"
            )
        self.save_hyperparameters(logger=False, ignore=["teacher", "student"])

        # build models
        self.forward_with_student = student_forward

        self.teachers = build_models(teacher, teacher_ckpt, num_teacher, map_location)
        self.students = build_models(student, student_ckpt, num_student, map_location)
        self.all_models = [*self.teachers, *self.students]  # reference only, not copy, not torch module

        assert self.teachers[0].tape == self.students[0].tape
        self.tape = self.students[0].tape

        if feature_distiller is not None:  # need all hidden features
            for each in self.all_models:
                each.tape = True
            self.tape = True

        self.student_forward4distill = self.dispatch_forward4distill(self.students[0])
        self.teacher_forward4distill = self.dispatch_forward4distill(self.teachers[0])
        # initialize all distillers
        fd_kwargs = fd_kwargs or {}
        ld_kwargs = ld_kwargs or {}
        off_wd_kwargs = off_wd_kwargs or {}
        on_wd_kwargs = on_wd_kwargs or {}

        self.feature_distiller = distill_resolver(feature_distiller, **fd_kwargs)
        self.logit_distiller = distill_resolver(logit_distiller, **ld_kwargs)
        self.check_logit_fea_distill_strategy(self.feature_distiller)
        self.check_logit_fea_distill_strategy(self.logit_distiller)

        # control metric logging, e.g., accuracy logging
        self.log_verbose = log_verbose
        self.log_distill = log_distill

        # Set the metric of target task.
        # Examples: 1. cross-entropy for classification 2. mse for regression
        self.aim_criterion = loss_resolver(aim_criterion)

        # accuracies
        self.train_acc = None
        self.val_acc = None
        self.val_acc_best = None
        self.test_all_acc = None
        self.test_acc = None

        # set metrics to log
        self.nclass = student.out_channels
        self.build_2d_metrics()

        # Fix the ModelCheckpoint Callback issue arisen from manual optimization at Lightning==2.0.0
        # https://github.com/Lightning-AI/lightning/issues/17167
        # Remember to wrap naive pytorch optimizer with `LightningOptimizer`!!!
        self.automatic_optimization = False

    # ########################## Some utils ###################################################################

    def check_logit_fea_distill_strategy(self, distiller):
        if distiller is None:
            return

        if self.is_teacher_learning:
            if distiller.single_direction:
                raise TypeError(
                    f"The distiller {distiller} works in a double-direction way,i.e., teachers <--> students, "
                    f"please also set optimizers for teacher models during instantiating like "
                    f"`{self.__class__.__name__}(optimizer_s=partial(Adam,lr=0.01), optimizer_t=partial(Adam,lr=0.01)`")
        else:
            if not distiller.single_direction:
                raise TypeError(
                    f"The distiller {distiller} works in a single-direction way,i.e., teachers  --> students, "
                    f"please do NOT set optimizers for teacher models during instantiating and initialize like "
                    f"`{self.__class__.__name__}(optimizer_s=partial(Adam,lr=0.01), optimizer_t=None)`")

    @cached_property
    def num_students(self):
        return len(self.students)

    @cached_property
    def num_teachers(self):
        return len(self.teachers)

    @cached_property
    def is_teacher_learning(self):
        return self.hparams.optimizer_t is not None

    def build_2d_metrics(self):
        model_group_names, model_nums, log_teacher = self.log_describer("train")
        self.train_acc = build_2d_metric_dict(Accuracy, model_group_names, model_nums,
                                              task="multiclass",
                                              num_classes=self.nclass)

        model_group_names, model_nums, log_teacher = self.log_describer("val")
        self.val_acc = build_2d_metric_dict(Accuracy, model_group_names, model_nums,
                                            task="multiclass",
                                            num_classes=self.nclass)

        #  Logging best so far validation accuracy
        self.val_acc_best = build_2d_metric_dict(MaxMetric, model_group_names, model_nums)

        model_group_names, model_nums, log_teacher = self.log_describer("test")
        self.test_all_acc = build_2d_metric_dict(Accuracy, model_group_names, model_nums,
                                                 task="multiclass",
                                                 num_classes=self.nclass)

        self.test_acc = build_2d_metric_dict(Accuracy, model_group_names, model_nums,
                                             num_dataloader=2,
                                             task="multiclass",
                                             num_classes=self.nclass)

    @lru_cache
    def log_describer(self, stage, return_flag=False):
        model_group_names = ["students", "teachers"]
        model_nums = [self.num_students, self.num_teachers]
        log_teacher = 1
        if self.is_teacher_learning:
            pass
        elif self.log_verbose == 0 and stage in ("val", "test", "valid"):
            model_group_names = ["students"]
            model_nums = [self.num_students]
            log_teacher = 0

        elif self.log_verbose == 1 and stage == "test":
            model_group_names = ["students"]
            model_nums = [self.num_students]
            log_teacher = 0
        else:
            pass

        if not return_flag:
            return model_group_names, model_nums, log_teacher
        else:
            return log_teacher

    def dispatch_forward4distill(self, model):
        if isinstance(model, NewBasicGNN):
            return self.step_gnn4distill
        elif isinstance(model, MLP):
            return self.step_mlp4distill
        else:
            raise ValueError

    # ################  Train and eval models ######################################################
    def forward(self, x, model_idx=0, *args, **kwargs):
        if self.forward_with_student:
            out = self.students[model_idx](x, *args, **kwargs)
        else:
            out = self.teachers[model_idx](x, *args, **kwargs)
        return out[-1] if self.student.tape else out

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure val_acc_best doesn't store
        # accuracy from these checks
        reset_2d_metrics(self.val_acc_best["students"])
        reset_2d_metrics(self.val_acc["students"])
        if self.log_describer("val", return_flag=True):
            reset_2d_metrics(self.val_acc_best["teachers"])
            reset_2d_metrics(self.val_acc["teachers"])

    def get_lr_schedulers(self):
        # wrap single scheduler to [scheduler] to handle general multiple students and teachers cases
        sch = self.lr_schedulers()
        if (sch is not None) and not isinstance(sch, Sequence):
            return [sch]
        else:
            return sch

    def get_pl_optimizer_list(self):
        opt = self.optimizers()
        if isinstance(opt, LightningOptimizer):
            return [opt]
        elif isinstance(opt, Sequence):
            return opt
        else:
            raise TypeError(f"List[LightningOptimizer] or `LightningOptimizer` is expected, but got `{type(opt)}`")

    def on_train_epoch_end(self) -> None:
        schedulers = self.get_lr_schedulers()
        num_s = self.num_students
        if schedulers is not None:
            schedulers_s, schedulers_t = schedulers[:num_s], schedulers[num_s:]
        else:
            schedulers_s, schedulers_t = list(), list()

        if len(schedulers_s) > 0:
            step_iter(schedulers_s)

        if len(schedulers_t) > 0:
            step_iter(schedulers_t)

        reset_2d_metrics(self.train_acc["students"])
        if self.log_describer("train", return_flag=True):
            reset_2d_metrics(self.train_acc["teachers"])

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     # will be used for the user to aggregate the outputs from training_step
    #     # https://github.com/Lightning-AI/lightning/issues/5550
    #     pass

    def step_gnn4distill(self, model, data, split) -> ChoirOutputElm:
        # consider only mini-batch training since MLP prefer this style
        bs = data.get("batch_size", None)
        topo = data.get("adj_t", data.get("edge_index", None))
        mask = data.get(f"{split}_mask")
        if bs is None:  # SAGE with Full Graph
            xs = model.forward(data.x, topo)
            if self.tape:
                logit4distill = xs[-1]
                xs4distill = xs[:-1]
            else:
                logit4distill = xs
                xs4distill = None
            logit4aim = logit4distill[mask]
            y4aim = data.y[mask]
        else:  # SAGE with Neighbor Sampler
            # wrong op1: [:bs] y/xs of val or test nodes will be output in the training
            # stage as the `distill_dataloader` will iterate over all observed nodes
            # (including train, val, transductive test nodes)

            # wrong op2:[split_mask] `train_mask` may include nodes out of the batch and
            # thus the sampled graph is not `complete` in the eyes of these training
            # nodes

            # correct op: [:bs][split_mask[:bs]]
            xs = model.forward(data.x, topo)
            if self.tape:  # tape the hidden features and logits
                xs = [x[:bs] for x in xs]
                logit4distill = xs[-1]
                xs4distill = xs[:-1]
            else:  # only the logits
                logit4distill = xs[:bs]
                xs4distill = None
            logit4aim = logit4distill[mask[:bs]]
            y4aim = data.y[:bs][mask[:bs]]

        if logit4aim.numel() == 0:  # in this case only distill with `logit` or/and `xs`
            logit4aim = None
            y4aim = None

        return logit4aim, y4aim, logit4distill, xs4distill

    def step_mlp4distill(self, model, data, split) -> ChoirOutputElm:
        bs = data.get("batch_size", None)
        mask = data.get(f"{split}_mask")
        if bs is None:  # MLP with Full Graph
            xs = model.forward(data.x)
            if self.tape:
                logit4distill = xs[-1]
                xs4distill = xs[:-1]
            else:
                logit4distill = xs
                xs4distill = None
            logit4aim = logit4distill[mask]
            y4aim = data.y[mask]

        else:  # MLP with Neighbor Sampler
            # wrong op1: [:bs] y/xs of val or test nodes will be output in the training
            # stage as the `distill_dataloader` will iterate over all observed nodes
            # (including train, val, transductive test nodes)

            # wrong op2:[split_mask] `train_mask` may include nodes out of the batch and
            # thus the sampled graph is not `complete` in the eyes of these training
            # nodes

            # correct op: [:bs][split_mask[:bs]]
            xs = model.forward(data.x[:bs])
            if self.tape:  # tape the hidden features and logits
                logit4distill = xs[-1]
                xs4distill = xs[:-1]
            else:  # only the logits
                logit4distill = xs
                xs4distill = None
            logit4aim = logit4distill[mask[:bs]]
            y4aim = data.y[:bs][mask[:bs]]

        if logit4aim.numel() == 0:  # in this case only distill with `logit` or/and `xs`
            logit4aim = None
            y4aim = None

        return logit4aim, y4aim, logit4distill, xs4distill

    @staticmethod
    def step4many(models: Union[Sequence[torch.nn.Module], ModuleList], data, split, run_func):
        """
        Parameters
        ----------
        models: Union[Sequence[torch.nn.Module], ModuleList]
        data:   Data
        split:  str
            `train`,`val`,`test`, or any str attributes stored in `data`
        run_func: The function to run a model

        Returns
        -------
            - List[logit, y, batch_size=len(y)] If `run_func=step_{gnn/mlp}4distill`
            - List[logit4aim, y4aim, logit4distill, xs4distill, bs4aim] If `run_func=step_wo_distill`
        """
        outputs = []
        for model in models:
            outputs.append(run_func(model, data, split))
        return outputs

    def compute_all_loss4all_models(self,
                                    student_outputs: ChoirOutput,
                                    teacher_outputs: ChoirOutput,
                                    stage: str,
                                    dataloader_idx: int = 0,
                                    ):
        logit4aim_s, y4aim_s, logit4distill_s, xs4distill_s = zip(*student_outputs)
        logit4aim_t, y4aim_t, logit4distill_t, xs4distill_t = zip(*teacher_outputs)
        bs4distill = logit4distill_s[0].size(0)
        bs4aim = y4aim_s[0].size(0)

        # aim loss
        if (bs4aim == 0) or (self.lambda_aim_s == self.lambda_aim_t == 0):
            aim_loss_s = aim_loss_t = 0.

        else:
            if self.lambda_aim_s != 0:
                # loss and acc logging is performed within `self.compute_log_aim_acc_loss`
                aim_loss_s = self.compute_log_aim_acc_loss(logit4aim_s, y4aim_s,
                                                           group2erd_dict=self.train_acc,
                                                           md_group="students",
                                                           stage=stage, batch_size=bs4aim,
                                                           dataloader_idx=dataloader_idx)
            else:
                aim_loss_s = 0.
            if self.lambda_aim_t != 0:
                aim_loss_t = self.compute_log_aim_acc_loss(logit4aim_t, y4aim_t,
                                                           group2erd_dict=self.train_acc,
                                                           md_group="teachers",
                                                           stage=stage, batch_size=bs4aim,
                                                           dataloader_idx=dataloader_idx)
            else:
                aim_loss_t = 0

        # logit distill loss
        if (self.logit_distiller is None) or (self.lambda_kd_logit_s == self.lambda_kd_logit_t == 0):
            logit_loss_s = logit_loss_t = 0.
        else:
            logit_loss_s, logit_loss_t = self.logit_distiller(logit4distill_s, logit4distill_t)

            if self.log_distill:
                self.log4many(f"{stage}_loss_logit/students", logit_loss_s, bs4distill, dataloader_idx,
                              on_step=False, on_epoch=True)
                if logit_loss_t is not None:
                    self.log4many(f"{stage}_loss_logit/teachers", logit_loss_t, bs4distill, dataloader_idx,
                                  on_step=False, on_epoch=True)
            logit_loss_t = logit_loss_t if logit_loss_t is not None else 0.

        # feature distill loss
        if (self.feature_distiller is None) or (self.lambda_kd_fea_s == self.lambda_kd_fea_t == 0):
            fea_loss_s = fea_loss_t = 0.
        else:
            fea_loss_s, fea_loss_t = self.feature_distiller(xs4distill_s, xs4distill_t)

            if self.log_distill:
                self.log4many(f"{stage}_loss_fea/students", fea_loss_s, bs4distill, dataloader_idx,
                              on_step=False, on_epoch=True)

                if fea_loss_t is not None:
                    self.log4many(f"{stage}_loss_fea/teachers", fea_loss_t, bs4distill, dataloader_idx,
                                  on_step=False, on_epoch=True)
            fea_loss_t = fea_loss_t if fea_loss_t is not None else 0.

        # total loss

        loss_s = self.lambda_aim_s * aim_loss_s + self.lambda_kd_logit_s * logit_loss_s + self.lambda_kd_fea_s * fea_loss_s
        loss_t = self.lambda_aim_t * aim_loss_t + self.lambda_kd_logit_t * logit_loss_t + self.lambda_kd_fea_t * fea_loss_t
        if isinstance(loss_s, float) and loss_s == 0:
            loss_s = torch.zeros(self.num_students, device=self.device, dtype=self.dtype, requires_grad=False)
        if isinstance(loss_t, float) and loss_t == 0:
            loss_t = torch.zeros(self.num_teachers, device=self.device, dtype=self.dtype, requires_grad=False)

        self.log4many(f"{stage}_loss/students", loss_s, bs4aim, dataloader_idx, on_step=False, on_epoch=True)
        self.log4many(f"{stage}_loss/teachers", loss_t, bs4aim, dataloader_idx, on_step=False, on_epoch=True)
        return loss_s, loss_t

    def training_step(self, data, batch_idx, dataloader_idx: int = 0):
        student_outputs = self.step4many(self.students, data, split="train",
                                         run_func=self.student_forward4distill)
        teacher_outputs = self.step4many(self.teachers, data, split="train",
                                         run_func=self.teacher_forward4distill)

        loss_s, loss_t = self.compute_all_loss4all_models(student_outputs, teacher_outputs,
                                                          "train", dataloader_idx)

        opts = self.get_pl_optimizer_list()
        opts_s, opts_t = opts[:self.num_students], opts[self.num_students:]

        # student backward
        zero_grad_iter(opts_s)
        self.manual_backward(loss_s.sum())
        step_iter(opts_s)

        if self.log_describer("train", return_flag=True) and len(opts_t) > 0:
            zero_grad_iter(opts_t)
            self.manual_backward(loss_t.sum())
            step_iter(opts_t)

    @staticmethod
    def step_wo_distill(gnn_or_mlp, data, split):
        bs = data.get("batch_size", None)
        topo = data.get("adj_t", data.get("edge_index", None))
        if bs is None:  # SAGE or MLP with Full Graph
            split_mask = data.get(split + "_mask")
            logit = gnn_or_mlp.forward(data.x, topo)
            logit = logit[-1][split_mask] if gnn_or_mlp.tape else logit[split_mask]
            y = data.y[split_mask]
        else:  # SAGE or MLP with Neighbor Sampler
            logit = gnn_or_mlp.forward(data.x, topo)
            logit = logit[-1][:bs] if gnn_or_mlp.tape else logit[:bs]
            y = data.y[:bs]
        return logit, y

    def log4many(self, name_stem: str, value1d, batch_size: int = None, dataloader_idx: int = 0, *args, **kwargs):
        value_dict = {f"{name_stem}_{i}_{dataloader_idx}": value for i, value in enumerate(value1d)}
        self.log_dict(value_dict, batch_size=batch_size, add_dataloader_idx=False, *args, **kwargs)

    def compute_log_aim_acc_loss(self, logits, targets, group2erd_dict, md_group: str, stage: str,
                                 batch_size=None,
                                 dataloader_idx: int = 0, loader_merged_dict=None):
        bs = batch_size if batch_size is not None else targets[0].size(0)
        acc1d = []
        loss1d = []
        for i, (logit, target) in enumerate(zip(logits, targets)):
            aim_loss = self.aim_criterion(logit, target)
            loss1d.append(aim_loss)

            pred = logit.argmax(dim=-1)
            sep_metric = group2erd_dict[md_group][i][dataloader_idx]
            sep_metric(pred, target)
            acc1d.append(sep_metric)

            if loader_merged_dict is not None:
                # Note: Though the metric merged from multiple test dataloader is updated here,
                #       the USER is in charge of logging these merged metric out of this function.
                #       In this LightningModule, the logging of merged `self.test_all_acc` happens
                #       at the end of test epoch.
                merge_metric = loader_merged_dict[md_group][i][0]
                merge_metric(pred, target)

        self.log4many(f"{stage}_acc/{md_group}", acc1d, bs, dataloader_idx, on_step=False, on_epoch=True)
        self.log4many(f"{stage}_loss_aim/{md_group}", loss1d, bs, dataloader_idx, on_step=False, on_epoch=True)

        return torch.stack(loss1d)

    def eval_step(self, data, batch_idx, stage, dataloader_idx: int = 0, loader_merged_dict=None):
        student_outputs = self.step4many(self.students, data, split=stage, run_func=self.step_wo_distill)
        logit4aim, y4aim = zip(*student_outputs)
        bs4aim = y4aim[0].size(0)
        loss_s = self.compute_log_aim_acc_loss(logit4aim, y4aim,
                                               group2erd_dict=getattr(self, f"{stage}_acc"),
                                               md_group="students",
                                               stage=stage, batch_size=bs4aim,
                                               dataloader_idx=dataloader_idx,
                                               loader_merged_dict=loader_merged_dict)
        if self.log_describer(stage, return_flag=True):
            teacher_outputs = self.step4many(self.teachers, data, split=stage, run_func=self.step_wo_distill)
            logit4aim, y4aim = zip(*teacher_outputs)
            loss_t = self.compute_log_aim_acc_loss(logit4aim, y4aim,
                                                   group2erd_dict=getattr(self, f"{stage}_acc"),
                                                   md_group="teachers",
                                                   stage=stage, batch_size=bs4aim,
                                                   dataloader_idx=dataloader_idx,
                                                   loader_merged_dict=loader_merged_dict)
        else:
            loss_t = None
        return loss_s, loss_t

    def validation_step(self, data, batch_idx, dataloader_idx: int = 0):
        return self.eval_step(data, batch_idx, "val", dataloader_idx)

    def reset_val_acc(self, md_group):
        for md_idx, acc1d in enumerate(self.val_acc[md_group]):  # noqa
            for dl_idx, acc1 in enumerate(acc1d):
                acc_scalar = acc1.compute()  # get val accuracy from current epoch
                val_acc_best = self.val_acc_best[md_group][md_idx][dl_idx]
                val_acc_best.update(acc_scalar)
                self.log(f"val_acc_best/{md_group}_{md_idx}_{dl_idx}",
                         val_acc_best.compute(), add_dataloader_idx=False,
                         on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.reset_val_acc("students")
        if self.log_describer("val", return_flag=True):
            self.reset_val_acc("teachers")

    def test_step(self, data, batch_idx, dataloader_idx: int = 0):
        self.eval_step(data, batch_idx, "test", dataloader_idx, loader_merged_dict=self.test_all_acc)

    def reset_test_acc(self, md_group):
        all_acc = [list1d[0].compute() for list1d in self.test_all_acc[md_group]]
        self.log4many(f"test_all_acc/{md_group}", all_acc, dataloader_idx=0, on_step=False, on_epoch=True)
        reset_2d_metrics(self.test_all_acc[md_group])
        reset_2d_metrics(self.test_acc[md_group])

    def on_test_epoch_end(self) -> None:
        self.reset_test_acc("students")
        if self.log_describer("test", return_flag=True):
            self.reset_test_acc("teachers")

    def configure_optimizers(self):
        opts_student = [self.hparams.optimizer_s(params=student.parameters()) for student in self.students]
        if self.hparams.optimizer_t is None:
            opts = opts_student
            opts_teacher = None
        else:
            opts_teacher = [self.hparams.optimizer_t(params=teacher.parameters()) for teacher in
                            self.teachers]
            opts = [*opts_student, *opts_teacher]

        # parse schedulers
        schedulers_s = initialize_sch(self.hparams.scheduler_s, self.hparams.scheduler_s_kwargs, opts_student)

        schedulers_t = initialize_sch(self.hparams.scheduler_t, self.hparams.scheduler_t_kwargs, opts_teacher)

        schedulers = [*schedulers_s, *schedulers_t]
        return opts, schedulers

    @torch.no_grad()
    def inference(self):
        pass


def initialize_sch(sch, sch_kwargs, opts):
    sch_kwargs = sch_kwargs or {}
    if (sch is None) or (opts is None) or (len(opts) == 0):
        schedulers = []
    elif isinstance(sch, str):
        sch_cls = lr_scheduler_resolver(sch, only_return_cls=True)
        schedulers = [sch_cls(opt, **sch_kwargs) for opt in opts]
    else:
        schedulers = [sch(opt, **sch_kwargs) for opt in opts]

    return schedulers
