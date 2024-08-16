from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as func

DirectedReturn = Tuple[Tensor, None]
UndirectedReturn = Tuple[Tensor, Tensor]


class KnowledgeDistiller(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def wrap1d(out):
        """
        Wrap the following kinds to the last format
        - out of model0 --> [ out of model0]
        - [ out of model0,  out of model1 ,...]
        """
        if not isinstance(out, Sequence):
            return (out,)
        else:
            return out

    @staticmethod
    def wrap2d(out):
        """
        Wrap the following kinds to the last format
        - [xs_s0.h0, xs_s0.h1, ...] of model0  --> [ [xs_s0.h0, xs_s0.h1, ...] ]
        - [ [xs_s0.h0, xs_s0.h1, ...] of model0,  [xs_s0.h0, xs_s0.h1, ...] of model1 ,...]
        """
        if not isinstance(out[0], Sequence):
            return (out,)
        else:
            return out


class TwoGroupKnowledgeDistiller(KnowledgeDistiller):
    def __init__(self, single_direction: bool, intra: bool, exclude_self: bool = None):
        """
        Two groups can distill knowledge with an inter and/or intra-group way
        """
        super().__init__()
        self.single_direction = single_direction
        self.intra = intra
        if exclude_self is None:  # By default, exclude self when distilling within the same group, e.g., students
            self.exclude_self = True if intra else False
        else:
            self.exclude_self = exclude_self


class DirectionController:
    @staticmethod
    def compute_loss(out_s: Sequence[Tensor], out_t: Sequence[Tensor], loss_func, single_direction, intra,
                     loss_kwargs_s=None, loss_kwargs_t=None) -> Union[DirectedReturn, UndirectedReturn]:
        if single_direction:
            if intra:
                return DirectionController.intra_directed_loss(out_s, out_t, loss_func, loss_kwargs_s)
            else:
                return DirectionController.directed_loss(out_s, out_t, loss_func, loss_kwargs_s)
        else:
            if intra:
                return DirectionController.intra_undirected_loss(out_s, out_t, loss_func, loss_kwargs_s,
                                                                 loss_kwargs_t)
            else:
                return DirectionController.undirected_loss(out_s, out_t, loss_func, loss_kwargs_s,
                                                           loss_kwargs_t)

    @staticmethod
    def directed_loss(out_s: Sequence[Tensor], out_t: Sequence[Tensor], loss_func,
                      loss_kwargs_s) -> DirectedReturn:
        loss_s = torch.zeros(len(out_s), len(out_t), device=out_s[0].device)
        for i, ls in enumerate(out_s):
            for j, lt in enumerate(out_t):
                if (ls is not None) and (lt is not None):
                    loss_s[i, j] = loss_func(ls, lt, **loss_kwargs_s)
        return loss_s, None

    @staticmethod
    def undirected_loss(out_s: Sequence[Tensor], out_t: Sequence[Tensor], loss_func, loss_kwargs_s,
                        loss_kwargs_t) \
            -> UndirectedReturn:
        loss_s = torch.zeros(len(out_s), len(out_t), device=out_s[0].device)
        loss_t = torch.zeros(len(out_t), len(out_s), device=out_s[0].device)
        for i, ls in enumerate(out_s):
            for j, lt in enumerate(out_t):
                if (ls is not None) and (lt is not None):
                    loss_s[i, j] = loss_func(ls, lt, **loss_kwargs_s)
                    loss_t[j, i] = loss_func(lt, ls, **loss_kwargs_t)
        return loss_s, loss_t

    @staticmethod
    def intra_directed_loss(out_s: Sequence[Tensor], out_t: Sequence[Tensor], loss_func,
                            loss_kwargs_s) -> DirectedReturn:
        num_s = len(out_s)
        num_t = len(out_t)
        loss_s = torch.zeros(num_s, num_t + num_s, device=out_s[0].device)

        for i, ls in enumerate(out_s):  # teacher2student
            for j, lt in enumerate(out_t):
                if (ls is not None) and (lt is not None):
                    loss_s[i, j] = loss_func(ls, lt, **loss_kwargs_s)

        for i, ls in enumerate(out_s):  # intra-student
            for j, other_ls in enumerate(out_s):
                if i != j:
                    if (ls is not None) and (other_ls is not None):
                        loss_s[i, num_t + j] = loss_func(ls, other_ls, **loss_kwargs_s)

        return loss_s, None

    @staticmethod
    def intra_undirected_loss(out_s: Sequence[Tensor], out_t: Sequence[Tensor], loss_func,
                              loss_kwargs_s, loss_kwargs_t) -> UndirectedReturn:
        num_s = len(out_s)
        num_t = len(out_t)
        loss_s = torch.zeros(num_s, num_t + num_s, device=out_s[0].device)
        loss_t = torch.zeros(num_t, num_s + num_t, device=out_s[0].device)
        for i, ls in enumerate(out_s):  # student2teacher and teacher2student
            for j, lt in enumerate(out_t):
                if (ls is not None) and (lt is not None):
                    loss_s[i, j] = loss_func(ls, lt, **loss_kwargs_s)
                    loss_t[j, i] = loss_func(lt, ls, **loss_kwargs_t)

        for i, ls in enumerate(out_s):  # intra-student
            for j, other_ls in enumerate(out_s):
                if i != j and (ls is not None) and (other_ls is not None):
                    loss_s[i, num_t + j] = loss_func(ls, other_ls, **loss_kwargs_s)

        for i, lt in enumerate(out_t):  # intra-teacher
            for j, other_lt in enumerate(out_t):
                if i != j and (lt is not None) and (other_lt is not None):
                    loss_t[i, num_s + j] = loss_func(lt, other_lt, **loss_kwargs_t)
        return loss_s, loss_t

    @staticmethod
    def loss_mt2vec(loss2dmat=None, weight=None, exclude_self=False):
        if loss2dmat is None:
            return None

        assert loss2dmat.ndim == 2, f"loss matrix is expected as a 2D tensor, but got {loss2dmat.ndim} one"
        num_preachers = loss2dmat.size(-1)
        if exclude_self:
            num_preachers = num_preachers - 1

        if weight is None:
            return loss2dmat.sum(-1) / num_preachers

        if weight.ndim == 1:
            assert weight.size(0) == loss2dmat.size(-1), \
                f"The weight length doesn't mathc the number of preachers {loss2dmat.size(-1)}"
        elif weight.ndim == 2:
            assert weight.shape == loss2dmat.shape, \
                f"The weight shape {weight.shape} doesn't mathc the shape " \
                f"of true number of worshippers and preachers {loss2dmat.shape}"
        else:
            raise ValueError(f"`weight` is expected to have a shape of either 1d ({loss2dmat.size(-1)},) "
                             f"or 2d {loss2dmat.shape}, but got {weight.shape}")
        return torch.sum(loss2dmat * weight, dim=-1) / num_preachers


def soft_target_loss(logit_s, logit_t, temperature=1., detach_t=True):
    logit_t = logit_t.detach() if detach_t else logit_t
    kld = func.kl_div(
        func.log_softmax(logit_s / temperature, dim=-1),
        func.softmax(logit_t / temperature, dim=-1),
        reduction="batchmean",
    )
    loss = kld * temperature * temperature
    return loss


def mse_loss(logit_s, logit_t, detach_t=True):
    logit_t = logit_t.detach() if detach_t else logit_t
    return func.mse_loss(logit_s, logit_t, reduction="mean")


class SoftTarget(TwoGroupKnowledgeDistiller, DirectionController):
    """Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf."""

    def __init__(self, temperature: float = 4.0):
        super().__init__(single_direction=True, intra=False, exclude_self=False)
        self.temperature = temperature
        self.loss_kwargs_s = {"temperature": self.temperature}

    def forward(self, out_s, out_t, weight_s=None):
        loss_s, loss_t = self.compute_loss(self.wrap1d(out_s), self.wrap1d(out_t), soft_target_loss,
                                           single_direction=self.single_direction, intra=self.intra,
                                           loss_kwargs_s=self.loss_kwargs_s)

        return self.loss_mt2vec(loss_s, weight_s, self.exclude_self), self.loss_mt2vec(loss_t)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class DML(TwoGroupKnowledgeDistiller, DirectionController):
    """Deep Mutual Learning
    https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf."""

    def __init__(self, temperature: float = 4.0, temperature_t: float = 1.):
        super().__init__(single_direction=False, intra=True)
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.loss_kwargs_s = {"temperature": self.temperature}
        self.loss_kwargs_t = {"temperature": self.temperature_t}

    def forward(self, out_s, out_t, weight_s=None, weight_t=None):
        loss_s, loss_t = self.compute_loss(self.wrap1d(out_s), self.wrap1d(out_t), soft_target_loss,
                                           single_direction=self.single_direction, intra=self.intra,
                                           loss_kwargs_s=self.loss_kwargs_s,
                                           loss_kwargs_t=self.loss_kwargs_t)
        return (self.loss_mt2vec(loss_s, weight_s, self.exclude_self),
                self.loss_mt2vec(loss_t, weight_t, self.exclude_self))


class RSD(TwoGroupKnowledgeDistiller, DirectionController):
    pass


class MSELogits(TwoGroupKnowledgeDistiller, DirectionController):
    """Do Deep Nets Really Need to be Deep?

    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    """

    def __init__(self, single_direction=False, intra=True):
        super().__init__(single_direction, intra)

    def forward(self, out_s, out_t, weight_s=None, weight_t=None):
        loss_s, loss_t = self.compute_loss(self.wrap1d(out_s), self.wrap1d(out_t), mse_loss,
                                           single_direction=self.single_direction, intra=self.intra)
        return (self.loss_mt2vec(loss_s, weight_s, self.exclude_self),
                self.loss_mt2vec(loss_t, weight_t, self.exclude_self))


class TwoGroupLayerWiseDistiller(TwoGroupKnowledgeDistiller):
    def __init__(self, single_direction=True, intra=False, distill_layers=None, detach_t: bool = True):
        super().__init__(single_direction, intra)
        self.detach_t = detach_t
        if distill_layers is None:
            self.distill_layers = None
        elif isinstance(distill_layers, Sequence):
            self.distill_layers = distill_layers
        elif isinstance(distill_layers, int):
            self.distill_layers = [distill_layers]
        else:
            raise TypeError(f"{distill_layers} should be None/Sequence/int")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_distill_layers(self, xs_s1md, xs_t1md):
        assert not isinstance(xs_s1md[0], Sequence)
        assert not isinstance(xs_t1md[0], Sequence)
        if self.distill_layers is None:
            mimic_layers = range(min(len(xs_s1md), len(xs_t1md)))
        else:
            mimic_layers = self.distill_layers
        return mimic_layers


class FeatureMimic(TwoGroupLayerWiseDistiller, DirectionController):
    def __init__(self, single_direction=True, intra=False, distill_layers=None, detach_t: bool = True):
        super().__init__(single_direction, intra, distill_layers, detach_t)

    def forward(self, xs_s, xs_t, weight_s=None, weight_t=None):
        xs_s, xs_t = self.wrap2d(xs_s), self.wrap2d(xs_t)
        mimic_layers = self.get_distill_layers(xs_s[0], xs_t[0])
        loss_s, loss_t = self.compute_loss(xs_s, xs_t, feature_mimic_loss, self.single_direction, self.intra,
                                           loss_kwargs_s={"mimic_layers": mimic_layers,
                                                          "detach_t": self.detach_t},
                                           loss_kwargs_t={"mimic_layers": mimic_layers,
                                                          "detach_t": self.detach_t})
        return (self.loss_mt2vec(loss_s, weight_s, self.exclude_self),
                self.loss_mt2vec(loss_t, weight_t, self.exclude_self))


def feature_mimic_loss(xs_s: Sequence[Tensor], xs_t: Sequence[Tensor], mimic_layers, detach_t):
    loss = 0.
    for i in mimic_layers:
        loss = loss + func.mse_loss(xs_s[i], xs_t[i].detach() if detach_t else xs_t[i])
    return loss
