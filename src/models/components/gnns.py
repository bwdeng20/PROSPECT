"""This module is adopted from torch_geometric.nn.models.basic_gnn."""
import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils.parametrize import remove_parametrizations
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCN2Conv,
    GCNConv,
    Linear,
    MessagePassing,
    SAGEConv,
)
from torch_geometric.typing import Adj

from .addin import spectral_norm
from .resolvers import activation_resolver, norm_resolver


class NewBasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (str or Callable, optional): The normalization operator to use.
            (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`). BN+ReLU,
            Sigmoid+BN
        act_kwargs (Dict[str,Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            normalization layer defined by :obj:`norm`.
            (default: :obj:`None`)
        tape (bool): If set to :obj:`True`, the list of output of all layers is
            returned. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        norm: Union[str, Callable, None] = None,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        tape: bool = False,
        spec_norm: Union[bool, str, float] = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.tape = tape
        self.sn, self.sn_arg = self.parse_spec_norm_arg(spec_norm)

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, self.sn, **kwargs)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, self.sn, **kwargs)
            )
        self.convs.append(
            self.init_conv(hidden_channels, out_channels, self.sn, **kwargs)
        )

        self.norms = None
        if norm is not None:
            norm_kwargs = norm_kwargs or {}
            norm_kwargs["in_channels"] = hidden_channels
            norm = norm_resolver(norm, **(norm_kwargs))
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))

        self.reset_parameters()
        self.has_sn = True if self.sn else False

    def init_conv(
        self, in_channels: int, out_channels: int, sn: bool = False, **kwargs
    ):
        """set a specific graph convolutional layer type."""
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

    def add_sn(self, scale):
        assert scale is not False
        if self.has_sn:
            return
        self._add_sn(scale)
        self.has_sn = True
        return self

    def _add_sn(self, scale):
        raise NotImplementedError

    def _rm_sn(self):
        raise NotImplementedError

    def rm_sn(self):
        if not self.has_sn:
            return
        self._rm_sn()
        self.has_sn = False
        return self

    @staticmethod
    def parse_spec_norm_arg(sn):
        if sn is False:
            return False, None
        elif sn is True:
            return True, 1.0
        elif isinstance(sn, (float, int)):
            return True, sn
        elif sn == "auto":
            return True, "auto"
        else:
            raise ValueError(f"{sn} is not either 'auto', float, or bool")

    def forward(
        self, x: Tensor, edge_index: Adj, *args, **kwargs
    ) -> Union[List[Tensor], Tensor]:
        # if self.spec_norm:
        #     means = x.mean(dim=1, keepdim=True)
        #     stds = x.std(dim=1, keepdim=True)
        #     x = (x - means) / stds
        # if self.spec_norm:
        #     x / x.norm(dim=1, keepdim=True)
        """"""
        xs: List[Tensor] = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.tape:
                xs.append(x)
            if i == self.num_layers - 1:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        return xs if self.tape else x

    def __repr__(self) -> str:
        extra = (
            f"[in={self.in_channels}, "
            f"hid={self.hidden_channels}, "
            f"out={self.out_channels}, #layer={self.num_layers}]"
        )
        return extra + super().__repr__()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        super().load_state_dict(state_dict, strict)

    def default_optimizer_scheme(self):
        return [{"params": self.parameters()}]


class GCN(NewBasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
             act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (str or Callable, optional): The normalization operator to use.
            (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`). BN+ReLU,
            Sigmoid+BN
        act_kwargs (Dict[str,Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            normalization layer defined by :obj:`norm`.
            (default: :obj:`None`)
        tape (bool): If set to :obj:`True`, the list of output of all layers is
            returned. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """

    def init_conv(
        self, in_channels: int, out_channels: int, sn: bool = False, **kwargs
    ) -> MessagePassing:
        conv = GCNConv(in_channels, out_channels, **kwargs)
        if sn:
            conv.lin = spectral_norm(conv.lin, scale=self.sn_arg)
        return conv

    def _add_sn(self, scale):
        for conv in self.convs:
            conv.lin = spectral_norm(
                conv.lin, "weight", scale=self.parse_spec_norm_arg(scale)[-1]
            )

    def _rm_sn(self):
        for conv in self.convs:
            remove_parametrizations(conv.lin, "weight")


class GraphSAGE(NewBasicGNN):
    """
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, str, optional): The normalization operator to use.
            (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            normalization layer defined by :obj:`norm`.
            (default: :obj:`None`)
        tape (bool): If set to :obj:`True`, the list of output of all layers is
            returned. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
            1. aggr (string, optional): The aggregation scheme to use
                (:obj:`"mean"`, :obj:`"max"`, :obj:`"lstm"`).
                (default: :obj:`"add"`)
            2.  normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\\ell_2`-normalized, *i.e.*,
            :math:`\frac{\\mathbf{x}^{\\prime}_i}
            {\\| \\mathbf{x}^{\\prime}_i\\ |_2}`.
            (default: :obj:`False`)
            ...
    """

    def init_conv(
        self, in_channels: int, out_channels: int, sn: bool = False, **kwargs
    ) -> MessagePassing:
        conv = SAGEConv(in_channels, out_channels, **kwargs)
        if sn:
            conv.lin_l = spectral_norm(conv.lin_l)
            if conv.root_weight:
                conv.lin_r = spectral_norm(conv.lin_r, scale=self.sn_arg)
        return conv

    def _add_sn(self, scale):
        s = self.parse_spec_norm_arg(scale)[-1]
        for conv in self.convs:
            conv.lin_l = spectral_norm(conv.lin_l, scale=s)
            if conv.root_weight:
                conv.lin_r = spectral_norm(conv.lin_r, scale=s)

    def _rm_sn(self):
        for conv in self.convs:
            remove_parametrizations(conv.lin_l, "weight")
            conv.lin_l = spectral_norm(conv.lin_l)
            if conv.root_weight:
                remove_parametrizations(conv.lin_r, "weight")


class GAT(NewBasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (str or Callable, optional): The normalization operator to use.
            (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`). BN+ReLU,
            Sigmoid+BN
        act_kwargs (Dict[str,Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            normalization layer defined by :obj:`norm`.
            (default: :obj:`None`)
        tape (bool): If set to :obj:`True`, the list of output of all layers is
            returned. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """

    def init_conv(
        self, in_channels: int, out_channels: int, sn: bool = False, **kwargs
    ) -> MessagePassing:

        v2 = kwargs.pop("v2", False)
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, "_is_conv_to_out", False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GATConv' (got '{out_channels}') is divisible "
                f"by the number of heads (got '{heads}')"
            )

        if concat:
            out_channels = out_channels // heads

        conv_layer = GATConv if not v2 else GATv2Conv
        conv = conv_layer(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            dropout=self.dropout,
            **kwargs,
        )
        if sn:
            if isinstance(in_channels, int):  # only support GATv1
                conv.lin_src = spectral_norm(conv.lin_src, scale=self.sn_arg)
            else:
                conv.lin_src = spectral_norm(conv.lin_src, scale=self.sn_arg)
                conv.lin_dst = spectral_norm(conv.lin_dst, scale=self.sn_arg)
        return conv


class GCN2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        norm: Union[str, Callable, None] = None,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        tape: bool = False,
        spec_norm: bool = False,
        alpha: float = 0.1,
        theta: float = 0.5,
        shared_weights=True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers  # the GCN2conv layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.tape = tape
        self.sn, self.sn_arg = NewBasicGNN.parse_spec_norm_arg(spec_norm)

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.lins = torch.nn.ModuleList()
        lin_in = Linear(in_channels, hidden_channels, bias=True)
        lin_out = Linear(hidden_channels, self.out_channels, bias=True)
        self.lins.append(spectral_norm(lin_in) if self.sn else lin_in)
        self.lins.append(spectral_norm(lin_out) if self.sn else lin_out)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                self.init_conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights,
                    self.sn,
                    **kwargs,
                )
            )

        self.norms = None
        if norm is not None:
            # GCN2 lin_in->(GCN2Conv->bn)->(GCN2Conv->bn)->lin_out
            norm_kwargs = norm_kwargs or {}
            norm_kwargs["in_channels"] = hidden_channels
            norm = norm_resolver(norm, **(norm_kwargs))
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.norms.append(copy.deepcopy(norm))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def init_conv(
        self,
        hidden_channels,
        alpha,
        theta,
        layer,
        shared_weights,
        sn: bool = False,
        **kwargs,
    ):
        conv = GCN2Conv(hidden_channels, alpha, theta, layer, shared_weights, **kwargs)
        if sn:
            conv = spectral_norm(conv, name="weight1", scale=self.sn_arg)
        return conv

    def forward(
        self, x: Tensor, edge_index: Adj, *args, **kwargs
    ) -> Union[List[Tensor], Tensor]:
        xs: List[Tensor] = [x]
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.lins[0](x)
        if self.tape:
            xs.append(x)
        x = x0 = self.act(x)
        for i in range(self.num_layers):
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = self.convs[i](x, x0, edge_index, *args, **kwargs)
            if self.tape:
                xs.append(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.lins[1](x)
        if self.tape:
            xs.append(x)
        return xs if self.tape else x

    def default_optimizer_scheme(self):
        # you need to tune these weight decay hyperparameters
        param_dict_list = [
            {"params": self.convs.parameters(), "weight_decay": 0.0},
            {"params": self.lins.parameters(), "weight_decay": 0.0},
        ]
        return param_dict_list


# if __name__ == "__main__":
#     gcn = GCN(2, 4, 2, 1, norm="batchnorm", act="leakyrelu")
#     gat = GAT(2, 4, 2, 1, norm="layernorm", act="selu")
#     sage = GraphSAGE(2, 4, 2, 1, norm="batchnorm", aggr="add")
#     print(gcn)
#     print(gat)
#     print(sage)
