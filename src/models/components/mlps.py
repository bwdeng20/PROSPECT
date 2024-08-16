import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn.dense import Linear

from src.models.components.resolvers import activation_resolver, norm_resolver


class MLP(torch.nn.Module):
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
            :class:`torch_geometric.nn.dense.Linear.
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
            bias: bool = True,
            tape: bool = False,
            spec_norm: bool = False,
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
        self.spec_norm = spec_norm

        self.out_channels = (
            out_channels if out_channels is not None else hidden_channels
        )

        self.lins = torch.nn.ModuleList()
        self.lins.append(
            self.get_linear(in_channels, hidden_channels, bias=bias, **kwargs)
        )
        for _ in range(num_layers - 2):
            self.lins.append(
                self.get_linear(hidden_channels, hidden_channels, bias=bias, **kwargs)
            )
        self.lins.append(
            self.get_linear(hidden_channels, out_channels, bias=bias, **kwargs)
        )

        self.norms = None
        if norm is not None:
            norm_kwargs = norm_kwargs or {}
            norm_kwargs["in_channels"] = hidden_channels
            norm = norm_resolver(norm, **norm_kwargs)
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))

        self.reset_parameters()

    def get_linear(self, hidden_channels, out_channels, bias, **kwargs):
        lin = Linear(hidden_channels, out_channels, bias=bias, **kwargs)
        if not self.spec_norm:
            return lin
        else:
            return spectral_norm(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

    def forward(self, x: Tensor, *args, **kwargs) -> Union[List[Tensor], Tensor]:
        """Compatible with `Data` from PyG dataloader."""
        xs: List[Tensor] = [x]
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i == self.num_layers - 1:
                if self.tape:
                    xs.append(x)
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            if self.tape:
                xs.append(x)
        return xs if self.tape else x


class JKMLP(MLP):
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
            bias: bool = True,
            tape: bool = False,
            spec_norm: bool = False,
            **kwargs,
    ):
        super(JKMLP, self).__init__(in_channels, hidden_channels, num_layers, out_channels,
                                    dropout, act, norm, act_first, act_kwargs, norm_kwargs, bias, tape,
                                    spec_norm, **kwargs)

        self.jk_projector = Linear(hidden_channels, out_channels)
        self.jk_projector.reset_parameters()

    def forward(self, x: Tensor, *args, **kwargs) -> Union[List[Tensor], Tensor]:
        """ Compatible with `Data` from PyG dataloader."""
        assert not self.tape, "Don't tape JKMLP"
        x_res: List[Tensor] = list()
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i == self.num_layers - 1:
                x = sum(x_res) + x
                break

            x_res.append(self.jk_projector(x))
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        return x


class ResMLP(MLP):
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
            bias: bool = True,
            **kwargs,
    ):
        super(ResMLP, self).__init__(in_channels, hidden_channels, num_layers, out_channels,
                                     dropout, act, norm, act_first, act_kwargs, norm_kwargs, bias,
                                     tape=False, **kwargs)

    def forward(self, x: Tensor, *args, **kwargs) -> Union[List[Tensor], Tensor]:
        x = self.lins[0](x)
        for i, lin in enumerate(self.lins[1:-1]):
            identity = x
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i + 1](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = x + identity
        x = self.lins[-1](x)
        return x


if __name__ == "__main__":
    mlp = MLP(2, 4, 2, 1, norm="batchnorm", act="selu")
    print(mlp)
    input = torch.rand(10, 2)
    print(mlp(input).shape)

    mlp = ResMLP(2, 4, 2, 1, norm="batchnorm", act="selu")
    print(mlp)
    input = torch.rand(10, 2)
    print(mlp(input).shape)
