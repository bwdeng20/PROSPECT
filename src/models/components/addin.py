from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.utils import parametrize


class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        scale: Union[float, str] = 1.0,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if isinstance(scale, str) and scale == "auto":
            self.is_scale_fixed = False
            self.scale = torch.nn.Parameter(torch.Tensor([0]))
        elif isinstance(scale, (float, int)):
            scale = float(scale)
            assert 0 <= scale <= 1
            self.is_scale_fixed = True
            self.scale = scale
        else:
            raise TypeError(
                f"{scale} is not a valid  arg, 'auto' for learnable,"
                f"numbers for fixed scale factor"
            )
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything
            # (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
        weight_mat = self._reshape_weight_to_matrix(weight)
        h, w = weight_mat.size()

        u = weight_mat.new_empty(h).normal_(0, 1)
        v = weight_mat.new_empty(w).normal_(0, 1)
        self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
        self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

        # Start with u, v initialized to some reasonable values by performing
        # number of iterations of the power method
        self._power_method(weight_mat, 15)

    def get_scaled_value(self, v1, v2):
        if self.is_scale_fixed:
            scale = self.scale
        else:
            scale = torch.sigmoid(self.scale)

        return (v2 - v1) * scale + v1

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(
                torch.mv(weight_mat, self._v),
                # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._u,
            )  # type: ignore[has-type]
            self._v = F.normalize(
                torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v
            )  # type: ignore[has-type]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            scaled_sigma = self.get_scaled_value(1.0, sigma)
            return weight / scaled_sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    scale: float = 1.0,
) -> Module:
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module, name, _SpectralNorm(weight, n_power_iterations, dim, eps, scale)
    )
    return module
