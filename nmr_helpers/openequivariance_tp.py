"""CUDA backend for the externally weighted NequIP message tensor products."""

from functools import lru_cache

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode


@lru_cache(maxsize=16)
def _cuda_kernel(device, irreps_in1, irreps_in2, irreps_out, instructions):
    try:
        import openequivariance as oeq
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 04 requires OpenEquivariance in the ARC environment."
        ) from exc

    with torch.cuda.device(device):
        problem = oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            list(instructions),
            shared_weights=False,
            internal_weights=False,
            irrep_normalization="component",
            path_normalization="element",
        )
        return oeq.TensorProduct(problem, torch_op=True)


@compile_mode("script")
class OpenEquivarianceMessageTensorProduct(o3.TensorProduct):
    """Keep the e3nn state layout and dispatch eager CUDA calls to OEQ."""

    def __init__(self, reference: o3.TensorProduct):
        if reference.internal_weights or reference.shared_weights:
            raise ValueError("Expected externally supplied, per-edge weights")
        if any(mul != 1 for mul, _ in reference.irreps_in2):
            raise ValueError("Expected edge irreps with multiplicity one")
        if any(i.connection_mode != "uvu" or not i.has_weight for i in reference.instructions):
            raise ValueError("Expected weighted uvu instructions")

        # graph-pes uses unit input path weights. e3nn's stored instructions
        # contain normalised weights, so rebuild from the raw five fields.
        instructions = tuple(
            (i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight)
            for i in reference.instructions
        )
        super().__init__(
            reference.irreps_in1,
            reference.irreps_in2,
            reference.irreps_out,
            list(instructions),
            internal_weights=False,
            shared_weights=False,
            irrep_normalization="component",
            path_normalization="element",
        )
        if self.instructions != reference.instructions or self.weight_numel != reference.weight_numel:
            raise ValueError("Message tensor product normalisation or weight layout changed")
        self.load_state_dict(reference.state_dict(), strict=True)
        self._oeq_spec = (
            str(reference.irreps_in1),
            str(reference.irreps_in2),
            str(reference.irreps_out),
            instructions,
        )

    @torch.jit.unused
    def _forward_cuda(self, x, y, weights):
        if x.dtype != torch.float32 or y.dtype != torch.float32 or weights.dtype != torch.float32:
            raise TypeError("Experiment 04 requires float32 message tensor products")
        # Keep compiled CUDA objects out of the serialised model and state dict.
        kernel = _cuda_kernel(x.device, *self._oeq_spec)
        return kernel(x.contiguous(), y.contiguous(), weights.contiguous())

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and x.is_cuda and x.shape[0] > 0:
            return self._forward_cuda(x, y, weight)
        return self._compiled_main_left_right(x, y, weight)
