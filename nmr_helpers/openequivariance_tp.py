"""Device-independent tensor products, including legacy Experiment 04 weights."""

from copy import copy
from functools import lru_cache
from math import prod
import warnings

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode


_LEGACY_FLIPPED_COUPLINGS = {(1, 2, 2), (2, 1, 2), (2, 2, 1)}


def _legacy_weight_to_e3nn(product):
    """OEQ 0.3.0 -> e3nn 0.4.4, independently audited from their CG tables."""
    if e3nn.__version__ != "0.4.4":
        raise RuntimeError(
            "Convert untagged Experiment 04 weights using e3nn 0.4.4 first; "
            "their original coefficient convention cannot be assumed in another version."
        )
    if any(ir.l > 2 for irreps in (product.irreps_in1, product.irreps_in2,
                                  product.irreps_out) for _, ir in irreps):
        raise RuntimeError("Legacy Experiment 04 conversion was audited only through l=2")
    signs = product.weight.new_ones(product.weight_numel)
    offset = 0
    for instruction in product.instructions:
        size = prod(instruction.path_shape)
        signature = (product.irreps_in1[instruction.i_in1].ir.l,
                     product.irreps_in2[instruction.i_in2].ir.l,
                     product.irreps_out[instruction.i_out].ir.l)
        if signature in _LEGACY_FLIPPED_COUPLINGS:
            signs[offset:offset + size] = -1
        offset += size
    if offset != product.weight_numel:
        raise RuntimeError("Unexpected legacy tensor-product weight layout")
    return signs


def _warn_legacy():
    warnings.warn(
        "Recovered the original OpenEquivariance 0.3.0 convention for an untagged "
        "Experiment 04 checkpoint. CPU/TorchScript inference now preserves its GPU "
        "function. Do not also apply the former notebook sign hooks.",
        UserWarning, stacklevel=3,
    )


@lru_cache(maxsize=16)
def _cuda_kernel(device, irreps_in1, irreps_in2, irreps_out, instructions, expected_paths):
    try:
        import openequivariance as oeq
        from openequivariance.implementations.e3nn_lite import wigner_3j
    except ImportError as exc:
        raise RuntimeError("Experiment 04 requires OpenEquivariance; use the pinned 0.3.0 environment") from exc

    with torch.cuda.device(device):
        problem = oeq.TPProblem(
            irreps_in1, irreps_in2, irreps_out, list(instructions),
            shared_weights=False, internal_weights=False,
            irrep_normalization="component", path_normalization="element",
        )
        actual_paths = tuple((i.path_weight, i.path_shape) for i in problem.instructions)
        if len(actual_paths) != len(expected_paths) or any(
            shape != expected_shape or abs(weight - expected_weight) > 1e-12
            for (weight, shape), (expected_weight, expected_shape)
            in zip(actual_paths, expected_paths)
        ):
            raise RuntimeError("OpenEquivariance path normalisation/layout differs from e3nn")

        # Derive signs from installed coefficient tables, never from model targets.
        a, b, c = map(o3.Irreps, (irreps_in1, irreps_in2, irreps_out))
        signs = []
        for instruction in problem.instructions:
            signature = (a[instruction.i_in1].ir.l, b[instruction.i_in2].ir.l,
                         c[instruction.i_out].ir.l)
            cpu = o3.wigner_3j(*signature, dtype=torch.float64, device="cpu")
            cuda = torch.as_tensor(wigner_3j(*signature), dtype=torch.float64)
            if torch.allclose(cpu, cuda, rtol=0, atol=1e-12):
                sign = 1.0
            elif torch.allclose(cpu, -cuda, rtol=0, atol=1e-12):
                sign = -1.0
            else:
                raise RuntimeError(f"Unsupported coefficient-basis difference for {signature}")
            signs.extend([sign] * prod(instruction.path_shape))
        return oeq.TensorProduct(problem, torch_op=True), torch.tensor(signs, device=device, dtype=torch.float32)


@compile_mode("script")
class OpenEquivarianceMessageTensorProduct(o3.TensorProduct):
    """New weights use e3nn conventions; old OEQ weights carry an explicit map.

    The persistent sign buffer defines what externally supplied weights mean.
    It is included in state_dict checkpoints and TorchScript exports. Untagged
    old wrapper objects/states are migrated as the original Experiment 04 OEQ
    convention. Use load_e3nn_state_dict for untagged *e3nn* source states.
    """

    def __init__(self, reference: o3.TensorProduct):
        if reference.internal_weights or reference.shared_weights:
            raise ValueError("Expected externally supplied, per-edge weights")
        if any(mul != 1 for mul, _ in reference.irreps_in2):
            raise ValueError("Expected edge irreps with multiplicity one")
        if any(i.connection_mode != "uvu" or not i.has_weight for i in reference.instructions):
            raise ValueError("Expected weighted uvu instructions")
        instructions = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight)
                        for i in reference.instructions]
        super().__init__(reference.irreps_in1, reference.irreps_in2,
                         reference.irreps_out, instructions,
                         internal_weights=False, shared_weights=False,
                         irrep_normalization="component", path_normalization="element")
        if self.instructions != reference.instructions or self.weight_numel != reference.weight_numel:
            raise ValueError("Message tensor product normalisation or weight layout changed")
        mapping = getattr(reference, "_oeq_weight_to_e3nn", self.weight.new_ones(self.weight_numel))
        self.register_buffer("_oeq_weight_to_e3nn", mapping.detach().clone())
        state = copy(reference.state_dict())
        state["_oeq_weight_to_e3nn"] = self._oeq_weight_to_e3nn
        self.load_state_dict(state, strict=True)
        self._refresh_spec()

    def _refresh_spec(self):
        if e3nn.__version__ != "0.4.4":
            raise RuntimeError("Experiment 04's stored e3nn convention requires e3nn 0.4.4; convert before upgrading")
        self._oeq_spec = (
            str(self.irreps_in1), str(self.irreps_in2), str(self.irreps_out),
            tuple((i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight)
                  for i in self.instructions),
        )
        self._oeq_expected_paths = tuple((i.path_weight, i.path_shape) for i in self.instructions)
        mapping = self._oeq_weight_to_e3nn
        if mapping.shape != (self.weight_numel,) or not bool(((mapping == 1) | (mapping == -1)).all()):
            raise RuntimeError("Invalid tensor-product convention metadata")
        self._oeq_cpu_needs_conversion = bool((mapping != 1).any())

    def __setstate__(self, state):
        super().__setstate__(state)
        if "_oeq_weight_to_e3nn" not in self._buffers:
            self.register_buffer("_oeq_weight_to_e3nn", _legacy_weight_to_e3nn(self))
            _warn_legacy()
        self._refresh_spec()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "_oeq_weight_to_e3nn"
        if key not in state_dict:
            state_dict[key] = _legacy_weight_to_e3nn(self)
            _warn_legacy()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._refresh_spec()

    @torch.jit.unused
    def _forward_cuda(self, x, y, weights):
        if x.dtype != torch.float32 or y.dtype != torch.float32 or weights.dtype != torch.float32:
            raise TypeError("Experiment 04 requires float32 message tensor products")
        kernel, e3nn_to_cuda = _cuda_kernel(x.device, *self._oeq_spec, self._oeq_expected_paths)
        mapped = weights * (self._oeq_weight_to_e3nn * e3nn_to_cuda)
        return kernel(x.contiguous(), y.contiguous(), mapped.contiguous())

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and x.is_cuda and x.shape[0] > 0:
            return self._forward_cuda(x, y, weight)
        if self._oeq_cpu_needs_conversion:
            weight = weight * self._oeq_weight_to_e3nn
        return self._compiled_main_left_right(x, y, weight)


def load_e3nn_state_dict(model, state_dict, strict=True):
    """Explicitly load canonical e3nn weights into a model with OEQ wrappers."""
    state = copy(state_dict)
    for name, module in model.named_modules():
        if isinstance(module, OpenEquivarianceMessageTensorProduct):
            key = (name + "." if name else "") + "_oeq_weight_to_e3nn"
            if key in state and not bool((state[key] == 1).all()):
                raise ValueError("Source state is tagged with a non-e3nn convention")
            state[key] = torch.ones_like(module._oeq_weight_to_e3nn)
    return model.load_state_dict(state, strict=strict)
