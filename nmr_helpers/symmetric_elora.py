"""Path-wise ELoRA for the transferred 0e+2e symmetric NMR model."""

import math

import torch
from e3nn import o3

from .elora import PathELoRA
from .symmetric_shielding import SymmetricTensorNequIP


def add_elora(model, rank=4):
    """Train 12 rank-r path-wise adapters plus the fresh tensor head/scaler.

    Original embedding, radial-network and interaction weights remain frozen.
    Adapter B factors start at zero, so the initial predictions match full
    transfer with the same fresh readout. CPU RNG state is preserved to keep
    the head and subsequent data-order seed comparable across methods.
    """
    if not isinstance(model, SymmetricTensorNequIP):
        raise TypeError("Expected a direct SymmetricTensorNequIP destination")
    if len(model.layers) != 4 or o3.Irreps(model.target_tensor_irreps) != o3.Irreps("0e + 2e"):
        raise ValueError("Expected the four-layer symmetric 0e+2e model")
    if not hasattr(model, "_symmetric_transfer_source_sha256"):
        raise ValueError("Transfer the E/F backbone before adding symmetric ELoRA")
    if bool(model._has_been_pre_fit) or hasattr(model, "_symmetric_elora_rank"):
        raise ValueError("Add ELoRA once, before pre-fitting the fresh NMR scaler")
    if type(rank) is not int or rank < 1:
        raise ValueError("ELoRA rank must be a positive integer")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Construct ELoRA on CPU before the trainer moves it to CUDA")

    operations = []
    for layer in model.layers:
        if layer.self_interaction is None or not hasattr(layer.self_interaction, "tensor_product"):
            raise ValueError("Expected tensor-product self-interactions in every layer")
        operations.extend([(layer, "pre_message_linear"), (layer, "post_message_linear"),
                           (layer.self_interaction, "tensor_product")])
    # Fail before freezing/replacing modules if an operator/rank is incompatible.
    for parent, name in operations:
        base = getattr(parent, name)
        if not isinstance(base, (o3.Linear, o3.FullyConnectedTensorProduct)):
            raise TypeError(f"Unsupported ELoRA operator {type(base).__name__}")
        if not base.internal_weights or not base.shared_weights:
            raise ValueError("ELoRA requires internally stored, shared channel weights")
        if isinstance(base, o3.Linear) and base.bias_numel:
            raise ValueError("Expected bias-free equivariant linears")
        for weight in base.weight_views():
            if rank > min(math.prod(weight.shape[:-1]), weight.shape[-1]):
                raise ValueError(f"Rank {rank} exceeds channel block {tuple(weight.shape)}")

    model.requires_grad_(False)
    model.tensor_readout.requires_grad_(True)
    model.scaler.requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        for parent, name in operations:
            setattr(parent, name, PathELoRA(getattr(parent, name), rank))
    model._symmetric_elora_rank = rank
    model._symmetric_transfer_backbone_frozen = True
    model.train()
    print(f"Added rank-{rank} ELoRA to {len(operations)} operators across four symmetric-tensor layers.", flush=True)
    print("Trainable: path-wise A/B factors and fresh tensor readout/scales/offsets.", flush=True)
    return model
