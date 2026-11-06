"""Copy the corrected E/F backbone into a fresh symmetric shielding model."""

from pathlib import Path
import hashlib

import torch
from e3nn import o3
from graph_pes.models import NequIP

from .symmetric_shielding import SymmetricTensorNequIP


BACKBONE_PARTS = ("Z_embedding", "initial_node_embedding", "edge_embedding", "layers")


def from_energy_force_model(model, checkpoint, freeze_backbone=False, expected_sha256=None):
    """Transfer embeddings and all four layers; initialise the NMR head afresh.

    The source is the trusted, corrected Experiment 04 model-e3nn.pt export.
    E/F readout, energy scales/offsets, optimizer state and pre-fit flags are
    excluded. The normal training pipeline fits the fresh IrrepAffine only
    on the selected NMR fitting structures. All transfer validation precedes
    state copying, and no parameters are shared with the loaded source.
    """
    if not isinstance(model, SymmetricTensorNequIP):
        raise TypeError("Expected a direct SymmetricTensorNequIP destination")
    if o3.Irreps(model.target_tensor_irreps) != o3.Irreps("0e + 2e"):
        raise ValueError("Destination must predict exactly 0e + 2e")
    if bool(model._has_been_pre_fit):
        raise ValueError("Destination must be a fresh, un-prefitted NMR model")
    if hasattr(model, "_symmetric_transfer_source_sha256"):
        raise ValueError("This destination already received a transferred backbone")
    if type(freeze_backbone) is not bool:
        raise TypeError("freeze_backbone must be a boolean")

    path = Path(checkpoint)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}")
    source = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(source, "models") or "nequip" not in source.models:
        raise TypeError("Expected an E/F AdditionModel with a named nequip backbone")
    old = source.models["nequip"]
    if not isinstance(old, NequIP):
        raise TypeError("Expected a NequIP E/F backbone")
    if len(old.layers) != 4 or len(model.layers) != 4:
        raise ValueError("Experiments 05/06 require four interaction layers")
    if any(type(layer.message_tensor_product) is not o3.TensorProduct for layer in old.layers):
        raise ValueError("Use the corrected model-e3nn.pt export, not the raw OEQ checkpoint")
    if not torch.equal(old.cutoff.cpu(), model.cutoff.cpu()):
        raise ValueError("Source and destination cutoffs differ")
    if not torch.equal(old.Z_embedding.Z_to_idx.cpu(), model.Z_embedding.Z_to_idx.cpu()):
        raise ValueError("Source and destination element ordering differs")
    if repr(old.edge_embedding) != repr(model.edge_embedding):
        raise ValueError("Source and destination edge representations differ")
    for index, (before, after) in enumerate(zip(old.layers, model.layers)):
        if before.irreps_in != after.irreps_in or before.irreps_out != after.irreps_out:
            raise ValueError(f"Interaction-layer irreps differ at layer {index}")
        if repr(before.aggregation) != repr(after.aggregation):
            raise ValueError(f"Neighbour aggregation differs at layer {index}")
        if before.message_tensor_product.instructions != after.message_tensor_product.instructions:
            raise ValueError(f"Message tensor-product paths differ at layer {index}")

    # Check every part before changing any destination state.
    for name in BACKBONE_PARTS:
        before, after = getattr(old, name).state_dict(), getattr(model, name).state_dict()
        if before.keys() != after.keys():
            raise ValueError(f"Backbone state keys differ in {name}")
        for key, value in before.items():
            if not torch.is_tensor(value) or value.shape != after[key].shape:
                raise ValueError(f"Backbone state shape/type differs in {name}.{key}")
    for name in BACKBONE_PARTS:
        getattr(model, name).load_state_dict(getattr(old, name).state_dict(), strict=True)
    model.requires_grad_(True)
    if freeze_backbone:
        for name in BACKBONE_PARTS:
            getattr(model, name).requires_grad_(False)
    # Plain Python provenance is pickled with model.pt; it is not a fitted flag.
    model._symmetric_transfer_source_sha256 = digest
    model._symmetric_transfer_backbone_frozen = freeze_backbone
    model.train()
    print(f"Transferred four-layer 0e+2e backbone from {path}", flush=True)
    print(f"Source SHA256: {digest}; frozen backbone: {freeze_backbone}", flush=True)
    print("Fresh symmetric NMR readout/scales/offsets; no E/F head or optimizer state transferred.", flush=True)
    return model
