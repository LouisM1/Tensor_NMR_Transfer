"""Convert OEQ-backed E/F models to ordinary e3nn for inference and transfer."""

from copy import deepcopy

import torch
from e3nn import o3
from graph_pes.utils.nn import HaddamardProduct, MLP

from .openequivariance_tp import OpenEquivarianceMessageTensorProduct


def canonical_e3nn_model(model):
    """Preserve the learned function while removing every OEQ wrapper.

    Fold each fixed convention sign into the final radial MLP's output rows
    and bias, then use plain e3nn tensor products. New canonical models have
    unit signs and require no parameter change. Original files are untouched.
    Use a fresh optimizer for subsequent training of a converted model: this
    function does not convert saved Adam moments.
    """
    result = deepcopy(model)
    for parent in list(result.modules()):
        if not hasattr(parent, "layers"):
            continue
        for layer in parent.layers:
            product = getattr(layer, "message_tensor_product", None)
            if not isinstance(product, OpenEquivarianceMessageTensorProduct):
                continue
            if layer.weight_generator._forward_hooks:
                raise ValueError("Remove old analysis sign hooks before converting the model")
            generator = layer.weight_generator
            if not (isinstance(generator, HaddamardProduct) and len(generator.components) == 2
                    and isinstance(generator.components[0], torch.nn.Sequential)
                    and len(generator.components[0]) == 2
                    and isinstance(generator.components[0][1], MLP)):
                raise ValueError("Unexpected radial-network layout; no conversion was performed")
            mlp = generator.components[0][1]
            final = mlp.linear_layers[-1]
            if mlp.activate_last or final.out_features != product.weight_numel:
                raise ValueError("Expected a linear final radial output with one row per TP weight")
            signs = product._oeq_weight_to_e3nn
            with torch.no_grad():
                final.weight.mul_(signs.to(final.weight).unsqueeze(1))
                if final.bias is not None:
                    final.bias.mul_(signs.to(final.bias))
            instructions = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight)
                            for i in product.instructions]
            plain = o3.TensorProduct(product.irreps_in1, product.irreps_in2,
                                     product.irreps_out, instructions,
                                     internal_weights=False, shared_weights=False,
                                     irrep_normalization="component", path_normalization="element")
            plain = plain.to(device=product.weight.device, dtype=product.weight.dtype)
            state = {key: value for key, value in product.state_dict().items()
                     if key != "_oeq_weight_to_e3nn"}
            plain.load_state_dict(state, strict=True)
            layer.message_tensor_product = plain
    if any(isinstance(module, OpenEquivarianceMessageTensorProduct) for module in result.modules()):
        raise ValueError("An OEQ tensor product outside the supported interaction layers remains")
    return result
