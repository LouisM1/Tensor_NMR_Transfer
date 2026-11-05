"""Export a legacy Experiment 04 model in the canonical e3nn convention."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import nmr_helpers
from graph_pes.utils.lammps import deploy_model
from nmr_helpers.openequivariance_compatibility import canonical_e3nn_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    output = args.output or source.with_name(source.stem + "-e3nn.pt")
    deployment = output.with_name(output.stem + "-lammps.pt")
    manifest = output.with_suffix(".json")
    if any(path.exists() for path in (output, deployment, manifest)):
        raise FileExistsError("Use new output names; original and previous exports are retained")
    torch.set_num_threads(2)
    model = torch.load(source, map_location="cpu", weights_only=False).eval()
    converted = canonical_e3nn_model(model).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    deploy_model(converted, deployment)
    record = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": str(output), "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "deployment": str(deployment),
        "convention": "e3nn 0.4.4; signs folded into final radial linear weights and bias",
        "original_files_retained": True,
        "optimizer": "Original Adam state is not converted; use a fresh optimizer for this export",
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
