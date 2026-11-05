"""Check legacy loading, portable exports and cached Experiment 04 predictions."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import numpy as np
import torch
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.lammps import as_lammps_data, deploy_model
import nmr_helpers
from nmr_helpers.openequivariance_tp import OpenEquivarianceMessageTensorProduct
from nmr_helpers.openequivariance_compatibility import canonical_e3nn_model


def compare(actual,expected):
    torch.testing.assert_close(actual['energy'],expected['energy'],rtol=0,atol=0.008)
    torch.testing.assert_close(actual['forces'],expected['forces'],rtol=5e-5,atol=1e-5)
    return {'energy_max_abs_eV':float((actual['energy']-expected['energy']).abs().max()),
            'force_max_abs_eV_per_A':float((actual['forces']-expected['forces']).abs().max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688'))
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    run=args.run
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original_hash=digest(run/'model.pt')
    source=torch.load(run/'model.pt',map_location='cpu',weights_only=False).eval()
    source.requires_grad_(False)
    mapping=[int((layer.message_tensor_product._oeq_weight_to_e3nn<0).sum()) for layer in source.models['nequip'].layers]
    assert mapping==[0,384,768,768]
    canonical=torch.load(run/'model-e3nn.pt',map_location='cpu',weights_only=False).eval()
    canonical.requires_grad_(False)
    assert not any(isinstance(module,OpenEquivarianceMessageTensorProduct) for module in canonical.modules())
    # Converting an already canonical export must not flip its weights again.
    again=canonical_e3nn_model(canonical)
    for key,value in canonical.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,again.state_dict()[key],rtol=0,atol=0)
    # Legacy state_dict loads preserve convention as well as ordinary parameters.
    best=torch.load(run/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
    old_state={key[6:]:value for key,value in best['state_dict'].items() if key.startswith('model.')}
    restored=deepcopy(source)
    for layer in restored.models['nequip'].layers:
        layer.message_tensor_product._oeq_weight_to_e3nn.fill_(1)
    restored.load_state_dict(old_state,strict=True)
    for key,value in source.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    with TemporaryDirectory() as temp:
        # The original class and its new convention metadata must round-trip.
        path=Path(temp)/'tagged.pt'
        torch.save(source,path)
        tagged=torch.load(path,map_location='cpu',weights_only=False).eval()
        tagged_state=Path(temp)/'tagged-state.pt'
        torch.save(source.state_dict(),tagged_state)
        restored.load_state_dict(torch.load(tagged_state,weights_only=False),strict=True)
        legacy_deploy=Path(temp)/'legacy-lammps.pt'
        deploy_model(tagged,legacy_deploy)
        scripted_legacy=torch.jit.load(str(legacy_deploy))
        scripted_canonical=torch.jit.load(str(run/'model-e3nn-lammps.pt'))
        data=ROOT/'experiments/04-EF-Pretrain-TensorBackbone-OpenEquivariance-data'
        checks=[]
        for split,indices,cache in [('valid',[0,25,49],'validation-04-native-phase.npz'),
                                     ('test',[0,75,149],'test-04-native-phase.npz')]:
            frames=read(ROOT/f'data/{split}.xyz',':')
            saved=np.load(data/cache)
            offsets=np.r_[0,np.cumsum(saved['counts'])]
            for index in indices:
                graph=AtomicGraph.from_ase(frames[index],cutoff=6.,property_mapping={'QM_energy':'energy','QM_forces':'forces'})
                expected={'energy':torch.tensor(saved['predicted_energy'][index],dtype=torch.float64),
                          'forces':torch.from_numpy(saved['predicted_forces'][offsets[index]:offsets[index+1]]).double()}
                with torch.enable_grad():
                    migrated=source.predict(graph,['energy','forces'])
                    converted=canonical.predict(graph,['energy','forces'])
                a=compare({key:value.detach().double() for key,value in migrated.items()},expected)
                b=compare({key:value.detach().double() for key,value in converted.items()},expected)
                record={'split':split,'index':index,'migrated_vs_cache':a,'canonical_vs_cache':b}
                if split=='valid' and index==0:
                    with torch.enable_grad():
                        legacy_script=scripted_legacy(as_lammps_data(graph))
                        canonical_script=scripted_canonical(as_lammps_data(graph))
                    record['scripted_legacy_vs_cache']=compare(legacy_script,expected)
                    record['scripted_canonical_vs_cache']=compare(canonical_script,expected)
                checks.append(record)
                print(record,flush=True)
    # Exact strict copy into the symmetric 0e+2e target's backbone.
    import yaml
    from graph_pes.config.shared import instantiate_config_from_dict,parse_model
    from graph_pes.config.training import TrainingConfig
    raw=yaml.safe_load((ROOT/'configs/tensor/01-SymmetricTensor-Scratch-N100.yaml').read_text())
    _,config=instantiate_config_from_dict(raw,TrainingConfig)
    target=parse_model(config.model)
    target_net = target.models['nequip'] if hasattr(target, 'models') else target
    for name in ('Z_embedding','initial_node_embedding','edge_embedding','layers'):
        getattr(target_net,name).load_state_dict(getattr(canonical.models['nequip'],name).state_dict(),strict=True)
    assert digest(run/'model.pt')==original_hash
    report={'source_sha256':original_hash,'canonical_sha256':digest(run/'model-e3nn.pt'),
            'negative_components_by_layer':mapping,'old_state_dict_load':True,'tagged_state_dict_load':True,
            'pickle_roundtrip':True,'canonical_conversion_idempotent':True,'strict_symmetric_backbone_copy':True,
            'cached_prediction_checks':checks,'native_cuda_comparison_run':False,
            'native_cuda_limitation':'No local CUDA; ARC SSH authentication unavailable in this session'}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('Legacy migration, state checkpoints, eager/scripted exports, cached predictions and strict tensor-backbone copy passed.',flush=True)


if __name__=='__main__':main()
