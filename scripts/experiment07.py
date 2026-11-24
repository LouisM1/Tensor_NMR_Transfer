"""Validate, smoke-test and audit the Experiment 7 sweep (run from project root)."""
import argparse
from copy import deepcopy
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import nmr_helpers
import numpy as np
import torch
import yaml
from graph_pes.atomic_graph import AtomicGraph, to_batch
from graph_pes.config.shared import (TorchConfig, instantiate_config_from_dict,
                                    parse_model, resolve_config_dict)
from graph_pes.config.training import TrainingConfig
from graph_pes.data.datasets import file_dataset
from graph_pes.scripts.utils import configure_general_options
from graph_pes.training.callbacks import SaveBestModel
from graph_pes.utils.nn import PerElementParameter
from nmr_helpers.elora import PathELoRA
from nmr_helpers.experiment07 import expected_trainable, install_fixes, sha
from nmr_helpers.shared_head_training import train_from_config

MANIFEST = ROOT/'provenance/experiment07/manifest.json'


def manifest():
    return json.loads(MANIFEST.read_text())


def fingerprint():
    import hashlib
    files = [MANIFEST, ROOT/'provenance/experiment07/pending.tsv', ROOT/'train.sbatch',
             ROOT/'experiment07.sbatch']
    for pattern in ('nmr_helpers/*.py', 'scripts/*.py', 'configs/tensor/07-LearningCurves/*.yaml'):
        files += sorted(ROOT.glob(pattern))
    data = {str(p.relative_to(ROOT)):sha(p) for p in files}
    data.update(manifest()['files'])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def check_files():
    for path, expected in manifest()['files'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Input hash mismatch: {path}')


def make_model(raw):
    resolved = resolve_config_dict(deepcopy(raw), TrainingConfig)
    configure_general_options(TorchConfig(**resolved['general']['torch']), resolved['general']['seed'])
    _, config = instantiate_config_from_dict(resolved, TrainingConfig)
    return parse_model(config.model), config


def check_mask(model, method):
    for name, p in model.named_parameters():
        assert p.requires_grad == expected_trainable(name, method), name
    wrappers = [m for m in model.modules() if isinstance(m, PathELoRA)]
    assert len(wrappers) == (12 if method=='elora' else 0)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate():
    check_files()
    for package, pinned in [('graph-pes','1.0.2'), ('e3nn','0.4.4'), ('pytorch-lightning','2.5.1')]:
        assert version(package)==pinned, (package,version(package),pinned)
    m = manifest()
    records=[]
    previous=set()
    for n in m['sizes']:
        ds = file_dataset('data/nmr_symm/train.xyz', cutoff=6.0, n=n, shuffle=True,
                          seed=42, pre_transform=False, property_map={'MS_Symm_Irreps':'tensor'})
        indices = list(map(int,ds.graphs.structures.indices))
        assert indices == m['subsets'][str(n)]
        assert previous.issubset(set(indices))
        previous=set(indices)
        assert ds[0].properties['tensor'].shape==(144,6)
    valid=file_dataset('data/nmr_symm/valid.xyz',cutoff=6.,pre_transform=False,
                       property_map={'MS_Symm_Irreps':'tensor'})
    assert len(valid)==50
    base=yaml.safe_load((ROOT/'configs/tensor/01-SymmetricTensor-Scratch-N100.yaml').read_text())
    model_specs={}
    counts={}
    for row in m['runs']:
        raw=yaml.safe_load((ROOT/row['config']).read_text())
        assert raw['data']['test'] is None
        assert raw['data']['train']=={**base['data']['train'],'n':row['n']}
        assert raw['data']['valid']==base['data']['valid']
        assert raw['loss']==base['loss']
        assert raw['fitting']['max_n_pre_fit'] is None
        assert raw['fitting']['loader_kwargs']==base['fitting']['loader_kwargs']
        assert raw['fitting']['trainer_kwargs']==base['fitting']['trainer_kwargs']
        method=row['method']
        if method not in model_specs:
            model,_=make_model(raw)
            counts[method]=check_mask(model,method)
            model_specs[method]=raw['model']
            del model
            gc.collect()
        assert raw['model']==model_specs[method]
        assert raw['fitting']['optimizer']['lr']==(.0001 if method=='full' else .003)
        records.append({**row,'trainable_tensor_entries':counts[method]})
    assert len([r for r in m['runs'] if not r['reuse']])==22
    print('PASS: all 25 configurations, masks, input hashes, nested subsets and recipe invariants.',flush=True)
    return records


class SmokeAudit(__import__('pytorch_lightning').Callback):
    def __init__(self, method, n):
        self.method,self.n=method,n
        self.steps=0

    def on_train_start(self,trainer,task):
        model=task.model
        check_mask(model,self.method)
        ds=trainer.train_dataloader.dataset
        indices=manifest()['subsets'][str(self.n)]
        if hasattr(ds.graphs,'structures'):
            assert list(map(int,ds.graphs.structures.indices))==indices
        else:
            # pre_transform=True replaces the converter with a graph list.
            # Check its ordered coordinates/labels against the declared subset.
            from ase.io import read
            frames=read(ROOT/'data/nmr_symm/train.xyz',':')
            assert len(ds.graphs)==len(indices)
            for graph,index in zip(ds.graphs,indices):
                atoms=frames[index]
                torch.testing.assert_close(graph.R,torch.as_tensor(atoms.positions,dtype=graph.R.dtype))
                torch.testing.assert_close(graph.properties['tensor'],
                    torch.as_tensor(atoms.arrays['MS_Symm_Irreps'],dtype=graph.properties['tensor'].dtype))
        assert len(trainer.val_dataloaders.dataset)==50
        batch=to_batch(list(ds.graphs))
        y,z=batch.properties['tensor'],batch.Z
        for element in (8,14):
            v=y[z==element]
            torch.testing.assert_close(model.scaler.offsets[element,0].detach().cpu(),v[:,0].mean())
            torch.testing.assert_close(model.scaler.scales[element,0].detach().cpu(),v[:,0].std(unbiased=False).clamp_min(.001))
            torch.testing.assert_close(model.scaler.scales[element,1].detach().cpu(),v[:,1:].square().mean().sqrt().clamp_min(.001))
        self.before={name:p.detach().cpu().clone() for name,p in model.named_parameters()}

    def on_before_optimizer_step(self,trainer,task,optimizer):
        for name,p in task.model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None,name
            elif p.grad is not None:
                assert torch.isfinite(p.grad).all(),name
        self.steps+=1

    def on_train_end(self,trainer,task):
        assert self.steps==trainer.global_step==2
        changed=[]
        for name,p in task.model.named_parameters():
            different=not torch.equal(p.detach().cpu(),self.before[name])
            if not p.requires_grad:
                assert not different,name
            if different:
                changed.append(name)
        assert any(n.startswith('tensor_readout.') for n in changed)
        assert any(n.startswith('scaler.') for n in changed)
        if self.method=='last':
            assert any(n.startswith('layers.3.') for n in changed)
        if self.method in ('scratch','full'):
            assert all(any(n.startswith(f'layers.{i}.') for n in changed) for i in range(4))
        if self.method=='elora':
            for prefix,module in task.model.named_modules():
                if isinstance(module,PathELoRA):
                    for factor in ('A','B'):
                        assert any(n.startswith(prefix+'.'+factor+'.') for n in changed),(prefix,factor)


def audit_export(directory):
    directory=Path(directory)
    best=torch.load(directory/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
    last=torch.load(directory/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False).eval()
    expected={k[6:]:v for k,v in best['state_dict'].items() if k.startswith('model.')}
    assert expected.keys()==saved.state_dict().keys()
    for key,value in saved.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value,expected[key]),key
            if value.is_floating_point(): assert torch.isfinite(value).all(),key
        else:
            assert value==expected[key],key
    assert all('param_names' in group for group in last['optimizer_states'][0]['param_groups'])
    return saved,best,last


def resume_training(directory, max_epochs=None):
    """Resume without re-fitting scalers, retaining Adam/scheduler/stopping state."""
    from graph_pes.config.shared import parse_loss
    from graph_pes.data.loader import GraphDataLoader
    from graph_pes.scripts.train import trainer_from_config
    from graph_pes.training.tasks import TrainingTask, get_all_eval_metrics
    from graph_pes.utils.logger import log_to_file
    directory=Path(directory).resolve()
    audit_export(directory)
    checkpoint=directory/'checkpoints/last.ckpt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    old_groups=state['optimizer_states'][0]['param_groups']
    raw=yaml.safe_load((directory/'train-config.yaml').read_text())
    if max_epochs is not None:
        raw['fitting']['trainer_kwargs']['max_epochs']=max_epochs
    model,config=make_model(raw)
    data=config.get_data(model)
    data.train.prepare_data();data.valid.prepare_data()
    data.train.setup();data.valid.setup()
    task=TrainingTask.load_from_checkpoint(str(checkpoint),map_location='cpu',
        model=model,loss=parse_loss(config.loss),optimizer=config.fitting.get_optimizer(),
        scheduler=config.fitting.get_scheduler(),eval_metrics=get_all_eval_metrics([data.train,data.valid]))
    for p in task.model.parameters():
        if isinstance(p,PerElementParameter):p.register_elements([8,14])
    check=config.fitting.get_optimizer()(task.model)
    assert [g['param_names'] for g in check.param_groups]==[g['param_names'] for g in old_groups]
    del check,state
    log_to_file(directory)
    trainer=trainer_from_config(config,directory)
    # NMR has no need for an energy/LAMMPS export.
    for cb in trainer.callbacks:
        if isinstance(cb,SaveBestModel):cb.try_to_deploy=False
    kwargs=dict(config.fitting.loader_kwargs);kwargs.pop('shuffle',None)
    trainer.fit(task,GraphDataLoader(data.train,shuffle=True,**kwargs),
                GraphDataLoader(data.valid,shuffle=False,**kwargs),ckpt_path=str(checkpoint))


def smoke(device, output, large=True):
    os.environ['WANDB_MODE']='disabled'
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    records=validate()
    if device=='cuda':
        assert torch.cuda.is_available(),'ARC smoke requires a usable CUDA allocation'
        print('GPU:',torch.cuda.get_device_name(0),flush=True)
        assert (torch.ones(4,device='cuda')*2).tolist()==[2.]*4
    install_fixes()
    import graph_pes.scripts.train as training
    original_factory,original_tester=training.trainer_from_config,training.test_with_lightning
    results=[]
    selected=[r for r in manifest()['runs'] if r['n']==10]
    if large:
        selected += [r for r in manifest()['runs'] if r['n']==800 and r['method']=='head']
    for row in selected:
        probe=yaml.safe_load((ROOT/row['config']).read_text())
        probe['general'].update(root_dir=str(output),run_id=row['name'])
        probe['wandb']=None
        probe['fitting']['trainer_kwargs'].update(accelerator='gpu' if device=='cuda' else 'cpu',
            max_epochs=1,limit_train_batches=8,limit_val_batches=2,enable_model_summary=False)
        # Full selected-N pre-fit and production batch/accumulation are retained.
        # The callback installs the production in-process graph cache, so
        # pre_transform=True exercises the actual array-job data-loading path.
        audit=SmokeAudit(row['method'],row['n'])
        def factory(config,directory):
            trainer=original_factory(config,directory)
            trainer.callbacks.append(audit)
            for cb in trainer.callbacks:
                if isinstance(cb,SaveBestModel): cb.try_to_deploy=False
            return trainer
        training.trainer_from_config=factory
        training.test_with_lightning=lambda *a,**k:None
        try:
            train_from_config(probe)
        finally:
            training.trainer_from_config,training.test_with_lightning=original_factory,original_tester
        directory=output/row['name']
        saved,best,last=audit_export(directory)
        rebuilt,config=make_model(probe)
        rebuilt.load_state_dict({k[6:]:v for k,v in last['state_dict'].items() if k.startswith('model.')},strict=True)
        for p in rebuilt.parameters():
            if isinstance(p,PerElementParameter):p.register_elements([8,14])
        opt=config.fitting.get_optimizer()(rebuilt)
        groups=last['optimizer_states'][0]['param_groups']
        assert [g['param_names'] for g in opt.param_groups]==[g['param_names'] for g in groups]
        opt.load_state_dict(last['optimizer_states'][0])
        # Check output and O(3) covariance after checkpoint reload, using a training atom set.
        from ase.io import read
        from scripts.smoke_symmetric_transfer import symmetry
        atoms=read(ROOT/'data/nmr_symm/train.xyz',index=manifest()['subsets'][str(row['n'])][0])
        sym=symmetry(saved,atoms)
        history=np.genfromtxt(directory/'epoch-metrics.csv',delimiter=',',names=True)
        assert history.size==1
        results.append({'method':row['method'],'n':row['n'],'optimizer_steps':audit.steps,
                        'export_matches_best':True,'optimizer_state_reload':True,
                        'correct_freeze_mask_and_updates':True,'selected_N_scaler_fit':True,
                        'symmetry_max_errors_ppm':sym})
        print(f"PASS smoke: {row['method']} N={row['n']}",flush=True)
        del saved,rebuilt,opt,best,last
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    # Exercise the exact production resume function, including callback state.
    resume_row=next(r for r in selected if r['method']=='last')
    resume_training(output/resume_row['name'],max_epochs=2)
    _,_,last=audit_export(output/resume_row['name'])
    assert last['global_step']==4
    report={'passed':True,'device':device,'torch':torch.__version__,
            'versions':{p:version(p) for p in ('graph-pes','e3nn','pytorch-lightning')},
            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'fingerprint':fingerprint(),'configurations':len(records),'runs':results,
            'lightning_resume_to_four_steps':True,
            'held_out_test_predictions_used':False,
            'limits':{'epochs':1,'train_batches':8,'validation_batches':2,
                      'pre_fit':'all selected N','wandb':False,'post_fit_scoring':False}}
    (output/'smoke-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL EXPERIMENT 7 SMOKE CHECKS PASSED:',output/'smoke-results.json',flush=True)


def receipt(path):
    check_files()
    report=json.loads(Path(path).read_text())
    assert report['passed'] and report['device']=='cuda'
    assert report['fingerprint']==fingerprint(),'Code or inputs changed after GPU smoke; rerun the pipeline'
    print('PASS: matching GPU smoke receipt')


def complete(path):
    directory=Path(path).resolve()
    saved,best,last=audit_export(directory)
    raw=yaml.safe_load((directory/'train-config.yaml').read_text())
    cb=next(x['+nmr_helpers.experiment07.CurveAudit'] for x in raw['fitting']['callbacks']
            if '+nmr_helpers.experiment07.CurveAudit' in x)
    check_mask(saved,cb['method'])
    result={'passed':True,'method':cb['method'],'n':cb['n'],'best_epoch':int(best['epoch'])+1,
            'last_epoch':int(last['epoch'])+1,'model_sha256':sha(directory/'model.pt'),
            'best_sha256':sha(directory/'checkpoints/best.ckpt'),
            'fingerprint':fingerprint()}
    (directory/'experiment07-complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified completed run:',directory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('validate')
    p=sub.add_parser('smoke');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',required=True);p.add_argument('--skip-large-prefit',action='store_true')
    p=sub.add_parser('receipt');p.add_argument('path')
    p=sub.add_parser('complete');p.add_argument('path')
    p=sub.add_parser('resume');p.add_argument('path')
    args=parser.parse_args()
    torch.set_num_threads(min(8,int(os.environ.get('OMP_NUM_THREADS','2'))))
    install_fixes()
    if args.command=='validate': validate()
    elif args.command=='smoke':smoke(args.device,args.output,not args.skip_large_prefit)
    elif args.command=='receipt':receipt(args.path)
    elif args.command=='complete':complete(args.path)
    elif args.command=='resume':resume_training(args.path)


if __name__=='__main__': main()
