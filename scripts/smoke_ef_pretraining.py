"""Bounded E/F optimisation and compatibility check; outputs are temporary."""
from pathlib import Path
import os,sys,json,copy,tempfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(tempfile.mkdtemp(prefix='tensor-ef-smoke-'))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tensor-nmr-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/tensor-ef-cache')
os.environ['WANDB_MODE']='disabled';sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np,torch,yaml,nmr_helpers
from e3nn import o3
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.config.shared import parse_model,parse_loss,instantiate_config_from_dict
from graph_pes.config.training import TrainingConfig
from graph_pes.scripts import train as training
from graph_pes.utils.nn import count_used_parameters
from graph_pes.utils.lammps import deploy_model
from nmr_helpers.shared_head_training import train_from_config
torch.set_num_threads(2);torch.set_default_dtype(torch.float32);torch.manual_seed(42)
cp=ROOT/'configs/tensor/02-EF-Pretrain-TensorBackbone.yaml'
raw=yaml.safe_load(cp.read_text());_,cfg=instantiate_config_from_dict(raw,TrainingConfig)
model=parse_model(cfg.model)
target_raw=yaml.safe_load((ROOT/'configs/tensor/01-SymmetricTensor-Scratch-N100.yaml').read_text())
_,target_cfg=instantiate_config_from_dict(target_raw,TrainingConfig)
target=parse_model(target_cfg.model)
net=model.models['nequip'];parts=('Z_embedding','initial_node_embedding','edge_embedding','layers')
for p in parts:
 a=getattr(net,p).state_dict();b=getattr(target,p).state_dict()
 assert a.keys()==b.keys(),p
 for k in a:
  if torch.is_tensor(a[k]):assert a[k].shape==b[k].shape,(p,k)
for a,b in zip(net.layers,target.layers):assert a.irreps_in==b.irreps_in and a.irreps_out==b.irreps_out
atoms=read(ROOT/'data/train.xyz',':2')
graphs=[AtomicGraph.from_ase(a,cutoff=6.,property_mapping={'QM_energy':'energy','QM_forces':'forces'}) for a in atoms]
assert all(set(g.properties)=={'energy','forces'} and not g.other for g in graphs)
model.pre_fit_all_components(graphs)
model.train();captured={}
def capture(module,args,output):
 output.retain_grad();captured['h']=output
hook=net.layers[-1].register_forward_hook(capture)
pred=model.predict(graphs[0],['energy','forces'])
loss=parse_loss(cfg.loss)(model,graphs[0],pred).loss_value
assert torch.isfinite(loss)
loss.backward();hook.remove()
n0=net.layers[-1].irreps_out.count('0e')
gradient_norms=[float(captured['h'].grad[:,:n0].norm()),float(captured['h'].grad[:,n0:].norm())]
assert all(np.isfinite(gradient_norms)) and min(gradient_norms)>0,gradient_norms
model.zero_grad();model.eval()
base=model.predict(graphs[0],['energy','forces'])
symmetry=[]
for R in [o3.rand_matrix(),torch.diag(torch.tensor([-1.,1.,1.]))]:
 a=atoms[0].copy();a.positions=a.positions@R.numpy().T;a.set_cell(a.cell.array@R.numpy().T,scale_atoms=False)
 rotated=model.predict(AtomicGraph.from_ase(a,cutoff=6.),['energy','forces'])
 torch.testing.assert_close(rotated['energy'],base['energy'],rtol=1e-6,atol=.02)
 torch.testing.assert_close(rotated['forces'],base['forces']@R.T,rtol=3e-4,atol=.003)
 symmetry.append(dict(energy_abs_error=float((rotated['energy']-base['energy']).abs().max()),force_max_error=float((rotated['forces']-base['forces']@R.T).abs().max())))
print('COMPATIBILITY AND E/F GRADIENTS PASSED',gradient_norms,symmetry,flush=True)

smoke=copy.deepcopy(raw);smoke['general'].update(root_dir=str(OUT),run_id='ef-smoke')
smoke['wandb']=None;smoke['fitting']['max_n_pre_fit']=2
smoke['fitting']['trainer_kwargs'].update(accelerator='cpu',max_epochs=1,limit_train_batches=4,limit_val_batches=1,accumulate_grad_batches=2,enable_model_summary=False)
for split in ('train','valid'):smoke['data'][split]['pre_transform']=False
# graph-pes creates a separate post-fit tester, ignoring the training
# trainer's limit_test_batches; bound that separate tester explicitly.
original=training.test_with_lightning
def bounded_test(tester,*args,**kwargs):
 tester.limit_test_batches=1
 return original(tester,*args,**kwargs)
training.test_with_lightning=bounded_test
try:train_from_config(smoke)
finally:training.test_with_lightning=original
run=OUT/'ef-smoke'
best=torch.load(run/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
last=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False).eval()
assert best['global_step']==2
assert all(torch.isfinite(v['exp_avg']).all() for v in last['optimizer_states'][0]['state'].values() if 'exp_avg' in v)
for p in parts:getattr(target,p).load_state_dict(getattr(saved.models['nequip'],p).state_dict(),strict=True)
target.eval()
assert torch.isfinite(target.predict(graphs[0],['tensor'])['tensor']).all()
for k,v in saved.state_dict().items():
 if torch.is_tensor(v):assert torch.equal(v,best['state_dict']['model.'+k]),k
deploy_model(saved,path=OUT/'deployment-check.pt')
report=dict(config=str(cp),config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256((ROOT/'nmr_helpers/tensor_ef_pretraining.py').read_bytes()).hexdigest(),
    backbone_shape_irrep_match=True,strict_four_layer_copy=True,final_irreps=str(net.layers[-1].irreps_out),
    scalar_anisotropic_output_gradient_norms=gradient_norms,symmetry_errors=symmetry,
    active_parameters=count_used_parameters(model,only_learnable=True),optimizer_updates=2,
    export_matches_best=True,deployment_passed=True,saved_ef_predictions_finite=all(torch.isfinite(v).all() for v in saved.predict(graphs[0],['energy','forces']).values()),
    held_out_test_accessed=False,smoke_overrides='CPU; 1 epoch; 4 training batches; accumulation 2; 1 validation batch; 2 prefit structures; 1 postfit batch per train/valid dataset; W&B disabled',
    temporary_results=str(OUT))
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('SMOKE PASSED',json.dumps(report),flush=True)
