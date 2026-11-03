from pathlib import Path
import sys,os,json,copy
os.environ['MPLCONFIGDIR']='/private/tmp/tensor-nmr-mpl'
os.environ['XDG_CACHE_HOME']='/private/tmp/tensor-nmr-smoke-cache'
os.environ['WANDB_MODE']='disabled'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
import tempfile
OUT=Path(tempfile.mkdtemp(prefix='symmetric-nmr-smoke-'))
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np,torch,yaml
import nmr_helpers
from ase.io import read
from e3nn import o3
from e3nn.io import CartesianTensor
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.config.shared import parse_model,parse_loss,instantiate_config_from_dict
from graph_pes.config.training import TrainingConfig
from nmr_helpers.shared_head_training import train_from_config
torch.set_num_threads(2);torch.set_default_dtype(torch.float32);torch.manual_seed(42)
raw=yaml.safe_load((ROOT/'configs/tensor/01-SymmetricTensor-Scratch-N100.yaml').read_text())
_,cfg=instantiate_config_from_dict(raw,TrainingConfig)
m=parse_model(cfg.model)
aa=read(ROOT/'data/train.xyz',':2')
for a in aa:
    t=torch.tensor(a.arrays['ms'].reshape(-1,3,3),dtype=torch.float64)
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    a.set_array('MS_Symm_Irreps',CartesianTensor('ij=ji').from_cartesian((t+t.transpose(-1,-2))/2).numpy())
    torch.set_default_dtype(old)
graphs=[AtomicGraph.from_ase(a,cutoff=6.,property_mapping={'MS_Symm_Irreps':'tensor'}) for a in aa]
m.pre_fit_all_components(graphs)
g=graphs[0];m.eval()
ct=CartesianTensor('ij=ji')
with torch.no_grad():p=m(g)['tensor']
assert p.shape==(144,6) and torch.isfinite(p).all() and p[:,1:].abs().max()>0
reference=ct.to_cartesian(p)
errors=[]
for R in [o3.rand_matrix(),torch.diag(torch.tensor([-1.,1.,1.]))]:
 a=aa[0].copy();a.positions=a.positions@R.numpy().T;a.set_cell(a.cell.array@R.numpy().T,scale_atoms=False)
 gr=AtomicGraph.from_ase(a,cutoff=6.)
 with torch.no_grad():actual=ct.to_cartesian(m(gr)['tensor'])
 expected=R@reference@R.T
 error=float((actual-expected).abs().max());errors.append(error)
 torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-3)
loss=parse_loss(cfg.loss)
value=loss(m,g,{'tensor':p})
print('MODEL',sum(x.numel() for x in m.parameters()),'parameters; symmetry errors',errors,flush=True)
# Same model/loss/optimizer/callbacks as YAML; only execution budget and logging overridden.
smoke=copy.deepcopy(raw)
smoke['general']['root_dir']=str(OUT/'smoke-runs');smoke['general']['run_id']='symmetric-smoke'
smoke['wandb']=None
fit=smoke['fitting'];fit['max_n_pre_fit']=2
fit['trainer_kwargs'].update(accelerator='cpu',max_epochs=1,limit_train_batches=8,limit_val_batches=2,limit_test_batches=0,enable_model_summary=False)
for split in ('train','valid'):smoke['data'][split]['pre_transform']=False
train_from_config(smoke)
run=OUT/'smoke-runs/symmetric-smoke'
ck=torch.load(run/'checkpoints/best.ckpt',weights_only=False,map_location='cpu')
saved=torch.load(run/'model.pt',weights_only=False,map_location='cpu')
saved.eval()
with torch.no_grad():pr=saved(g)['tensor']
assert pr.shape==p.shape and torch.isfinite(pr).all()
assert ck['global_step']==2,ck['global_step']
last=torch.load(run/'checkpoints/last.ckpt',weights_only=False,map_location='cpu')
moments=[v['exp_avg'] for v in last['optimizer_states'][0]['state'].values() if 'exp_avg' in v]
assert moments and all(torch.isfinite(v).all() for v in moments) and any(v.abs().max()>0 for v in moments)
selected=next(v for k,v in last['callbacks'].items() if k.startswith('ModelCheckpoint') and v.get('monitor')=='valid/loss/total')
assert torch.isfinite(selected['best_model_score'])
assert any(not torch.equal(saved.state_dict()[k],m.state_dict()[k]) for k in m.state_dict() if torch.is_tensor(m.state_dict()[k]) and m.state_dict()[k].is_floating_point())
report=dict(production_config=str(ROOT/'configs/tensor/01-SymmetricTensor-Scratch-N100.yaml'),
    output_irreps=str(m.target_tensor_irreps),shape=list(p.shape),symmetry_max_abs_errors=errors,
    optimizer_updates=ck['global_step'],parameters=sum(x.numel() for x in m.parameters()),
    smoke_overrides='CPU, 1 epoch, 8 training batches, 2 validation batches, 2 prefit structures, W&B disabled; original N100 subset retained',
    saved_model_finite=True)
(OUT/'smoke-results.json').write_text(json.dumps(report,indent=2)+'\n')
print('SMOKE PASSED',json.dumps(report),flush=True)
