"""Verify this local environment without training or modifying inputs."""
from pathlib import Path
import sys,os,json,hashlib,tempfile,importlib
from importlib.metadata import distributions

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tensor-nmr-mpl'))
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve(), 'Run with this repo .venv/bin/python'
installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions()}
expected={}
for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name,version=line.split('==');expected[name.lower().replace('_','-')]=version
assert expected==installed, 'Installed versions differ from bootstrap lock'
modules=['torch','graph_pes','e3nn','ase','numpy','pandas','matplotlib','pytorch_lightning','soprano','sklearn','wandb','ipykernel',
         'nmr_helpers.shielding_transfer','nmr_helpers.elora','nmr_helpers.shared_head_training','nmr_helpers.frozen_head_training','nmr_helpers.training_fixes']
for name in modules:
    m=importlib.import_module(name)
    assert str(ROOT) in str(m.__file__),(name,m.__file__)
import torch
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
from e3nn.io import CartesianTensor
torch.set_num_threads(2)
torch.set_default_dtype(torch.float32)
path=ROOT/'pretrained/Kavanagh-Med-9059603/model.pt'
model=torch.load(path,map_location='cpu',weights_only=False).eval()
a=read(ROOT/'data/train.xyz',index=0)
graph=AtomicGraph.from_ase(a,cutoff=6.)
with torch.no_grad():energy=model.predict(graph,['energy'])['energy']
assert torch.isfinite(energy).all()
ct=CartesianTensor('ij=ji')
t=torch.as_tensor(a.arrays['ms'].reshape(-1,3,3),dtype=torch.float32)
s=(t+t.transpose(-1,-2))/2
y=ct.from_cartesian(s)
assert str(ct)=='1x0e+1x2e' and y.shape==(len(a),6)
torch.testing.assert_close(ct.to_cartesian(y),s,rtol=1e-5,atol=1e-4)
assert 'NequIP_experiments/.venv' not in '\n'.join(sys.path)
report=dict(python=sys.executable,prefix=sys.prefix,packages=len(installed),
    versions={x:installed[x] for x in ['torch','graph-pes','e3nn','numpy','pandas','soprano']},
    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    energy=float(energy),symmetric_tensor_irreps=str(ct),tensor_shape=list(y.shape),
    success=True)
print(json.dumps(report,indent=2))
