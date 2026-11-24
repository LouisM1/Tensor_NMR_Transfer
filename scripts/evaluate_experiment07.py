"""Export fixed-set tensor predictions, Soprano scores and six learning panels."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import nmr_helpers
import numpy as np
import pandas as pd
import torch
import yaml
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
from analysis_helpers.nmr_parameters import reconstruct_symmetric, extract_soprano
from nmr_helpers.experiment07 import sha
from scripts.experiment07 import audit_export, manifest, check_files

METHODS=['full','elora','last','head','scratch']
LABEL={'full':'Full fine-tuning','elora':'ELoRA','last':'Last layer + NMR head',
       'head':'NMR head only','scratch':'Direct training'}
COLOR={'full':'#7548A3','elora':'#008C83','last':'#377BB5','head':'#7D858D','scratch':'#D18B35'}
MARKER={'full':'D','elora':'o','last':'s','head':'v','scratch':'^'}


def chosen_runs():
    rows=[]
    for row in manifest()['runs']:
        if row['reuse']:
            directory=ROOT/'graph-pes-results'/row['reuse']
        else:
            choices=[p for p in (ROOT/'graph-pes-results').glob(row['name']+'-*')
                     if (p/'experiment07-complete.json').exists()]
            if len(choices)!=1:
                raise ValueError(f"{row['name']}: expected one completed run, found {choices}. "
                                 'Use --run-map with explicit method:N -> run-directory choices.')
            directory=choices[0]
        rows.append((row,directory))
    return rows


def predict(directory,frames,device):
    model,best,last=audit_export(directory)
    model=model.to(device).eval()
    targets=[];predictions=[];numbers=[];structure=[]
    with torch.no_grad():
        for index,atoms in enumerate(frames):
            graph=AtomicGraph.from_ase(atoms,cutoff=float(model.cutoff)).to(device)
            pred=model(graph)['tensor'].detach().cpu().numpy()
            assert pred.shape==(len(atoms),6) and np.isfinite(pred).all()
            target=np.asarray(atoms.arrays['MS_Symm_Irreps'])
            # Use original DFT shielding tensors for the Soprano reference.
            ms=np.asarray(atoms.arrays['ms']).reshape(-1,3,3)
            symmetric=(ms+ms.transpose(0,2,1))/2
            np.testing.assert_allclose(reconstruct_symmetric(target),symmetric,atol=1e-5,rtol=1e-6)
            targets.append(symmetric);predictions.append(pred);numbers.append(atoms.numbers)
            structure.append(np.full(len(atoms),index))
    return {'target_cartesian':np.concatenate(targets),'prediction':np.concatenate(predictions),
            'Z':np.concatenate(numbers),'structure':np.concatenate(structure)}


def score(arrays):
    reference=extract_soprano(arrays['target_cartesian'],arrays['Z'])
    prediction=extract_soprano(reconstruct_symmetric(arrays['prediction']),arrays['Z'])
    records=[]
    for element,z in [('Si',14),('O',8)]:
        mask=arrays['Z']==z
        for quantity in ('isotropy','anisotropy','asymmetry'):
            residual=prediction[quantity][mask]-reference[quantity][mask]
            assert np.isfinite(residual).all()
            records.append({'element':element,'quantity':quantity,'atoms':int(mask.sum()),
                            'rmse':float(np.sqrt(np.mean(residual**2)))})
    return records


def plot(table,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,ScalarFormatter,NullFormatter
    names={'isotropy':'Isotropic shielding','anisotropy':'Shielding anisotropy Δσ','asymmetry':'Asymmetry η'}
    fig,axes=plt.subplots(3,2,figsize=(10,10),layout='constrained')
    for row,quantity in enumerate(names):
        for col,element in enumerate(('Si','O')):
            ax=axes[row,col]
            for method in METHODS:
                values=table[(table['method']==method)&(table.element==element)&(table.quantity==quantity)]
                values=values.sort_values('n')
                assert values.n.tolist()==[10,50,100,200,800]
                ax.plot(values.n,values.rmse,color=COLOR[method],marker=MARKER[method],label=LABEL[method],ms=5)
            ax.set(xscale='log',xlabel='Training structures, N',title=f'{element} · {names[quantity]}',
                   ylabel='RMSE (dimensionless)' if quantity=='asymmetry' else 'RMSE (ppm)')
            ax.xaxis.set_major_locator(FixedLocator([10,50,100,200,800]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Symmetric tensor NMR · single-seed test learning curves',fontsize=13)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'07-test-learning-curves.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/07-LearningCurves')
    parser.add_argument('--run-map',type=Path,help='JSON dictionary e.g. {"full:10":"graph-pes-results/..."}; specify all 25')
    args=parser.parse_args()
    torch.set_num_threads(min(8,int(os.environ.get('OMP_NUM_THREADS','2'))))
    check_files()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.run_map:
        mapping=json.loads(args.run_map.read_text())
        rows=[(r,ROOT/mapping[f"{r['method']}:{r['n']}"]) for r in manifest()['runs']]
    else:
        rows=chosen_runs()
    # Resolve and validate every model before reading any held-out targets.
    selections={}
    for row,directory in rows:
        model,_,_=audit_export(directory)
        raw=yaml.safe_load((directory/'train-config.yaml').read_text())
        expected=yaml.safe_load((ROOT/row['config']).read_text())
        assert raw['data']['train']==expected['data']['train']
        assert raw['data']['valid']==expected['data']['valid']
        assert raw['data']['test'] is None and raw['loss']==expected['loss']
        if row['method']!='scratch':
            assert model._symmetric_transfer_source_sha256==manifest()['source_sha256']
        del model
        selections[f"{row['method']}:{row['n']}"]=str(directory)
    (output/'selected-runs.json').write_text(json.dumps(selections,indent=2)+'\n')
    all_scores=[]
    for split,filename,count in [('validation','valid.xyz',50),('test','test.xyz',150)]:
        frames=read(ROOT/'data/nmr_symm'/filename,':')
        assert len(frames)==count
        for row,directory in rows:
            cache=output/f"predictions-{row['method']}-N{row['n']}-{split}.npz"
            identity={'model_sha256':sha(directory/'model.pt'),
                      'data_sha256':sha(ROOT/'data/nmr_symm'/filename),
                      'method':row['method'],'n':row['n'],'split':split,'device':args.device}
            if cache.exists():
                with np.load(cache,allow_pickle=False) as stored:
                    assert json.loads(str(stored['identity']))==identity,f'Stale cache: {cache}'
                    arrays={k:stored[k] for k in ('target_cartesian','prediction','Z','structure')}
            else:
                arrays=predict(directory,frames,args.device)
                temporary=cache.with_name(cache.name+'.tmp.npz')
                np.savez_compressed(temporary,**arrays,identity=json.dumps(identity,sort_keys=True))
                temporary.replace(cache)
            assert arrays['prediction'].shape==(count*144,6)
            for metric in score(arrays):
                all_scores.append({'method':row['method'],'n':row['n'],'split':split,**metric})
            print(f"Scored {row['method']} N{row['n']} {split}",flush=True)
    table=pd.DataFrame(all_scores)
    table.to_csv(output/'metrics.csv',index=False)
    plot(table[table.split=='test'],output)
    print('Saved predictions, Soprano scores and six-panel curves:',output)


if __name__=='__main__':main()
