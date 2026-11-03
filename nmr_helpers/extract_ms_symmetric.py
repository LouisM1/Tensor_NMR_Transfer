"""Create data/nmr_symm copies; never overwrite originals or existing outputs.

Run: .venv/bin/python -m nmr_helpers.extract_ms_symmetric
MS_Symm_QM stores flattened Cartesian S=(ms+ms.T)/2, shape (atoms,9).
MS_Symm_Irreps stores e3nn CartesianTensor('ij=ji') coefficients, shape (atoms,6).
The first coefficient equals sqrt(3)*trace(ms)/3 in this orthonormal basis.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch
from ase.io import iread, write
from e3nn.io import CartesianTensor

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--output-dir',type=Path,default=Path('data/nmr_symm'))
    args=parser.parse_args(); out=args.output_dir
    sources=[args.data_dir/f'{s}.xyz' for s in ('raw','train','valid','test')]
    outputs=[out/p.name for p in sources]
    for p in sources: assert p.is_file(),p
    for p in [*outputs,out/'manifest.json']: assert not p.exists(),f'Refusing to overwrite {p}'
    assert not any(p.resolve()==q.resolve() for p in sources for q in outputs)
    hashes={str(p):sha(p) for p in sources}
    out.mkdir(parents=True,exist_ok=True)
    # Build the change-of-basis coefficients in float64, not float32 cast to double.
    torch.set_default_dtype(torch.float64)
    ct=CartesianTensor('ij=ji')
    rtp=ct.reduced_tensor_products(torch.zeros(3,3))
    created=[]; records=[]
    try:
        for src,dst in zip(sources,outputs):
            frames=atoms=0
            with dst.open('x') as f:
                created.append(dst)
                for a in iread(src,':'):
                    assert 'MS_Symm_QM' not in a.arrays and 'MS_Symm_Irreps' not in a.arrays
                    t=np.asarray(a.arrays['ms'],dtype=np.float64).reshape(len(a),3,3)
                    assert np.isfinite(t).all()
                    s=(t+t.transpose(0,2,1))/2
                    coeff=ct.from_cartesian(torch.from_numpy(s),rtp=rtp).numpy()
                    np.testing.assert_allclose(ct.to_cartesian(torch.from_numpy(coeff),rtp=rtp).numpy(),s,rtol=0,atol=1e-9)
                    np.testing.assert_allclose(coeff[:,0]/np.sqrt(3),np.trace(t,axis1=1,axis2=2)/3,rtol=0,atol=1e-9)
                    a.set_array('MS_Symm_QM',s.reshape(len(a),9))
                    a.set_array('MS_Symm_Irreps',coeff)
                    write(f,a,format='extxyz');frames+=1;atoms+=len(a)
            # Re-read written files to check precision and retention of every original array.
            for a,b in zip(iread(src,':'),iread(dst,':'),strict=True):
                np.testing.assert_array_equal(a.numbers,b.numbers)
                np.testing.assert_allclose(a.positions,b.positions,rtol=0,atol=1e-8)
                np.testing.assert_allclose(a.cell.array,b.cell.array,rtol=0,atol=1e-8)
                np.testing.assert_array_equal(a.pbc,b.pbc)
                for key in a.arrays:
                    np.testing.assert_array_equal(a.arrays[key],b.arrays[key])
                assert a.info==b.info
                t=a.arrays['ms'].reshape(len(a),3,3);s=(t+t.transpose(0,2,1))/2
                np.testing.assert_allclose(b.arrays['MS_Symm_QM'].reshape(-1,3,3),s,rtol=0,atol=1e-8)
                np.testing.assert_allclose(ct.to_cartesian(torch.from_numpy(b.arrays['MS_Symm_Irreps']),rtp=rtp).numpy(),s,rtol=0,atol=2e-8)
            records.append(dict(source=str(src),output=str(dst),source_sha256=hashes[str(src)],output_sha256=sha(dst),structures=frames,atoms=atoms))
            print(dst,frames,'structures;',atoms,'atoms',flush=True)
        assert all(sha(p)==hashes[str(p)] for p in sources)
        import e3nn
        (out/'manifest.json').write_text(json.dumps(dict(
            format='extxyz',units='ppm',symmetry='S=(ms+ms.T)/2',
            cartesian_array='MS_Symm_QM: row-major flattened 3x3',
            irrep_array="MS_Symm_Irreps: e3nn.io.CartesianTensor('ij=ji'), 1x0e+1x2e",
            scalar_coefficient='sqrt(3)*isotropic shielding',e3nn_version=e3nn.__version__,
            originals_unchanged=True,records=records),indent=2)+'\n')
    except BaseException:
        for p in created:p.unlink()
        raise

if __name__=='__main__':main()
