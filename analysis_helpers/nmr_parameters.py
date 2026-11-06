"""Soprano shielding-parameter evaluation of the saved N100 tensor predictions.

MSIsotropy -> shielding sigma_iso (ppm), MSAnisotropy -> Haeberlen Delta
(ppm), MSAsymmetry -> eta (dimensionless). No shift referencing, model
selection, fitting, sign alignment, clipping or filtering is performed.
"""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from soprano.properties.nmr import (MSIsotropy, MSAnisotropy, MSAsymmetry,
                                    MSReducedAnisotropy, MSSpan)

from .anisotropic_components import traceless_cartesian
from .symmetric_transfer_review import (COLORS, LABELS, RUNS, Z, load_cache, load_review,
                                        metrics, sha256, style)

PARAMETERS = ['isotropy', 'anisotropy', 'asymmetry']
NAMES = {'isotropy': 'Isotropic shielding', 'anisotropy': 'Shielding anisotropy Δ',
         'asymmetry': 'Asymmetry η', 'span': 'Shielding span Ω',
         'reduced_anisotropy': 'Reduced anisotropy ζ'}
UNITS = {'isotropy': 'ppm', 'anisotropy': 'ppm', 'asymmetry': 'dimensionless',
         'span': 'ppm', 'reduced_anisotropy': 'ppm'}


def reconstruct_symmetric(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return (traceless_cartesian(coefficients[:, 1:])
            + coefficients[:, 0, None, None]/np.sqrt(3)*np.eye(3))


def extract_soprano(tensors, numbers):
    """Use fresh ASE objects so cached diagonalisation never crosses tensors."""
    tensors = np.asarray(tensors, dtype=np.float64)
    assert tensors.shape == (len(numbers), 3, 3)
    # Geometry is unnecessary for these tensor-only getters. All source
    # structures and cached model outputs remain untouched.
    atoms = Atoms(numbers=np.asarray(numbers))
    atoms.new_array('ms', tensors.copy())
    out = {
        'isotropy': np.asarray(MSIsotropy.get(atoms, tag='ms', save_array=False)),
        'anisotropy': np.asarray(MSAnisotropy.get(atoms, tag='ms', force_recalc=True)),
        'asymmetry': np.asarray(MSAsymmetry.get(atoms, tag='ms')),
        'reduced_anisotropy': np.asarray(MSReducedAnisotropy.get(atoms, tag='ms')),
        'span': np.asarray(MSSpan.get(atoms, tag='ms')),
    }
    symmetric = (tensors+tensors.transpose(0, 2, 1))/2
    ascending = np.linalg.eigvalsh(symmetric)
    out['ascending_principal_values'] = ascending
    out['centred_principal_values'] = ascending-ascending.mean(axis=1, keepdims=True)
    hsort = atoms.get_array('ms_diagonal_evals_hsort')
    out['haeberlen_principal_values'] = hsort
    # Check the getters against their stated definitions, including sign.
    isotropy = symmetric.trace(axis1=1, axis2=2)/3
    zeta = hsort[:, 2]-isotropy
    delta = hsort[:, 2]-(hsort[:, 0]+hsort[:, 1])/2
    eta = (hsort[:, 1]-hsort[:, 0])/np.where(zeta == 0, np.inf, zeta)
    np.testing.assert_allclose(out['isotropy'], isotropy, atol=1e-10)
    np.testing.assert_allclose(out['anisotropy'], delta, atol=1e-10)
    np.testing.assert_allclose(out['reduced_anisotropy'], zeta, atol=1e-10)
    np.testing.assert_allclose(out['asymmetry'], eta, atol=1e-10)
    np.testing.assert_allclose(out['anisotropy']*2/3, out['reduced_anisotropy'], atol=1e-10)
    assert np.all(out['asymmetry'] >= -1e-10) and np.all(out['asymmetry'] <= 1+1e-10)
    assert all(np.isfinite(array).all() for array in out.values())
    return out


def prepare_parameters(root, output=None):
    root = Path(root)
    parent = root / 'experiments/symmetric-transfer-N100-data'
    output = Path(output) if output is not None else parent / 'nmr-parameters'
    manifest, pooled = load_review(root, parent)
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {f'predictions-{label}-{split}.npz': sha256(parent / f'predictions-{label}-{split}.npz')
                    for label in RUNS for split in ['train', 'validation', 'test']}
    records, diagnostic_records, sign_bins, shape_records = [], [], [], []
    atom_records = []
    basis_difference = {}
    for split in ['train', 'validation', 'test']:
        data_name = 'valid' if split == 'validation' else split
        frames = read(root / f'data/nmr_symm/{data_name}.xyz', ':')
        if split == 'train':
            frames = [frames[i] for i in manifest['training_indices_zero_based']]
        raw_ms = np.concatenate([a.arrays['ms'].reshape(-1, 3, 3) for a in frames])
        numbers = np.concatenate([a.numbers for a in frames])
        # DFT parameters come from original ms tensors, as in the established
        # isotropy extractor. Soprano diagonalises their symmetric part.
        reference = extract_soprano(raw_ms, numbers)
        base = load_cache(parent, '01', split)
        np.testing.assert_array_equal(numbers, base['Z'])
        reconstructed_reference = extract_soprano(reconstruct_symmetric(base['target']), numbers)
        basis_difference[split] = {}
        for parameter in ['isotropy', 'anisotropy', 'asymmetry', 'span']:
            difference = np.max(np.abs(reference[parameter]-reconstructed_reference[parameter]))
            basis_difference[split][parameter] = float(difference)
            np.testing.assert_allclose(reference[parameter], reconstructed_reference[parameter], rtol=1e-7, atol=1e-7)
        for label in RUNS:
            cache = load_cache(parent, label, split)
            for key in ['Z', 'target', 'frame_index', 'n_atoms']:
                np.testing.assert_array_equal(cache[key], base[key])
            prediction = extract_soprano(reconstruct_symmetric(cache['prediction']), numbers)
            saved = {'Z': numbers, 'frame_index': cache['frame_index'], 'n_atoms': cache['n_atoms']}
            for key in reference:
                saved['reference_' + key] = reference[key]
                saved['prediction_' + key] = prediction[key]
            np.savez_compressed(output / f'parameters-{label}-{split}.npz', **saved)
            for element, atomic_number in Z.items():
                mask = numbers == atomic_number
                for parameter in PARAMETERS + ['span', 'reduced_anisotropy']:
                    result = metrics(reference[parameter][mask], prediction[parameter][mask])
                    result['Bias'] = float(np.mean(prediction[parameter][mask]-reference[parameter][mask]))
                    records.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                                    'Element': element, 'Parameter': parameter, 'Units': UNITS[parameter], **result})
                iso_rmse = records[-5]['RMSE']
                expected_iso = float(pooled[(pooled.Experiment == label) & (pooled.Split == split)
                                           & (pooled.Element == element) & (pooled.Block == 'iso')].RMSE.iloc[0])
                np.testing.assert_allclose(iso_rmse, expected_iso, rtol=1e-7, atol=1e-8)
                delta_y, delta_p = reference['anisotropy'][mask], prediction['anisotropy'][mask]
                eta_y = reference['asymmetry'][mask]
                zeta_y = reference['reduced_anisotropy'][mask]
                zeta_p = prediction['reduced_anisotropy'][mask]
                residual = delta_p-delta_y
                flipped = delta_y*delta_p < 0
                squared = residual**2
                diagnosis = {
                    'Experiment': label, 'Method': LABELS[label], 'Split': split, 'Element': element,
                    'N': int(mask.sum()), 'Opposite_sign_count': int(flipped.sum()),
                    'Opposite_sign_percent': float(flipped.mean()*100),
                    'Delta_MSE_from_opposite_sign_percent': float(100*squared[flipped].sum()/squared.sum()),
                    'Reference_eta_at_least_0.90_among_opposite_sign': int(np.sum(eta_y[flipped] >= .9)),
                    'Reference_eta_at_least_0.95_among_opposite_sign': int(np.sum(eta_y[flipped] >= .95)),
                    'Delta_RMSE_same_sign_only_diagnostic': float(np.sqrt(np.mean(squared[~flipped]))) if (~flipped).any() else None,
                    'Reference_abs_zeta_min_ppm': float(np.min(np.abs(zeta_y))),
                    'Prediction_abs_zeta_min_ppm': float(np.min(np.abs(zeta_p))),
                    'Reference_abs_zeta_below_1ppm': int(np.sum(np.abs(zeta_y) < 1)),
                    'Prediction_abs_zeta_below_1ppm': int(np.sum(np.abs(zeta_p) < 1)),
                    'Reference_exactly_isotropic': int(np.sum(zeta_y == 0)),
                    'Prediction_exactly_isotropic': int(np.sum(zeta_p == 0)),
                    'Reference_abs_zeta_below_5ppm': int(np.sum(np.abs(zeta_y) < 5)),
                }
                diagnostic_records.append(diagnosis)
                principal_error = prediction['centred_principal_values'][mask]-reference['centred_principal_values'][mask]
                shape_records.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                                      'Element': element, 'Centred_principal_value_RMSE_ppm': float(np.sqrt(np.mean(principal_error**2))),
                                      'Centred_principal_value_MAE_ppm': float(np.mean(np.abs(principal_error)))})
                for lower in np.arange(0, 1, .1):
                    upper = lower+.1
                    subset = (eta_y >= lower) & ((eta_y < upper) if upper < .999999 else (eta_y <= 1+1e-10))
                    sign_bins.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                                      'Element': element, 'Reference_eta_lower': float(lower),
                                      'Reference_eta_upper': float(upper), 'Reference_eta_mid': float(lower+.05),
                                      'N': int(subset.sum()), 'Opposite_sign_count': int(flipped[subset].sum()),
                                      'Opposite_sign_percent': float(flipped[subset].mean()*100) if subset.any() else None})
                # Retain matched per-atom quantities for follow-up without
                # requiring a re-run of either Soprano or model inference.
                indices = np.flatnonzero(mask)
                offsets = np.r_[0, np.cumsum(cache['n_atoms'])[:-1]]
                for parameter in PARAMETERS:
                    error = prediction[parameter][mask]-reference[parameter][mask]
                    for i in np.argsort(np.abs(error))[-10:][::-1]:
                        index = int(indices[i])
                        frame = int(cache['frame_index'][index])
                        atom_records.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                            'Element': element, 'Parameter': parameter, 'Frame_index_zero_based': frame,
                            'Atom_index_in_frame_zero_based': int(index-offsets[frame]),
                            'DFT': float(reference[parameter][index]), 'Predicted': float(prediction[parameter][index]),
                            'Error': float(error[i]), 'Reference_eta': float(reference['asymmetry'][index]),
                            'Prediction_eta': float(prediction['asymmetry'][index]),
                            'Reference_Delta_ppm': float(reference['anisotropy'][index]),
                            'Prediction_Delta_ppm': float(prediction['anisotropy'][index]),
                            'Opposite_Delta_sign': bool(reference['anisotropy'][index]*prediction['anisotropy'][index] < 0)})
            print('Soprano parameters:', label, split, len(numbers), 'atoms', flush=True)
    pd.DataFrame(records).to_csv(output / 'parameter-metrics.csv', index=False)
    pd.DataFrame(diagnostic_records).to_csv(output / 'convention-diagnostics.csv', index=False)
    pd.DataFrame(sign_bins).to_csv(output / 'sign-differences-by-asymmetry.csv', index=False)
    pd.DataFrame(shape_records).to_csv(output / 'principal-value-metrics.csv', index=False)
    pd.DataFrame(atom_records).to_csv(output / 'largest-parameter-errors.csv', index=False)
    result = {'analysis_date': '2026-11-06', 'soprano_version': version('soprano'),
              'parent_manifest_sha256': sha256(parent / 'manifest.json'), 'input_sha256': input_hashes,
              'data_sha256': manifest['data_sha256'], 'helper_sha256': sha256(Path(__file__)),
              'parameters': {'isotropy': 'MSIsotropy.get; sigma_iso; ppm; no reference -> shielding',
                             'anisotropy': 'MSAnisotropy.get; Delta_sigma = sigma_ZZ - (sigma_XX + sigma_YY)/2; ppm',
                             'asymmetry': 'MSAsymmetry.get; eta = (sigma_YY - sigma_XX)/(sigma_ZZ - sigma_iso); dimensionless',
                             'reduced_anisotropy': 'MSReducedAnisotropy.get; zeta_sigma = 2 Delta_sigma/3; ppm',
                             'span': 'MSSpan.get; Omega = sigma_max - sigma_min; ppm'},
              'ordering': 'Haeberlen: |sigma_ZZ-iso| >= |sigma_XX-iso| >= |sigma_YY-iso|',
              'reference': 'Original stored DFT ms tensors, diagonalised through their symmetric part; verified against reconstructed coefficient targets.',
              'reconstruction_max_abs_difference': basis_difference,
              'fresh_ASE_object_for_each_tensor_set': True,
              'original_isotropic_rmse_reproduced': True,
              'primary_metrics': 'All atom predictions retained, signed Delta unmodified, no clipping, reference fitting, sign alignment or sample exclusions.',
              'diagnostics': 'Opposite Delta signs can include a Haeberlen ordering-boundary crossing and genuine tensor error; no claim all are harmless. Same-sign-only score is diagnostic, never the headline.',
              'near_isotropy': 'Counts reported at |zeta| < 1 and < 5 ppm; exact-isotropy eta uses Soprano zero convention. No filtering.',
              'shape_orientation': 'These scalar parameters ignore principal-axis orientation; retain original full-tensor errors separately.',
              'no_inference_or_retraining': True,
              'output_sha256': {p.name: sha256(p) for p in output.iterdir() if p.suffix in ['.csv', '.npz']}}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return output


def load_parameter_metrics(root, output=None):
    root = Path(root)
    parent = root / 'experiments/symmetric-transfer-N100-data'
    output = Path(output) if output is not None else parent / 'nmr-parameters'
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert sha256(parent / name) == digest, name
    for name, digest in manifest['output_sha256'].items():
        assert sha256(output / name) == digest, name
    return pd.read_csv(output / 'parameter-metrics.csv', dtype={'Experiment': str})


def parameter_cache(data, label, split):
    with np.load(Path(data) / f'parameters-{label}-{split}.npz', allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def parameter_limits(data, parameter, element, split):
    if parameter == 'asymmetry':
        return -.035, 1.035
    pairs = []
    for label in RUNS:
        cache = parameter_cache(data, label, split)
        mask = cache['Z'] == Z[element]
        pairs.extend([cache['reference_'+parameter][mask], cache['prediction_'+parameter][mask]])
    low, high = min(a.min() for a in pairs), max(a.max() for a in pairs)
    padding = .045*(high-low)
    return float(low-padding), float(high+padding)


def parameter_panel(ax, cache, element, parameter, limits, color, title):
    mask = cache['Z'] == Z[element]
    y = cache['reference_'+parameter][mask]
    p = cache['prediction_'+parameter][mask]
    flipped = y*p < 0 if parameter == 'anisotropy' else np.zeros(y.shape, dtype=bool)
    ax.plot(limits, limits, '--', lw=1, color='#82909B', zorder=1)
    ax.scatter(y[~flipped], p[~flipped], s=6, alpha=.19, marker='o', color=color,
               linewidths=0, edgecolors='none', rasterized=True, zorder=2)
    if flipped.any():
        ax.scatter(y[flipped], p[flipped], s=6, alpha=.38, marker='o', color='#B74757',
                   linewidths=0, edgecolors='none', rasterized=True, zorder=2)
    unit = '' if parameter == 'asymmetry' else ' ppm'
    axisunit = 'dimensionless' if parameter == 'asymmetry' else 'ppm'
    ax.set(xlim=limits, ylim=limits, title=title,
           xlabel=f'DFT ({axisunit})', ylabel=f'Predicted ({axisunit})')
    ax.set_aspect('equal', adjustable='box')
    m = metrics(y, p)
    text = f"RMSE  {m['RMSE']:.3f}{unit}\nMAE     {m['MAE']:.3f}{unit}\nR²        {m['R2']:.3f}"
    ax.text(.035, .965, text, transform=ax.transAxes, va='top', ha='left', fontsize=8.7,
            bbox={'facecolor':'white', 'edgecolor':'#D8E1E7', 'alpha':.96, 'boxstyle':'round,pad=.35'}, zorder=3)
    ax.grid(alpha=.13, zorder=0)


def sign_legend(figure):
    from matplotlib.lines import Line2D
    point = Line2D([], [], ls='', marker='o', ms=4, color='#B74757',
                   label='Opposite signs of predicted and DFT anisotropy')
    figure.legend(handles=[point], loc='lower center', bbox_to_anchor=(.5,-.025), frameon=False, fontsize=9)


def plot_parameter_parity(data, parameter, split='test'):
    import matplotlib.pyplot as plt
    style()
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 8.1), layout='constrained')
    for row, element in enumerate(['Si', 'O']):
        limits = parameter_limits(data, parameter, element, split)
        for col, label in enumerate(RUNS):
            parameter_panel(axes[row,col], parameter_cache(data,label,split), element, parameter,
                            limits, COLORS[label], f'{LABELS[label]} · {element}')
    name = NAMES[parameter][0].lower()+NAMES[parameter][1:]
    figure.suptitle(f'{split.capitalize()} {name} parity at N100', fontsize=13)
    if parameter == 'anisotropy':
        sign_legend(figure)
    return figure


def plot_run_parameter_parity(data, label, split='test'):
    import matplotlib.pyplot as plt
    style()
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), layout='constrained')
    cache = parameter_cache(data, label, split)
    for row, element in enumerate(['Si', 'O']):
        for col, parameter in enumerate(PARAMETERS):
            parameter_panel(axes[row,col], cache, element, parameter,
                            parameter_limits(data,parameter,element,split), COLORS[label],
                            f'{element} · {NAMES[parameter]}')
    figure.suptitle(f'{LABELS[label]}: {split} NMR parameter parity at N100', fontsize=13)
    sign_legend(figure)
    return figure


def plot_parameter_rmse(table, split='test'):
    import matplotlib.pyplot as plt
    style()
    frame = table[(table.Split == split) & table.Parameter.isin(PARAMETERS)]
    figure, axes = plt.subplots(2,3,figsize=(12.1,7.2),layout='constrained')
    for row, element in enumerate(['Si','O']):
        for col, parameter in enumerate(PARAMETERS):
            ax = axes[row,col]
            values = [float(frame[(frame.Element==element)&(frame.Parameter==parameter)&(frame.Experiment==label)].RMSE.iloc[0]) for label in RUNS]
            bars = ax.bar(range(3), values, color=list(COLORS.values()), width=.65)
            ax.bar_label(bars, fmt='%.3f' if parameter=='asymmetry' else '%.2f', padding=4, fontsize=10)
            maximum = float(frame[frame.Parameter==parameter].RMSE.max())
            ax.set(title=f'{element} · {NAMES[parameter]}', ylim=(0,maximum*1.22),
                   ylabel='RMSE (dimensionless)' if parameter=='asymmetry' else 'RMSE (ppm)')
            ax.set_xticks(range(3), ['Direct\ntraining','Full fine\ntuning','ELoRA'],fontsize=9)
            ax.grid(axis='y',alpha=.16)
            ax.set_axisbelow(True)
    figure.suptitle(f'{split.capitalize()} NMR parameter RMSE at N100',fontsize=13)
    return figure


def plot_sign_diagnostics(data, split='test'):
    import matplotlib.pyplot as plt
    style()
    frame = pd.read_csv(Path(data)/'sign-differences-by-asymmetry.csv',dtype={'Experiment':str})
    frame = frame[frame.Split==split]
    maximum = max(1., float(frame.Opposite_sign_percent.max()))
    figure, axes = plt.subplots(1,2,figsize=(10.8,3.8),sharey=True,layout='constrained')
    for ax,element in zip(axes,['Si','O']):
        for label in RUNS:
            values=frame[(frame.Element==element)&(frame.Experiment==label)].sort_values('Reference_eta_mid')
            ax.plot(values.Reference_eta_mid,values.Opposite_sign_percent,'o-',ms=4,lw=1.6,color=COLORS[label],label=LABELS[label])
        ax.set(title=element,xlabel='DFT asymmetry η (binned)',ylabel='Opposite anisotropy signs (%)',xlim=(0,1),ylim=(0,min(100,maximum*1.12)))
        ax.grid(alpha=.16)
    axes[0].legend(frameon=False,fontsize=9)
    figure.suptitle(f'{split.capitalize()} anisotropy sign differences by asymmetry',fontsize=13)
    return figure
