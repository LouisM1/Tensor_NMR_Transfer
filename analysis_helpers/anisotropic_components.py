"""Component-resolved symmetric-traceless NMR errors from saved predictions.

Fixed e3nn 0.4.4 basis; physical values in ppm. Component diagnostics depend
on the stored coordinate frame. The pooled five-component RMSE remains the
rotation-independent summary. No inference, fitting or calibration occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .symmetric_transfer_review import COLORS, LABELS, RUNS, Z, load_cache, load_review, sha256, style

SQRT2 = np.sqrt(2.)
SQRT6 = np.sqrt(6.)
BASIS = np.zeros((5, 3, 3), dtype=np.float64)
BASIS[0, 0, 2] = BASIS[0, 2, 0] = 1 / SQRT2
BASIS[1, 0, 1] = BASIS[1, 1, 0] = 1 / SQRT2
BASIS[2] = np.diag([-1 / SQRT6, 2 / SQRT6, -1 / SQRT6])
BASIS[3, 1, 2] = BASIS[3, 2, 1] = 1 / SQRT2
BASIS[4] = np.diag([-1 / SQRT2, 0, 1 / SQRT2])
COMPONENT_NAMES = ['q1', 'q2', 'q3', 'q4', 'q5']
COMPONENT_FORMULAS = ['sqrt(2) A_xz', 'sqrt(2) A_xy', '(2 A_yy - A_xx - A_zz) / sqrt(6)',
                      'sqrt(2) A_yz', '(A_zz - A_xx) / sqrt(2)']
COMPONENT_LABELS = [r'$q_1$' + '\n' + r'$\sqrt{2}A_{xz}$',
                    r'$q_2$' + '\n' + r'$\sqrt{2}A_{xy}$',
                    r'$q_3$' + '\n' + r'$(2A_{yy}-A_{xx}-A_{zz})/\sqrt{6}$',
                    r'$q_4$' + '\n' + r'$\sqrt{2}A_{yz}$',
                    r'$q_5$' + '\n' + r'$(A_{zz}-A_{xx})/\sqrt{2}$']


def traceless_cartesian(q):
    """Reconstruct A from signed q values before calculating Cartesian errors."""
    q = np.asarray(q, dtype=np.float64)
    assert q.shape[-1] == 5
    return np.einsum('...k,kij->...ij', q, BASIS)


def statistics(reference, prediction):
    residual = prediction - reference
    squared = residual**2
    count = len(residual)
    tail_count = max(1, int(np.ceil(.01 * count)))
    total = squared.sum()
    sst = np.sum((reference-reference.mean())**2)
    return {'N': int(count), 'RMSE_ppm': float(np.sqrt(squared.mean())),
            'MAE_ppm': float(np.abs(residual).mean()), 'Bias_ppm': float(residual.mean()),
            'P95_absolute_error_ppm': float(np.quantile(np.abs(residual), .95)),
            'P99_absolute_error_ppm': float(np.quantile(np.abs(residual), .99)),
            'Reference_RMS_ppm': float(np.sqrt(np.mean(reference**2))),
            'R2': float(1-total/sst) if sst else None,
            'Worst_1pct_atoms': int(tail_count),
            'Worst_1pct_MSE_share_percent': float(100*np.sort(squared)[-tail_count:].sum()/total) if total else 0.}


def prepare_components(root, output=None):
    root = Path(root)
    parent = root / 'experiments/symmetric-transfer-N100-data'
    output = Path(output) if output is not None else parent / 'anisotropic-components'
    parent_manifest, pooled = load_review(root, parent)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {f'predictions-{label}-{split}.npz': sha256(parent / f'predictions-{label}-{split}.npz')
              for label in RUNS for split in ['train', 'validation', 'test']}
    inputs['metrics.csv'] = sha256(parent / 'metrics.csv')
    # Confirm the explicit Cartesian basis against the installed e3nn converter.
    import torch
    from e3nn.io import CartesianTensor
    coefficients = torch.zeros((5, 6), dtype=torch.float64)
    coefficients[:, 1:] = torch.eye(5, dtype=torch.float64)
    library_basis = CartesianTensor('ij=ji').to_cartesian(coefficients).numpy()
    np.testing.assert_allclose(BASIS, library_basis, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.einsum('kij,lij->kl', BASIS, BASIS), np.eye(5), atol=1e-14)

    coefficient_rows, cartesian_rows, outlier_rows = [], [], []
    for split in ['train', 'validation', 'test']:
        for label in RUNS:
            cache = load_cache(parent, label, split)
            for element, atomic_number in Z.items():
                indices = np.flatnonzero(cache['Z'] == atomic_number)
                reference = cache['target'][indices, 1:]
                prediction = cache['prediction'][indices, 1:]
                error = prediction-reference
                reference_a, prediction_a = traceless_cartesian(reference), traceless_cartesian(prediction)
                error_a = prediction_a-reference_a
                np.testing.assert_allclose(error_a, error_a.transpose(0, 2, 1), atol=1e-12)
                np.testing.assert_allclose(np.trace(error_a, axis1=1, axis2=2), 0, atol=1e-12)
                np.testing.assert_allclose(np.sum(error_a**2, axis=(1, 2)), np.sum(error**2, axis=1), rtol=1e-12, atol=1e-10)
                component_mse = np.mean(error**2, axis=0)
                expected = float(pooled[(pooled.Experiment == label) & (pooled.Split == split)
                                        & (pooled.Element == element) & (pooled.Block == 'aniso')].RMSE.iloc[0])
                np.testing.assert_allclose(np.sqrt(component_mse.mean()), expected, rtol=1e-12)
                offsets = np.r_[0, np.cumsum(cache['n_atoms'])[:-1]]
                for k, name in enumerate(COMPONENT_NAMES):
                    values = statistics(reference[:, k], prediction[:, k])
                    coefficient_rows.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                        'Element': element, 'Component': name, 'Formula': COMPONENT_FORMULAS[k], **values,
                        'Component_MSE_share_percent': float(100*component_mse[k]/component_mse.sum())})
                    worst = np.argsort(np.abs(error[:, k]))[-10:][::-1]
                    for rank, index in enumerate(worst, 1):
                        global_index = int(indices[index])
                        frame = int(cache['frame_index'][global_index])
                        outlier_rows.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                            'Element': element, 'Component': name, 'Error_rank': rank,
                            'Frame_index_zero_based': frame,
                            'Atom_index_in_frame_zero_based': int(global_index-offsets[frame]),
                            'DFT_ppm': float(reference[index, k]), 'Predicted_ppm': float(prediction[index, k]),
                            'Signed_error_ppm': float(error[index, k])})
                for i, axis_i in enumerate('xyz'):
                    for j, axis_j in enumerate('xyz'):
                        cartesian_rows.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                            'Element': element, 'Row': axis_i, 'Column': axis_j,
                            'Component': 'A_' + axis_i + axis_j,
                            **statistics(reference_a[:, i, j], prediction_a[:, i, j])})
    pd.DataFrame(coefficient_rows).to_csv(output / 'coefficient-metrics.csv', index=False)
    pd.DataFrame(cartesian_rows).to_csv(output / 'cartesian-metrics.csv', index=False)
    pd.DataFrame(outlier_rows).to_csv(output / 'largest-component-errors.csv', index=False)
    manifest = {'analysis_date': '2026-11-06', 'parent_manifest_sha256': sha256(parent / 'manifest.json'),
                'input_sha256': inputs, 'helper_sha256': sha256(Path(__file__)),
                'basis': COMPONENT_FORMULAS, 'basis_matches_installed_e3nn': True,
                'pooled_rmse_recovered_from_component_mse': True,
                'traceless_symmetric_and_frobenius_checks_passed': True,
                'coordinate_scope': 'Component scores refer to the stored DFT coordinate frame; individual scores are not rotation-invariant.',
                'cartesian_scope': 'A = S - trace(S)/3 I; no isotropic contribution. Reconstruct signed tensor errors before taking RMSE.',
                'tail_definition': 'For each element/model/component separately, ceil(1% of atoms) with largest squared residual; divide their SSE by component SSE.',
                'outlier_indices': 'Zero-based frame in the stored split (selected seed-42 order for train) and zero-based atom within that frame.',
                'models_or_original_predictions_changed': False,
                'output_sha256': {p.name: sha256(p) for p in output.glob('*.csv')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return output


def load_components(root, output=None):
    root = Path(root)
    parent = root / 'experiments/symmetric-transfer-N100-data'
    output = Path(output) if output is not None else parent / 'anisotropic-components'
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert sha256(parent / name) == digest, name
    for name, digest in manifest['output_sha256'].items():
        assert sha256(output / name) == digest, name
    return (pd.read_csv(output / 'coefficient-metrics.csv', dtype={'Experiment': str}),
            pd.read_csv(output / 'cartesian-metrics.csv', dtype={'Experiment': str}))


def plot_component_rmse(components, split='test'):
    import matplotlib.pyplot as plt
    style()
    frame = components[components.Split == split]
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), sharey=True, layout='constrained')
    x = np.arange(5)
    maximum = float(frame.RMSE_ppm.max())
    for ax, element in zip(axes, ['Si', 'O']):
        for offset, label in enumerate(RUNS):
            values = frame[(frame.Element == element) & (frame.Experiment == label)].set_index('Component').loc[COMPONENT_NAMES].RMSE_ppm
            bars = ax.bar(x+(offset-1)*.25, values, width=.23, color=COLORS[label], label=LABELS[label])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8.2)
        ax.set_xticks(x, COMPONENT_LABELS, fontsize=8.7)
        ax.set(title=f'{element} anisotropic component RMSE', ylabel='RMSE (ppm)', ylim=(0, maximum*1.3))
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    figure.suptitle(f'{split.capitalize()} anisotropic coefficient errors at N100', fontsize=13)
    return figure


def plot_cartesian_rmse(cartesian, split='test'):
    import matplotlib.pyplot as plt
    style()
    frame = cartesian[cartesian.Split == split]
    maximum = float(frame.RMSE_ppm.max())
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.7), layout='constrained')
    image = None
    for row, element in enumerate(['Si', 'O']):
        for col, label in enumerate(RUNS):
            ax = axes[row, col]
            values = frame[(frame.Element == element) & (frame.Experiment == label)].pivot(index='Row', columns='Column', values='RMSE_ppm').loc[list('xyz'), list('xyz')].to_numpy()
            image = ax.imshow(values, vmin=0, vmax=maximum, cmap='YlOrRd')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=12,
                            color='white' if values[i,j] > maximum*.63 else '#243647')
            ax.set(xticks=[0,1,2], xticklabels=list('xyz'), yticks=[0,1,2], yticklabels=list('xyz'),
                   title=f'{LABELS[label]} · {element}')
            ax.tick_params(length=0)
    figure.colorbar(image, ax=axes, shrink=.83, label='Anisotropic Cartesian RMSE (ppm)', pad=.025)
    figure.suptitle(f'{split.capitalize()} traceless tensor component RMSE at N100', fontsize=13)
    return figure


def plot_error_concentration(components, split='test'):
    import matplotlib.pyplot as plt
    style()
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 3.65), layout='constrained')
    frame = components[components.Split == split]
    image = None
    for ax, element in zip(axes, ['Si', 'O']):
        values = frame[frame.Element == element].pivot(index='Experiment', columns='Component', values='Worst_1pct_MSE_share_percent').loc[list(RUNS), COMPONENT_NAMES].to_numpy()
        image = ax.imshow(values, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
        for i in range(3):
            for j in range(5):
                ax.text(j, i, f'{values[i,j]:.1f}%', ha='center', va='center', fontsize=10.5,
                        color='white' if values[i,j] > 63 else '#243647')
        ax.set(xticks=range(5), xticklabels=COMPONENT_NAMES, yticks=range(3),
               yticklabels=list(LABELS.values()), title=element)
        ax.tick_params(length=0)
    figure.colorbar(image, ax=axes, shrink=.9, label='Squared error from worst 1% of atoms (%)', pad=.02)
    figure.suptitle(f'{split.capitalize()} anisotropic error concentration at N100', fontsize=13)
    return figure
