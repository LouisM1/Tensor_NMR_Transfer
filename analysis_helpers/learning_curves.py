"""Experiment 07 learning curves for Soprano NMR parameters.

Reads the metrics written by ``scripts/evaluate_experiment07.py`` and adds an
element-balanced standardised MSE per parameter:

    balanced = (MSE_Si / var_Si + MSE_O / var_O) / 2

where each variance is that parameter's DFT variance over the full
800-structure training pool. The fixed denominators mirror the training loss
normalisers and use no validation or test labels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nmr_parameters import NAMES, UNITS, extract_soprano, reconstruct_symmetric
from .symmetric_transfer_review import style

METHODS = ['full', 'elora', 'last', 'head', 'scratch']
LABEL = {'full': 'Full fine-tuning', 'elora': 'ELoRA', 'last': 'Last layer + NMR head',
         'head': 'NMR head only', 'scratch': 'Direct training'}
COLOR = {'full': '#7548A3', 'elora': '#008C83', 'last': '#377BB5', 'head': '#7D858D',
         'scratch': '#D18B35'}
MARKER = {'full': 'D', 'elora': 'o', 'last': 's', 'head': 'v', 'scratch': '^'}
SIZES = [10, 50, 100, 200, 800]
ELEMENTS = {'Si': 14, 'O': 8}
ELEMENT_NAMES = {'Si': 'Silicon', 'O': 'Oxygen'}
ALL_PARAMETERS = ['isotropy', 'abs_anisotropy', 'asymmetry', 'skew', 'anisotropy']

__all__ = ['METHODS', 'LABEL', 'SIZES', 'ALL_PARAMETERS', 'training_pool_variance',
           'score_saved_predictions', 'load_curves', 'sign_flip_table', 'run_summary',
           'rmse_table', 'plot_parameter_curves', 'save_figure']


def training_pool_variance(root, data):
    """Per-element DFT variance of each parameter over the 800 training structures."""
    cache = Path(data) / 'training-pool-parameter-variance.json'
    if cache.exists():
        stored = json.loads(cache.read_text())
        if all(p in stored.get(el, {}) for el in ELEMENTS for p in ALL_PARAMETERS):
            return stored
    from ase.io import read
    frames = read(Path(root) / 'data/nmr_symm/train.xyz', ':')
    ms = np.concatenate([np.asarray(a.arrays['ms']).reshape(-1, 3, 3) for a in frames])
    numbers = np.concatenate([a.numbers for a in frames])
    params = extract_soprano((ms + ms.transpose(0, 2, 1)) / 2, numbers)
    result = {'structures': len(frames)}
    for element, z in ELEMENTS.items():
        mask = numbers == z
        result[element] = {p: float(np.var(params[p][mask])) for p in ALL_PARAMETERS}
    cache.write_text(json.dumps(result, indent=2) + '\n')
    return result


def score_saved_predictions(data):
    """Compute all Soprano parameter RMSEs and sign-flip counts from cached .npz files."""
    data = Path(data)
    selected = data / 'selected-runs.json'
    if not selected.exists():
        selected = data / 'selected-runs-partial.json'
    runs = json.loads(selected.read_text())
    metric_rows, flip_rows = [], []
    ref_cache = {}
    for split in ('validation', 'test'):
        for key in runs:
            method, n_str = key.split(':')
            n = int(n_str)
            cache = data / f'predictions-{method}-N{n}-{split}.npz'
            with np.load(cache, allow_pickle=False) as stored:
                Z_arr = stored['Z']
                if split not in ref_cache:
                    ref_cache[split] = extract_soprano(stored['target_cartesian'], Z_arr)
                reference = ref_cache[split]
                prediction = extract_soprano(reconstruct_symmetric(stored['prediction']), Z_arr)
            for element, z in ELEMENTS.items():
                mask = Z_arr == z
                for quantity in ALL_PARAMETERS:
                    residual = prediction[quantity][mask] - reference[quantity][mask]
                    assert np.isfinite(residual).all()
                    metric_rows.append({
                        'method': method, 'n': n, 'split': split,
                        'element': element, 'quantity': quantity,
                        'atoms': int(mask.sum()),
                        'rmse': float(np.sqrt(np.mean(residual**2))),
                    })
                flipped = (reference['anisotropy'][mask] * prediction['anisotropy'][mask]) < 0
                flip_rows.append({
                    'method': method, 'n': n, 'split': split,
                    'element': element, 'atoms': int(mask.sum()),
                    'opposite_sign_count': int(flipped.sum()),
                    'opposite_sign_percent': float(100.0 * flipped.mean()),
                })
    metrics_df = pd.DataFrame(metric_rows)
    flips_df = pd.DataFrame(flip_rows)
    # Cross-check against metrics.csv written by evaluate_experiment07.py if present.
    base_path = data / 'metrics.csv'
    if base_path.exists():
        base = pd.read_csv(base_path)
        merged = base.merge(metrics_df, on=['method', 'n', 'split', 'element', 'quantity'],
                            suffixes=('_base', '_ext'))
        assert len(merged) == len(base)
        np.testing.assert_allclose(merged.rmse_base, merged.rmse_ext, rtol=1e-12, atol=1e-12)
    metrics_df.to_csv(data / 'metrics-all-parameters.csv', index=False)
    flips_df.to_csv(data / 'anisotropy-sign-flips.csv', index=False)
    return metrics_df, flips_df


def load_curves(root, data, recompute_extended=False):
    """Long table: method, n, split, element (Si, O or balanced), quantity, value, metric."""
    data = Path(data)
    ext_path = data / 'metrics-all-parameters.csv'
    base_path = data / 'metrics.csv'
    if recompute_extended or not ext_path.exists() or (
        base_path.exists() and ext_path.stat().st_mtime < base_path.stat().st_mtime
    ):
        table, _ = score_saved_predictions(data)
        source_name = ext_path.name
    else:
        table = pd.read_csv(ext_path)
        source_name = ext_path.name
    variance = training_pool_variance(root, data)
    table['metric'] = 'rmse'
    table['value'] = table['rmse']
    table['standardised_mse'] = [r.rmse**2 / variance[r.element][r.quantity]
                                 for r in table.itertuples()]
    balanced = (table.groupby(['method', 'n', 'split', 'quantity'], as_index=False)
                .agg(value=('standardised_mse', 'mean'), elements=('element', 'nunique')))
    assert (balanced.elements == 2).all()
    balanced = balanced.drop(columns='elements').assign(element='balanced', metric='balanced_mse')
    columns = ['method', 'n', 'split', 'element', 'quantity', 'metric', 'value']
    return pd.concat([table[columns], balanced[columns]], ignore_index=True), source_name


def sign_flip_table(data, split='test'):
    """Opposite-sign Δσ percentage per element, method and N."""
    flips = pd.read_csv(Path(data) / 'anisotropy-sign-flips.csv')
    frame = flips[flips.split == split]
    pivot = frame.pivot_table(index=['element', 'method'], columns='n', values='opposite_sign_percent')
    pivot = pivot.reindex(columns=SIZES)
    order = [(e, m) for e in ['Si', 'O'] for m in METHODS if (e, m) in pivot.index]
    pivot = pivot.loc[order]
    pivot.index = pd.MultiIndex.from_tuples([(e, LABEL[m]) for e, m in order],
                                            names=['Element', 'Method'])
    pivot.columns.name = 'N'
    return pivot


def run_summary(root, data):
    """Selected run per method and N with validation-selected and final epochs."""
    data = Path(data)
    selected = data / 'selected-runs.json'
    if not selected.exists():
        selected = data / 'selected-runs-partial.json'
    runs = json.loads(selected.read_text())
    rows = []
    for key, directory in runs.items():
        method, n = key.split(':')
        directory = Path(directory)
        audit = directory / 'experiment07-complete.json'
        best = last = None
        if audit.exists():
            record = json.loads(audit.read_text())
            best, last = record['best_epoch'], record['last_epoch']
        rows.append({'Method': LABEL[method], 'method': method, 'N': int(n),
                     'Run': directory.name, 'Best epoch': best, 'Last epoch': last})
    for method in METHODS:
        for n in SIZES:
            if f'{method}:{n}' not in runs:
                rows.append({'Method': LABEL[method], 'method': method, 'N': n,
                             'Run': 'missing (did not finish)', 'Best epoch': None,
                             'Last epoch': None})
    frame = pd.DataFrame(rows)
    frame['order'] = frame.method.map(METHODS.index)
    return frame.sort_values(['order', 'N']).drop(columns=['order', 'method']).reset_index(drop=True)


def rmse_table(curves, quantity, split='test'):
    frame = curves[(curves.split == split) & (curves.quantity == quantity)]
    pivot = frame.pivot_table(index=['element', 'method'], columns='n', values='value')
    pivot = pivot.reindex(columns=SIZES)
    order = [(e, m) for e in ['Si', 'O', 'balanced'] for m in METHODS if (e, m) in pivot.index]
    pivot = pivot.loc[order]
    pivot.index = pd.MultiIndex.from_tuples([(e, LABEL[m]) for e, m in order],
                                            names=['Element', 'Method'])
    pivot.columns.name = 'N'
    return pivot


def _log_axes(ax, y_values):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(SIZES))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(8, 1000)
    lo, hi = np.nanmin(y_values), np.nanmax(y_values)
    lo, hi = lo / 1.25, hi * 1.25
    ax.set_ylim(lo, hi)
    # Label 1-2-5 steps; add 3 and 7 when the range is under one decade.
    steps = (1, 2, 5) if hi / lo > 8 else (1, 1.5, 2, 3, 5, 7)
    ticks = [m * 10.0**k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
             for m in steps if lo <= m * 10.0**k <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='both', axis='y', alpha=.16)
    ax.grid(which='major', axis='x', alpha=.12)
    ax.set_axisbelow(True)


def plot_parameter_curves(curves, quantity, split='test'):
    """Silicon RMSE, oxygen RMSE and balanced standardised MSE versus N."""
    import matplotlib.pyplot as plt
    style()
    frame = curves[(curves.split == split) & (curves.quantity == quantity)]
    unit = UNITS[quantity]
    panels = [('Si', ELEMENT_NAMES['Si'], f'RMSE ({unit})'),
              ('O', ELEMENT_NAMES['O'], f'RMSE ({unit})'),
              ('balanced', 'Balanced error', 'Element-balanced standardised MSE')]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    figure.subplots_adjust(left=.06, right=.99, bottom=.12, top=.78, wspace=.28)
    for ax, (element, title, ylabel) in zip(axes, panels):
        values = frame[frame.element == element]
        for method in METHODS:
            line = values[values.method == method].sort_values('n')
            ax.plot(line.n, line.value, color=COLOR[method], marker=MARKER[method],
                    label=LABEL[method], lw=2, ms=6.5, mec='white', mew=.8)
        _log_axes(ax, values.value.to_numpy())
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold')
        ax.set(xlabel='Training structures, N', ylabel=ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(.055, .915), ncol=5,
                  frameon=False, fontsize=10.5, handlelength=2.4)
    figure.suptitle(f'{NAMES[quantity]}: {split} error versus training-set size',
                    x=.06, y=.985, ha='left', fontsize=16, fontweight='bold')
    return figure


def save_figure(figure, directory, name):
    """Save learning-curve figures as high-DPI PNG, PDF and SVG."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / (name + '.png'), dpi=240, bbox_inches='tight')
    figure.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    figure.savefig(directory / (name + '.svg'), bbox_inches='tight')
