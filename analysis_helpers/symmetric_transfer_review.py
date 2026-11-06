"""Reproducible N100 symmetric-NMR evaluation and transfer diagnostics.

Models and data are read-only. Predictions are from validation-selected exports.
0e is reported as isotropic shielding (coefficient / sqrt(3)); 2e pools the
five orthonormal traceless coefficients. No fitting or calibration occurs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import nmr_helpers  # graph-pes/e3nn compatibility registration before e3nn.
import numpy as np
import pandas as pd
import torch
import yaml
from ase.io import read
from e3nn.io import CartesianTensor
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.nn import count_used_parameters, PerElementParameter
from graph_pes.utils.sampling import SequenceSampler


RUNS = {
    '01': '01-SymmetricTensor-Scratch-N100-9245885',
    '05': '05-SymmetricTensor-FullFineTune-N100-9274115',
    '06': '06-SymmetricTensor-ELoRA-N100-R4-9274116',
}
LABELS = {'01': 'Direct training', '05': 'Full fine-tuning', '06': 'ELoRA'}
COLORS = {'01': '#65758B', '05': '#CB7044', '06': '#187F83'}
SOURCE = 'graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model-e3nn.pt'
SOURCE_SHA = 'c5229e19d27f1ed965bb507e48caa21bf0d837b5613f8efb44915fcf39513654'
BACKBONE = ('Z_embedding', 'initial_node_embedding', 'edge_embedding', 'layers')
PANELS = [('Si', 'iso'), ('Si', 'aniso'), ('O', 'iso'), ('O', 'aniso')]
Z = {'Si': 14, 'O': 8}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def same_state(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_state(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same_state(x, y)
    else:
        assert a == b


def architecture(config):
    model = config['model']
    while '+nmr_helpers.symmetric_shielding.SymmetricTensorNequIP' not in model:
        assert len(model) == 1
        model = next(iter(model.values()))['model']
    return model


def load_best(root, label):
    directory = Path(root) / 'graph-pes-results' / RUNS[label]
    model = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False).eval()
    best = torch.load(directory / 'checkpoints/best.ckpt', map_location='cpu', weights_only=False)
    last = torch.load(directory / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    expected = {k[6:]: v for k, v in best['state_dict'].items() if k.startswith('model.')}
    same_state(model.state_dict(), expected)
    # PerElementParameter's used-element registry is Python metadata lost on
    # pickle reload. Restore it for accurate parameter counts, without fitting.
    for parameter in model.parameters():
        if isinstance(parameter, PerElementParameter):
            parameter.register_elements([8, 14])
    same_state(model.state_dict(), expected)
    stopping = next(v for k, v in last['callbacks'].items() if k.startswith('EarlyStopping'))
    summary = yaml.safe_load((directory / 'summary.yaml').read_text())
    config = yaml.safe_load((directory / 'train-config.yaml').read_text())
    assert stopping['wait_count'] == stopping['patience'] == 300
    assert np.isclose(float(stopping['best_val_loss']), summary['early_stopping/best_valid_loss'])
    assert config['data']['test'] is None
    if label != '01':
        assert model._symmetric_transfer_source_sha256 == SOURCE_SHA
    audit = {
        'run_id': RUNS[label], 'label': LABELS[label],
        'model_sha256': sha256(directory / 'model.pt'),
        'checkpoint_sha256': sha256(directory / 'checkpoints/best.ckpt'),
        'last_checkpoint_sha256': sha256(directory / 'checkpoints/last.ckpt'),
        'config_sha256': sha256(directory / 'train-config.yaml'),
        'export_exactly_matches_best_checkpoint': True,
        'element_registry_restored_without_refitting_or_state_change': True,
        'best_epoch': best['epoch'] + 1, 'last_epoch': last['epoch'] + 1,
        'best_global_step': best['global_step'], 'last_global_step': last['global_step'],
        'best_recorded_validation_loss': float(stopping['best_val_loss']),
        'last_recorded_validation_loss': summary['valid/loss/total'],
        'early_stopping_checks_since_best': stopping['wait_count'],
        'early_stopping_patience': stopping['patience'], 'early_stopping_reached': True,
        'initial_lr': config['fitting']['optimizer']['lr'],
        'final_lr': last['optimizer_states'][0]['param_groups'][0]['lr'],
        'parameters': count_used_parameters(model),
        'trainable_parameters': count_used_parameters(model, only_learnable=True),
        'wandb_runtime_seconds': summary['_runtime'],
    }
    assert audit['trainable_parameters'] == summary['n_learnable_parameters']
    return model, config, audit


def metrics(reference, prediction):
    reference, prediction = np.asarray(reference).ravel(), np.asarray(prediction).ravel()
    residual = prediction - reference
    sst = np.square(reference - reference.mean()).sum()
    return {'N': int(reference.size), 'RMSE': float(np.sqrt(np.mean(residual**2))),
            'MAE': float(np.mean(np.abs(residual))),
            'R2': float(1 - np.sum(residual**2) / sst) if sst else None}


def panel_arrays(cache, element, block):
    mask = cache['Z'] == Z[element]
    if block == 'iso':
        return cache['target'][mask, 0] / np.sqrt(3), cache['prediction'][mask, 0] / np.sqrt(3)
    return cache['target'][mask, 1:].ravel(), cache['prediction'][mask, 1:].ravel()


def geometry_hash(atoms):
    h = hashlib.sha256()
    for array in (atoms.numbers, np.round(atoms.positions, 6), np.round(atoms.cell.array, 6), atoms.pbc):
        h.update(np.asarray(array).tobytes())
    return h.hexdigest()


def capture(model):
    features, handles = {}, []
    for i, layer in enumerate(model.layers):
        def collect(module, args, output, index=i):
            features[index] = {}
            for name in ('0e', '2e'):
                blocks = [output[:, s].detach().cpu().numpy().copy()
                          for (_, ir), s in zip(module.irreps_out, module.irreps_out.slices())
                          if str(ir) == name]
                assert blocks
                features[index][name] = np.concatenate(blocks, axis=1)
        handles.append(layer.register_forward_hook(collect))
    return features, handles


def audit_backbone(model, source, label):
    result = {}
    if label == '06':
        state = model.state_dict()
        compared = 0
        for part in BACKBONE:
            for key, value in getattr(source, part).state_dict().items():
                candidate = f'{part}.{key}'
                if candidate not in state:
                    for operator in ('pre_message_linear', 'post_message_linear', 'self_interaction.tensor_product'):
                        candidate = candidate.replace('.' + operator + '.', '.' + operator + '.base.')
                assert candidate in state, candidate
                same_state(value, state[candidate])
                compared += value.numel()
        adapters = [(n, p) for n, p in model.named_parameters() if '.A.' in n or '.B.' in n]
        operators = [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == 'PathELoRA']
        assert len(operators) == 12
        assert all(torch.isfinite(p).all() for _, p in adapters)
        zero_blocks = []
        for name, operator in operators:
            norms = [float((b @ a).norm()) for a, b in zip(operator.A, operator.B)]
            assert any(norm > 0 for norm in norms), name
            for index, norm in enumerate(norms):
                if norm != 0:
                    continue
                instruction = operator.base.instructions[index]
                assert hasattr(operator.base, 'irreps_in')
                before = str(operator.base.irreps_in[instruction.i_in].ir)
                after = str(operator.base.irreps_out[instruction.i_out].ir)
                zero_blocks.append({'operation': name, 'path_index': index,
                                    'input_irrep': before, 'output_irrep': after,
                                    'path_shape': list(instruction.path_shape)})
        # The layer-3 0o output cannot reach the final 0e/2e via edge irreps:
        # 0o x 0e -> 0o, 0o x 1o -> 1e, 0o x 2e -> 2o. Its two adapter
        # blocks therefore have zero task gradient, while all 12 operators
        # have learned nonzero updates through other paths.
        assert all(row['input_irrep'] == row['output_irrep'] == '0o' for row in zero_blocks)
        result = {'original_backbone_state_exactly_matches_source': True,
                  'backbone_state_values_compared': compared, 'adapter_operators': 12,
                  'all_adapter_operators_have_nonzero_updates': True,
                  'zero_adapter_blocks': zero_blocks}
    elif label == '05':
        changed = []
        for i, (a, b) in enumerate(zip(model.layers, source.layers)):
            changed.append(any(not torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        assert all(changed)
        result['all_four_backbone_layers_changed'] = True
    return result


def evaluate(root, output, history_directory, resume=False):
    root, output, history_directory = Path(root), Path(output), Path(history_directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    assert sha256(root / SOURCE) == SOURCE_SHA
    source_model = torch.load(root / SOURCE, map_location='cpu', weights_only=False).eval()
    source = source_model.models['nequip'].requires_grad_(False)
    configs = {label: yaml.safe_load((root / 'graph-pes-results' / rid / 'train-config.yaml').read_text())
               for label, rid in RUNS.items()}
    for config in configs.values():
        assert config['data'] == configs['01']['data']
        assert config['loss'] == configs['01']['loss']
        assert architecture(config) == architecture(configs['01'])
        fitting = deepcopy(config['fitting'])
        fitting['optimizer']['lr'] = configs['01']['fitting']['optimizer']['lr']
        assert fitting == configs['01']['fitting']
        assert config['general']['seed'] == 42
    data_paths = {'train': root / configs['01']['data']['train']['path'],
                  'validation': root / configs['01']['data']['valid']['path'],
                  'test': root / 'data/nmr_symm/test.xyz'}
    all_train = read(data_paths['train'], ':')
    selected = SequenceSampler(all_train).shuffled(42)[:100]
    prior = json.loads((root / 'provenance/symmetric-transfer-pilots/setup-checks.json').read_text())
    assert list(selected.indices) == prior['training_subset_indices_zero_based']
    frames = {'train': list(selected), 'validation': read(data_paths['validation'], ':'),
              'test': read(data_paths['test'], ':')}
    assert [len(frames[s]) for s in frames] == [100, 50, 150]
    hashes = {s: {geometry_hash(a) for a in fs} for s, fs in frames.items()}
    full_train_hashes = {geometry_hash(a) for a in all_train}
    assert not (full_train_hashes & hashes['validation'] or full_train_hashes & hashes['test']
                or hashes['validation'] & hashes['test'])
    reference, graphs = {}, {}
    basis = CartesianTensor('ij=ji')
    for split, fs in frames.items():
        reference[split] = {
            'Z': np.concatenate([a.numbers for a in fs]),
            'target': np.concatenate([a.arrays['MS_Symm_Irreps'] for a in fs]).astype(np.float64),
            'frame_index': np.repeat(np.arange(len(fs)), [len(a) for a in fs]),
            'n_atoms': np.array([len(a) for a in fs]),
        }
        stored = np.concatenate([a.arrays['MS_Symm_QM'].reshape(-1, 3, 3) for a in fs])
        converted = basis.to_cartesian(torch.from_numpy(reference[split]['target']).float()).numpy()
        np.testing.assert_allclose(converted, stored, rtol=1e-6, atol=8e-5)
        graphs[split] = [AtomicGraph.from_ase(a, cutoff=6., property_mapping={'MS_Symm_Irreps': 'tensor'}) for a in fs]
    print('Matched N100/50/150; target basis and geometry separation checked.', flush=True)

    source_features, handles = capture(source)
    saved_features = []
    with torch.no_grad():
        for g in graphs['validation']:
            source(g)
            saved_features.append(deepcopy(source_features))
    for h in handles:
        h.remove()
    print('Captured E/F 0e and 2e features on all 50 validation structures.', flush=True)

    records, tensor_records, frame_records, drift_records, audits = [], [], [], [], {}
    for label in RUNS:
        model, config, audit = load_best(root, label)
        audit.update(audit_backbone(model, source, label))
        reuse = False
        if resume and (output / f'audit-{label}.json').exists():
            old = json.loads((output / f'audit-{label}.json').read_text())
            for key in ['model_sha256', 'checkpoint_sha256', 'config_sha256']:
                assert old[key] == audit[key]
            reuse = True
        features, handles = capture(model)
        model.requires_grad_(False)
        normalisers = next(iter(config['loss']['shielding'].values()))
        feature_sums = {}
        for split, gs in graphs.items():
            predicted, frame_losses = [], []
            cached = None
            # Transfer validation is recomputed to recover layer diagnostics.
            if reuse and (label == '01' or split != 'validation'):
                with np.load(output / f'predictions-{label}-{split}.npz', allow_pickle=False) as file:
                    cached = {key: file[key] for key in file.files}
                for key, values in reference[split].items():
                    np.testing.assert_array_equal(cached[key], values)
                assert np.isfinite(cached['prediction']).all()
                print(label, split, 'reusing checked prediction cache', flush=True)
            for i, graph in enumerate(gs):
                if cached is None:
                    with torch.no_grad():
                        prediction = model.predict(graph, ['tensor'])['tensor']
                    assert prediction.shape == graph.properties['tensor'].shape
                    assert torch.isfinite(prediction).all()
                    array = prediction.detach().cpu().numpy().astype(np.float64)
                else:
                    array = cached['prediction'][cached['frame_index'] == i]
                predicted.append(array)
                target = reference[split]['target'][reference[split]['frame_index'] == i]
                z_values = reference[split]['Z'][reference[split]['frame_index'] == i]
                terms = []
                for element, block in PANELS:
                    mask = z_values == Z[element]
                    err = (array[mask, 0] - target[mask, 0]) / np.sqrt(3) if block == 'iso' else array[mask, 1:] - target[mask, 1:]
                    mse = float(np.mean(err**2))
                    scale = normalisers[element.lower() + ('_iso_std' if block == 'iso' else '_aniso_rms')]
                    terms.append(mse / scale**2)
                    frame_records.append({'Experiment': label, 'Split': split, 'Frame': i,
                                          'Element': element, 'Block': block, 'MSE': mse})
                frame_losses.append(float(np.mean(terms)))
                if split == 'validation' and label != '01':
                    for layer in range(4):
                        for block in ('0e', '2e'):
                            before, after = saved_features[i][layer][block], features[layer][block]
                            for element, atomic_number in Z.items():
                                mask = z_values == atomic_number
                                x, y = before[mask].astype(np.float64), after[mask].astype(np.float64)
                                key = layer, block, element
                                sums = feature_sums.setdefault(key, np.zeros(4))
                                sums += [np.sum((y-x)**2), np.sum(x*x), np.sum(y*y), np.sum(x*y)]
                if (i + 1) % 50 == 0:
                    print(label, split, i + 1, '/', len(gs), flush=True)
            cache = {**reference[split], 'prediction': np.concatenate(predicted)}
            np.savez_compressed(output / f'predictions-{label}-{split}.npz', **cache)
            audit[f'{split}_structures'] = len(gs)
            audit[f'{split}_atoms'] = len(cache['Z'])
            audit[f'{split}_balanced_loss_cpu'] = float(np.mean(frame_losses))
            for element, block in PANELS:
                y, p = panel_arrays(cache, element, block)
                records.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                                'Element': element, 'Block': block, 'Units': 'ppm', **metrics(y, p)})
            for element, atomic_number in Z.items():
                mask = cache['Z'] == atomic_number
                error = cache['prediction'][mask] - cache['target'][mask]
                aniso_sse = float(np.sum(error[:, 1:]**2))
                zero_sse = float(np.sum(cache['target'][mask, 1:]**2))
                tensor_records.append({'Experiment': label, 'Method': LABELS[label], 'Split': split,
                    'Element': element,
                    'Full tensor Frobenius RMSE': float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    'Anisotropic Frobenius RMSE': float(np.sqrt(np.mean(np.sum(error[:, 1:]**2, axis=1)))),
                    'Anisotropic relative RMS error': float(np.sqrt(aniso_sse / zero_sse)),
                    'Anisotropic skill versus zero': float(1 - aniso_sse / zero_sse)})
        for h in handles:
            h.remove()
        relative = abs(audit['validation_balanced_loss_cpu'] / audit['best_recorded_validation_loss'] - 1)
        assert relative < .001, (label, relative)
        audit['cpu_vs_recorded_validation_loss_relative_difference'] = relative
        for (layer, block, element), (error, before, after, dot) in feature_sums.items():
            drift_records.append({'Experiment': label, 'Method': LABELS[label], 'Layer': layer + 1,
                                  'Irrep': block, 'Element': element,
                                  'Relative RMS feature change': float(np.sqrt(error / before)),
                                  'Cosine similarity': float(dot / np.sqrt(before * after)),
                                  'Feature RMS ratio': float(np.sqrt(after / before))})
        history = json.loads((history_directory / f'history-{label}.json').read_text())
        history = [{k: v for k, v in row.items() if v is not None} for row in history]
        json_write(output / f'history-{label}.json', history)
        valid = pd.DataFrame(history).dropna(subset=['valid/loss/total', 'epoch']).copy()
        assert len(valid) == audit['last_epoch'] and valid['epoch'].is_unique
        best_row = valid.loc[valid['valid/loss/total'].idxmin()]
        assert int(best_row['epoch']) + 1 == audit['best_epoch']
        audit['minutes_to_best'] = float(best_row['_runtime'] / 60)
        common = valid['epoch'].between(5, 99)
        audit['median_seconds_per_epoch_6_to_100'] = float(valid['_runtime'].diff()[common].median())
        timing = pd.DataFrame(history).dropna(subset=['timer/step_duration_ms/train', 'epoch'])
        timing = timing[timing['epoch'].between(5, 99)]
        audit['median_logged_ms_per_training_batch_6_to_100'] = float(timing['timer/step_duration_ms/train'].median())
        metadata = json.loads((history_directory / f'metadata-{label}.json').read_text())
        audit['wandb_state'] = metadata['state']
        audit['gpu'] = metadata.get('metadata', {}).get('gpu')
        audit['wandb_url'] = metadata['url']
        assert audit['wandb_state'] == 'finished'
        audits[label] = audit
        json_write(output / f'audit-{label}.json', audit)
        print(label, 'validation CPU:', audit['validation_balanced_loss_cpu'], 'test:', audit['test_balanced_loss_cpu'], flush=True)
    pd.DataFrame(records).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(tensor_records).to_csv(output / 'tensor-metrics.csv', index=False)
    pd.DataFrame(frame_records).to_csv(output / 'per-structure-mse.csv', index=False)
    pd.DataFrame(drift_records).to_csv(output / 'feature-change.csv', index=False)

    frame_table = pd.DataFrame(frame_records)
    random = np.random.default_rng(20261106)
    bootstrap_indices = random.integers(0, 150, size=(2000, 150))
    bootstrap = []
    for element, block in PANELS:
        table = frame_table[(frame_table.Split == 'test') & (frame_table.Element == element) & (frame_table.Block == block)]
        baseline = table[table.Experiment == '01'].sort_values('Frame').MSE.to_numpy()
        for label in ('05', '06'):
            values = table[table.Experiment == label].sort_values('Frame').MSE.to_numpy()
            improvement = 100 * (1 - np.sqrt(values[bootstrap_indices].mean(axis=1) / baseline[bootstrap_indices].mean(axis=1)))
            lower, upper = np.percentile(improvement, [2.5, 97.5])
            bootstrap.append({'Experiment': label, 'Element': element, 'Block': block,
                              'RMSE reduction versus direct (%)': float(100*(1-np.sqrt(values.mean()/baseline.mean()))),
                              '95% structure-bootstrap lower (%)': float(lower),
                              '95% structure-bootstrap upper (%)': float(upper)})
    pd.DataFrame(bootstrap).to_csv(output / 'test-bootstrap.csv', index=False)
    manifest = {'analysis_date': '2026-11-06', 'runs': audits, 'source_sha256': SOURCE_SHA,
                'data_sha256': {s: sha256(p) for s, p in data_paths.items()},
                'training_indices_zero_based': list(selected.indices),
                'matched_architecture_data_loss_and_fitting_except_lr': True,
                'target_cartesian_reconstruction_verified': True,
                'geometry_hash_overlap_train_pool_validation_test': False,
                'geometry_hash_scope': 'Stored atom order; positions/cell rounded to 1e-6 Angstrom. Parent-trajectory independence is not established.',
                'evaluation': 'Unmodified validation-selected models; CPU float32, highest matmul precision; float64 metric accumulation.',
                'bootstrap': '2000 paired resamples of test structures, fixed seed 20261106; conditional on these trained models and stored split; no seed/trajectory uncertainty.',
                'feature_change': 'Validation-only, fixed-channel coordinates relative to the corrected E/F source; a descriptive adaptation measure, not a feature-utility or causal transfer test.',
                'no_fitting_or_test_based_checkpoint_selection': True}
    manifest['analysis_helper_sha256'] = sha256(Path(__file__))
    manifest['artifact_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir())
                                   if p.suffix in ('.npz', '.csv') or p.name.startswith('history-')}
    json_write(output / 'manifest.json', manifest)
    print('Evaluation and provenance complete:', output, flush=True)
    return manifest


def load_review(root, data=None):
    root = Path(root)
    data = Path(data) if data is not None else root / 'experiments/symmetric-transfer-N100-data'
    manifest = json.loads((data / 'manifest.json').read_text())
    for name, digest in manifest.get('artifact_sha256', {}).items():
        assert sha256(data / name) == digest, f'Analysis artifact changed: {name}'
    assert sha256(root / SOURCE) == manifest['source_sha256']
    for label, audit in manifest['runs'].items():
        directory = root / 'graph-pes-results' / RUNS[label]
        assert sha256(directory / 'model.pt') == audit['model_sha256']
        assert sha256(directory / 'checkpoints/best.ckpt') == audit['checkpoint_sha256']
    for split, digest in manifest['data_sha256'].items():
        name = 'valid' if split == 'validation' else split
        assert sha256(root / f'data/nmr_symm/{name}.xyz') == digest
    return manifest, pd.read_csv(data / 'metrics.csv', dtype={'Experiment': str})


def load_cache(data, label, split):
    with np.load(Path(data) / f'predictions-{label}-{split}.npz', allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def history(data, label):
    frame = pd.DataFrame(json.loads((Path(data) / f'history-{label}.json').read_text()))
    valid = frame.dropna(subset=['valid/loss/total', 'epoch']).sort_values('epoch').copy()
    valid['Epoch'] = valid['epoch'] + 1
    valid['Minutes'] = valid['_runtime'] / 60
    train = frame.dropna(subset=['train/loss/total', 'epoch']).groupby('epoch', as_index=False).mean(numeric_only=True)
    train['Epoch'] = train['epoch'] + 1
    return valid, train


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.labelcolor': '#243647',
                         'text.color': '#243647', 'xtick.color': '#445462', 'ytick.color': '#445462',
                         'savefig.facecolor': 'white', 'axes.titleweight': 'medium'})


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / (name + '.png'), dpi=180, bbox_inches='tight')
    figure.savefig(directory / (name + '.pdf'), bbox_inches='tight')


def plot_run_history(data, label):
    import matplotlib.pyplot as plt
    style()
    valid, train = history(data, label)
    audit = json.loads((Path(data) / f'audit-{label}.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.9), layout='constrained')
    axes[0].plot(train.Epoch, train['train/loss/total'], color='#5F7795', lw=1.4, label='Logged training loss')
    axes[0].plot(valid.Epoch, valid['valid/loss/total'], color=COLORS[label], lw=1.6, label='Validation loss')
    axes[0].axvline(audit['best_epoch'], color='#80909D', ls=':', lw=1.2, label='Selected epoch')
    axes[0].set(title='Training and validation loss', xlabel='Epoch', ylabel='Balanced loss', yscale='log')
    block_colors = ['#4477AA', '#228833', '#CC6677', '#AA3377']
    for (element, block), color in zip(PANELS, block_colors):
        axes[1].plot(valid.Epoch, valid[f'valid/metrics/{element}_{block}_rmse'], color=color, lw=1.5,
                     label=f"{element} {'isotropic (0e)' if block == 'iso' else 'anisotropic (2e)'}")
    axes[1].set(title='Validation RMSE', xlabel='Epoch', ylabel='RMSE (ppm)', yscale='log')
    for ax in axes:
        ax.grid(alpha=.16)
        ax.legend(frameon=False, fontsize=8.6)
    return fig


def parity_limits(data, split, element, block):
    values = [panel_arrays(load_cache(data, label, split), element, block) for label in RUNS]
    low = min(min(y.min(), p.min()) for y, p in values)
    high = max(max(y.max(), p.max()) for y, p in values)
    padding = .045 * (high-low)
    return low-padding, high+padding


def parity_panel(ax, cache, element, block, limits, color, title=None, fontsize=9.3):
    y, p = panel_arrays(cache, element, block)
    # Match the tiny circular markers in the earlier Experiment 01 notebook.
    ax.scatter(y, p, s=6, alpha=0.19, marker='o', color=color,
               linewidths=0, edgecolors='none', rasterized=True, zorder=2)
    ax.plot(limits, limits, '--', color='#81909A', lw=1, zorder=1)
    ax.set(xlim=limits, ylim=limits, xlabel='DFT (ppm)', ylabel='Predicted (ppm)')
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.13, zorder=0)
    m = metrics(y, p)
    ax.text(.035, .965, f"RMSE  {m['RMSE']:.3f} ppm\nMAE     {m['MAE']:.3f} ppm\nR²        {m['R2']:.3f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=fontsize, zorder=3,
            bbox={'facecolor': 'white', 'edgecolor': '#D8E1E7', 'alpha': .96,
                  'boxstyle': 'round,pad=.35'})


def plot_parity(data, label, split):
    import matplotlib.pyplot as plt
    style()
    cache = load_cache(data, label, split)
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 8.5), layout='constrained')
    for ax, (element, block) in zip(axes.flat, PANELS):
        title = f"{element} {'isotropic shielding (0e)' if block == 'iso' else 'anisotropic coefficients (2e)'}"
        parity_panel(ax, cache, element, block, parity_limits(data, split, element, block), COLORS[label], title)
    fig.suptitle(f"{LABELS[label]}: {split} parity at N100", fontsize=13)
    return fig


def plot_learning_comparison(data):
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (element, block) in zip(axes.flat, PANELS):
        for label in RUNS:
            valid, _ = history(data, label)
            ax.plot(valid.Epoch, valid[f'valid/metrics/{element}_{block}_rmse'], color=COLORS[label],
                    lw=1.55, label=LABELS[label], alpha=.92)
        ax.set(title=f"{element} {'isotropic shielding (0e)' if block == 'iso' else 'anisotropic coefficients (2e)'}",
               xlabel='Epoch', ylabel='Validation RMSE (ppm)', yscale='log')
        ax.grid(alpha=.16)
    axes[0, 0].legend(frameon=False, fontsize=9)
    return fig


def plot_error_comparison(data):
    import matplotlib.pyplot as plt
    style()
    table = pd.read_csv(Path(data) / 'metrics.csv', dtype={'Experiment': str})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    x = np.arange(4)
    for ax, split in zip(axes, ('validation', 'test')):
        for offset, label in enumerate(RUNS):
            values = [float(table[(table.Experiment == label) & (table.Split == split)
                                  & (table.Element == element) & (table.Block == block)].RMSE.iloc[0])
                      for element, block in PANELS]
            bars = ax.bar(x + (offset-1)*.245, values, width=.225, color=COLORS[label], label=LABELS[label])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.set_xticks(x, ['Si 0e', 'Si 2e', 'O 0e', 'O 2e'])
        maximum = float(table[table.Split == split].RMSE.max())
        ax.set(title=f'{split.capitalize()} RMSE at N100', ylabel='RMSE (ppm)', ylim=(0, maximum*1.28))
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.7, loc='upper left')
    return fig


def plot_anisotropic_parity(data, split='test'):
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.1), layout='constrained')
    for col, label in enumerate(RUNS):
        cache = load_cache(data, label, split)
        for row, element in enumerate(('Si', 'O')):
            parity_panel(axes[row, col], cache, element, 'aniso', parity_limits(data, split, element, 'aniso'),
                         COLORS[label], f'{LABELS[label]} · {element} 2e', fontsize=8.4)
    fig.suptitle(f'Anisotropic NMR {split} parity at N100', fontsize=13)
    return fig


def plot_feature_change(data):
    import matplotlib.pyplot as plt
    style()
    frame = pd.read_csv(Path(data) / 'feature-change.csv', dtype={'Experiment': str})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), layout='constrained')
    for row, element in enumerate(('Si', 'O')):
        for col, block in enumerate(('0e', '2e')):
            ax = axes[row, col]
            for label in ('05', '06'):
                values = frame[(frame.Experiment == label) & (frame.Element == element) & (frame.Irrep == block)]
                ax.plot(values.Layer, 100*values['Relative RMS feature change'], 'o-', ms=4,
                        color=COLORS[label], lw=1.8, label=LABELS[label])
            ax.set(title=f'{element} {block} feature change', xlabel='Interaction layer',
                   ylabel='RMS change / E/F feature RMS (%)', xticks=[1, 2, 3, 4])
            ax.grid(alpha=.16)
    axes[0, 0].legend(frameon=False, fontsize=9)
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse checked partial prediction caches after an interrupted analysis')
    args = parser.parse_args()
    evaluate(args.root, args.output, args.history, resume=args.resume)
