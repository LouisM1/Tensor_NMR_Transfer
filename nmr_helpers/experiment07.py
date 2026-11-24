"""Experiment 7 transfer masks and durable, local training-curve logging."""
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch
from pytorch_lightning import Callback
from graph_pes.training.opt import Optimizer
from .symmetric_transfer import from_energy_force_model
from .training_fixes import install_training_fixes


def restricted_transfer(model, checkpoint, expected_sha256, last_layer=False):
    """Start independently from E/F, then expose only the requested parameters."""
    model = from_energy_force_model(model, checkpoint, freeze_backbone=True,
                                   expected_sha256=expected_sha256)
    if last_layer:
        model.layers[-1].requires_grad_(True)
    return model


def expected_trainable(name, method):
    if method in ('scratch', 'full'):
        return True
    if name.startswith(('tensor_readout.', 'scaler.')):
        return True
    if method == 'last':
        return name.startswith('layers.3.')
    if method == 'elora':
        return '.A.' in name or '.B.' in name
    if method == 'head':
        return False
    raise ValueError(method)


def ordered_optimizer(self, model):
    # Identical ordering/groups to train.sbatch, including frozen parameters.
    named = sorted(model.named_parameters())
    non_decayable = {id(p) for p in model.non_decayable_parameters()}
    groups = []
    for label, use_non_decay in [('non-decayable', True), ('normal', False)]:
        members = [(n, p) for n, p in named if (id(p) in non_decayable) == use_non_decay]
        if members:
            group = dict(name=label, params=[p for _, p in members],
                         param_names=[n for n, _ in members])
            if use_non_decay:
                group['weight_decay'] = 0.0
            groups.append(group)
    return self.optimizer_class(groups, **self.kwargs)


def install_fixes():
    install_training_fixes()
    Optimizer.__call__ = ordered_optimizer


def install_job_local_graph_cache():
    # locache 4.0.2 writes beside installed source files and has no cache-dir
    # environment option. Avoid concurrent writes to the shared ARC environment.
    # Preserve pre_transform=True and reuse prepared graphs within this process.
    import graph_pes.data.datasets as datasets
    current = datasets.get_all_graphs_and_cache_to_disk
    if not getattr(current, '_experiment07_cache', False):
        cached = lru_cache(maxsize=None)(current.__wrapped__)
        cached._experiment07_cache = True
        datasets.get_all_graphs_and_cache_to_disk = cached


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class CurveAudit(Callback):
    """Check the fit mask and log loss/RMSE even without a W&B connection."""
    def __init__(self, method, n):
        self.method, self.n = method, n
        self.audit_steps = 0
        install_job_local_graph_cache()

    def on_train_start(self, trainer, task):
        if len(trainer.train_dataloader.dataset) != self.n:
            raise ValueError('Wrong training subset size')
        for name, p in task.model.named_parameters():
            if p.requires_grad != expected_trainable(name, self.method):
                raise ValueError(f'Unexpected freeze mask: {name}')
        self.directory = Path(trainer.checkpoint_callbacks[0].dirpath).parent
        if trainer.ckpt_path:
            # SaveBestModel does not serialize its best threshold. A worse
            # resumed epoch must never replace the historical best model.pt.
            from graph_pes.training.callbacks import SaveBestModel
            state = torch.load(trainer.ckpt_path, map_location='cpu', weights_only=False)
            scores = [v['best_model_score'] for k,v in state['callbacks'].items()
                      if k.startswith('ModelCheckpoint') and v.get('monitor')=='valid/loss/total']
            if len(scores)!=1:
                raise ValueError('Resume checkpoint lacks a unique validation best score')
            for cb in trainer.callbacks:
                if isinstance(cb, SaveBestModel): cb.best_val_loss=float(scores[0])
        details = {'method': self.method, 'n': self.n,
                   'trainable_names': [k for k, p in task.model.named_parameters() if p.requires_grad],
                   'source_sha256': getattr(task.model, '_symmetric_transfer_source_sha256', None)}
        (self.directory / 'experiment07-fit-audit.json').write_text(json.dumps(details, indent=2)+'\n')

    def on_before_optimizer_step(self, trainer, task, optimizer):
        if self.audit_steps >= 2:
            return
        for name, p in task.model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f'Non-finite gradient: {name}')
            if not p.requires_grad and p.grad is not None:
                raise RuntimeError(f'Frozen parameter received a gradient: {name}')
        self.audit_steps += 1

    def on_train_epoch_end(self, trainer, task):
        # Runs after validation; the existing RMSE callback is placed first.
        values = {'epoch': int(trainer.current_epoch), 'global_step': int(trainer.global_step)}
        keys = [f'{s}/loss/total' for s in ('train', 'valid')]
        keys += [f'{s}/metrics/{e}_{b}_rmse' for s in ('train', 'valid')
                 for e in ('Si', 'O') for b in ('iso', 'aniso')]
        for key in keys:
            value = trainer.callback_metrics.get(key)
            if value is None:
                raise RuntimeError(f'Missing epoch metric: {key}')
            value = float(value)
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError(key)
            values[key] = value
        values['lr'] = trainer.optimizers[0].param_groups[0]['lr']
        path = self.directory / 'epoch-metrics.csv'
        exists = path.exists() and path.stat().st_size > 0
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values))
            if not exists:
                writer.writeheader()
            writer.writerow(values)
