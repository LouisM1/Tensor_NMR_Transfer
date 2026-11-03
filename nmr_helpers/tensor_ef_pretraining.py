"""E/F pretraining with the exact four-layer symmetric-tensor backbone.

The energy readout uses scalar channels and squared norms of each 2e
channel. This is rotationally invariant but gives the final 2e features an
E/F gradient, unlike a scalar-only linear readout. The readout and E/F
scales/offsets are task-specific and must not be transferred to NMR.
"""
import torch
from torch import nn
from e3nn import o3
from graph_pes.models import NequIP
from .symmetric_shielding import SymmetricTensorNequIP

class InvariantTensorEnergyReadout(nn.Module):
    def __init__(self,irreps,hidden_features=128):
        super().__init__()
        irreps=o3.Irreps(irreps)
        self.scalars=irreps.count('0e')
        self.rank_two=irreps.count('2e')
        if irreps!=o3.Irreps(f'{self.scalars}x0e + {self.rank_two}x2e') or not self.rank_two:
            raise ValueError('Expected contiguous 0e and 2e feature blocks')
        self.network=nn.Sequential(nn.Linear(self.scalars+self.rank_two,hidden_features),
                                   nn.SiLU(),nn.Linear(hidden_features,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        q=x[:,self.scalars:].reshape(-1,self.rank_two,5)
        invariants=torch.cat((x[:,:self.scalars],q.square().mean(dim=-1)),dim=-1)
        return self.network(invariants)

class TensorCompatibleEFNequIP(NequIP):
    def __init__(self,elements,features,cutoff=6.,layers=4,radial_features=64,
                 backbone_target_irreps='0e + 2e',readout_hidden_features=128,
                 direct_force_predictions=False):
        if o3.Irreps(backbone_target_irreps)!=o3.Irreps('0e + 2e'):
            raise ValueError('The shared backbone must match the symmetric tensor pilot')
        if direct_force_predictions:
            raise ValueError('Forces must be energy gradients for this experiment')
        common=dict(elements=elements,features=features,cutoff=cutoff,layers=layers,
                    radial_features=radial_features)
        # Construct the same backbone as the already-running NMR baseline.
        # Base NequIP supplies its tested energy/force prediction machinery;
        # its temporary backbone is replaced before the model is used.
        tensor=SymmetricTensorNequIP(**common,prune_last_layer=True)
        super().__init__(**common,prune_last_layer=False,direct_force_predictions=False)
        for name in ('Z_embedding','initial_node_embedding','edge_embedding','layers'):
            setattr(self,name,getattr(tensor,name))
        self.energy_readout=InvariantTensorEnergyReadout(self.layers[-1].irreps_out,readout_hidden_features)
