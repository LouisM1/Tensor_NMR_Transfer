"""Symmetric shielding baseline with equivariant affine scaling and balanced loss."""
import math
import torch
from torch import nn
from e3nn import o3
from graph_pes.models import TensorNequIP
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.nn import PerElementParameter
from graph_pes.training.loss import Loss
from .shared_head_training import ElementRMSELogger

class SymmetricTarget(str):
    """graph-pes 1.0.2 expects both a string readout and iterable pruning irreps."""
    def __new__(cls):return super().__new__(cls,'0e + 2e')
    def __iter__(self):return iter(o3.Irreps(str(self)))

class IrrepAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales=PerElementParameter.of_shape((2,),default_value=1.)
        self.offsets=PerElementParameter.of_shape((1,),default_value=0.)
    def forward(self,x: torch.Tensor,graph: AtomicGraph) -> torch.Tensor:
        scales=self.scales[graph.Z]
        return torch.cat((x[:,:1]*scales[:,:1]+self.offsets[graph.Z],x[:,1:]*scales[:,1:]),dim=1)
    @torch.no_grad()
    def pre_fit(self,graph):
        y=graph.properties['tensor']
        assert y.shape==(len(graph.Z),6)
        for z in torch.unique(graph.Z).tolist():
            v=y[graph.Z==z]
            self.offsets[z]=v[:,0].mean()
            self.scales[z,0]=v[:,0].std(unbiased=False).clamp_min(.001)
            self.scales[z,1]=v[:,1:].square().mean().sqrt().clamp_min(.001)
    def non_decayable_parameters(self):return [self.scales,self.offsets]

class SymmetricTensorNequIP(TensorNequIP):
    def __init__(self,target_tensor_irreps='0e + 2e',**kwargs):
        if o3.Irreps(target_tensor_irreps)!=o3.Irreps('0e + 2e'):
            raise ValueError('This model requires exactly one 0e and one 2e output')
        super().__init__(target_tensor_irreps=SymmetricTarget(),target_method='direct',props='tensor',**kwargs)
        self.scaler=IrrepAffine()

class SymmetricBalancedMSE(Loss):
    """Equal Si/O and scalar/anisotropic contributions; constants from training only.

    0e coefficient is sqrt(3)*sigma_iso; the five 2e coefficients are an
    orthonormal representation of the Cartesian traceless symmetric tensor.
    """
    def __init__(self,si_iso_std,o_iso_std,si_aniso_rms,o_aniso_rms,weight=1.):
        super().__init__(weight=weight,is_per_atom=True)
        values=[si_iso_std,o_iso_std,si_aniso_rms,o_aniso_rms]
        if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Scales must be finite and positive')
        self.iso=[si_iso_std,o_iso_std];self.aniso=[si_aniso_rms,o_aniso_rms]
    @property
    def required_properties(self):return ['tensor']
    @property
    def name(self):return 'symmetric_element_block_balanced_mse'
    def forward(self,model,graph,predictions):
        p,y=predictions['tensor'],graph.properties['tensor']
        if p.shape!=y.shape or y.shape!=(len(graph.Z),6):raise ValueError('Expected (atoms,6) irreps')
        if not bool(((graph.Z==14)|(graph.Z==8)).all()):raise ValueError('Only Si/O supported')
        loss=p.new_zeros(())
        for i,z in enumerate((14,8)):
            mask=graph.Z==z
            if not bool(mask.any()):raise ValueError('Both Si and O must be present')
            err=(p[mask]-y[mask]).square()
            loss=loss+.25*(err[:,0].mean()/(3*self.iso[i]**2)+err[:,1:].mean()/self.aniso[i]**2)
        return loss

class SymmetricRMSELogger(ElementRMSELogger):
    def _update(self,stage,task,batch):
        def collect(module,args):
            _,graph,predictions=args
            err=(predictions['tensor'].detach()-graph.properties['tensor']).double().square()
            values=torch.stack([v for z in (14,8) for v in
                (err[graph.Z==z,0].sum()/3,err[graph.Z==z,1:].sum()/5,(graph.Z==z).sum())])
            previous=getattr(self,'_'+stage)
            setattr(self,'_'+stage,values if previous is None else previous+values)
        self._hook=task.total_loss.register_forward_pre_hook(collect)
    def _log(self,stage,trainer,task):
        values=getattr(self,'_'+stage)
        if values is None or trainer.sanity_checking:return
        for i,element in enumerate(('Si','O')):
            for j,block in enumerate(('iso','aniso')):
                task.log(f'{stage}/metrics/{element}_{block}_rmse',(values[3*i+j]/values[3*i+2]).sqrt(),on_step=False,on_epoch=True)
