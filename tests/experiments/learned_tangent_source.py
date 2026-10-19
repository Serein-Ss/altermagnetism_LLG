"""Sample-fitted conditional source for the unchanged sphere flow backbone."""
import torch
from torch import nn
from scripts.model.sphere import tangent_project, sphere_exp
from scripts.model.gate_f import path_interpolate
from tests.experiments.local_equivariant_transition import neighbors
from tests.experiments.projected_midpoint_sampler import midpoint_step
from tests.experiments.reflection_augmentation import augment_pairs


class LearnedTangentSource(nn.Module):
    def __init__(self, fitted):
        super().__init__()
        if fitted['lag'] != .01:
            raise ValueError('Only fitted physical lag .01 is supported')
        self.register_buffer('temperatures',torch.tensor([r['theta'] for r in fitted['rows']],dtype=torch.float64))
        self.register_buffer('coefficients',torch.tensor([r['coefficients'] for r in fitted['rows']],dtype=torch.float64))
        self.register_buffer('sigmas',torch.tensor([r['sigma'] for r in fitted['rows']],dtype=torch.float64))

    def from_noise(self, initial, theta, axis, noise):
        match=theta[:,None]==self.temperatures.to(theta)[None]
        if not bool((match.sum(-1)==1).all()):
            raise ValueError('Temperature outside fitted source conditions')
        index=match.long().argmax(-1)
        left,right=neighbors(initial)
        neighbor=tangent_project(initial,left+right)
        axis=axis[:,None]
        director=tangent_project(initial,(initial*axis).sum(-1,keepdim=True)*axis)
        basis=torch.stack([neighbor,torch.linalg.cross(initial,neighbor),director,
                           torch.linalg.cross(initial,director)],-2)
        mean=(basis*self.coefficients.to(initial)[index,None,:,None]).sum(-2)
        tangent=mean+tangent_project(initial,noise)*self.sigmas.to(initial)[index,None,None]
        return sphere_exp(initial,tangent)

    def draw(self, initial, theta, axis, generator):
        noise=torch.randn(initial.shape,dtype=initial.dtype,device=initial.device,generator=generator)
        return self.from_noise(initial,theta,axis,noise)

    def train_step(self, model, optimizer, initial, target, theta, lag, axis,
                   generator, scale, augmentation_generator):
        a,b,mask=augment_pairs(initial,target,augmentation_generator)
        optimizer.zero_grad(set_to_none=True)
        z=self.draw(a,theta,axis,generator)
        tau=torch.rand(len(a),dtype=a.dtype,device=a.device,generator=generator)
        x,v,diagnostics=path_interpolate(torch.stack((a,z),1),torch.stack((a,b),1),tau,generator)
        loss=(model(x[:,1],tau,a,theta,lag,axis)-v[:,1]).square().sum(-1).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite conditional-source loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()),float(grad),diagnostics,int(mask.sum())

    @torch.no_grad()
    def sample(self, model, initial, theta, lag, axis, generator, scale, steps):
        if steps<1:raise ValueError('Positive flow integration steps required')
        x=self.draw(initial,theta,axis,generator)
        def field(state,time):
            return model(state,initial.new_full((len(initial),),time),initial,theta,lag,axis)
        for step in range(steps):
            x=midpoint_step(field,x,step/steps,1/steps)
        return x
