"""Independent chain physics and generator software checks, with explicit scope."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from scripts.workflow.bauer_campaign import graph, save
from scripts.model.gate_f import build_model, matching_loss, sample, reference_path, path_interpolate
from scripts.model.sphere import sphere_exp
from scripts.literature.bauer_2011.model import OpenChain
from scripts.core.bauer_fast import ensemble


def rotate(x, rotation):
    return torch.einsum('...j,ij->...i', x, rotation)


def run(root, device):
    if device == 'cuda' and (not torch.cuda.is_available() or not __import__('os').getenv('SLURM_JOB_ID')):
        raise RuntimeError('CUDA checks require a Slurm GPU allocation')
    torch.set_num_threads(1); torch.manual_seed(916)
    start = time.monotonic(); rows = []
    def check(name, value, tolerance, detail=''):
        finite = bool(np.isfinite(value)); passed = finite and value <= tolerance
        rows.append(dict(name=name, value=float(value) if finite else None, tolerance=tolerance,
                         status='PASS' if passed else 'FAIL', detail=detail))
    for dtype in (torch.float64, torch.float32):
        label = str(dtype).split('.')[-1]; tol = 1e-8 if dtype==torch.float64 else 1e-4
        initial = torch.randn(3,1,25,1,3,device=device,dtype=dtype)
        initial /= initial.norm(dim=-1,keepdim=True)
        g = graph(25,device,dtype); axis = torch.tensor([0.,0.,1.],device=device,dtype=dtype).expand(3,-1)
        source = reference_path(initial,11,torch.Generator(device=device).manual_seed(11))
        y = reference_path(initial,11,torch.Generator(device=device).manual_seed(12))
        condition = torch.tensor([[.11,.1],[.13,.1],[.12,.1]],device=device,dtype=dtype)
        times = torch.linspace(0,1,11,device=device,dtype=dtype).expand(3,-1)
        flat = initial[:,0,:,0].detach().clone().requires_grad_(True)
        chain = OpenChain(dict(equation_convention='bauer_ll',exchange=1.,anisotropy=.1))
        field = -torch.autograd.grad(chain.energy(flat).sum(),flat)[0]
        check(label+'/energy_gradient',float((field-chain.field(flat)).abs().max()),tol)
        check(label+'/graph_field',float((g.field(flat[:,None],axis)[:,0]-field).abs().max()),tol)
        check(label+'/graph_energy',float((g.energy(flat[:,None],axis)[:,0]-chain.energy(flat)).abs().max()),tol)
        check(label+'/open_edges',float(abs(len(g.edges)-24)),0.)
        ground = torch.zeros_like(flat);ground[...,2]=1
        check(label+'/ground_energy',float((chain.energy(ground)+24+2.5).abs().max()),tol)
        check(label+'/end_field',float((chain.field(ground)[:,0,2]-1.2).abs().max()),tol)
        check(label+'/interior_field',float((chain.field(ground)[:,1:-1,2]-2.2).abs().max()),tol)
        for tau_value in (0.,.3,1.):
            tau = torch.full((3,),tau_value,device=device,dtype=dtype)
            x,v,diag = path_interpolate(source,y,tau,torch.Generator(device=device).manual_seed(13))
            check(label+f'/interpolation_norm_{tau_value}',float((x.norm(dim=-1)-1).abs().max()),tol)
            check(label+f'/interpolation_tangent_{tau_value}',float((x*v).sum(-1).abs().max()),tol)
            check(label+f'/interpolation_anchor_{tau_value}',float((x[:,0]-initial).abs().max()),0.)
            if tau_value in (0.,1.):
                check(label+f'/interpolation_endpoint_{tau_value}',float((x-(source if tau_value==0 else y)).abs().max()),tol)
        antipodal = -source.clone();antipodal[:,0]=initial
        x,v,diag = path_interpolate(source,antipodal,torch.full((3,),.5,device=device,dtype=dtype),
                                    torch.Generator(device=device).manual_seed(14))
        check(label+'/antipodal_finite',float(not (torch.isfinite(x).all() and torch.isfinite(v).all())),0.)
        for kind in ('rfm','deterministic','euclidean','autoregressive'):
            net = build_model(kind,width=16,blocks=2,condition_mean=(.12,.1),condition_std=(.01,1.)).to(device=device,dtype=dtype)
            gen = torch.Generator(device=device).manual_seed(15)
            loss,diag = matching_loss(net,y,condition,times,g,axis,gen,kind=kind)
            loss.backward()
            check(label+'/'+kind+'/finite_loss_gradient',float(not(torch.isfinite(loss) and all(
                p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))),0.)
            check(label+'/'+kind+'/nonzero_gradient',float(sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)==0),0.)
            tau=torch.full((3,),.4,device=device,dtype=dtype)
            output=net(source,tau,initial,condition,times,g,axis,tangent=kind!='euclidean')
            errors=[];absolute=[]
            for transform in range(20):
                q,_=torch.linalg.qr(torch.randn(3,3,device=device,dtype=dtype));q[:,0]*=torch.linalg.det(q)
                rotated=net(rotate(source,q),tau,rotate(initial,q),condition,times,g,rotate(axis,q),tangent=kind!='euclidean')
                expected=rotate(output,q);diff=(rotated-expected).norm()
                errors.append(float(diff/max(float(expected.norm()),1e-8*math.sqrt(expected.numel()))))
                absolute.append(float((rotated-expected).abs().max()))
            check(label+'/'+kind+'/20_joint_SO3',max(errors),tol,detail=f'max_absolute={max(absolute)}')
            # Controlled zero-velocity fixture tests sampler identities. Arbitrary
            # untrained ambient fields need not define a stable finite transport.
            if kind=='euclidean':
                with torch.no_grad():
                    net.output.weight.zero_();net.output.bias.zero_()
            prediction=sample(net,initial,condition,times,g,axis,gen,steps=8,kind=kind)
            check(label+'/'+kind+'/sample_finite',float(not torch.isfinite(prediction).all()),0.)
            check(label+'/'+kind+'/hard_initial',float((prediction[:,0]-initial).abs().max()),1e-10 if dtype==torch.float64 else 1e-5)
            if kind!='euclidean':
                check(label+'/'+kind+'/sample_norm',float((prediction.norm(dim=-1)-1).abs().max()),1e-10 if dtype==torch.float64 else 1e-5)
            else:
                rows.append(dict(name=label+'/euclidean/norm_diagnostic',status='DIAGNOSTIC',
                                 value=float((prediction.norm(dim=-1)-1).abs().max()),
                                 detail='Ambient baseline intentionally not normalized.'))
            if kind=='rfm':
                q,_=torch.linalg.qr(torch.randn(3,3,device=device,dtype=dtype));q[:,0]*=torch.linalg.det(q)
                p1=sample(net,initial,condition,times,g,axis,gen,steps=8,kind=kind,source=source)
                p2=sample(net,rotate(initial,q),condition,times,g,rotate(axis,q),gen,steps=8,kind=kind,source=rotate(source,q))
                check(label+'/rfm/source_coupled_sample_SO3',float((p2-rotate(p1,q)).norm()/p1.norm()),tol)
                doubled=sample(net,initial,condition,times,g,axis,gen,steps=16,kind=kind,source=source)
                rows.append(dict(name=label+'/rfm/untrained_step_doubling',status='DIAGNOSTIC',
                                 value=float((p1-doubled).square().mean().sqrt()),
                                 detail='Wiring/finite check only; trained-model distribution convergence remains required.'))
    # Independent deterministic physical limits, compiled CPU even in a GPU software job.
    rng=np.random.default_rng(915);s=rng.normal(size=(4,8,3));s/=np.linalg.norm(s,axis=-1,keepdims=True)
    def energy(x):return -(x[:,:,:-1]*x[:,:,1:]).sum((-2,-1))-.1*(x[...,2]**2).sum(-1)
    conserved,_=ensemble(s,np.arange(4)+21,.001,1000,10,.1,0.,0.)
    damped,_=ensemble(s,np.arange(4)+21,.001,1000,10,.1,.1,0.)
    check('zero_temperature/no_damping_energy',float(abs(energy(conserved)-energy(conserved)[:,:1]).max()),1e-8)
    check('zero_temperature/damped_energy_monotone',float(max(0,np.diff(energy(damped),axis=1).max())),1e-8)
    save(Path(root)/'software/result.json',dict(status='PASS' if all(r['status']!='FAIL' for r in rows) else 'FAIL',
         checks=rows, device=device, wall_seconds=time.monotonic()-start,
         scope='Physics identities, random-network gradients/covariance and controlled zero-field samplers; no learned physical distribution certificate.',
         cuda_device=torch.cuda.get_device_name() if device=='cuda' else None))
