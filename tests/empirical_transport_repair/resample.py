"""Independent latent replicates after training; never changes a checkpoint."""
from .run import (argparse,hashlib,json,time,h5py,np,torch,HERE,ROOT,STOP,empirical,make_net,
                  tasks,source,path_interpolate,conditional,integrate,branch_summary,dump,require_gpu,
                  comparison,diagnostic_comparison)
from .supervision import tasks as supervision_tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    require_gpu('cuda');torch.set_num_threads(1)
    for file in ('source_lock.json','extended_lock.json'):
        for name,digest in json.loads((HERE/file).read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    seed=(11,22,33)[a.index];out=HERE/f'{"resample_smoke" if a.smoke else "resample"}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False)
    row,y=empirical();times=torch.arange(11,device='cuda')*.1;n=16 if a.smoke else 512
    cases=[]
    for i,t in enumerate(tasks()):
        if t['seed']==seed:cases.append((t,HERE/f'train/{i:02d}/step6000.pt'))
    for i,t in enumerate(supervision_tasks()):
        if t['seed']==seed:cases.append((t,HERE/f'supervision_train/{i:02d}/checkpoint.pt'))
    if a.smoke:cases=[(dict(domain='bauer',arm='hierarchical',seed=seed,repair='software_only'),None)]
    results=[];missing=[];began=time.perf_counter()
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.create_dataset('empirical_targets',data=y.cpu().numpy())
        for task,path in cases:
            if path is not None and not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            net=make_net(task)
            if path is not None:net.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model'])
            net.eval();initial=y[:1,0].expand(n,-1,-1)
            for rep in range(1 if a.smoke else 3):
                z=source(initial,11,torch.Generator(device='cuda').manual_seed(929800+rep),2.)
                zr=source(initial,11,torch.Generator(device='cuda').manual_seed(929900+rep),2.)
                reference=path_interpolate(zr,y[torch.arange(n,device='cuda')%8],z.new_full((n,),STOP),torch.Generator(device='cuda').manual_seed(929950+rep))[0]
                for steps in ([16] if a.smoke else ([128,512] if rep==0 else [128])):
                    parts=[];tick=time.perf_counter()
                    for start in range(0,n,64):
                        init=initial[start:start+64]
                        parts.append(integrate(lambda x,t:net.velocity(x.float(),t.float(),init.float(),times).double(),z[start:start+64],0.,STOP,steps,'euler'))
                    x=torch.cat(parts);torch.cuda.synchronize()
                    group=h.create_group(f'{task["repair"]}_rep{rep}_steps{steps}')
                    group.create_dataset('generated',data=x.cpu().numpy(),compression='lzf')
                    group.create_dataset('reference',data=reference.cpu().numpy(),compression='lzf')
                    group.create_dataset('time_ps',data=np.arange(11)*.1)
                    results.append(dict(arm=task['repair'],replicate=rep,steps=steps,seconds=time.perf_counter()-tick,
                        **branch_summary(x,y),physical=comparison(x.cpu().numpy(),reference.cpu().numpy()),
                        diagnostic=diagnostic_comparison(x.cpu().numpy(),reference.cpu().numpy(),0,929),
                        norm_error=float((x.norm(dim=-1)-1).abs().max()),anchor_error=float((x[:,0]-initial).abs().max())))
            del net
        h.attrs['complete']=not missing
    dump(out/'complete.json',dict(seed=seed,parent=row,rows=results,missing=missing,total_seconds=time.perf_counter()-began,
        scope='tau.95 empirical-law diagnostic;3 independent latent replicates512paths each; not3 independent LLG datasets; no real-path physics certification',scientific_certification=False))
    if missing:raise RuntimeError('missing checkpoints; see complete.json')


if __name__=='__main__':main()
