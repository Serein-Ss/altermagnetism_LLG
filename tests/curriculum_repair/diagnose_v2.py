"""Corrected diagnostic: training parents have eight branches, not sixteen."""
from .run import (argparse,hashlib,json,time,h5py,np,torch,F,HERE,ROOT,load,references,
                  fm_loss,flat_gradient,source,differentiable_sample,joint_loss,
                  gradient_relation,generate,dump,diagnostic_comparison,require_gpu)

def diagnose(seed,out,smoke=False):
    task=dict(domain='bauer',arm='hierarchical',seed=seed)
    times=torch.arange(11,device='cuda')*.1;rows=[]
    parents=[(r,x) for r,x in references('bauer','train') if r['L']==16]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for kind in ('baseline','joint'):
            net,_=load(task,kind)
            for j,(row,raw) in enumerate(parents[:1] if smoke else parents):
                reference=F.normalize(torch.tensor(raw,device='cuda'),dim=-1)
                for repeat in range(1 if smoke else 2):
                    gs={k:torch.Generator(device='cuda').manual_seed(926100+j*100+repeat*10+i)
                        for i,k in enumerate(('source','tau'))}
                    # Whole train branches, identical branch subsets and sources across checkpoints.
                    y=reference
                    assert len(y)>=2
                    loss,_=fm_loss(net,y,times,gs);gf=flat_gradient(loss,net)
                    initial=reference[:1,0].expand(8,-1,-1)
                    z=source(initial,11,torch.Generator(device='cuda').manual_seed(926200+j*10+repeat),2.)
                    for steps in (32,64):
                        x=differentiable_sample(net,initial,times,z,steps)
                        aux,parts=joint_loss(x,reference);ga=flat_gradient(aux,net)
                        relation=gradient_relation(gf,ga)
                        if not torch.isfinite(ga).all() or not torch.isfinite(gf).all():raise RuntimeError('nonfinite diagnostic gradient')
                        rows.append(dict(checkpoint=kind,parent=row,repeat=repeat,steps=steps,reference_branches=len(reference),
                                         fm_loss=float(loss.detach()),aux_loss=float(aux.detach()),**parts,**relation))
                        del x,aux,ga
                initial=reference[:1,0].expand(32,-1,-1)
                z=source(initial,11,torch.Generator(device='cuda').manual_seed(926300+j),2.)
                for steps in (32,128,256):
                    x=generate(net,initial,times,None,steps,source=z).cpu().numpy()
                    group=h.create_group(f'{kind}/parent{j}/steps{steps}')
                    group.attrs['parent']=json.dumps(row)
                    group.create_dataset('generated',data=x,compression='lzf')
                    group.create_dataset('reference',data=raw,compression='lzf')
                    group.create_dataset('time_ps',data=np.arange(11)*.1)
                    dump(out/f'{kind}_parent{j}_steps{steps}.json',diagnostic_comparison(x,raw,0,seed))
            del net
        h.attrs['complete']=True
    dump(out/'gradients.json',rows)
    return dict(seed=seed,gradient_rows=len(rows),scientific_certification=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    require_gpu('cuda');torch.set_num_threads(1)
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    parent=HERE/f'diagnose/{a.index:02d}';out=parent/'retry2';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();result=diagnose((11,22,33)[a.index],out)
    result.update(total_seconds=time.perf_counter()-start,output=str(out),reference_branches=8,
                  repeat_interpretation='Same8 train futures; independent latent/tau draws, not additional physical samples')
    dump(out/'complete.json',result)
    dump(parent/'gradients.json',json.loads((out/'gradients.json').read_text()))
    dump(parent/'complete.json',result)


if __name__=='__main__':main()

