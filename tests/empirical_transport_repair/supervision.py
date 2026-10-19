"""Matched CFM vs exact empirical conditional-velocity supervision, tau<.95."""
from .run import (argparse,hashlib,json,time,math,np,torch,HERE,ROOT,STOP,empirical,make_net,
                  rate,source,path_interpolate,conditional,evaluate,dump,require_gpu)


def tasks():
    rows=[dict(domain='bauer',arm='hierarchical',seed=s,repair=a,updates=6000)
          for s in (11,22,33) for a in ('paired_cfm','exact_velocity')]
    np.random.default_rng(92904).shuffle(rows);return rows


def run(task,out,smoke):
    net=make_net(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    _,y=empirical();times=torch.arange(11,device='cuda')*.1
    gs=torch.Generator(device='cuda').manual_seed(929400+task['seed'])
    gt=torch.Generator(device='cuda').manual_seed(929500+task['seed'])
    history=[];stream=hashlib.sha256();start=time.perf_counter();total=32 if smoke else 6000
    for step in range(1,total+1):
        z=source(y[:,0],11,gs,2.);tau=torch.rand(len(y),device='cuda',dtype=y.dtype,generator=gt)*STOP
        x,v,cut=path_interpolate(z,y,tau,gs)
        if cut['unresolved_cut_count']:raise RuntimeError('unresolved cut invalidates exact target comparison')
        if task['repair']=='exact_velocity':
            with torch.no_grad():v,_,bad=conditional(x,tau,y,y[:,0])
            if bad.any():raise RuntimeError('exact target outside support')
        pred=net.velocity(x.float(),tau.float(),y[:,0].float(),times)
        loss=(pred[:,1:]-v[:,1:].float()).square().sum(-1).mean()
        if step<=32:
            for val in (x,tau):stream.update(val.cpu().numpy().tobytes())
        lr=rate('fresh_decay',step,total)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(grad) or not torch.isfinite(loss):raise RuntimeError('nonfinite training')
        opt.step()
        if step%100==0 or smoke:history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=lr))
        if step%500==0:dump(out/'history.json',history);print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;dump(out/'history.json',history)
    torch.save(dict(model=net.state_dict(),task=task,step=total,optimizer=opt.state_dict(),source_rng=gs.get_state(),tau_rng=gt.get_state(),transport_training_max=STOP),out/'checkpoint.pt')
    net.eval();result=evaluate(net,out,smoke)
    for row in result['rows']:row['outside_trained_transport_range']=row['stop_tau']>STOP
    result.update(task=task,train_seconds=elapsed,first32_input_hash=stream.hexdigest(),
        transport_training_max=STOP,peak_bytes=torch.cuda.max_memory_allocated(),
        scope='Compare ONLY matched paired_cfm vs exact_velocity at tau.95. tau1 output is unsupported extrapolation. Target-conditioned teacher is empirical diagnostic only.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    require_gpu('cuda');torch.set_num_threads(1)
    for file in ('source_lock.json','extended_lock.json'):
        for name,digest in json.loads((HERE/file).read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'supervision_{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=run(tasks()[a.index],out,a.action=='smoke');result['total_seconds']=time.perf_counter()-start;dump(out/'complete.json',result)


if __name__=='__main__':main()
