"""Paired learning-rate intervention on frozen P2 checkpoints; development only."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import ROOT, Model, dump
from tests.bauer_p2_20260923.run import evaluate
from tests.bauer_closed_loop_20260919.run import batch
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_p2_fit_diagnostic_20260923.run import paired
from tests.bauer_p2_diagnostics_20260923.run import table

HERE=Path(__file__).resolve().parent
P2=HERE.parent/'bauer_p2_20260923'
ARMS={'control':.001,'low_lr':.0001}


def prepare():
    lock=json.loads((P2/'source_lock.json').read_text())
    for name,digest in lock.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for seed in (11,22,33):
        p=P2/'runs'/f'seed_{seed}'/'best.pt'
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    protocol=dict(arms=ARMS,seeds=[11,22,33],updates=3000,snapshots=[0,1000,2000,3000],
        data='unchanged full P2 train L16/25/32, half anchors/half windows',
        pairing='same parent checkpoint, optimizer moments, data and flow RNG sequences within seed',
        selection='fixed final step primary; intermediate snapshots descriptive, no new physical tolerance',
        evaluation='36 dev parents only, 16 independent branches each, 1ps at 0.1ps grid, 128 flow steps; final also 256',
        hypothesis='smaller optimizer steps reduce oscillatory fitting and improve joint statistics without changing architecture or loss',
        interpretation='lower loss alone not success; require time/energy/increment metrics together, report tradeoffs and all seeds',
        estimate_minutes_per_job=[20,30],time_limit_minutes=60,
        scope='bounded optimizer intervention, not size transfer or physical certification',
        skill_reference='Experimental Design, Scientific Agent Skills, doi:10.48550/arXiv.2609.00065')
    dump(HERE/'protocol.json',protocol)
    for p in HERE.glob('*.py'):
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    lock[str((HERE/'protocol.json').relative_to(ROOT))]=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)
    (HERE/'logs').mkdir(exist_ok=True)


def optimizer_for(net, checkpoint, lr):
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    for group in optimizer.param_groups:group['lr']=lr
    return optimizer


def run(index,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    # Adjacent array entries form a same-seed pair.
    seed=protocol['seeds'][index//2];arm=list(ARMS)[index%2];lr=ARMS[arm]
    torch.set_num_threads(1);torch.manual_seed(seed+92330)
    ck=torch.load(P2/'runs'/f'seed_{seed}'/'best.pt',map_location=device,weights_only=False)
    config=copy.deepcopy(ck['config']);config['learning_rate']=lr
    net=Model(config).to(device);net.load_state_dict(ck['model'])
    optimizer=optimizer_for(net,ck,lr)
    rng=np.random.default_rng(seed+92330)
    generator=torch.Generator(device=device).manual_seed(seed+92331)
    data=[PathWindows([ROOT/config['data']/f'L{n}'],'train',config['duration_ps'],stride_frames=10) for n in config['sizes']]
    out=HERE/'runs'/f'{arm}_seed_{seed}';out.mkdir(parents=True,exist_ok=False)
    history=[];snapshots=[];start=time.perf_counter();train_seconds=0.;stream=hashlib.sha256()
    for step in range(protocol['updates']+1):
        if step:
            net.train();torch.cuda.synchronize();tick=time.perf_counter()
            y,times=batch(data,rng,config,device,'B')
            z=F.normalize(torch.randn(y.shape,device=device,generator=generator),dim=-1);z[:,0]=y[:,0]
            tau=torch.rand(len(y),device=device,generator=generator)
            x,target,cut=path_interpolate(z,y,tau,generator)
            if step<=10 or step%1000==0:
                for value in (y,z,tau):stream.update(value.detach().cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss=(net.velocity(x,tau,y[:,0],times)[:,1:]-target[:,1:]).square().sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(grad)
            optimizer.step();torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
            history.append(dict(step=step,loss=float(loss),gradient_norm_before_clip=float(grad),**cut))
        if step not in protocol['snapshots']:continue
        result=evaluate(net,config,'dev',128,92332,out/f'generated_{step}',repeats=200)
        dump(out/f'dev_{step}.json',result)
        snapshots.append(dict(step=step,metrics=result['mean_parent_RMSE'],sample_stream_sha256=stream.hexdigest()))
        torch.save(dict(model=net.state_dict(),config=config,optimizer=optimizer.state_dict(),seed=seed,
            parent_step=ck['step'],additional_step=step,numpy_rng=rng.bit_generator.state,
            flow_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'step_{step}.pt')
        dump(out/'history.json',history)
        dump(out/'summary.json',dict(seed=seed,arm=arm,lr=lr,parent_step=ck['step'],completed_step=step,
            snapshots=snapshots,train_seconds=train_seconds,elapsed_seconds=time.perf_counter()-start,
            gpu=torch.cuda.get_device_name(),status='DEVELOPMENT_DIAGNOSTIC_NOT_CERTIFIED'))
        print(seed,arm,step,result['mean_parent_RMSE'],flush=True)
    result=evaluate(net,config,'dev',256,92332,out/'generated_final256',repeats=200)
    dump(out/'dev_final256.json',result)
    curves=[]
    for file in sorted((out/'generated_3000').glob('*.npz')):
        with np.load(file) as z:
            curves.append(dict(parent_id=file.stem,curves=paired(z['generated'],z['reference'])))
    dump(out/'decomposition.json',curves)
    dump(out/'complete.json',dict(seconds=time.perf_counter()-start,step=step,sample_stream_sha256=stream.hexdigest()))


def report():
    parts=['<h1>P2 学习率配对修正测试</h1><p>相同P2 checkpoint、优化器动量、数据窗口和潜变量；只比较学习率1e-3与1e-4。每个seed追加3000更新。主比较固定最终步，保留起点和中间结果，不按事后最优点宣称通过。</p>']
    all_runs=[]
    for seed in (11,22,33):
        pair=[]
        for arm in ARMS:
            out=HERE/'runs'/f'{arm}_seed_{seed}';path=out/'summary.json'
            if not path.exists():parts.append(f'<p>{arm} seed {seed}：尚无结果。</p>');continue
            r=json.loads(path.read_text());all_runs.append(r);pair.append(r)
            keys=list(r['snapshots'][0]['metrics'])
            entries=[[s['step']]+[f'{s["metrics"][k]:.5f}' for k in keys] for s in r['snapshots']]
            parts.append(f'<h2>{arm} seed {seed}：记录至{r["completed_step"]}步</h2>'+table(['step']+keys,entries))
            final=out/'dev_final256.json'
            if final.exists():
                f=json.loads(final.read_text())
                parts.append('<p>最终256 flow步：</p>'+table(keys,[[f'{f["mean_parent_RMSE"][k]:.5f}' for k in keys]]))
        if len(pair)==2:
            a={s['step']:s for s in pair[0]['snapshots']};b={s['step']:s for s in pair[1]['snapshots']}
            for step in a.keys() & b.keys():
                assert a[step]['sample_stream_sha256']==b[step]['sample_stream_sha256'],'paired input streams differ'
            parts.append(f'<p>seed {seed} 已记录共同步的数据/潜变量抽样指纹一致。</p>')
    parts.append('<p>逐父状态及完整分支bootstrap点态区间见各run的dev JSON与decomposition JSON。没有预设物理等价容差，不能把较小RMSE或区间含零称为认证。若只有loss下降或时间/能量之间存在权衡，本次修正仍未解决问题。原P2、原始数据和旧诊断保持冻结。</p>')
    out=HERE/'assets';out.mkdir(exist_ok=True)
    body=''.join(parts)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P2优化器对照</title><style>body{font:15px/1.6 sans-serif;margin:24px}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:7px;border:1px solid #bbc}</style>'+body)
    dump(out/'summary.json',dict(runs=all_runs,status='DIAGNOSTIC_ONLY'))
    # Rebuild the previous diagnosis and dashboard before embedding this new page.
    import tests.bauer_p2_fit_diagnostic_20260923.publish
    page=ROOT/'tests/index.html';text=page.read_text()
    page.write_text(text.replace('</main>','<section id="p2-optimizer"><details><summary>P2最新：学习率修正对照</summary>'+body+'</details></section></main>',1))
    path=ROOT/'tests/progress_dashboard/snapshot.json';snapshot=json.loads(path.read_text())
    snapshot['sources'][str((out/'index.html').relative_to(ROOT))]=hashlib.sha256((out/'index.html').read_bytes()).hexdigest()
    dump(path,snapshot)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--index',type=int);p.add_argument('--device',choices=['cuda']);a=p.parse_args()
    if a.mode=='run':run(a.index,a.device)
    else:globals()[a.mode]()
