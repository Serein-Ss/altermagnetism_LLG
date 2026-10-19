"""Re-evaluate frozen checkpoints; no training, no DEV-future supervision."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet
from tests.training_restart_repair.data import load_train,rows
from tests.optimizer_response_repair.run import verify as verify_previous,source
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.bauer_p2_20260923.common import dump
from .probe import draft,oriented,scores

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARMS=('before','restore_low','reset_low','restore_high','reset_high')
def checkpoint(seed,arm):
    if arm=='before':return source(seed)/'checkpoint.pt'
    i=(11,22,33).index(seed)*4+ARMS.index(arm)-1
    return ROOT/f'tests/optimizer_response_repair/train/{i:02d}/checkpoint.pt'

def prepare():
    assert not (HERE/'source_lock.json').exists()
    verify_previous();_,hashes=load_train()
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    paths += [checkpoint(seed,arm) for seed in (11,22,33) for arm in ARMS]
    for p in paths:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes);(HERE/'logs').mkdir(exist_ok=True)

def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

@torch.no_grad()
def run(index,smoke):
    require_gpu('cuda');seed=(11,22,33)[index]
    folder=HERE/('smoke' if smoke else 'results')/f'{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    nets={}
    for arm in ARMS:
        ck=torch.load(checkpoint(seed,arm),map_location='cpu',weights_only=True)
        assert ck['step']==(54000 if arm=='before' else 54512)
        assert ck['task']['seed']==seed
        net=TransitionNet().cuda();net.load_state_dict(ck['model']);net.eval();nets[arm]=net
    bank,hashes=load_train(smoke);meta=rows(smoke)
    records=[];stream=hashlib.sha256();tick=time.perf_counter()
    for sid,b in enumerate(bank):
        b=F.normalize(torch.tensor(b,device='cuda'),dim=-1)
        for rep in range(1 if smoke else 2):
            for k in ((0,5,9) if smoke else range(10)):
                x,t,s,c,angle,diag=draft(b,sid,rep,k)
                for a in (x,t,s,c,angle):stream.update(a.cpu().numpy().tobytes())
                for aug in (False,True):
                    xx,ss,cc=oriented(x,s,c,angle,aug)
                    for arm,net in nets.items():
                        pred=net.velocity(xx,t,xx[:,0],xx.new_tensor([0.,.1]))
                        assert torch.isfinite(pred).all() and float(pred[:,0].abs().max())==0
                        tangent=float((pred[:,1:]*xx[:,1:]).sum(-1).abs().max())
                        assert tangent<2e-4
                        records.append(dict(state=sid,source_group=meta[sid]['source_group'],
                            family=meta[sid]['parent']['family'],rep=rep,bin=k,augmented=aug,
                            arm=arm,values=scores(pred,ss,cc),tangent_max=tangent,
                            teacher_count=diag['teacher_count']))
    torch.cuda.synchronize()
    dump(folder/'complete.json',dict(seed=seed,smoke=smoke,records=records,probe_hash=stream.hexdigest(),
        seconds=time.perf_counter()-tick,device=torch.cuda.get_device_name(),data_hashes=hashes,
        columns=['hybrid_teacher_risk','sampled_label_risk','label_residual_power','cross_term','field_power'],
        scientific_certification=False))

def aggregate(smoke):
    import html
    seeds=(11,) if smoke else (11,22,33)
    runs=[json.loads((HERE/('smoke' if smoke else 'results')/str(s)/'complete.json').read_text()) for s in seeds]
    assert len({r['probe_hash'] for r in runs})==1
    summary=[];strata=[]
    for r in runs:
        expected=(2*1*3*2*5) if smoke else (48*2*10*2*5)
        assert len(r['records'])==expected
        for arm in ARMS:
            for aug in (False,True):
                for k in ((0,5,9) if smoke else range(10)):
                    rr=[x for x in r['records'] if x['arm']==arm and x['augmented']==aug and x['bin']==k]
                    strata.append(dict(seed=r['seed'],arm=arm,augmented=aug,bin=k,
                        values=np.mean([x['values'] for x in rr],axis=0).tolist()))
                rr=[x for x in r['records'] if x['arm']==arm and x['augmented']==aug]
                summary.append(dict(seed=r['seed'],arm=arm,augmented=aug,
                    values=np.mean([x['values'] for x in rr],axis=0).tolist()))
    out=HERE/('smoke_assets' if smoke else 'assets');out.mkdir(exist_ok=True)
    dump(out/'summary.json',dict(summary=summary,strata=strata,paired_probe_hash=runs[0]['probe_hash'],
        columns=runs[0]['columns'],scientific_certification=False))
    body='<h1>Flow目标分层复核（不是物理认证）</h1><p>不训练模型。全部64条TRAIN分支×10个均匀tau分层×2次新latent；48个状态来自24个来源父状态。增广与不增广分别报告。tau是flow运输时间，不是物理时间。这里只审计restart目标，不代表native与restart之和。</p>'
    body+='<p>旧探针前8分支与8个tau一一配对；旧结果保留。新探针去除该混杂。tau≤0.9为有限TRAIN库条件平均标签，之后沿用采样标签；有限经验分布不等于真实LLG规律。</p>'
    body+='<table border="1"><tr><th>seed/arm</th><th>旋转增强</th><th>混合教师风险</th><th>采样标签风险</th><th>标签残差功率</th><th>交叉项</th><th>场功率</th></tr>'
    for r in summary:
        body+='<tr><td>'+html.escape(str(r['seed'])+'/'+r['arm'])+'</td><td>'+str(r['augmented'])+'</td>'+''.join(f'<td>{v:.6g}</td>' for v in r['values'])+'</tr>'
    body+='</table><p>采样风险=教师风险+标签残差+交叉项。有限样本交叉项不必为零。两种风险均不是磁化、能量或动态相关误差。不能根据本表宣称物理修复。</p><p><a href="summary.json">逐时间层与种子统计</a>；<a href="../../optimizer_response_repair/assets/index.html">已有独立物理验证</a></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Transport risk audit</title>'+body)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        verify()
        run(a.index,a.smoke) if a.action=='run' else aggregate(a.smoke)
