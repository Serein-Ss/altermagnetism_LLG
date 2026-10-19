"""Separate deployable sampling from reference-informed transport localization."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.reference_resolution.report import compare,brief
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,parents,CUTS,dump


def geometry(x):
    # Per whole path; these are geometry features of transport states, not LLG-time validation.
    return np.stack([(x[:,1:]*x[:,:1]).sum(-1).mean((1,2)),
                     (x[:,1:,1:]*x[:,1:,:-1]).sum(-1).mean((1,2)),
                     (x[:,2:]*x[:,1:-1]).sum(-1).mean((1,2))],axis=-1)


def geometry_comparison(x,y):
    a=geometry(x);b=geometry(y);rng=np.random.default_rng(93777)
    # Paired source and base-future draws are the independent diagnostic units.
    diff=a-b;boot=np.array([diff[rng.integers(len(diff),size=len(diff))].mean(0) for _ in range(100)])
    return dict(features=['S0_overlap','neighbor_alignment','adjacent_frame_alignment'],
                on_policy_mean=a.mean(0).tolist(),teacher_mean=b.mean(0).tolist(),
                difference=diff.mean(0).tolist(),pointwise95=np.quantile(boot,[.025,.975],axis=0).tolist())


def analyze(index,smoke):
    out=HERE/('smoke' if smoke else 'run')/f'{index:02d}';run=json.loads((out/'complete.json').read_text())
    rows=[];transport=[];solver=[]
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete']
        for name,g in h.items():
            i=int(name.removeprefix('parent'));parent=parents()[i]
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as ref:
                assert ref.attrs['complete'];base=ref['base/spins'][:];fine=ref['fine/spins'][:]
            arrays={key:v[:] for key,v in g['final'].items()}
            for key,x in arrays.items():
                informed=bool(g[f'final/{key}'].attrs['reference_informed'])
                cf=compare(x,fine)
                # Base used to build restart states: intentionally omit misleading independent CI.
                cb=None if informed else compare(x,base)
                rows.append(dict(parent=parent,parent_index=i,sampler=key,reference_informed=informed,
                                 base=cb,fine=cf,count=len(x)))
            for tau in CUTS:
                transport.append(dict(parent=parent,tau=tau,**geometry_comparison(g[f'on_policy/tau{tau}'][:],g[f'teacher/tau{tau}'][:])))
            low=8 if smoke else 128;high=low*4
            for a,b in ((f'uniform{low}',f'late{low}'),(f'uniform{low}',f'uniform{high}'),
                        (f'late{low}',f'late{high}'),(f'uniform{high}',f'late{high}')):
                solver.append(dict(parent=parent,pair=[a,b],same_latent_path_rms=float(np.sqrt(np.mean((arrays[a]-arrays[b])**2)))))
    dump(out/'report.json',dict(task=tasks()[index],run=run,rows=rows,transport=transport,solver=solver,scientific_certification=False))


def aggregate(smoke):
    results=[];missing=[];errors=[];view=[];transport=[];sampling=[]
    for i in ([0] if smoke else range(len(tasks()))):
        try:
            path=HERE/('smoke' if smoke else 'run')/f'{i:02d}/report.json'
            if not path.exists():missing.append(i);continue
            r=json.loads(path.read_text());results.append(r);s=r['task']['seed']
            for row in r['rows']:
                p=row['parent'];view.append([s,p['L'],p['family'],p['parent_id'],row['sampler'],
                    '参考辅助，仅诊断' if row['reference_informed'] else '正常生成',*[round(v,7) for v in brief(row['fine'])]])
            for row in r['transport']:
                p=row['parent'];transport.append([s,p['L'],p['parent_id'],row['tau'],*[round(v,7) for v in row['difference']]])
            for row in r['solver']:
                p=row['parent'];sampling.append([s,p['L'],p['parent_id'],' / '.join(row['pair']),row['same_latent_path_rms']])
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,scientific_certification=False))
    body=f'<h1>冻结模型：运输误差定位与非均匀采样验证</h1><p>完成{len(results)}；缺失{missing}；错误{html.escape(str(errors))}。'+('冒烟测试，不用于科学结论。' if smoke else '开发诊断，不是最终独立测试。')+'</p>'
    body+='<p>冻结control step24000三组seed，Bauer开放链L16/25/32；1ps、0.1ps输出。每父状态128条生成，参考base128与fine128。正常采样比较均匀/末段加密128与512步，每步两次网络调用。只改变辅助运输积分，不改变物理时间。不新增训练。</p>'
    body+='<p>参考辅助重启从真实未来与latent的球面插值开始，无法部署；更接近目标和更短剩余积分本身就可能改善结果，不能独自证明分布偏移因果，也不能当成修复成功。比较对象fine参考未用于重启。中间transport状态的几何差异不是物理轨迹误差。</p>'
    body+='<h2>终态路径统计（相对独立fine参考，数值越小偏差越小）</h2>'+table(['训练seed','L','初态族','父状态','采样','用途','增量方差RMSE','两时关联RMSE','平均能量RMSE'],view)
    body+='<h2>运输过程中：正常生成 − 参考插值的几何均值差</h2>'+table(['seed','L','父状态','运输tau','初态重合度差','邻点对齐差','相邻帧对齐差'],transport)
    body+='<h2>相同潜变量的采样积分差异（非模型对LLG误差）</h2>'+table(['seed','L','父状态','采样器对','路径RMS'],sampling)
    body+='<p>完整JSON保留100次整条分支bootstrap逐点区间、各物理指标和耗时。时间/格点不是独立样本；区间含零不证明等价。参考—参考差异不是容差。不自动选择最好模型或晋升采样器。硬约束通过与路径统计可信分开报告。</p>'
    body+='<p><a href="summary.json">完整指标与不确定区间JSON</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        dashboard=ROOT/'tests/index.html';text=dashboard.read_text();start='<!-- transport-path-repair:start -->';end='<!-- transport-path-repair:end -->'
        body=body.replace('href="summary.json"','href="transport_path_repair/assets/summary.json"')
        block=start+'<section id="transport-path-repair">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=dashboard.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(dashboard)
    if errors or missing:raise RuntimeError('Incomplete diagnostic: see summary')


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--smoke',action='store_true');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
