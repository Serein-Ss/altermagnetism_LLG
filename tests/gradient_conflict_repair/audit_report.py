"""Report frozen gradient evidence without claiming a repaired physical model."""
import argparse
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import HERE,ROOT,dump
from tests.transport_audit.report import table


def main(smoke):
    results=[];rows=[]
    for i in range(3):
        p=HERE/('extended_audit_smoke' if smoke else 'extended_audit')/f'{i:02d}/summary.json'
        r=json.loads(p.read_text());assert r['seed']==(11,22,33)[i] and r['frozen_weights']
        assert r['batches']==(32 if smoke else 8192)
        final=next(x for x in r['blocks'] if x['batches']==r['batches'])
        rows.append([r['seed'],r['batches'],final['cosine'],r['same_objective_half_cosine']['0'][0],r['same_objective_half_cosine']['1'][0],final['native_norm'],final['replay_norm'],r['seconds']])
        results.append(r)
    assets=HERE/('audit_smoke_assets' if smoke else 'audit_assets');assets.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharey=True)
    for ax,r in zip(axes,results):
        widths=sorted({b['batches'] for b in r['blocks']})
        for w in widths:
            v=[b['cosine'] for b in r['blocks'] if b['batches']==w]
            ax.scatter(np.full(len(v),w),v,s=9,alpha=.35,color='#0072B2')
            ax.scatter(w,np.mean(v),marker='D',s=35,color='#D55E00')
        ax.set_xscale('log',base=2);ax.axhline(0,color='black',lw=.8);ax.set_ylim(-1.05,1.05)
        ax.set_title(f"Seed {r['seed']}");ax.set_xlabel('Batches averaged per block')
    axes[0].set_ylabel('Native / restart gradient cosine')
    fig.suptitle('Frozen TRAIN gradients; dots=disjoint blocks; diamonds=descriptive mean'+(' (SMOKE)' if smoke else ''))
    fig.savefig(assets/'gradient_convergence.png',dpi=180);plt.close(fig)
    dump(assets/'summary.json',dict(results=results,table=rows,physical_repair_validated=False,projection_training_submitted=False))
    body='<h1>梯度冲突检查：先分离随机噪声，再决定修复</h1>'
    body+='<p>'+('仅32批次软件冒烟。' if smoke else '三训练种子各8192批次；模型权重冻结。')+'本页不是新模型的物理结果，也不代表路径分布已修复。</p>'
    body+='<p>初次128批次中，两类监督的负夹角比例45%–51%，与同类监督换批次的41%–52%相近；128批平均cosine为−0.133、−0.143、+0.157，方向不稳定。因此未提交9组投影长训练，而是补充本次高精度诊断。保留通过22项回归与GPU冒烟的sum/project/rescale候选。</p>'
    body+='<img style="max-width:100%" src="gradient_convergence.png"><p>横轴是估计一次平均梯度所合并的批次数；纵轴小于零表示夹角相反。蓝点是该尺度互不重叠的批次块，橙点是描述平均；不同尺度复用同一数据，不是独立重复，点的散布不是物理置信区间。</p>'
    body+=table(['训练seed','总批次','全部平均梯度cosine','原始监督前后半一致性','重启监督前后半一致性','原始平均梯度范数','重启平均梯度范数','计算秒'],rows)
    body+='<p>只有平均梯度在重复块/种子上有可重复的冲突，且同类梯度自身可稳定估计，才支持进一步测试针对性投影。否则不能把小批次噪声当作已定位病因。即使冲突稳定，仍需sum、方向投影、等范数缩放三组物理统计对照；梯度证据本身不说明生成均值、涨落、能量或1ps路径已正确。</p>'
    body+='<p>不新增LLG数据，不使用开发参考训练，不改变Hamiltonian、边界、时间网格或网络。<a href="summary.json">全部诊断数据</a>；<a href="../smoke_assets/index.html">修复候选的软件冒烟（不是正式模型结果）</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- gradient-conflict-audit:start -->';end='<!-- gradient-conflict-audit:end -->'
        section=start+'<section id="gradient-conflict-audit"><h2>最新：梯度冲突与训练噪声诊断</h2><p>冻结模型诊断，不是新模型物理认证。<a href="gradient_conflict_repair/audit_assets/index.html">查看诊断、后续修复依据和原始数据</a></p><img style="max-width:100%" src="gradient_conflict_repair/audit_assets/gradient_convergence.png"></section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,text,flags=re.S)
        else:text=text.replace('</body>',section+'</body>') if '</body>' in text else text+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
