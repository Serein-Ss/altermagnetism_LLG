import html
import json
import re
import h5py
import numpy as np
import torch
from tests.moment_bias_repair.report import analyze
from tests.reference_resolution.report import compare,brief
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];view=[];expanded_view=[]
    for i,task in enumerate(tasks()):
        out=HERE/f'train/{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            run=json.loads((out/'complete.json').read_text());ps=analyze(out)
            assert len([p for p in ps if p['split']=='dev' and p['group'].endswith('steps128')])==36
            assert all(p['norm_error']<2e-6 and p['anchor_error']<2e-6 for p in ps)
            for split in ('train','dev'):
                chosen=[p for p in ps if p['split']==split and p['group'].endswith('steps128')]
                view.append([task['seed'],task['repair'],split,*np.mean([brief(p) for p in chosen],axis=0).tolist()])
            expanded=[]
            with h5py.File(out/'expanded.h5') as h:
                assert h.attrs['complete'] and len(h)==18
                for name,g in h.items():
                    j=int(name.removeprefix('parent'));parent=json.loads(g.attrs['parent'])
                    with h5py.File(ROOT/f'tests/reference_resolution/reference/{j:02d}/paths.h5') as r:
                        assert r.attrs['complete'];refs={k:r[f'{k}/spins'][:] for k in ('base','fine')}
                    for sampler in ('euler128','midpoint128'):
                        x=g[sampler][:]
                        assert len(x)==256 and np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6
                        assert np.max(abs(x[:,0]-refs['base'][:1,0]))<2e-6
                        comparisons={k:compare(x,v) for k,v in refs.items()}
                        expanded.append(dict(parent=parent,sampler=sampler,**comparisons))
            for sampler in ('euler128','midpoint128'):
                for family in ('tilted','interior_domain','single_wall'):
                    chosen=[p for p in expanded if p['sampler']==sampler and p['parent']['family']==family]
                    expanded_view.append([task['seed'],task['repair'],sampler,family,
                        *np.mean([brief(p['fine']) for p in chosen],axis=0).tolist(),
                        float(np.mean([p['fine']['diagnostic']['increment_variance']['difference'][0] for p in chosen]))])
            dump(out/'diagnostics.json',dict(parents=ps,expanded=expanded))
            results.append(dict(index=i,task=task,run=run,parents=ps,expanded=expanded))
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    for seed in (11,22,33):
        hashes=[r['run']['stream_hash'] for r in results if r['task']['seed']==seed]
        if len(set(hashes))>1:errors.append(dict(seed=seed,error='unpaired data/source/tau'))
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,overview=view,expanded_overview=expanded_view,scientific_certification=False))
    body=f'<h1>初态逐块调制／相对物理时间偏置修复</h1><p>完成{len(results)}/9；缺失{missing}；错误{html.escape(str(errors))}。</p>'
    body+='<p>原模型control、初态逐块FiLM anchor、时间注意力相对时间偏置relative_time：各三个seed。均从原control step18000零扰动扩展，续训6000步；新增模块不接收真实未来或Hamiltonian。相同数据/源噪声/运输时刻、相同更新次数；参数与计算不完全相同，不能直接归因为唯一机制。</p>'
    body+='<h2>原18训练／36开发初态</h2>'+table(['seed','方案','集合','增量方差RMSE','两时关联RMSE','能量均值RMSE'],view)
    body+='<h2>扩充参考：每初态256条生成，对比独立128条半步长LLG</h2>'+table(['seed','方案','采样器','初态类别','增量方差RMSE','两时关联RMSE','能量均值RMSE','首段增量方差差：生成−参考'],expanded_view)
    body+='<p>完整结果同时保存原步长与半步长参考比较、按父状态误差与100次整分支bootstrap。Euler128与中点128使用配对潜变量；中点每步两次网络调用，不等成本。逐点区间含零不证明等价，无自动通过分数；这些是反复使用的开发数据，不是最终独立检验。</p>'
    body+='<details><summary>参数量、耗时、完整路径统计及区间</summary><pre>'+html.escape(json.dumps(results,ensure_ascii=False,indent=2))+'</pre></details>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- condition-time-repair:start -->';end='<!-- condition-time-repair:end -->'
        block=start+'<section id="condition-time-repair">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if missing or errors:raise RuntimeError('Incomplete experiment, see summary')


if __name__=='__main__':main()
