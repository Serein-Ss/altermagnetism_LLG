"""Failure-aware repair dashboard, using the frozen physical evaluator."""
import html
import json
import re
import numpy as np
import torch
from tests.spin_path_transformer.report import analyze
from tests.cross_dimension_20260924.report import image_tag
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    body=['<h1>早期 flow 与时间联合统计：训练修复对照</h1>',
        '<p>同一细粗交互骨干、相同参考数据和物理网格、三个训练seed。不是新增网络模块；不改旧结果，不自动宣称物理通过。</p>',
        '<h2>本轮改变什么</h2>',
        table(['方案','改动'],[['baseline','原均匀运输时间FM，同期重新训练'],['importance','早期多采样，使用密度比校正，保持原期望FM目标'],
            ['early','同样早期多采样，不校正，明确提高早期权重'],['joint','原FM加训练父状态的生成集合时间统计约束'],['early_joint','早期加权加集合时间统计约束']]),
        '<p>合成任务比较前三项。一维比较baseline/early/joint/early_joint，形成2×2对照。所有版本的数据与潜变量流按seed配对；tau和辅助样本流独立。相同更新次数不代表相同算力。</p>',
        '<h2>辅助项的物理含义与限制</h2><p>只比较同一训练初态下的条件增量方差与两时刻自旋相关矩阵。每16更新，8条模型路径通过32步可微球面采样，与该训练父状态16条参考分支比较；不用真实未来作为推理输入，不逐条配对，也不强制能量守恒。</p>',
        '<p>辅助项只使用原始训练父状态的首1ps。有限分支平方统计差有抽样偏差，32步采样有离散误差，因此必须独立检查128/256步结果与未优化的能量、磁化及空间统计；不能用辅助loss下降代替验证。</p>',
        '<p>物理范围不变：Bauer开放L16/25/32、63.824849668K、1ps与0.1ps保存网格；合成时间无ps单位，运输止于0.95。全部是开发证据，不是新最终留出测试。</p>',
        '<p>合成模式理论概率为0.5，不把有限参考比例当精确目标。256条生成分支/父状态，逐父状态Wilson区间；物理指标100次整条分支bootstrap。各seed单列，无自动等价性门槛。</p>']
    results=[];missing=[];errors=[];overview=[]
    for i,task in enumerate(tasks()):
        out=HERE/f'run/{i:02d}';file=out/'complete.json'
        if not file.exists():missing.append(i);continue
        info=json.loads(file.read_text())
        try:
            cache=out/'diagnostics.json'
            if cache.exists():diagnostics=json.loads(cache.read_text())
            else:diagnostics=analyze(i,task,out,assets);dump(cache,diagnostics)
            result=dict(index=i,run=info,diagnostics=diagnostics);results.append(result)
            parents=[p for p in diagnostics['parents'] if p['group'].endswith('steps128')]
            if task['domain']=='bauer':
                ratio=np.mean([p['diagnostic']['increment_variance']['generated'][0]/p['diagnostic']['increment_variance']['reference'][0] for p in parents])
                corr=np.mean([p['diagnostic']['two_time_spin_correlation']['rmse'] for p in parents])
                overview.append([i,'Bauer',task['repair'],task['seed'],round(ratio,4),round(corr,5),'首步方差比/两时RMSE'])
            else:
                evals=[r for r in info['evaluation'] if r['steps']==128]
                overview.append([i,'合成',task['repair'],task['seed'],[round(r['positive_fraction'],4) for r in evals],
                    [round(r['direction_coherence'],4) for r in evals],'每父状态正向比例/方向一致性'])
            body.append('<h2>'+html.escape(str(dict(index=i,**task)))+'</h2>')
            body.append('<p>训练 '+str(round(info['train_seconds'],1))+'秒；GPU '+html.escape(info['device'])+'；早期tau比例 '+str(round(info['early_tau_fraction'],3))+'。</p>')
            for image in diagnostics['images']:body.append(image_tag(assets/image))
            if task['domain']=='toy':body.append(table(['父状态','flow步数','正向比例','逐点Wilson95%'],[[r['parent'],r['steps'],round(r['positive_fraction'],4),[round(v,4) for v in r['positive_wilson95']]] for r in info['evaluation']]))
            body.append('<details><summary>完整分项物理指标与不确定性</summary><pre>'+html.escape(json.dumps(diagnostics['parents'],ensure_ascii=False,indent=2))+'</pre></details>')
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    body.insert(1,f'<p>完成并分析 {len(results)}/21；缺失 {missing}；分析错误 {errors}。</p>')
    body.insert(2,table(['任务','体系','方案','seed','指标1','指标2','说明'],overview))
    section='\n'.join(body)
    text='<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+section+'</body></html>'
    tmp=assets/'index.html.tmp';tmp.write_text(text);tmp.replace(assets/'index.html')
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,scientific_certification=False))
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- flow-training-repair:start -->';end='<!-- flow-training-repair:end -->'
        block=start+'<section id="flow-training-repair">'+section+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        tmp=dashboard.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(dashboard)
    if errors:raise RuntimeError('analysis failures recorded')


if __name__=='__main__':main()
