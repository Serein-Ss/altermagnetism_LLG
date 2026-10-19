"""Show optimizer controls separately from reference-informed diagnostics."""
import html
import json
import re
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def main():
    assets=HERE/'assets';assets.mkdir(exist_ok=True);results=[];missing=[];rows=[];errors=[]
    oracle_path=HERE/'oracle/00/complete.json'
    oracle=json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    if oracle is None:errors.append('oracle missing')
    for i,task in enumerate(tasks()):
        f=HERE/f'train/{i:02d}/complete.json'
        if not f.exists():missing.append(i);continue
        r=json.loads(f.read_text());results.append(dict(index=i,**r))
        for x in r['rows']:
            if x['start_tau']==0 and x['steps']==128:
                rows.append([i,task['seed'],task['repair'],x['stop_tau'],round(x['nearest_path_mse'],6),
                    round(x['diagnostic']['increment_variance']['rmse'],6),round(x['diagnostic']['two_time_spin_correlation']['rmse'],6)])
    for seed in (11,22,33):
        hashes=[r['first32_source_hash'] for r in results if r['task']['seed']==seed]
        if len(set(hashes))>1:errors.append(f'seed{seed} unpaired latent streams')
    dump(assets/'summary.json',dict(results=results,oracle=oracle,missing=missing,errors=errors,scientific_certification=False))
    body='<h1>单初态经验路径：训练与生成运输衔接诊断</h1>'
    body+=f'<p>训练完成 {len(results)}/12；缺失 {missing}；汇总错误 {html.escape(str(errors))}。</p>'
    body+='<p>固定一个Bauer L16训练初态、8条经验路径。fresh从头训练，warm从原全数据FM6000步恢复；各增加6000步。constant保持1e-3，decay余弦降至3e-5。warm额外用过历史数据和计算，不能解释为总成本相同。</p>'
    body+='<p>物理路径仍是1ps、0.1ps网格。运输tau0.95是有残余潜变量的平滑经验分布，绝不能冒充原tau1目标。tau1结果单列。参考来自相同偏置归一化高斯source的解析有限混合运输，不是真实LLG的解析概率律。</p>'
    body+=table(['索引','seed','方案','运输终点','最近经验路径MSE','增量方差RMSE','两时关联RMSE'],rows)
    body+='<p>最近路径标签是8条经验样本的编号，不是物理翻转机制。restart>0使用真实经验路径构造中间分布，只能定位误差，不是可部署推理改进。Oracle越界近邻延拓次数必须检查，不自动认定积分参考可靠。</p>'
    for title,obj in [('匹配潜变量的解析运输数值核验',oracle),('各运输时间误差、采样收敛和重启定位',results)]:
        body+='<details><summary>'+title+'</summary><pre>'+html.escape(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre></details>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>'
    (assets/'index.html').write_text(page)
    p=ROOT/'tests/index.html'
    if p.exists():
        text=p.read_text();start='<!-- empirical-transport-repair:start -->';end='<!-- empirical-transport-repair:end -->'
        block=start+'<section id="empirical-transport-repair">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('incomplete experiment; see summary.json')


if __name__=='__main__':main()
