"""Numerical diagnostic report; preserves previous dashboard sections."""
import html
import json
import re
from .run import HERE,ROOT,tasks,dump


def table(headers,rows):
    return '<table border="1" cellpadding="6"><tr>'+''.join('<th>'+html.escape(str(k))+'</th>' for k in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def main():
    assets=HERE/'assets';assets.mkdir(exist_ok=True);done=[];missing=[]
    body=['<h1>冻结 checkpoint：运输阶段、采样器与训练阶段诊断</h1>',
        '<p>不重训、不改物理参数。正常生成和借用正确中间分布的诊断严格分开。tau是flow运输时间，不是ps。</p>',
        '<p>合成实验每个固定父状态使用256个新随机源，理论正向概率为0.5。Wilson区间仅描述该固定模型/父状态的生成抽样误差；各训练seed分开报告，不作多重比较通过判定。</p>',
        '<p>一维使用L16每种制备家族的第一个开发父状态：128生成/16真实分支，比较3000和6000更新的已有模型。物理跨度仍1ps、保存间隔0.1ps。不是新最终测试。</p>',
        '<p>Euler512与midpoint256均为512次网络求值。先检查数值差异，再解释物理偏差；新积分器仅为测试实现。</p>',
        '<p>中间分布重启使用已知目标轨迹，不能部署、不能当作模型正常性能，更不是将真实未来加入正常推理条件。</p>']
    for i,task in enumerate(tasks()):
        file=HERE/f'run/{i:02d}/complete.json'
        if not file.exists():missing.append(i);continue
        result=json.loads(file.read_text());done.append(result)
        body.append('<h2>'+html.escape(str(task))+'</h2>')
        if task['domain']=='toy':
            rows=[[r['parent'],r['start_tau'],r['method'],r['steps'],round(r['positive_fraction'],4),
                   [round(v,4) for v in r['positive_wilson95']],round(r['direction_coherence'],4),round(r['rotation_amplitude'],4)] for r in result['rows']]
            body.append(table(['父状态序号','起点tau（非0=借助参考）','积分器','步数','正向比例','逐点95%区间','方向一致性','幅度'],rows))
            body.append(table(['父状态序号','tau','精确条件速度MSE','支持域外数'],[[r['parent'],r['tau'],round(r['mse'],6),r['unsupported']] for r in result['velocity_errors']]))
        else:
            rows=[]
            for r in result['rows']:
                d=r['diagnostic'];v=d['increment_variance']
                rows.append([r['checkpoint_step'],r['family'],r['method'],r['steps'],round(v['generated'][0]/v['reference'][0],4),round(v['rmse'],5),round(d['two_time_spin_correlation']['rmse'],5)])
            body.append(table(['训练更新','制备族','积分器','步数','首步方差比','整窗增量方差RMSE','两时相关RMSE'],rows))
        body.append('<details><summary>完整统计、逐点区间与耗时</summary><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></details>')
    body.insert(1,f'<p>完成 {len(done)}/9；缺失 {missing}。无自动科学通过声明。</p>')
    section='\n'.join(body)
    text='<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+section+'</body></html>'
    (assets/'index.html').write_text(text);dump(assets/'summary.json',dict(results=done,missing=missing,scientific_certification=False))
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- transport-audit:start -->';end='<!-- transport-audit:end -->'
        block=start+'<section id="transport-audit">'+section+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        tmp=dashboard.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(dashboard)


if __name__=='__main__':main()
