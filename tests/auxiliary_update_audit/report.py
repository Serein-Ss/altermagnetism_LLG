import argparse,json,re
from pathlib import Path
import numpy as np
from tests.transport_audit.report import table
from tests.bauer_p2_20260923.common import dump
from .run import verify,HERE,ROOT


def main(smoke):
    verify();reports=[json.loads((HERE/('smoke' if smoke else 'audit')/f'{i:02d}/report.json').read_text()) for i in range(1 if smoke else 9)]
    if not smoke:
        for seed in (11,22,33):
            group=[r for r in reports if r['seed']==seed]
            for key in ('native_input_hash','auxiliary_input_hash'):assert len({r[key] for r in group})==1,key
    rows=[];norms=[]
    for report in reports:
        assert report['actual_model_updates']==0
        for r in report['rows']:
            label=[report['seed'],report['arm'],r['role'],r['state']]
            norms.append(label+[r['base_norm'],r['local_norm'],r['spatial_norm'],r['local_spatial_cosine'],r['baseline_update_norm']])
            for v in r['variants']:rows.append(label+[v['mode'],v['weight'],v['weighted_gradient_ratio'],v['raw_frequency_normalized_ratio'],v['cosine_base_aux'],v['update_increment_ratio'],v['clip_factor'],v['first_order_aux_change'],v['baseline_first_order_aux_change']])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(reports=reports,rows=rows,norms=norms,pairing_errors=[],actual_model_updates=0,scientific_certification=False))
    body='<h1>Bauer辅助损失：梯度—裁剪—Adam影子更新诊断</h1><p>上一轮9项训练及报告全部完成，但局部/空间辅助与control物理指标几乎重合，没有稳定增量改善。本轮修复原日志只报非零梯度、没有跟踪实际更新作用的缺口。所有源checkpoint冻结，零实际训练更新。不是训练新模型或物理修复成功。</p>'
    body+='<p>9个126144步checkpoint、三个seed和三旧训练臂；每模型8个固定TRAIN快照（真实/模型角色0/2，状态0/12/24/36）。同seed输入和latent配对，64未来128步实际采样，局部矩和空间矩分别反传。影子参数及Adam复制来自真实checkpoint，原权重/优化器不变，裁剪10。权重0.1和1仅影子诊断，不自动采用weight1训练。</p>'
    if smoke:body+='<p>仅checkpoint0、一个状态、8flow步的软件预检，不能解释正式128步梯度大小或物理准确性。</p>'
    body+=table(['seed','old arm','role','state','native+.25teacher norm','local norm','spatial norm','local/spatial cosine','baseline update norm'],norms)
    body+=table(['seed','old arm','role','state','aux mode','weight','weighted grad/base','ratio divided64','base/aux cosine','Adam delta increment/base','clip factor','aux grad dot shadow delta','aux grad dot base delta'],rows)
    body+='<p>Adam delta increment/base比较加入辅助后参数更新差与原更新大小，比非零梯度更接近实际作用。dot负只是一阶下降预测，不证明有限步loss或物理统计改善。ratio/64是原始梯度时间平均量级，不能替代Adam动量、频率、非线性更新。只有终态Adam和lr，不能还原6000步全过程；需要说明这一限制而非宣称权重失效已被证实。</p>'
    body+='<p>原生数据流恢复；teacher本轮在每probe角色内选三个相邻bank ID各16分支，而非历史跨角色完整48状态调度，用于有限配对梯度诊断。旧native/teacher冲突实验是早期不同模型，不覆盖本轮部署矩损失。本轮TRAIN固定状态/latent不是独立物理总体统计，不把参数或格点当样本，不做显著性或等价验收。没有DEV训练、数据/split/温度/边界/时间修改、Hamiltonian系数输入、生产晋升或推送。</p><a href="summary.json">全部分项数据</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1700px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';s=path.read_text();start='<!-- auxiliary-update-audit:start -->';end='<!-- auxiliary-update-audit:end -->'
        block=start+'<section id="auxiliary-update-audit"><h2>辅助损失实际更新作用诊断</h2><a href="auxiliary_update_audit/assets/index.html">三seed冻结模型：梯度、裁剪、Adam影子更新</a></section>'+end
        s=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,s,flags=re.S) if start in s else s.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(s);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
