"""Versioned empirical correction and full-data development report."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.empirical_transport_repair.run import tasks as empirical_tasks
from tests.moment_bias_repair.report import analyze
from tests.transport_audit.report import table
from .metrics import compare
from .run import HERE, ROOT, tasks, dump


def correction():
    rows = []
    for i, task in enumerate(empirical_tasks()):
        path = ROOT/f'tests/empirical_transport_repair/train/{i:02d}/paths.h5'
        with h5py.File(path) as h:
            assert h.attrs['complete']
            reference = h['empirical_targets'][:]
            generated = h['start0.0_steps128_euler_stop1.0/generated'][:]
        rows.append(dict(index=i, task=task, source=str(path.relative_to(ROOT)),
            corrected=compare(generated, reference, 'empirical'),
            historical_estimand=compare(generated, reference, 'population', repeats=0)))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--correction-only', action='store_true'); args = p.parse_args()
    torch.set_num_threads(1); assets = HERE/'assets'; assets.mkdir(exist_ok=True)
    corrected = correction(); dump(assets/'empirical_correction_v2.json', corrected)
    body = '<h1>经验分布评估修正与完整数据学习率对照</h1>'
    body += '<p>固定8条路径的经验分布方差使用分母8；独立LLG总体的方差估计仍使用N−1。下表只重算原单父状态拟合实验的完整运输终点tau=1，不改变其1ps物理时长。历史报告保留；此版本替代其中经验拟合方差的解读，不把经验拟合当作物理认证。</p>'
    view = []
    for r in corrected:
        old = r['historical_estimand']['metrics']; new = r['corrected']['metrics']
        view.append([r['task']['seed'], r['task']['repair'],
            round(old['increment_variance']['rmse'], 6), round(new['increment_variance']['rmse'], 6),
            round(new['two_time_spin_correlation']['rmse'], 6)])
    body += table(['seed','方案','旧口径增量方差RMSE','经验目标修正RMSE','两时关联RMSE（不变）'], view)
    results = []; missing = []; errors = []; overview = []
    if not args.correction_only:
        for i, task in enumerate(tasks()):
            out = HERE/f'train/{i:02d}'
            if not (out/'complete.json').exists(): missing.append(i); continue
            try:
                info = json.loads((out/'complete.json').read_text()); parents = analyze(out)
                assert sum(p['split']=='dev' and p['group'].endswith('steps128') for p in parents)==36
                assert all(p['norm_error']<2e-6 and p['anchor_error']<2e-6 for p in parents)
                dump(out/'diagnostics_v2.json', dict(reference_kind='population', parents=parents))
                for split in ('train','dev'):
                    ps = [p for p in parents if p['split']==split and p['group'].endswith('steps128')]
                    vals = [np.mean([p['diagnostic'][k]['rmse'] for p in ps]) for k in ('increment_variance','two_time_spin_correlation')]
                    vals += [np.mean([p['physical'][k]['rmse'] for p in ps]) for k in ('energy_mean','spatial')]
                    overview.append([task['seed'],task['repair'],split,len(ps),*[round(float(v),6) for v in vals]])
                results.append(dict(index=i,task=task,run=info,parents=parents))
            except Exception as exc: errors.append(dict(index=i,error=repr(exc)))
        for seed in (11,22,33):
            hashes = [r['run']['full_data_source_tau_hash'] for r in results if r['task']['seed']==seed]
            if len(set(hashes))>1: errors.append(dict(seed=seed,error='Unpaired training streams'))
        dump(assets/'summary.json', dict(results=results, missing=missing, errors=errors, scientific_certification=False))
        body += f'<h2>完整数据续训：完成{len(results)}/6</h2><p>缺失：{missing}；错误：{html.escape(str(errors))}</p>'
        body += '<p>三seed配对，从各自原FM6000步继续6000步，唯一干预为恒定0.001或余弦降至0.00003。使用最终checkpoint，不按开发集挑选中间步。下表均为总体口径N−1；train只作重复使用的训练校准，dev是独立父状态的开发证据，不是新最终留出集。</p>'
        body += table(['seed','方案','集合','父状态','增量方差RMSE','两时关联RMSE','能量均值RMSE','空间RMSE'], overview)
        body += '<p>逐父状态保留100次整分支bootstrap、32/128/256步采样检查、模长与初帧误差。区间包含零不等于统计等价；未设置自动通过阈值。全部分项及物理时间保存在JSON和paths.h5中。</p>'
        body += '<details><summary>完整开发评估</summary><pre>'+html.escape(json.dumps(results,ensure_ascii=False,indent=2))+'</pre></details>'
    else:
        body += '<p>完整训练集三seed对照尚待计算；没有新独立开发结果。</p>'
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>'
    (assets/'index.html').write_text(page)
    dashboard = ROOT/'tests/index.html'
    if dashboard.exists():
        text = dashboard.read_text(); start='<!-- full-data-decay:start -->'; end='<!-- full-data-decay:end -->'
        block = start+'<section id="full-data-decay">'+body+'</section>'+end
        if start in text: text = re.sub(re.escape(start)+'.*?'+re.escape(end), lambda _:block, text, flags=re.S)
        else: text = text.replace('</body>', block+'</body>') if '</body>' in text else text+block
        temp = dashboard.with_suffix('.html.tmp'); temp.write_text(text); temp.replace(dashboard)
    if errors or missing: raise RuntimeError('Incomplete campaign; see summary.json')


if __name__ == '__main__': main()
