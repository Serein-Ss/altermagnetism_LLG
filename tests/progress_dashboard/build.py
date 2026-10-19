"""Build a read-only, offline progress snapshot from existing experiment reports."""
import base64
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from .project_sections import sections

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = TESTS/relative
    if not path.exists():
        return None
    record(path)
    return json.loads(path.read_text())


def esc(x):
    return html.escape(str(x), quote=True)


def link(path, title):
    path = Path(path)
    if path == TESTS/'bauer_closed_loop_20260919/assets/index.html' and (HERE/'styled/index.html').exists():
        path=HERE/'styled/index.html'
    if path == TESTS/'bauer_2d_review_20260918/assets/index.html' and (HERE/'styled/data_review.html').exists():
        path=HERE/'styled/data_review.html'
    if not path.exists():
        return f'<span class="muted">{esc(title)}（尚无文件）</span>'
    href = os.path.relpath(path, TESTS)
    return f'<a href="{esc(href)}">{esc(title)}</a>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def image(relative, caption):
    path = TESTS/relative
    if relative.startswith('bauer_closed_loop_20260919/assets/') and (HERE/'styled'/path.name).exists():
        path=HERE/'styled'/path.name
    if not path.exists():
        return f'<p class="warning">图片尚未生成：{esc(relative)}</p>'
    record(path)
    mime = 'gif' if path.suffix == '.gif' else 'png'
    return f'<figure><img loading="lazy" alt="{esc(caption)}" src="data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{esc(caption)}</figcaption></figure>'


def main():
    SOURCES.clear()
    campaign = read('bauer_two_stage_20260919/submission.json')
    campaign_status = read('bauer_two_stage_20260919/assets/status.json')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    closed = read('bauer_closed_loop_20260919/assets/report.json')
    audit = read('bauer_closed_loop_20260919/audit.json')
    miltat = read('miltat_1ns_20260919/results/report.json')
    inventory = []
    for length in (16, 25, 32):
        manifest = read(f'bauer_active_data_20260918/data/L{length}/manifest.json')
        complete = read(f'bauer_active_data_20260918/data/L{length}/complete.json')
        if manifest is None:
            inventory.append([length, '缺失', '—', '—', '—', '—'])
            continue
        for split in ('train', 'dev', 'test'):
            rows = [r for r in manifest if r['split'] == split]
            inventory.append([length, split, len({r['parent_id'] for r in rows}),
                              sum(r['branches'] for r in rows if r['arm']=='base'),
                              sum(r['branches'] for r in rows if r['arm']=='fine'),
                              complete['status'] if complete else '无完成报告'])
    parts = ['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>磁性随机路径研究 · 测试主网页</title>
<style>
:root{color-scheme:light;--ink:#182b3a;--muted:#526575;--line:#d7e0e7;--blue:#075985}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f2f5f8;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}
header{background:#15364c;color:white;padding:38px max(24px,calc((100vw - 1200px)/2))}header p{max-width:1000px;color:#e1edf5}
nav{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;flex-wrap:wrap;z-index:2}
main{max-width:1240px;margin:auto;padding:22px}section{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin:0 0 22px;scroll-margin-top:110px}
h1{font-size:30px;line-height:1.4}h2{font-size:24px;margin-top:0}h3{font-size:19px}a{color:var(--blue);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #a74900;outline-offset:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.card{padding:18px;background:#eef5f9;border-radius:8px}.card strong{display:block;font-size:24px}.muted,figcaption{color:var(--muted)}.warning{background:#fff5df;border-left:5px solid #a85a00;padding:14px}.good{background:#e8f5ee;border-left:5px solid #247445;padding:14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf3f7}code{overflow-wrap:anywhere;background:#eef2f6;padding:2px 5px}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:20px 0}img{width:100%;height:auto}figcaption{font-size:14px}summary{cursor:pointer;font-weight:650;padding:12px;background:#edf3f7;border-radius:6px}details{margin:12px 0}li{margin:7px 0}.formula{background:#f2f6fa;padding:12px;overflow-wrap:anywhere}
@media(max-width:650px){main{padding:10px}section{padding:16px}header{padding:25px 18px}h1{font-size:25px}nav{position:static;gap:12px}table{min-width:600px}}
</style></head><body><header><h1>磁性随机路径研究 · 测试主网页</h1><p>一个入口检查：主线 → 数据 → 模型 → 结果 → 问题 → 下一步。参考模拟、生成模型结果和历史探索分别标记。</p>''',
             f'<p>本地报告快照：{esc(stamp)}。不是实时任务监控；报告不存在时显示缺失，不填零、不推断成功。</p></header>',
             '<nav aria-label="页面导航">'+''.join(f'<a href="#{anchor}">{title}</a>' for anchor,title in [('overview','当前结论'),('plan','研究主线'),('data','一维数据'),('models','模型闭环'),('metrics','指标解释'),('figures','看图检查'),('systems','二维/三维'),('next','下一步'),('files','结果位置'),('history','历史归档')])+'</nav><main>',
             '<section id="overview"><h2>当前结论：流程已跑通，物理统计尚未完全通过</h2><p>用同一哈密顿量体系的小尺寸数据学习条件随机路径，验证在更大尺寸上的统计保真与计算价值。当前首例为 Bauer 一维开放链；不是跨材料通用 checkpoint。</p>']
    runs = closed.get('runs', []) if closed else []
    diagnosis = TESTS/'bauer_p2_diagnostics_20260923/assets/index.html'
    if diagnosis.exists():
        parts.append('<p>'+link(diagnosis,'P2固定模型：训练/开发、初态族和时间分解诊断')+'</p>')
    p2 = TESTS/'bauer_p2_20260923/assets/fragment.html'
    if p2.exists():
        record(p2)
        from tests.bauer_p2_20260923.architecture_web import embed
        parts.append(embed(p2.read_text()))
    current = TESTS/'literature_path_campaign_20260920/assets/fragment.html'
    if current.exists():
        record(current)
        parts.append(current.read_text())
    latest = TESTS/'cross_system_audit_20260920/assets/index.html'
    if latest.exists():
        record(latest)
        body = latest.read_text().split('<body>',1)[1].split('</body>',1)[0]
        body = body.replace('<img alt=', '<img loading="lazy" data-frames="11" alt=')
        parts.append('<div class="warning"><h2>已完成的前一轮原因诊断（保留全部主要图与二维生成GIF）</h2><p>以下是修正前的局部原型与一维诊断，不冒充新模型结果。</p></div>'+body)
    if campaign:
        parts.append('<p class="warning">新一轮两阶段实验已实现测试代码并提交，初态模型、路径噪声对照、四组闭环、L50/L100推广及速度比较按依赖执行。以下12次运行的数值仍是上一轮先导，不冒充新实验结果。'+link(TESTS/'bauer_two_stage_20260919/assets/index.html','查看新一轮任务与实际结果')+'</p>')
        if campaign_status:
            parts.append('<p>新一轮完整评估报告：'+('已有（是否通过请看报告）' if campaign_status.get('assessment_available') else '尚无，不能判断推广成功')+'。</p>')
    chosen = next((r for r in runs if r['name']==closed.get('selected_for_display')), None) if closed else None
    passed = sum(v['status']=='WITHIN_PILOT_MARGIN' for v in chosen['test']['statistics'].values()) if chosen else None
    parts.append('<div class="cards">'+''.join(f'<div class="card">{esc(label)}<strong>{esc(value)}</strong><span>{esc(note)}</span></div>' for label,value,note in [
        ('本轮完整运行', f'{len(runs)}/12', 'A/B/C/D × 三个训练 seed'),
        ('生成窗口','1 ps','11帧；保存间隔0.1 ps'),
        ('开发集展示模型',chosen['name'] if chosen else '未完成','不按测试效果选模型'),
        ('先导指标在线内',f'{passed}/10' if passed is not None else '未知','不是完整路径分布认证')])+'</div>')
    parts.append('<p class="warning">研究结论（人工说明，依据本轮报告）：部分磁化、能量和空间统计较接近；时间关联和局部增量仍需核验。真实参考之间的增量比较也超出原容差，因此不能把全部超线归因于模型。大尺寸、长时间、温度和边界迁移尚未证实。</p><p>只下载本 HTML 可看内嵌图，但子页面不会自动下载。需要完整跳转时，请下载离线ZIP，整体解压后打开其中的 index.html。不要只从压缩包取出一个文件。离线包不包含H5数据与checkpoint，相关入口会显示明确说明。</p></section>')
    parts.append('''<section id="plan"><h2>核心思想、创新假设与执行顺序</h2>
<p>学习 P[未来磁矩路径 | 完整初态、条件]，不是只拟合平均磁化。常规使用接口目标为尺寸、温度、逐轴边界、seed、总时长；体系及初态准备协议绑定模型包。当前先验证真实初态 → 生成未来，初态生成器后置。</p>
<ol><li><b>潜在贡献：</b>小体系局部随机规律能否迁移成大体系的时空统计与机制概率。</li><li><b>方法假设：</b>联合时间建模与共享多尺度空间交互能否提供可复现收益；不能仅因模块存在就认定学会长波模态。</li><li><b>计算价值：</b>比较匹配物理精度的端到端成本，包含数据和训练摊销；当前尚无最终加速证明。</li></ol>
<p>推进顺序：可信参考 → 小体系单块 → 同窗尺寸推广 / 小体系时间接续分别验证 → 大体系长轨迹 → 文献统计与成本。新的物理发现暂不前置。</p></section>''')
    parts.append('<nav style="position:static" aria-label="项目说明导航"><a href="#architecture">模型架构</a><a href="#novelty">创新对比</a><a href="#requirements">数据路线</a></nav>')
    parts.append(sections(table))
    parts.append('<section id="data"><h2>一维数据：长轨迹复现与训练分支库不是同一批数据</h2>')
    parts.append(table(['数据','尺寸与时长','用途','边界'],[
        ['Bauer图2参考','L100，3条10 ns','文献定性对照、观察成核/畴壁','不是模型输出；不是完整概率认证'],
        ['小体系分支库','L16/25/32，每条100 ps','同初态多噪声后继、训练与独立检验','本轮模型只生成1 ps'],
    ]))
    parts.append('<p>小体系参数：J=50 meV；K/J=0.1；阻尼0.1；约63.825 K；无外场；开放边界。基础积分0.25 fs，开发细步长0.125 fs，保存0.1 ps。以下数量直接读取manifest。</p>')
    parts.append(table(['L','划分','父状态数','基础轨迹','细步长轨迹','数据完成记录'],inventory))
    parts.append('''<h3>父状态如何选择？</h3><p>六类：正向、反向、倾斜、边缘反向域、内部反向域、单畴壁。每尺寸每族训练24候选选8个，开发4选2，测试4选2。训练以标准化局部描述符做最远点采样；开发/测试随机选，不按未来事件挑选。正反向先经历20 ps热演化，其余为带扰动人工构型，不冒充平衡样本。</p>
<p>每训练父状态8个真实噪声未来，每开发/测试父状态16个；开发另有16个细步长未来。每个模型评价生成16条；确定性D的同初态输出没有随机多样性。父来源先划分再切窗，同源窗口不是独立重复。</p>''')
    parts.append('<p>'+link(TESTS/'bauer_2d_review_20260918/assets/index.html','打开完整一维数据/二维文献检查页')+'</p>')
    parts.append('<details><summary>展开：六类父状态示例（L25）</summary>'+image('bauer_2d_review_20260918/assets/initial_L25.png','每行是一个初态，不是时间；列是格点，颜色是sz。只展示z分量不代表完整构型完全相同。')+'</details>')
    parts.append('<details><summary>展开：L100 三条完整10 ns参考轨迹</summary>')
    for b in range(3):
        parts.append(image(f'bauer_fig2_10ns_20260917/results/comparison/path_{b}_full_10ns.png',f'参考LLG轨迹{b}：完整10 ns；不是生成模型输出。看反向区域出现、扩张、消失及界面移动。'))
    parts.append('</details></section><section id="models"><h2>本轮一维模型闭环</h2>')
    parts.append(table(['模型','空间/时间结构','问题'],[['A','局部单步，滚动输出','基础转移与累积误差'],['B','局部空间＋块内时间交互','联合时间建模的收益'],['C','多尺度空间＋块内时间交互','空间多尺度的收益'],['D','同B类骨干的确定性预测','没有随机分布时的局限']]))
    parts.append('<p>各3000次更新、三个seed；开发集选checkpoint后再读测试未来。逐格点球面输出、首帧约束；未输入解析有效场。共同更新数不等于相同算力/目标帧曝光。易轴与磁矩共同旋转协变不等于固定易轴下任意旋转对称。</p>')
    modelrows=[]
    for r in runs:
        s, t = r['selection'], r['test']
        modelrows.append([r['name'],s['selected_step'],f'{s["dev_score"]:.4f}',f'{t["score"]:.4f}',
                          f'{sum(v["status"]=="WITHIN_PILOT_MARGIN" for v in t["statistics"].values())}/10',s['parameters']])
    parts.append(table(['运行','选中更新步','开发分数','测试分数↓','在线内指标','参数数'],modelrows))
    parts.append('<p>'+link(TESTS/'bauer_closed_loop_20260919/assets/index.html','打开本轮完整模型报告')+'</p></section>')
    meanings=[
        ('mz_mean','平均磁化','先格点平均sz，再分支平均；生成/参考时间曲线RMSE','整体趋势，不代表空间机制'),
        ('mz_var','磁化方差','每时刻跨分支的mz方差，比较时间曲线RMSE','随机未来分散程度；不是空间方差'),
        ('energy_mean','平均能量','E/(LJ)跨分支均值的时间RMSE','交换与易轴统计；有限温不要求能量守恒'),
        ('spatial_corr','空间相关','r=1/2/4的sz(i)sz(i+r)，分支平均后RMSE','几个距离的非连通z相关，不是所有长程关联'),
        ('temporal_corr','初态时间重叠','格点平均s(t)·s(0)，分支平均后时间RMSE','初态记忆；不是完整多滞后自相关'),
        ('increment_mean','增量均值','Δs=s(t+0.1ps)−s(t)，跨分支均值；时间/位置/分量RMSE','离散局部平均变化，不是直接测得瞬时漂移'),
        ('increment_cov','增量协方差','同格点同时间Δs的3×3分支协方差；比较矩阵RMSE','局部涨落幅度及分量关系；16分支估计噪声较大'),
        ('endpoint_w1','端点分布W1','1 ps时mz样本的一维Wasserstein距离','端点分布差异，不检查中间过程'),
        ('wall_proxy','畴壁代理','相邻sz异号键比例的条件均值时间RMSE','非严格畴壁数，不能直接解释为翻转事件'),
        ('path_energy_distance','路径特征分布距离','拼接多时刻磁化/能量/重叠/相关；归一化后energy distance','energy是统计名称；只检查选定观测量联合分布')]
    parts.append('<section id="metrics"><h2>指标怎么计算、为什么使用？</h2><p>先对每个父状态比较16个生成未来与16个参考未来，再对36个测试父状态平均。样本族人为等额，不是自然占据概率。图中区间是按尺寸/准备族分层的父状态bootstrap 95%区间，不消除未来分支数不足的问题。</p>')
    metricrows=[]
    for key,title,formula,meaning in meanings:
        v=chosen['test']['statistics'][key] if chosen else None
        metricrows.append([title,formula,meaning,f'{v["mean"]:.5f} / {v["margin"]:g}' if v else '缺失',
                           {'WITHIN_PILOT_MARGIN':'先导在线内','OUTSIDE_PILOT_MARGIN':'超出先导线','INCONCLUSIVE':'证据不足'}.get(v['status'],'未知') if v else '缺失'])
    parts.append(table(['指标','算法','物理意义与局限','展示模型误差/容差','状态'],metricrows))
    parts.append('''<p class="formula">综合分数 = (1/10) Σ[各指标的父状态平均误差 ÷ 该指标预设容差]。越低越好；不是准确率、不是基本物理量、不是通过概率。低分不能补偿单个关键指标失败。</p>
<p class="warning">容差是预先冻结的工程筛查线，不是文献标准。真实参考8+8比较与开发基础/细步长16+16比较中的局部增量也可能超线。先做同样本量参考校准，不能仅为通过而改松阈值。</p>
<details><summary>“每步变化较大，但终点更接近初态”为什么不矛盾？</summary><p>相邻重叠检查一步位移，终点重叠检查净位移。示意：0°→30°→−30°→5°，每步变化大，却比0°→10°→20°→30°更接近起点。该例仅解释关系，不认定模型已被证实为某种振荡机制。需要进一步检查多滞后关联、增量关联和准备族。</p></details></section>''')
    parts.append('<section id="figures"><h2>推荐看图顺序</h2><p>先看L25参考/生成路径，再看各模型指标，最后看采样步数。参考分支0和生成分支0不是同噪声配对，不要求逐像素相同。</p>')
    for L in (25,16,32):
        parts.append(f'<details{" open" if L==25 else ""}><summary>L{L}：路径、分支与时间重叠</summary>'+image(f'bauer_closed_loop_20260919/assets/paths_L{L}.png','上排：真实/生成sz时空图及16分支磁化；下排：端点分布、初态重叠、异号键比例。相同色标比较；没有事件不能证明寿命正确。')+'</details>')
    parts.append(image('bauer_closed_loop_20260919/assets/metrics.png','每个seed独立展示；误差越低越好。虚线是工程筛查线，不能单独解释为科学认证。'))
    parts.append('<details><summary>采样积分32/64/128步敏感性</summary>'+image('bauer_closed_loop_20260919/assets/sampling_steps.png','固定checkpoint和随机源比较采样步数；综合分数稳定不证明模型准确，逐路径差异另存sampler_paired.json。')+'</details></section>')
    parts.append('<section id="systems"><h2>二维、三维与通用架构预检查</h2>')
    if miltat:
        phases=miltat.get('phases',{})
        parts.append(table(['二维参考阶段','物理时长','计算用时','保存帧模长误差'],[[name,f'{v["duration_ps"]:g} ps',f'{v["seconds"]/60:.2f} min',f'{v["max_saved_norm_error_float64"]:.3g}'] for name,v in phases.items()]))
        parts.append('<p>二维 Miltat：12.5 K，阻尼0.5，46400磁矩，2.5 fs步长，完整偶极作用。50 ps显著扰动后的零温准备＋1 ns热模拟。此前主网页只接入完成报告，没有把该H5轨迹渲染成图；现在补齐展示，不是重新模拟。</p><p>模型参数：J=29 meV，D=−1.5 meV，K=0.4 meV，磁矩2.1 μB，晶格常数2.51 Å；200×116×2三角格点。文献入口：<a href="https://arxiv.org/abs/1806.03172">Miltat、Rohart、Thiaville：磁畴壁和斯格明子的布朗运动</a>。实际参数与来源核验项保留在原始report.json。</p><p class="warning">固定开放窗口，不是原文移动窗口；有限准备不保证平衡；1 ns不足以覆盖原文图10完整拟合区间。这是单条参考LLG路径，不是二维生成模型输出，也不构成完整文献扩散结论复现。</p>')
        diagnostic=read('progress_dashboard/miltat/summary.json')
        if diagnostic:
            d=diagnostic['thermal']
            parts.append(f'<p>已读取全部 {d["frames"]} 个热模拟保存帧（每1 ps一帧），未发现核心消失帧。按下述阈值定义的核心中心净位移为 {d["net_displacement_nm"]:.2f} nm；保存帧积分器双精度模长最大误差 {d["max_integrator_saved_norm_error"]:.2g}，float32轨迹文件复算误差 {d["max_stored_norm_error"]:.2g}。二者精度不同，不是磁矩失守恒。</p>')
            parts.append(image('progress_dashboard/miltat/textures.png','上排看完整膜面及受扰动初态，下排在相同固定区域放大0、0.5、1 ns。色标统一为sz∈[−1,1]；放大有助于观察小磁核，不改变模拟尺寸。'))
            parts.append(image('progress_dashboard/miltat/thermal.gif','二维动态参考：左侧为三维磁矩箭头，右侧为sx、sy、sz的二维空间色图。固定视野包含观测到的运动范围；箭头每4行/列抽取，色图保留视野内全部格点。GIF每20 ps显示一帧、共51帧；统计使用全部1001帧。'))
            parts.append(image('progress_dashboard/miltat/diagnostics.png','依次看核心中心轨迹、位移分量、核心等效半径、能量、模长误差、单轨迹时间平均MSD。MSD只展示至250 ps滞后；重叠时间起点不是独立样本，不据此拟合或认证扩散系数。'))
            parts.append('<p>核心判据：sz≥0.5的三角格六邻接最大连通区域，质心按真实格点坐标计算；等效半径=√(核心格点数×单格点面积/π)。这是纹理追踪代理，不是拓扑荷或严格畴壁位置；分裂时连通域可能切换，阈值敏感性尚未认证。</p>')
            parts.append('<details><summary>零温准备：显著扰动之后的50 ps松弛</summary>'+image('progress_dashboard/miltat/preparation.png','零温准备阶段的能量、阈值核心半径和模长。能量趋稳是局部准备诊断，不证明有限温分布已达到平衡。')+'</details>')
            parts.append('<p>'+link(HERE/'miltat/thermal_diagnostics.csv','全部热模拟逐帧诊断CSV')+' · '+link(HERE/'miltat/msd.csv','MSD数据')+' · '+link(HERE/'miltat/summary.json','可视化方法与数据摘要')+'</p>')
    else: parts.append('<p>二维1 ns报告尚缺失。</p>')
    rows=[]
    for system in ('bauer','miltat','hirst'):
        s=read(f'path_smoke_20260917/results/{system}/summary.json')
        rows.append([system,'已有报告' if s else '缺失',s.get('scientific_status','未知') if s else '未确认',str(s.get('checks',{}).get('passed','未知')) if s else '—'])
    parts.append(table(['体系原型','状态','证据范围','软件约束检查'],rows))
    parts.append('<p>三维Hirst/Mn2Au仍是前期参考与原型检查，不是训练规模数据或新模型统计验证。</p><p>'+link(TESTS/'physical_audit_20260917/results/index.html','早期三体系物理/零温展示页（历史短时数据）')+'</p></section>')
    parts.append('''<section id="next"><h2>下一步：先校准评价，再针对时间误差改进</h2><ol>
<li>先用现有开发数据校准参考—参考误差，按样本量与准备族分析；不足时定向增加独立分支。尚未提交新计算。</li>
<li>同步冻结初态准备协议、来源与权重，建立独立初态训练/开发/测试合同；六族覆盖库不直接充当自然初态分布。</li>
<li>按尺寸、准备族、时间间隔诊断：相邻帧过强变化来自哪里？初态记忆、时间谱和增量关联如何？尚待深入分析。</li>
<li>据诊断选择时间模块/目标/窗口的受控修改；不是直接无限增加训练预算。</li>
<li>实现并验证初态最小模型、独立seed及组合入口，完成真实/生成初态与LLG/模型后继的四组小体系闭环。</li>
<li>固定可信窗口，先L50/L100尺寸迁移；更长直接生成窗及时间接续独立验证，最终使用未参与开发的新测试集确认。</li>
</ol><p>本网页构建不会提交任务、重训模型、修改原始数据或推送GitHub。这里是当前工作清单，不是自动调度承诺。</p></section>''')
    parts.append('<section id="files"><h2>文件入口与如何更新主网页</h2><ul>')
    parts.append('<li>'+link(TESTS/'bauer_two_stage_20260919/assets/index.html','新一轮：初态＋路径、四组闭环、L50/L100推广与速度比较（以实际完成记录为准）')+'</li>')
    entries=[(ROOT/'GUIDE/INITIAL_STATE_PATH_PLAN_20260919.md','当前研究方案（修订12：两阶段生成与下一步）'),
             (ROOT/'GUIDE/MDGEN_MLIP_PATH_PLAN_20260918.md','上一版数据及模型合同（修订11历史状态）'),
             (ROOT/'GUIDE/MULTISCALE_STOCHASTIC_PATH_ARCHITECTURE.md','模型架构设计'),
             (TESTS/'bauer_active_data_20260918/data','一维原始分支库'),
             (TESTS/'bauer_closed_loop_20260919/config.yaml','冻结配置与容差'),
             (TESTS/'bauer_closed_loop_20260919/assets/report.json','全部闭环数值报告'),
             (TESTS/'bauer_closed_loop_20260919/audit.json','完整性与参考步长筛查'),
             (TESTS/'bauer_closed_loop_20260919/runs','全部checkpoint、生成轨迹、选模记录'),
             (TESTS/'bauer_closed_loop_20260919/logs','任务日志'),
             (TESTS/'miltat_1ns_20260919/results','二维1 ns原始数据及报告')]
    parts.extend('<li>'+link(p,t)+'</li>' for p,t in entries)
    parts.append('</ul><p>从项目根目录更新：</p><pre><code>bash ../run_zrs_mag.sh -m tests.progress_dashboard.build</code></pre><p>刷新已有报告会更新数字与图；新增研究分支需在构建脚本登记。它不是持续监控服务。更新前后的来源哈希可在 progress_dashboard/snapshot.json 检查。</p></section>')
    parts.append('<section id="history"><h2>历史探索归档：不与当前模型结论混用</h2><p>下列是早期方案与结果入口；仅供追溯，不作为本轮新架构的通过证据。不纳入临时测试、浏览器缓存或合成数据目录。</p><ul>')
    for p,t in [(TESTS/'paper_work/index.html','早期文章六图'),(TESTS/'paper_work/completion/index.html','早期补全结果'),(TESTS/'results','历史实验统计'),(TESTS/'experiments','历史测试代码'),(ROOT/'assets/literature_reproduction','文献复现归档'),(ROOT/'assets/research','研究结果归档')]:
        parts.append('<li>'+link(p,t)+'</li>')
    parts.append('</ul></section><footer><p>仅聚合现有结果；关键图原样内嵌，没有重绘、筛除不利运行或改动原数据。</p></footer></main></body></html>')
    snapshot=dict(generated_at=stamp, current_run_status=closed.get('status') if closed else None,
                  displayed_model=chosen['name'] if chosen else None, completed_runs=len(runs),
                  scientific_certification=False, sources=SOURCES,
                  limitations=['Static snapshot, not live scheduler state','Narrative is manually curated; new experiments require registration',
                               'Standalone HTML embeds key PNGs but relative external links need project directories'])
    (HERE/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    (TESTS/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(page=str(TESTS/'index.html'),sources=len(SOURCES),runs=len(runs),bytes=(TESTS/'index.html').stat().st_size),ensure_ascii=False))


if __name__=='__main__': main()
