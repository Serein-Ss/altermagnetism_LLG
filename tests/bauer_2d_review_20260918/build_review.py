"""Read-only data audit and offline inspection page; no simulations or training."""
import base64
import csv
import hashlib
import html
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'assets'
DATA = ROOT / 'tests/bauer_active_data_20260918/data'
FAMILIES = ['positive', 'negative', 'tilted', 'edge_domain', 'interior_domain', 'single_wall']
LABELS = ['正向准备', '反向准备', '倾斜', '边缘反向域', '内部反向域', '单畴壁']
font_manager.fontManager.addfont('/usr/share/fonts/wqy-microhei/wqy-microhei.ttc')
plt.rcParams.update({'font.family': 'WenQuanYi Micro Hei', 'axes.unicode_minus': False,
                     'font.size': 11, 'figure.dpi': 110, 'savefig.dpi': 130})


def save(fig, name):
    fig.savefig(OUT / name, facecolor='white')
    plt.close(fig)
    return OUT / name


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def img(path, caption):
    raw = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{raw}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def audit():
    inventory, qc, endpoints, dtrows, galleries = [], [], [], [], []
    seen_seeds, seen_states, parent_split = set(), {}, {}
    norm_curves = {}
    for length in [16, 25, 32]:
        root = DATA / f'L{length}'
        records = json.loads((root/'manifest.json').read_text())
        counts = Counter()
        norm_curves[length] = np.zeros(1001)
        dev = {}
        for row in records:
            path = root/row['file']
            assert digest(path) == row['sha256'], path
            assert parent_split.setdefault(row['parent_id'], row['split']) == row['split']
            with h5py.File(path) as h:
                assert h.attrs['complete'] and h.attrs['arm'] == row['arm']
                parent = json.loads(h.attrs['parent_json'])
                assert parent['parent_id'] == row['parent_id'] and parent['split'] == row['split']
                initial = h['initial'][:]
                state_sha = hashlib.sha256(initial.tobytes()).hexdigest()
                assert state_sha == parent['state_sha256']
                assert seen_states.setdefault(state_sha, row['parent_id']) == row['parent_id']
                seeds = h['seeds'][:].tolist()
                assert len(set(seeds)) == len(seeds) and not seen_seeds.intersection(seeds)
                seen_seeds.update(seeds)
                t = h['time_ps'][:]
                assert len(t) == 1001 and np.allclose(t, np.arange(1001)*.1)
                assert h['spins'].shape == (row['branches'], 1001, length, 3)
                normerr = anchorerr = mzerr = energyerr = 0.
                for branch in range(row['branches']):
                    s = h['spins'][branch].astype('f8')
                    assert np.isfinite(s).all()
                    normerr = max(normerr, float(abs(np.linalg.norm(s, axis=-1)-1).max()))
                    norm_curves[length] = np.maximum(norm_curves[length], abs(np.linalg.norm(s, axis=-1)-1).max(-1))
                    anchorerr = max(anchorerr, float(abs(s[0]-initial).max()))
                    mz = s[...,2].mean(-1)
                    e = -(s[:,:-1]*s[:,1:]).sum((-1,-2))/length-.1*(s[...,2]**2).mean(-1)
                    mzerr = max(mzerr, float(abs(mz-h['mean_mz'][branch]).max()))
                    energyerr = max(energyerr, float(abs(e-h['energy_per_spin_J'][branch]).max()))
                assert normerr < 1e-6 and anchorerr < 1e-7 and mzerr < 1e-6 and energyerr < 1e-6
                qc.append(dict(L=length, file=row['file'], split=row['split'], arm=row['arm'],
                               norm_error=normerr, anchor_error=anchorerr,
                               mz_recompute_error=mzerr, energy_recompute_error=energyerr,
                               sha256=row['sha256'], branches=row['branches']))
                counts[(row['split'],row['arm'])] += row['branches']
                # Test futures inspected for integrity only, never plotted or used for selection.
                if row['split'] != 'test':
                    for horizon in [1, 10, 100]:
                        k = int(horizon*10)
                        mz = h['mean_mz'][:,k]
                        en = h['energy_per_spin_J'][:,k]
                        if row['arm'] == 'base':
                            for b in range(len(mz)):
                                endpoints.append(dict(L=length,parent_id=row['parent_id'],split=row['split'],
                                    family=row['family'],time_ps=horizon,branch=b,mz=float(mz[b]),energy=float(en[b])))
                    if row['split'] == 'dev':
                        dev[(row['parent_id'], row['arm'])] = {
                            'mz': h['mean_mz'][:,[10,100,1000]],
                            'energy': h['energy_per_spin_J'][:,[10,100,1000]], 'family': row['family']}
        for split in ['train','dev','test']:
            parents = {r['parent_id'] for r in records if r['split']==split}
            inventory.append([length,split,len(parents),counts[(split,'base')],counts[(split,'fine')],100,1001])
        for parent in sorted({p for p,arm in dev}):
            for metric in ['mz','energy']:
                a, b = dev[(parent,'base')][metric], dev[(parent,'fine')][metric]
                for j, horizon in enumerate([1,10,100]):
                    se = np.sqrt(a[:,j].var(ddof=1)/len(a)+b[:,j].var(ddof=1)/len(b))
                    dtrows.append(dict(L=length,parent_id=parent,family=dev[(parent,'base')]['family'],
                        metric=metric,time_ps=horizon,base_mean=float(a[:,j].mean()),fine_mean=float(b[:,j].mean()),
                        difference=float(a[:,j].mean()-b[:,j].mean()),independent_mean_SE=float(se),
                        reference_half_difference=float(a[:8,j].mean()-a[8:,j].mean())))
        # One predeclared dev parent (lowest ID) per family; no future-outcome selection.
        for family,label in zip(FAMILIES,LABELS):
            chosen = min((r for r in records if r['split']=='dev' and r['arm']=='base' and r['family']==family),
                         key=lambda r:int(r['parent_id'].split('parent')[-1]))
            with h5py.File(root/chosen['file']) as h:
                fig, ax = plt.subplots(2,3,figsize=(15,7),layout='constrained')
                t = h['time_ps'][:]
                for b in range(3):
                    s=h['spins'][b]
                    im=ax[0,b].imshow(s[...,2].T,origin='lower',aspect='auto',interpolation='nearest',
                        extent=[0,100,-.5,length-.5],vmin=-1,vmax=1,cmap='RdBu_r')
                    ax[0,b].set(xlabel='时间 (ps)',ylabel='格点编号',title=f'分支{b} / seed {h["seeds"][b]}')
                fig.colorbar(im,ax=list(ax[0]),label='s_z')
                mz=h['mean_mz'][:]
                for b in range(len(mz)):
                    ax[1,0].plot(t,mz[b],lw=.55,alpha=.6)
                ax[1,0].plot(t,mz.mean(0),'k',lw=1.6,label='16分支均值')
                ax[1,0].set(xlabel='时间 (ps)',ylabel='平均 s_z',ylim=(-1,1))
                ax[1,0].legend()
                s=h['spins'][0]
                for k,style in zip(range(3),['-','--',':']):
                    ax[1,1].plot(t,s[...,k].mean(-1),style,label='xyz'[k])
                ax[1,1].set(xlabel='时间 (ps)',ylabel='分支0的空间平均分量',ylim=(-1,1)); ax[1,1].legend()
                for k,style in zip(range(3),['-','--',':']):
                    ax[1,2].plot(t,s[...,k].std(-1),style,label='xyz'[k])
                ax[1,2].set(xlabel='时间 (ps)',ylabel='分支0的空间标准差',ylim=(0,1)); ax[1,2].legend()
                fig.suptitle(f'L={length} / {label} / {chosen["parent_id"]}（准备族，不是终态分类）')
                filename=f'paths_L{length}_{family}.png'
                save(fig,filename)
                galleries.append(dict(L=length,family=family,parent_id=chosen['parent_id'],file=filename))
        print(f'L{length}: {len(records)} files verified and plotted',flush=True)
    dump('integrity_audit.json',dict(status='INTEGRITY_PASS_PHYSICAL_DISTRIBUTION_NOT_CERTIFIED',files=qc,
         independent_future_seeds=len(seen_seeds),parents=len(parent_split),test_use='integrity_only'))
    dump('gallery_selection.json',galleries)
    fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained')
    for length,style in zip([16,25,32],['-','--',':']):
        ax.plot(np.arange(1001)*.1,norm_curves[length],style,label=f'L={length}')
    ax.set(xlabel='时间 (ps)',ylabel='各文件/分支/格点的最大 |长度−1|',ylim=(0,1e-7),
           title='磁矩模长随时间：float32存储值以float64重算，所有划分仅做完整性检查')
    ax.legend();save(fig,'norm_over_time.png')
    for name, rows in [('development_endpoints.csv',endpoints),('development_dt_comparison.csv',dtrows)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return inventory,qc,endpoints,dtrows,galleries


def plots(endpoints,dtrows):
    for length in [16,25,32]:
        root=DATA/f'L{length}'
        parents=json.loads((root/'parents.json').read_text())
        with np.load(root/'candidates.npz') as data:
            s=data['spins']
            fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
            for family,label,ax in zip(FAMILIES,LABELS,axes.flat):
                ids=[p['id'] for p in parents if p['selected'] and p['split']=='train' and p['family']==family]
                im=ax.imshow(s[ids,:,2],vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto',interpolation='nearest')
                ax.set(title=label,xlabel='格点编号',ylabel='训练父状态编号')
                ax.set_yticks(range(len(ids)),[str(i) for i in ids])
            fig.colorbar(im,ax=list(axes.flat),label='初态 s_z')
            fig.suptitle(f'L={length}：全部48个已选训练初态；正／反向准备已经历20 ps热演化')
            save(fig,f'initial_L{length}.png')
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for i,length in enumerate([16,25,32]):
        for j,horizon in enumerate([1,10,100]):
            ax=axes[i,j]
            for split,style,col in [('train','-','#0072B2'),('dev','--','#D55E00')]:
                a=[r['mz'] for r in endpoints if r['L']==length and r['time_ps']==horizon and r['split']==split]
                ax.hist(a,bins=np.linspace(-1,1,31),density=True,histtype='step',linestyle=style,color=col,label=f'{split}, n={len(a)}')
            ax.set(title=f'L={length}, t={horizon} ps',xlabel='空间平均 s_z',ylabel='概率密度',xlim=(-1,1));ax.legend(fontsize=8)
    fig.suptitle('准备族平衡混合后的端点分布：不是自然平衡分布，也不是独立父状态数')
    save(fig,'endpoint_distributions.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,metric in enumerate(['mz','energy']):
        for j,horizon in enumerate([1,10,100]):
            ax=axes[i,j]
            for length,marker,col in [(16,'o','#0072B2'),(25,'s','#D55E00'),(32,'^','#009E73')]:
                rows=[r for r in dtrows if r['metric']==metric and r['time_ps']==horizon and r['L']==length]
                ax.errorbar([r['fine_mean'] for r in rows],[r['difference'] for r in rows],
                            yerr=[r['independent_mean_SE'] for r in rows],fmt=marker,ms=4,color=col,
                            capsize=2,alpha=.8,label=f'L={length}')
            ax.axhline(0,c='black',lw=.8)
            ax.set(title=f't={horizon} ps',xlabel='细步长：父状态条件均值',ylabel=('磁化' if metric=='mz' else 'E/(N J)')+'：基础−细步长')
            ax.legend(fontsize=8)
    fig.suptitle('开发集步长对照：每点一个父状态，各臂16个独立未来；误差棒为差值的1个标准误差')
    save(fig,'dt_comparison.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,length in zip(axes,[16,25,32]):
        d=json.loads((DATA/f'L{length}'/'selection_diagnostics.json').read_text())
        for name,marker in [('fps','o'),('random','s')]:
            vals=[next(r['max_pool_distance'] for r in d if r['family']==f and r['method']==name) for f in FAMILIES]
            ax.plot(range(6),vals,marker=marker,label=name)
        ax.set_xticks(range(6),LABELS,rotation=35,ha='right')
        ax.set(title=f'L={length}',ylabel='候选池最大最近距离（标准化描述符）');ax.legend()
    fig.suptitle('覆盖诊断仅涉及已有候选池；random仅一次抽样，不能宣称全面覆盖或普遍优胜')
    save(fig,'coverage.png')


def page(inventory,qc,dtrows,galleries):
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>一维数据检查与二维文献确认</title><style>
    body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#eef2f6;color:#162536;margin:0;line-height:1.7}
    main{max-width:1380px;margin:auto;padding:24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}
    nav{position:sticky;top:0;background:#142f49;padding:12px;z-index:2;display:flex;flex-wrap:wrap;gap:15px}nav a{color:white}
    a{color:#075a94}img{width:100%;height:auto}figure{margin:14px 0}figcaption{color:#34475c;font-size:.95rem}
    table{border-collapse:collapse;width:100%;font-size:.95rem}td,th{border:1px solid #c7d2df;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f8}
    .scroll{overflow:auto}.warning{background:#fff3cd;padding:14px;border-left:5px solid #ab6a00}
    details{margin:12px 0;border:1px solid #ccd7e2;padding:12px;border-radius:8px}summary{cursor:pointer;font-weight:600}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:800px){.cols{grid-template-columns:1fr}}
    h1,h2,h3{line-height:1.35}code{overflow-wrap:anywhere}button{padding:8px}a:focus,summary:focus{outline:3px solid #0072B2}
    </style><nav><a href="#status">结论边界</a><a href="#bauer">一维文献复现</a><a href="#data">新数据</a><a href="#parents">初态</a><a href="#branches">分支轨迹</a><a href="#dt">步长核验</a><a href="#two">二维文献逐图</a></nav><main>
    <h1>一维数据检查与二维文献确认</h1><p>2026-09-18｜只读分析；不生成新模拟、不训练模型。所有PNG内嵌，单独下载本HTML也能离线查看；一维不制作动画。</p>
    <section id="status"><h2>先看这一页的证据边界</h2>
    <p>一维10 ns轨迹是物理求解器生成的文献定性对照；100 ps分支库是为学习条件路径规律构造的数据；两者都不是生成模型输出。</p>
    <p class="warning">新数据完整性通过不等于数值收敛或训练集覆盖已充分。二维只有软件／短时物理诊断，没有复现原文ns扩散曲线。本文不自动放行正式训练。</p></section>''']
    base=ROOT/'tests/bauer_fig2_10ns_20260917/results'
    parts.append('<section id="bauer"><h2>一、一维Bauer图2：原文对照与三条完整10 ns轨迹</h2><p>L=100、开放边界、J=50 meV时间映射、K/J=0.1、阻尼0.1、温度约63.825 K；随机路径不要求与原文逐像素一致。重点看反向区域的出现、扩张、消退和畴壁移动。原文：<a href="https://doi.org/10.1088/0953-8984/23/39/394204">Bauer et al. 2011</a>。</p>')
    parts.append(img(base/'comparison/paper_figure2.png','原文图2；作者原图，不是本项目数据。以下自有三条轨迹按相同分段组织。'))
    for b in range(3):
        parts.append(f'<details><summary>轨迹 {b}：与原文相同分段布局</summary>')
        parts.append(img(base/f'comparison/path_{b}_paper_layout.png',f'自有LLG轨迹{b}的分段展示，仅用于形态对照。'))
        parts.append('</details>')
    for b in range(3):
        parts.append(img(base/f'comparison/path_{b}_full_10ns.png',f'自有轨迹{b}：时间完整覆盖10 ns，不截断。显示分辨率不等于积分步长；原始数据保留。'))
    parts.append('<details><summary>旧事件／畴壁诊断及短时步长检查</summary><p>这些是旧L100数据的探索性诊断，不是新L16/25/32库的验收。事件判据含阈值与保持时间，不能把磁化过零直接称为完成翻转。</p>')
    for p in ['basins.png','path_0_events.png','path_1_events.png','path_2_events.png','dt_check.png']:
        parts.append(img(base/'diagnostics_events'/p,'旧L100探索性诊断：'+p+'；判据和短时样本量限制仍适用。'))
    parts.append('</details></section><section id="data"><h2>二、新的一维小体系分支库</h2>')
    parts.append(table(['L','划分','父状态','基础分支','细步长分支','未来时长/ps','帧数/条'],inventory))
    parts.append('<p>共216个父状态、2304条基础分支＋576条细步长分支。每条100 ps、1001帧；基础步长0.25 fs，开发细步长0.125 fs，保存0.1 ps。六个准备族等额配置。每尺寸训练48父×8分支，开发12父×16分支，测试12父×16分支。细步长只用于开发诊断。</p>')
    parts.append('<p>保存路径：<code>tests/bauer_active_data_20260918/data/L16、L25、L32</code>。索引manifest.json；父状态parents.json；每个H5保存同父多分支、初态、时间、seed、磁化与能量。能量字段energy_per_spin_J表示E/(N·J交换)，不是焦耳。</p>')
    parts.append(table(['本轮重新检查','结果'],[
        ['逐文件SHA256',f'{len(qc)}个分支H5全部匹配'],['所有分支／帧有限性、时间、形状、父状态隔离','通过'],
        ['最大模长误差',f'{max(r["norm_error"] for r in qc):.3g}'],
        ['最大首帧分量误差',f'{max(r["anchor_error"] for r in qc):.3g}'],
        ['磁化重算最大误差',f'{max(r["mz_recompute_error"] for r in qc):.3g}'],
        ['E/(NJ)重算最大误差',f'{max(r["energy_recompute_error"] for r in qc):.3g}'],
        ['独立测试集','仅完整性检查；不展示未来统计，不参与调参']]))
    parts.append(img(OUT/'norm_over_time.png','全量数据磁矩模长误差随时间；这是存储与球面约束检查，不能证明随机动力学精度。'))
    parts.append('</section><section id="parents"><h2>三、初态到底是什么</h2><p>每张图覆盖该尺寸全部48个训练父状态。每一行是一个初态，每列一个格点，颜色是s_z，统一−1到1。正向／反向指准备协议；20 ps热演化后可能已经改变方向，不是标注错误。其余四族是带随机扰动的人工状态，不冒充平衡样本。</p>')
    for length in [16,25,32]:parts.append(img(OUT/f'initial_L{length}.png',f'L{length}所有已选训练初态；检查局部畴、边缘与空间变化是否有差异。'))
    parts.append(img(OUT/'coverage.png','覆盖候选池的描述符距离：越低表示对这批候选的几何覆盖更近，不等于覆盖所有物理状态。'))
    parts.append('</section><section id="branches"><h2>四、同一初态为何有多条未来</h2><p>按尺寸和准备族各展示一个开发父状态，共18个父状态。选择规则固定为该组最小父ID，不按是否翻转挑样本。上排为分支0–2的完整100 ps时空图；下排展示该父状态全部16条磁化曲线以及分支0的三分量均值和空间标准差。无需动画即可同时看成核、畴扩张和返回；曲线分离是LLG热噪声的真实多样性，尚不是模型能力。</p>')
    for g in galleries:
        parts.append(f'<details><summary>L={g["L"]} / {LABELS[FAMILIES.index(g["family"])]} / {g["parent_id"]}</summary>')
        parts.append(img(OUT/g['file'],'三条热噪声分支的时空图＋全部16分支磁化；不把分支均值当作典型轨迹。'))
        parts.append('</details>')
    parts.append(img(OUT/'endpoint_distributions.png','1、10、100 ps的端点磁化直方图：train/dev分开展示。样本混合权重来自人为准备族，不是自然占据概率；这些图不足以证明联合路径规律。'))
    parts.append('</section><section id="dt"><h2>五、新库的步长敏感性</h2><p>固定同一父状态，比较基础0.25 fs与细步长0.125 fs；两臂未来seed不同，因此不能逐条对齐比较。每点对应一个开发父状态（每尺寸12个），每臂16条。横轴是细步长条件均值，纵轴是基础减细步长，误差棒是独立均值差的1 SE，不是95%等价区间。</p>')
    parts.append(img(OUT/'dt_comparison.png','磁化和能量的条件均值差。围绕零且误差棒较大，表示当前分支数难以区分小偏差，不能据此宣布步长收敛。'))
    rows=[]
    for metric in ['mz','energy']:
        for horizon in [1,10,100]:
            r=[x for x in dtrows if x['metric']==metric and x['time_ps']==horizon]
            rows.append([metric,horizon,len(r),f'{np.mean([abs(x["difference"]) for x in r]):.5g}',
                         f'{np.mean([abs(x["reference_half_difference"]) for x in r]):.5g}',
                         sum(abs(x['difference'])>2*x['independent_mean_SE'] for x in r)])
    parts.append(table(['量','时间/ps','父状态数','平均绝对步长差','基础8+8绝对均值差','|步长差|>2SE点数'],rows))
    parts.append('<p>以上为描述性筛查，重复时间点相关、未做多重检验，不能把2SE计数当成正式通过／失败。8+8参考的样本量与16+16不同，不能直接当成同精度误差基线。下一步若需确认收敛，应预声明容差并增加独立分支／第三步长。原始逐父数值随完整目录提供CSV。</p></section>')
    parts.append('<section><h2>下载本页分析数值</h2><p>下列CSV/JSON同样内嵌，不依赖服务器路径；不包含原始大体量磁矩数据。</p>')
    for name in ['development_endpoints.csv','development_dt_comparison.csv','integrity_audit.json','gallery_selection.json']:
        raw=base64.b64encode((OUT/name).read_bytes()).decode()
        parts.append(f'<p><a download="{name}" href="data:application/octet-stream;base64,{raw}">{name}</a></p>')
    parts.append('</section>')
    parts.append((HERE/'two_dimensional.html').read_text())
    parts.append('<section><h2>二维关键原图所在页面</h2><p>下列为开放原文页面的原样渲染，未裁剪或修改曲线；各页保留原始图注。完整14图见上方原文PDF入口。</p>')
    for number,caption in [(5,'原文图6–8：畴壁阻尼、二维纹理与中心轨迹'),(6,'原文图9–10：位移分布与扩散'),(7,'原文图11及解析公式'),(8,'原文图12–13：半径频谱、尺寸与阻尼'),(9,'原文图14：热背景与耗散')]:
        parts.append(f'<details><summary>{caption}</summary>')
        parts.append(img(HERE/f'sources/pages-{number:02d}.png',caption+'。来源：Miltat et al. arXiv:1806.03172。'))
        parts.append('</details>')
    parts.append('</section>')
    parts.append('<section><h2>检查后如何继续</h2><ol><li>先确认一维六类准备状态及100 ps分支是否满足研究用途，并核验步长误差。</li><li>确认二维先做斯格明子纹理与25 K短轨迹，还是直接扩展到多温扩散统计；目前仅提供方案，不提交新计算。</li><li>二维需先解决初态纹理稳定性、DMI手性和时间约定，之后才生成训练规模数据。</li></ol><p>本页生成脚本：tests/bauer_2d_review_20260918/build_review.py。PNG为唯一静态图格式。完整性表、逐父CSV与选图规则保存在同目录；源数据只读。可视化遵循Scientific Visualization技能的共同色标、抽样披露和误差定义规范，文献解读使用Nature Paper Card证据规范；这些流程不是科学认证。</p></section></main></html>')
    (OUT/'index.html').write_text(''.join(parts))
    dump('review_summary.json',dict(inventory=inventory,file_count=len(qc),
          scope='read_only_integrity_and_descriptive_train_dev_analysis',
          max_norm_error=max(r['norm_error'] for r in qc),
          dt_description=rows,html_bytes=(OUT/'index.html').stat().st_size))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    inventory,qc,endpoints,dtrows,galleries=audit()
    plots(endpoints,dtrows)
    page(inventory,qc,dtrows,galleries)
    print(f'Finished in {time.time()-start:.1f}s: {OUT / "index.html"}',flush=True)
