"""Read-only static replot of completed one-dimensional paths in paper layout."""
import base64
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
MEDIA=OUT/'comparison'
PAPER_END_NS=750000*.6582119569/50/1000


def image_tag(path):
    return '<img class="figure" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def plot_path(t,mz,spins,edges,path,seed):
    segments=len(edges)-1
    fig,axes=plt.subplots(2*segments,1,figsize=(12,3.3*segments),
                          gridspec_kw={'height_ratios':[1,1.2]*segments})
    cmap=ListedColormap(['#1039e8','#00c97b'])
    for j,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
        ax,heat=axes[2*j:2*j+2]
        ix=np.flatnonzero((t>=start)&(t<=end))
        shown=ix[np.unique(np.linspace(0,len(ix)-1,min(6000,len(ix)),dtype=int))]
        ax.plot(t[ix],mz[ix],lw=.45,color='#1746b5')
        ax.set(xlim=(start,end),ylim=(-100,100),ylabel='Total Mz')
        ax.set_title(f'Seed {seed} | {start:.4f} - {end:.4f} ns',fontsize=10)
        heat.imshow((spins[shown,:,2]>=0).T,origin='lower',aspect='auto',interpolation='nearest',
                    extent=(t[shown[0]],t[shown[-1]],-.5,99.5),vmin=0,vmax=1,cmap=cmap)
        heat.set(xlim=(start,end),xlabel='Time (ns)',ylabel='Chain site')
    fig.subplots_adjust(left=.08,right=.98,top=.96,bottom=.08,hspace=.8)
    fig.savefig(path,dpi=170); plt.close(fig)


def main():
    MEDIA.mkdir(exist_ok=True)
    paper=MEDIA/'paper_figure2.png'
    if not paper.exists(): raise FileNotFoundError('extract the paper Figure 2 first')
    rows=[]; full=[]
    with h5py.File(OUT/'trajectory.h5','r') as h:
        assert h.attrs['complete'] and h['spins'].shape==(3,100001,100,3)
        t=h['time_ps'][:]/1000
        for b in range(3):
            s=h['spins'][b]; mz=h['M_total'][b,:,2]; seed=int(h['seed'][b])
            panel=MEDIA/f'path_{b}_paper_layout.png'
            plot_path(t,mz,s,np.linspace(0,PAPER_END_NS,4),panel,seed)
            whole=MEDIA/f'path_{b}_full_10ns.png'
            plot_path(t,mz,s,[0,10],whole,seed)
            rows.append(f'<article><h3>模拟轨迹 {b} · seed {seed}</h3>'+image_tag(panel)+'</article>')
            full.append(f'<article><h3>轨迹 {b}：连续 0–10 ns，不切段</h3>'+image_tag(whole)+'</article>')
    style='''body{max-width:1650px;margin:25px auto;padding:15px;font:16px/1.7 system-ui;background:#f4f6f8;color:#182534}article,section{background:white;padding:18px;margin:18px 0;border-radius:10px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}.paper{position:sticky;top:10px}.figure{width:100%;height:auto;cursor:zoom-in}canvas{display:block;width:100%;height:auto}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}button,select,input{font:inherit}button{padding:6px 12px;cursor:pointer}input[type=number]{width:90px}.scrub{width:100%}.warning{background:#fff1cb;padding:15px}.player:fullscreen{background:white;overflow:auto;padding:25px;box-sizing:border-box}dialog{max-width:96vw;max-height:96vh;padding:10px}dialog img{max-width:92vw;max-height:88vh;object-fit:contain}nav a{margin-right:20px}@media(max-width:850px){.compare{grid-template-columns:1fr}.paper{position:static}}'''
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Bauer图2：原文对照与连续轨迹</title><style>'+style+'</style><h1>Bauer 图2：原文与三条模拟轨迹对照</h1>',
          '<nav><a href="#comparison">原文对照</a><a href="#full">完整10 ns轨迹</a></nav>',
          '<p>一维链使用静态磁化曲线与位置–时间图展示演化，不设置动画播放器；二维平面研究时再加入动画演示。</p>',
          '<p>现有随机LL模拟结果，不是生成模型；本次仅重绘，未重新计算。100点开放链，K/J=0.1，kBT/J=0.11，λ=0.1，无外场。J=50 meV用于单位映射，对应约63.82 K。初态为+z后固定预演化20 ps；不筛选seed、不强制磁化数值。</p>',
          '<p class="warning">左右比较物理过程与图形结构，不要求随机事件逐一对应。原图横轴是ℏ/J单位，原文750000单位换算为9.873179 ns；右侧也使用这三个时间窗口，未按翻转时刻平移或对齐。余下约0.127 ns保留在末尾完整图中。右侧站点编号0–99，原图标至100；总磁化均是100个自旋之和。</p>',
          '<h2 id="comparison">一、原文在左，三条模拟结果在右</h2><p>图中蓝色表示 s_z&lt;0，绿色表示 s_z≥0。点击任一静态图可放大。原图A–F是作者标注，不对应本次轨迹中的固定事件。</p><div class="compare">',
          '<section class="paper"><h3>原文 Figure 2</h3>'+image_tag(paper)+
          '<p>Bauer et al., J. Phys.: Condens. Matter 23, 394204 (2011)。图像摘自项目内 arXiv:1010.4730v2 PDF 第4页，仅裁去周围正文，未重画原数据。<a href="https://arxiv.org/html/1010.4730v2#S2.F2">查看原文及图注</a> · <a href="https://doi.org/10.1088/0953-8984/23/39/394204">DOI</a></p><p>每段上方看总磁化是否从一个平台进入另一个平台；下方看反向区域从端部还是内部开始，随后扩展还是消失。孤立的单点变色也可能只是热涨落，不能直接认定为成核。</p></section><div>'+''.join(rows)+'</div></div>',
          '<details><summary>能量与模长检查</summary>'+image_tag(OUT/'diagnostics.png')+'</details>',
          '<h2 id="full">二、每条轨迹的完整0–10 ns记录</h2><p>以下每张图使用连续时间轴，不裁去等待段或失败尝试。总磁化使用全部100001个保存时刻；时空图等间隔抽取最多6000列用于显示，不把它当成全部保存帧的高分辨率事件检测。绿色/蓝色含义与上方一致。</p>',
          ''.join(full),'<dialog id="zoom"><button onclick="this.parentElement.close()">关闭</button><br><img alt="放大图"></dialog>',
          '<script>document.querySelectorAll(".figure").forEach(img=>{img.onclick=()=>{const d=document.getElementById("zoom");d.querySelector("img").src=img.src;d.showModal();};});</script></html>']
    target=OUT/'index.html'; backup=OUT/'index_before_comparison.html'
    if not backup.exists(): backup.write_bytes(target.read_bytes())
    tmp=OUT/'index.comparison.tmp.html'; tmp.write_text('\n'.join(page)); tmp.replace(target)
    (MEDIA/'manifest.json').write_text(json.dumps({'trajectory_sha256':hashlib.sha256((OUT/'trajectory.h5').read_bytes()).hexdigest(),
         'paper_end_ns':PAPER_END_NS,'paper_crop':{'pdf_page':4,'dpi':180,'x':126,'y':122,'width':1280,'height':670},
         'comparison_edges_ns':np.linspace(0,PAPER_END_NS,4).tolist(),'full_edges_ns':[0,10],
         'presentation':'static_1d','animation':False,'new_simulation':False},indent=2))
    print(target,flush=True)


if __name__=='__main__': main()
