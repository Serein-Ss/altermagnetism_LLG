"""Real-frame GIFs, offline HTML and measurements on frozen pilot trajectories."""
import base64
import html
import json
from pathlib import Path
import h5py
import numpy as np
import yaml
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
PILOT = BASE.parent / 'literature_preflight_20260917/results'
SMOKE = BASE.parent / 'path_smoke_20260917/results'


def triangular_charge(spins):
    nx, ny = spins.shape[2:4]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    pos = np.stack((x, np.sqrt(3)*y), -1)[...,None,:] + np.array([[0.,0.],[.5,np.sqrt(3)/2]])
    pos = pos.reshape(-1,2)
    tri = Delaunay(pos).simplices.copy()
    ab, ac = pos[tri[:,1]]-pos[tri[:,0]], pos[tri[:,2]]-pos[tri[:,0]]
    flip = ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0] < 0
    tri[flip] = tri[flip][:,[0,2,1]]
    center = pos[tri].mean(1)
    inside = ((center > pos.min(0)+3) & (center < pos.max(0)-3)).all(1)
    q, qi = [], []
    for state in spins:
        s = state.reshape(state.shape[0],-1,3)
        a,b,c = (s[:,tri[:,j]] for j in range(3))
        numerator = (a*np.cross(b,c)).sum(-1)
        denominator = 1+(a*b).sum(-1)+(a*c).sum(-1)+(b*c).sum(-1)
        solid_angle = 2*np.arctan2(numerator, denominator)
        q.append(solid_angle.sum(1)/(4*np.pi)); qi.append(solid_angle[:,inside].sum(1)/(4*np.pi))
    return np.array(q), np.array(qi)


def gif(paths, times, name, outfile, titles, fps, dpi):
    # paths: list of [time,*space,basis,3], never fabricated/interpolated frames.
    fig, axes = plt.subplots(2,len(paths),figsize=(6*len(paths),6),squeeze=False,dpi=dpi)
    frames = []
    means = [p.reshape(len(p),-1,3).mean(1) for p in paths]
    for k,t in enumerate(times):
        for col,(p,title) in enumerate(zip(paths,titles)):
            top,bottom = axes[:,col]; top.clear(); bottom.clear()
            if name == 'bauer':
                s=p[k,:,0]
                top.quiver(np.arange(len(s)),np.zeros(len(s)),s[:,0],s[:,2],s[:,2],cmap='coolwarm',clim=(-1,1),angles='xy',scale_units='xy',scale=1)
                top.set(xlim=(-1,len(s)),ylim=(-1.3,1.3),xlabel='Chain site',ylabel='Spin x-z projection')
            elif name == 'miltat':
                z=p[k,...,0,2]; cx,cy=np.array(z.shape)//2
                half=15; z=z[cx-half:cx+half,cy-half:cy+half]
                top.imshow(z.T,origin='lower',vmin=-1,vmax=1,cmap='coolwarm',extent=(-half*.251,half*.251,-half*.251*np.sqrt(3),half*.251*np.sqrt(3)))
                top.set(xlabel='x from window center (nm)',ylabel='y from window center (nm)')
                title += '\ncentral crop, basis 0; sz colors [-1,1]'
            else:
                z=p[k,:,:,p.shape[3]//2,0,2]
                top.imshow(z.T,origin='lower',vmin=-1,vmax=1,cmap='coolwarm')
                top.set(xlabel='Cell x',ylabel='Cell y')
                title += '\nmiddle-z slice, basis 0; sz colors [-1,1]'
            top.set_title(title)
            for c,label in enumerate(('mx','my','mz')):
                bottom.plot(times,means[col][:,c],label=label,alpha=.65)
                bottom.plot(times[k],means[col][k,c],'o')
            if name == 'hirst':
                n=(p*np.array([1,-1,-1,1])[None,None,None,None,:,None]).reshape(len(p),-1,3).mean(1)
                bottom.plot(times,np.linalg.norm(n,axis=-1),label='|Neel|',color='black')
            bottom.axvline(t,color='gray',ls=':'); bottom.set(xlabel='Time (ps)',ylabel='Order parameter',ylim=(-1.05,1.05)); bottom.legend(fontsize=8)
        fig.suptitle(f'{name}: t = {t:.4f} ps | saved frame {k+1}/{len(times)} | playback slowed down')
        fig.tight_layout(rect=(0,0,1,.94))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    frames[0].save(outfile,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0)
    plt.close(fig)


def embed(path, mime): return '<img style="max-width:100%" src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def main():
    settings=yaml.safe_load((BASE/'settings.yaml').read_text())
    source=yaml.safe_load((BASE/'source_audit.yaml').read_text())
    root=BASE/'results'; root.mkdir(exist_ok=True)
    page=['<!doctype html><meta charset="utf-8"><title>物理核验与真实轨迹动画</title>',
          '<style>body{max-width:1200px;margin:30px auto;font-family:sans-serif;line-height:1.8}pre{white-space:pre-wrap;background:#eee;padding:15px}h2{border-top:2px solid #777}</style>',
          '<h1>三体系物理核验：待人工检查，不放行正式数据生成</h1>',
          '<p>本页 GIF 全部由已保存的真实帧制作，没有插帧。时间显示为 ps；播放速度仅为观看设置，不是模拟速度。三维是明确标注的二维截面，并非完整三维可视化。</p>',
          '<p>完整参考动画选轨迹 0；模型对比使用独立开发轨迹和同初态的已保存模型输出（seed 71）。只含 3 帧的模型片段很短，不能证明长时间一致。全部媒体内嵌，可单独下载 HTML 离线查看。</p>',
          '<p>未自动启动统一数据集生成或 Bauer 训练闭环。方程测试通过 ≠ 文献复现通过；短时间样本不能给出可靠寿命或扩散常数。</p>']
    for name in ('bauer','miltat','hirst'):
        out=root/name; out.mkdir(exist_ok=True)
        with h5py.File(PILOT/name/'trajectory.h5') as h:
            s=h['spins'][:]; t=h['time_ps'][:]
            observed={k:{key:value[:] for key,value in h[k].items()} for k in ('observables_dt_div_1','observables_dt_div_2','observables_dt_div_4')}
        report=dict(shape=list(s.shape), finite=bool(np.isfinite(s).all()),norm_error=float(np.abs(np.linalg.norm(s,axis=-1)-1).max()),
                    actual_reference_time_ps=float(t[-1]),train_data_release=False)
        errors={}
        for key in ('magnetization','energy_per_spin'):
            fine=observed['observables_dt_div_4'][key]
            errors[key]=[float(np.sqrt(np.mean((observed['observables_dt_div_'+str(i)][key]-fine)**2))) for i in (1,2)]
        report['old_pilot_observable_rms_vs_finest']=errors
        fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for i in (1,2,4):
            o=observed['observables_dt_div_'+str(i)]
            val=np.linalg.norm(o['neel'],axis=-1) if name=='hirst' else o['magnetization'][...,2]
            ax[0].plot(t,val.mean(1),label='dt/'+str(i))
        ax[0].set(xlabel='Time (ps)',ylabel='|Neel|' if name=='hirst' else 'Mean mz',title='Old pilot: coupled-noise dt comparison'); ax[0].legend()
        if name=='miltat':
            q,qi=triangular_charge(s)
            report['topological_charge_full']=q.tolist(); report['topological_charge_interior']=qi.tolist()
            area=.251**2*np.sqrt(3)/2
            report['positive_sz_equivalent_radius_nm']=np.sqrt((s[...,2]>0).reshape(len(s),s.shape[1],-1).sum(-1)*area/np.pi).tolist()
            ax[1].plot(t,qi); ax[1].set(xlabel='Time (ps)',ylabel='Interior solid-angle charge Q',title='Open-boundary topology diagnostic')
            relaxation=out/'texture_relaxation.h5'
            if relaxation.exists():
                with h5py.File(relaxation) as h: rs=h['spins'][:]; rt=h['time_ps'][:]
                if np.all(np.linalg.norm(rs,axis=-1)>.99):
                    rq,rqi=triangular_charge(rs)
                    report['zero_T_relaxation']={'time_ps':rt.tolist(),'charge_interior':rqi.tolist(),
                          'core_sites':(rs[...,2]>=.5).reshape(len(rs),1,-1).sum(-1).tolist()}
                    report['zero_T_relaxation']['positive_sz_equivalent_diameter_nm']=(2*np.sqrt((rs[...,2]>0).reshape(len(rs),1,-1).sum(-1)*area/np.pi)).tolist()
                    fig_r,ar=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
                    ar[0].plot(rt,report['zero_T_relaxation']['core_sites']); ar[0].set(xlabel='Additional T=0 time (ps)',ylabel='Core sites (sz >= 0.5)',title='Texture still evolving')
                    ar[1].plot(rt,report['zero_T_relaxation']['positive_sz_equivalent_diameter_nm']); ar[1].set(xlabel='Additional T=0 time (ps)',ylabel='Area-equivalent sz>0 diameter (nm)',title='Operational size, not paper digitization')
                    fig_r.savefig(out/'texture_stability.png',dpi=160); plt.close(fig_r)
                    if not (out/'zero_T_relaxation.gif').exists():
                        gif([rs[:,0]],rt,name,out/'zero_T_relaxation.gif',['T=0 diagnostic, from old final state'],6,85)
        else:
            energy=observed['observables_dt_div_4']['energy_per_spin']
            ax[1].plot(t,energy); ax[1].set(xlabel='Time (ps)',ylabel='Energy / (spin E0)',title='Thermal energy: not required to be conserved')
        fig.savefig(out/'physical_diagnostics.png',dpi=160); plt.close(fig)
        (out/'audit.json').write_text(json.dumps(report,indent=2))
        anim=out/'reference.gif'
        if not anim.exists(): gif([s[:,0]],t,name,anim,['Reference trajectory 0 (old pilot)'],settings['animation']['fps'],settings['animation']['dpi'])
        paired=out/'reference_vs_generated.gif'
        if not paired.exists():
            d=np.load(SMOKE/name/'generated.npz')
            gif([d['reference'],d['spins'][0]],d['time_ps'],name,paired,['Held-out reference','Generated seed 71 (smoke only)'],2,85)
        page.extend(['<h2>'+name+'</h2>','<p>来源：<a href="'+source['sources'][name]+'">论文原文</a></p>',
                     '<pre>'+html.escape(yaml.safe_dump(source[name],allow_unicode=True))+'</pre>',
                     '<h3>完整已有参考片段</h3>',embed(anim,'image/gif'),
                     '<h3>参考与生成的同初态短片段（不是逐路径预测）</h3>',embed(paired,'image/gif'),
                     '<h3>物理诊断</h3>',embed(out/'physical_diagnostics.png','image/png')])
        if (out/'zero_T_relaxation.gif').exists():
            page.extend(['<h3>二维纹理零温继续弛豫</h3>',embed(out/'zero_T_relaxation.gif','image/gif'),embed(out/'texture_stability.png','image/png'),
                         '<p><b>未通过稳定纹理验收：</b>本次从已有末态继续零温弛豫 5 ps，核心格点数 10→62，末段仍增加。拓扑未消失不代表尺寸已稳定。当前数据包含纹理松弛的瞬态，不能直接解释为稳态纹理的热扩散样本。</p>'])
        if (out/'checks.json').exists(): page.append('<h3>独立解析极限与积分对照</h3><pre>'+html.escape((out/'checks.json').read_text())+'</pre>')
        else: page.append('<p>新的 GPU 物理核验尚无完成报告；不能标记通过。</p>')
        if (out/'checks.json').exists():
            check=json.loads((out/'checks.json').read_text())
            comparison=check['integrator_comparison']
            fig,aa=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
            aa[0].loglog(comparison['dt_fs'],comparison['rms_vs_midpoint'],'o-')
            aa[0].set(xlabel='Heun time step (fs)',ylabel='Spin RMS vs finite-dt midpoint',title='Smaller dt: closer integrator agreement')
            aa[1].errorbar(comparison['dt_fs'],comparison['energy_bias_per_spin'],yerr=np.array(comparison['energy_bias_se'])*2,fmt='o-')
            aa[1].axhline(0,color='gray',ls=':'); aa[1].set(xlabel='Heun time step (fs)',ylabel='Paired energy bias per spin (E0)',title='Mean difference +/- 2 paired SE')
            fig.savefig(out/'independent_checks.png',dpi=160); plt.close(fig)
            page.append(embed(out/'independent_checks.png','image/png'))
            page.append('<p>解析进动、短时热扩散、零温耗散检查：'+('通过' if check['basic_checks_passed'] else '未通过')+
                        '。积分器对照随步长减小更接近，但 midpoint 本身也有有限步长误差；这些小尺寸短时间结果不能替代原尺寸全时段统计收敛。</p>')
        if name=='bauer':
            page.append('<p><b>核验解释：</b>低/高能垒及时间换算一致；代码与论文的中轴 x/y 命名不同，可用整体正旋转对应。现有 50 K、2 ps 只用于短时诊断，未验证翻转寿命。当前仅能支持继续准备严格验收的数据生成方案，不能声称已经复现论文寿命规律。</p>')
        elif name=='miltat':
            page.append('<p><b>核验解释：</b>内部拓扑电荷约为 1，核心格点少不等于斯格明子消失。图中的半径是 sz>0 区域面积等效半径，不等于论文未完全同一定义的纹理半径。1 ps 不能拿来拟合论文 ns 级扩散。零温继续弛豫用于识别是否仍在明显变形。</p>')
        else:
            page.append('<p><b>核验解释：</b>需使用预测步和校正步都归一化的 Heun。原样本保留不覆盖；原尺寸同噪声重放单独量化该差异。晶格坐标/键壳层的独立来源和绝对时间参数仍有未确认项。</p>')
            if (out/'paper_heun_replay.json').exists():
                replay=json.loads((out/'paper_heun_replay.json').read_text())
                page.append('<h3>原尺寸、同噪声的归一化差异</h3><pre>'+html.escape(json.dumps({k:v for k,v in replay.items() if k not in ('neel_old','neel_paper_heun')},indent=2))+'</pre>')
                fig,aa=plt.subplots(figsize=(7,4),constrained_layout=True)
                aa.plot(t,np.linalg.norm(np.array(replay['neel_old']),axis=-1).mean(1),label='Old Heun')
                aa.plot(t,np.linalg.norm(np.array(replay['neel_paper_heun']),axis=-1).mean(1),'--',label='Paper-style Heun')
                aa.set(xlabel='Time (ps)',ylabel='|Neel|',title='Matched initial state and Wiener increments'); aa.legend()
                fig.savefig(out/'paired_heun.png',dpi=160); plt.close(fig)
                page.append(embed(out/'paired_heun.png','image/png'))
        page.append('<details><summary>原始诊断数值</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details>')
    (root/'index.html').write_text('\n'.join(page))
    print(root/'index.html',flush=True)


if __name__=='__main__': main()
