"""Self-contained review dashboard: physical models, inventory, 3D spin GIFs.

No CDN or external JavaScript is required. All depicted vectors are actual spins;
spatial subsampling is fixed across time and recorded in a JSON manifest.
"""
import base64
import html
import json
from pathlib import Path
import h5py
import numpy as np
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results'
PILOT=BASE.parent/'literature_preflight_20260917/results'
SMOKE=BASE.parent/'path_smoke_20260917/results'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def media(path):
    mime='image/gif' if path.suffix=='.gif' else 'image/png'
    return '<img loading="lazy" src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def select_sites(shape,name):
    indices=np.indices(shape[:-1]).reshape(len(shape)-1,-1).T
    if name=='bauer':
        start=max(0,(len(indices)-12)//2)
        ids=np.arange(start,min(start+12,len(indices))); pos=np.c_[indices[:,0],np.zeros((len(indices),2))]
    elif name=='miltat':
        pos=np.c_[indices[:,0]+.5*indices[:,2],np.sqrt(3)*(indices[:,1]+.5*indices[:,2]),np.zeros(len(indices))]
        center=pos.mean(0); ids=np.argsort(((pos-center)**2).sum(1))[:96]; pos-=center
    else:
        # Only basis 0 glyphs at cell indices: Mn fractional z is not invented.
        axes=[np.linspace(0,shape[i]-1,4,dtype=int) for i in range(3)]
        mask=(indices[:,3]==0)
        for i in range(3): mask &= np.isin(indices[:,i],axes[i])
        ids=np.flatnonzero(mask); pos=indices[:,:3].astype(float)
    return ids,pos[ids],indices[ids]


def component_grid(positions, vectors):
    """One cell per selected spin; tile z layers, never average overlaps."""
    x,ix=np.unique(positions[:,0],return_inverse=True)
    y,iy=np.unique(positions[:,1],return_inverse=True)
    z,iz=np.unique(positions[:,2],return_inverse=True)
    columns=ix+iz*(len(x)+1)
    grid=np.full((len(y),len(x)*len(z)+len(z)-1,3),np.nan)
    grid[iy,columns]=vectors
    return grid,iy,columns


def spin_movie(path,times,name,out,title):
    version='component_grids_v4'
    manifest=out.with_suffix('.json')
    if out.exists() and manifest.exists() and json.loads(manifest.read_text()).get('layout')==version: return
    ids,pos,labels=select_sites(path.shape[1:],name)
    flat=path.reshape(len(path),-1,3); spins=flat[:,ids]
    deviation=np.linalg.norm(flat,axis=-1)-1
    extrema=(deviation.min(1),deviation.mean(1),deviation.max(1))
    max_error=np.abs(deviation).max(1)
    arrow_scale=.75 if name!='hirst' else 3.
    low=pos.min(0)-max(2,arrow_scale*1.2); high=pos.max(0)+max(2,arrow_scale*1.2)
    fig=plt.figure(figsize=(16,11),dpi=90)
    gs=fig.add_gridspec(3,3,height_ratios=(1,1,1.05),hspace=.38,wspace=.28)
    ax_field=fig.add_subplot(gs[:2,:2],projection='3d')
    ax_norm=fig.add_subplot(gs[0,2]); ax_error=fig.add_subplot(gs[1,2])
    projections=[fig.add_subplot(gs[2,i]) for i in range(3)]
    cmap=plt.get_cmap('coolwarm').copy(); cmap.set_bad('#eeeeee')
    for ax,component in zip(projections,'xyz'):
        bar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1,1),cmap=cmap),
                        ax=ax,orientation='horizontal',fraction=.08,pad=.24,ticks=[-1,-.5,0,.5,1])
        bar.set_label('s'+component+' (signed component)',fontsize=9)
    frames=[]
    for k,t in enumerate(times):
        for ax in [ax_field,ax_norm,ax_error,*projections]: ax.clear()
        colors=plt.cm.coolwarm((spins[k,:,2]+1)/2)
        ax_field.quiver(*pos.T,*spins[k].T,length=arrow_scale,normalize=False,colors=colors,arrow_length_ratio=.28)
        ax_field.set(xlabel='cell/site x',ylabel='cell/site y',zlabel='cell z / spin z')
        ax_field.set_title('3D magnetic vector field | color = sz | no normalization\n'+('basis 0; cell coordinates, NOT Mn positions' if name=='hirst' else 'same fixed sites as the component grids'),fontsize=12)
        limits=np.ptp(pos,axis=0)
        for setter,lo,hi in zip((ax_field.set_xlim,ax_field.set_ylim,ax_field.set_zlim),low,high): setter(lo,hi)
        ax_field.set_box_aspect(high-low if name=='bauer' else np.maximum(limits,4)); ax_field.view_init(elev=25,azim=-55)
        ax_field.tick_params(labelsize=8)
        for values,label in zip(extrema,('min','mean','max')):
            ax_norm.plot(times,values,label=label)
            ax_norm.plot(t,values[k],'o',markersize=4)
        ax_norm.axhline(0,color='gray',ls=':'); ax_norm.axvline(t,color='gray',ls=':')
        ax_norm.set(xlabel='Physical time (ps)',ylabel='|s| - 1',title='Full-domain spin norm trend')
        ax_norm.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False); ax_norm.legend(fontsize=8)
        ax_error.plot(times,max_error,'o-',markersize=3)
        ax_error.axvline(t,color='gray',ls=':')
        ax_error.set(xlabel='Physical time (ps)',ylabel='max over all sites of abs(|s|-1)',title='Conservation error: saved frames')
        ax_error.set_ylim(0,max(1e-16,float(max_error.max())*1.25))
        ax_error.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
        grid,_,_=component_grid(pos,spins[k])
        for c,ax in enumerate(projections):
            ax.imshow(grid[:,:,c],origin='lower',interpolation='nearest',cmap=cmap,vmin=-1,vmax=1)
            ax.set_title('s'+'xyz'[c]+' : site component map',fontsize=11)
            ax.set_xticks(np.arange(grid.shape[1]+1)-.5,minor=True)
            ax.set_yticks(np.arange(grid.shape[0]+1)-.5,minor=True)
            ax.grid(which='minor',color='white',linewidth=.5)
            ax.tick_params(which='minor',bottom=False,left=False)
            if name=='hirst':
                nx=len(np.unique(pos[:,0])); z=np.unique(pos[:,2])
                ax.set_xticks(np.arange(len(z))*(nx+1)+(nx-1)/2)
                ax.set_xticklabels(['z='+str(int(v)) for v in z],fontsize=8)
                ax.set_xlabel('Separate XY layers; x increases within each tile',fontsize=8)
                ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(np.unique(pos[:,1]).astype(int),fontsize=8)
            else:
                xs=np.unique(pos[:,0]); ticks=np.unique(np.linspace(0,len(xs)-1,min(5,len(xs)),dtype=int))
                ax.set_xticks(ticks); ax.set_xticklabels([f'{xs[i]:g}' for i in ticks],fontsize=8)
                ax.set_xlabel('x position (one square per site)',fontsize=8)
                ax.set_yticks([])
        norm=max_error[k]
        fig.suptitle(f'{title} | t={t:.4f} ps | frame {k+1}/{len(times)} | max norm error={norm:.1e}',fontsize=11)
        fig.subplots_adjust(left=.05,right=.97,bottom=.07,top=.91)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=500 if len(times)==3 else 170,loop=0)
    frames[len(frames)//2].save(out.with_suffix('.png'))
    out.with_suffix('.json').write_text(json.dumps({'spatial_flat_indices':ids.tolist(),'spatial_multi_indices':labels.tolist(),
        'layout':version,'time_ps':times.tolist(),'interpolation':False,'arrow_scale':arrow_scale,
        'component_maps':['sx','sy','sz'],'color_limits':[-1,1],'component_values_renormalized':False,
        'grid_layout':'unique x/y coordinates; z layers tiled horizontally with a blank column; no averaging',
        'norm_scope':'all sites, saved physical frames only; no unrecorded flow steps',
        'hirst_geometry_warning':'cell coordinates and basis 0 only, not audited atomic positions'},indent=2))
    plt.close(fig)


def model_section():
    flow=['输入：固定初态 S₀、晶格连接、温度、归一化物理时间、晶体参考轴',
          '随机源：各后继帧从均匀球面噪声开始；初态帧固定',
          '9 个三维向量：当前态、初态、一跳/两跳邻域、前/后时间帧、3 根参考轴',
          '9×9 Gram 矩阵 → 81 个旋转不变量；加 3 个标量 → 84 维',
          'Linear 84→24 + SiLU → 2 个残差块（LayerNorm + FiLM）',
          'Linear 24→18：9 个直接向量权重 + 9 个叉乘权重',
          '组合三维速度 → 切平面投影 v−(s·v)s → 球面指数映射',
          '32 个 flow 积分步 → 联合生成 2 个后继帧；不是 32 个 LLG 物理步']
    blocks=''.join('<div class="block">'+html.escape(x)+'</div>'+('<div class="arrow">↓</div>' if i<len(flow)-1 else '') for i,x in enumerate(flow))
    return '<h2 id="model">五、当前模型到底是什么</h2><p>下面是实际已运行的 PathNet 最小原型，不是已完成的完整多尺度模型。三个体系分别训练三个模型，不是一个 checkpoint 跨三个哈密顿量。</p>'+blocks+table(['问题','当前实际实现'],[
        ['输入哈密顿量了吗？','不输入 J、D、K 或有效场；使用晶格邻接（包含不同键的连接，但忽略耦合权重）和参考轴。'],
        ['物理时间与 flow 时间','物理时间指定输出帧间隔；flow 时间 τ∈[0,1] 是噪声运输进度，两者不是一回事。'],
        ['旋转等变性','磁矩与晶体参考轴共同旋转的 SO(3) 协变；不是固定晶体下任意旋转磁矩仍物理等价。'],
        ['周期性','邻接运算正确跨越周期边界；不要求首尾自旋数值相等。'],
        ['模长','切空间速度＋球面更新；守恒的是每个自旋模长，不是总磁化。'],
        ['时间模块','当前只用了相邻帧向量特征，没有时间 Transformer 或长期记忆。'],
        ['多尺度模块','当前仅一跳/两跳局域信息；层级粗化、远程模块、跨尺寸效果尚未实现验证。'],
        ['训练规模','每体系 80 次更新；完整空间、3 帧窗口；Adam lr=0.002、宽度24、2残差块。'],
        ['缺少什么','初态生成器、多温度有效性验证、完整多尺度、长时间续写和严格路径分布验收。']])


def main():
    cfg=yaml.safe_load((BASE.parent/'literature_preflight_20260917/config.yaml').read_text())
    titles={'bauer':'Bauer：一维 Co/Pt 磁链','miltat':'Miltat：二维 Co 单层斯格明子','hirst':'Hirst：三维 Mn₂Au 反铁磁体'}
    formulas={
      'bauer':('H = −J Σ⟨ij⟩ sᵢ·sⱼ − Σᵢ(Kx sxᵢ² + Ky syᵢ² + Kz szᵢ²)，|sᵢ|=1，B=0。',
                '当前 (Kx,Ky,Kz)=(0.2,0,1.0) meV；论文中轴取 y，可取 (0,0.2,1.0)。坐标整体正旋转后等价，低/高能垒同为0.8/1 meV。最近邻交换，无偶极和 DMI。',
                '原文 Eq.(1)–(3)、Section III/IV、Appendix A。式(1)双重求和带1/2，当前每条无向键计一次。'),
      'miltat':('H = −J Σ⟨ij⟩ sᵢ·sⱼ − Σ⟨ij⟩ Dᵢⱼ·(sᵢ×sⱼ) − K Σᵢ szᵢ² + [μ₀ μs²/(4π)] Σᵢ&lt;ⱼ [sᵢ·sⱼ/rᵢⱼ³ − 3(sᵢ·rᵢⱼ)(sⱼ·rᵢⱼ)/rᵢⱼ⁵]。',
                '当前 Dᵢⱼ=d(ẑ×r̂ᵢⱼ)，rᵢⱼ从 i 指向 j；三角晶格，完整开放边界偶极和，无外场。该式是当前代码的明确表达，不冒充原文逐字公式。',
                '原文 Section III A 列出交换、DMI、各向异性、完整偶极及参数；本文没有从该段确认上述 DMI 有向键符号映射，故保留未核验项。'),
      'hirst':('H = −Σ⟨ij⟩ Jᵢⱼ sᵢ·sⱼ − dx Σᵢ sxᵢ² − dz Σᵢ szᵢ² − μs B·Σᵢsᵢ；B=0。',
                '当前四类交换 J1–J4；每点配位数4/4/4/1，四 Mn 基元。负 dz 对应易平面，正 dx 选出 x 易轴；没有加入 DMI 或偶极。',
                '正式发表版 Eq.(1)–(4)、Table I、Fig.1；温度跃迁协议见 Fig.7及其正文。')}
    original={
      'bauer':{'J':'50 meV','barrier_low':'0.8 meV','barrier_high':'1 meV','alpha':'0.1（作者估计）','temperature_K':'50 K 是本项目诊断点，非指定复现图点'},
      'miltat':{'J':'29 meV/bond','D':'−1.5 meV/bond；符号约定待核对','K':'0.4 meV/atom','moment':'2.1 μB','a':'2.51 Å','alpha':'0.5（Fig.10）','temperature_K':'25 K（Fig.10条件之一）'},
      'hirst':{'J1':'−5.3422 mRy','J2':'0.6484 mRy','J3':'−0.6341 mRy','J4':'−6.8986 mRy','d_z':'−0.0663 mRy','d_x':'0.0026 mRy','moment':'3.8663 μB','a':'3.330 Å','c':'8.537 Å','alpha':'0.01（温度跃迁）','temperature_K':'600 K为当前选定温度，完整曲线尚未逐点复现'}}
    differences={
      'bauer':'当前50点开放链、50 K、2 ps；论文讨论翻转寿命，当前窗口不足以验收寿命。原文使用弱随机 Runge–Kutta，旧样本为 Heun，不能称为同一积分实现。',
      'miltat':'当前200×116×2=46400自旋开放窗口，数量与论文一致，但窗口固定；原文窗口随纹理移动，扩散分析为ns尺度。当前仅1 ps且纹理尚在弛豫。γ数值来源与DMI符号仍未完全核实。',
      'hirst':'当前30³×4=108000自旋、三向周期边界；30³在论文平衡态段明确，Fig.7尺寸未明确。内部Mn坐标/键壳层对应及γ数值仍待核实。旧Heun漏预测步归一化，已补配对重放，但未覆盖旧数据。'}
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据与物理模型审查台</title>',
          '<style>body{font-family:system-ui,sans-serif;max-width:1320px;margin:28px auto;padding:0 16px;line-height:1.75;background:#f6f8fa;color:#17212b}section,article{background:white;padding:20px;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{border:1px solid #cbd3db;padding:9px;text-align:left;vertical-align:top}th{background:#e6edf4}img{display:block;max-width:100%;height:auto;margin:12px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:16px}nav a{margin-right:16px}.warning{background:#fff0cd;padding:18px;border-left:5px solid #dd8e00}.block{border:1px solid #8ca2b8;background:#edf4ff;border-radius:8px;padding:10px;text-align:center;max-width:900px;margin:auto}.arrow{text-align:center;font-size:22px}code{overflow-wrap:anywhere}details{margin:14px 0}h2{padding-top:16px}</style>',
          '<h1>数据规划、物理模型与三维磁矩审查台</h1><nav><a href="#plan">数据规划</a><a href="#inventory">数量清单</a><a href="#bauer">一维</a><a href="#miltat">二维</a><a href="#hirst">三维</a><a href="#model">模型结构与结果</a></nav>',
          '<p class="warning"><b>状态：未放行正式训练数据生成。</b>文件完整和基础方程检查通过，不代表三个体系已完全复现文献。二维初态稳定性、部分文献映射及长期统计仍未通过或未完成。这里的新增模拟仅用于你审查。</p>',
          '<section><h2 id="plan">一、数据怎样规划</h2><p>统一的是数据格式与来源记录，不是把不同哈密顿量不加区分地混合。三个体系先分别训练；先完成Bauer闭环，再逐步扩展。正式训练样本数、温度网格、尺寸和时长待你确认，当前不擅自冻结或生成。</p>',
          table(['体系','准备的数据类型','物理检验目标','当前前置条件'],[
            [titles['bauer'],'有序态、受控扰动态等条件下的短时随机路径；后续补翻转事件窗口','局域随机增量、空间/时间关联；后续翻转概率、时间、机制','步长/状态覆盖/样本数和时长验收；不能由2 ps判断寿命'],
            [titles['miltat'],'稳定纹理初态及其热运动路径','拓扑保持、形变、位移分布、扩散','先完成纹理制备与稳定性；确认DMI和时间约定'],
            [titles['hirst'],'有序AFM升温和受控扰动弛豫路径','亚晶格磁化、Néel序参量、纵横向弛豫与关联','晶格映射、时间标定及完整弛豫验收']]),
          '<p>未来统一字段：体系ID、配置与来源哈希、边界、尺寸/基元、初态协议、温度/阻尼、积分器/步长、物理时间ps、单位自旋、seed、原始轨迹ID和划分。按独立轨迹/初态组划分训练、开发和测试，禁止重叠窗口跨集合。</p></section>',
          '<section><h2 id="inventory">二、当前究竟有多少数据</h2><p>本清单盘点20260917这轮三个体系的核验数据，不把仓库中的历史实验和旧模型数据混入。</p>']
    inventory=[]
    for name in titles:
        row=cfg['systems'][name]; n=row['numerics']
        with h5py.File(PILOT/name/'trajectory.h5') as h: shape=h['spins'].shape
        cold=ROOT/name/'cold/report.json'
        count=json.loads(cold.read_text())['trajectories'] if cold.exists() else 0
        strong=ROOT/name/'cold_strong/report.json'
        strong_count=json.loads(strong.read_text())['trajectories'] if strong.exists() else 0
        inventory.append([name,'×'.join(map(str,shape[2:-1])),f'{shape[1]}条 / 每条{shape[0]}帧',f'{row["physics"]["temperature_K"]} K；α={row["physics"]["alpha"]}',f'{n["duration_ps"]} ps / {n["save_ps"]} ps',f'{n["dt_fs"]/4} fs',f'{strong_count}条显著扰动 + {count}条原态/弱扰动对照'])
    page.append(table(['体系','空间×基元','已保存完整热轨迹','条件','时长 / 帧间隔','保存自旋的积分步长','新增确定性数据'],inventory))
    page.append('<p>原始热参考共12条（8+2+2），不是三个步长乘3后的36条独立样本；三个步长共享噪声，只有最细步长保存完整自旋。另有：Hirst同噪声重放2条（非新增独立样本）；Miltat旧末态继续零温弛豫1条、5 ps；模型输出每体系2条、共6条。CPU I/O小样本、解析极限样本及小尺寸积分器诊断不计为正式数据。下面的新零温对照也不是热分布训练样本。</p></section>')
    for name,title in titles.items():
        out=ROOT/name; row=cfg['systems'][name]
        page.extend(['<section><h2 id="'+name+'">三、'+title+'</h2>', '<p><a href="https://doi.org/'+row['doi']+'">DOI：'+row['doi']+'</a> · <a href="'+row['source']+'">可核查原文</a></p>',
                     '<h3>哈密顿量与来源</h3><pre>'+formulas[name][0]+'</pre><p>'+formulas[name][1]+'</p><p><b>出处：</b>'+formulas[name][2]+'</p>'])
        rows=[]
        for key,value in row['physics'].items():
            current=str(value['value'])+' '+value['unit'] if isinstance(value,dict) else str(value)+(' K' if key=='temperature_K' else '')
            rows.append([key,current,original[name][key], 'Section IV / Eq.1–3' if name=='bauer' else ('Section III A / Fig.10' if name=='miltat' else ('Table I' if key not in ('alpha','temperature_K') else 'Fig.7及正文；当前温度选择另行标注'))])
        page.extend([table(['参数','当前配置','原文值或差异','原文位置'],rows),'<p class="warning">'+differences[name]+'</p>'])
        if name=='bauer':
            page.append('<p>动力学：ℏ ds/dt=−s×h−λs×(s×h)−s×f，h=−∂H/∂s；噪声仅进入进动项。代码时间尺度 t₀=ℏ/J，热强度 θ=kBT/J。磁矩模长已吸收到能量参数，不额外假设μs。</p>')
        else:
            page.append('<p>动力学采用 Gilbert 形式：ds/dt=−γ/[μs(1+α²)]{s×(h+ζ)+αs×[s×(h+ζ)]}，h=−∂H/∂s。零温把ζ严格置零，保留阻尼。代码 t₀=μs/(γE₀)，γ=1.76085963023×10¹¹ s⁻¹T⁻¹为当前常规电子值，尚不冒充原文明确数值；γ与γ₀=μ₀γ的场单位约定不同，不可直接混用。</p>')
        meta=json.loads((PILOT/name/'metadata.json').read_text())
        initial_description={
            'bauer':'全部磁矩沿+z，直接接入50 K热浴；8条共享同一确定初态、使用不同热噪声分支。',
            'miltat':'从半径参数1.25 nm、壁宽参数0.4 nm的径向纹理出发，先零温弛豫1 ps，再接入25 K热浴；2条共享制备初态。这个1 ps制备没有通过稳定性验收。',
            'hirst':'四基元沿x轴，符号(+,-,-,+)，从完全有序AFM瞬时接入600 K热浴；2条共享同一确定初态。'}[name]
        page.append('<p><b>当前热样本的确切条件：</b>'+initial_description+' 边界周期标志='+str(row['geometry']['periodic'])+'；批次随机seed='+str(meta['seed'])+'（批次内独立噪声流元素，不是每条轨迹都重置为相同seed）。</p>')
        page.append('<details><summary>代码实际约化参数与SI换算尺度</summary><pre>'+html.escape(json.dumps({'reduced':meta['reduced'],'scales':meta['scales']},indent=2))+'</pre></details>')
        with h5py.File(PILOT/name/'trajectory.h5') as h: thermal=h['spins'][:,0]; times=h['time_ps'][:]
        movie=out/'thermal_spin3d.gif'; spin_movie(thermal,times,name,movie,name+' | finite-T reference #0')
        page.extend(['<h3>有限温随机轨迹：三维磁矩场＋三个分量格点色彩图</h3><p>显示轨迹0。左上保留放大的三维磁矩场；下方三个格子图分别显示相同格点的 sₓ、sᵧ、s_z，已取消二维箭头。每个有效格子对应一个磁矩，不是XY/XZ/YZ平面的矢量长度。colorbar固定为[-1,1]：蓝色负向、红色正向、近白色接近零；不对每帧自动缩放、不重新归一化。灰色格子是没有显示磁矩的位置或分层间隔，不是零值。右侧统计全部格点的模长偏差最小/平均/最大值及最大绝对误差，零线对应模长1。只检查保存的物理时间帧，不能据此断言未记录的flow内部步骤也已逐步核验。</p>',media(movie)])
        if name=='hirst': page.append('<p><b>三维位置说明：</b>三维场及色彩图只显示相同的基元0固定格点，位置是计算单元索引，不是尚未核验的Mn原子分数坐标。色彩图将不同z层的XY格子横向平铺，层间留灰色空列；每层内x向右、y向上，标明z层位置，不对深度或不同基元做平均。这里的格子是抽样索引排布，不代表原子距离。全场模长统计包含四个基元的所有磁矩。</p>')
        if name=='miltat': page.append('<p>三维场及色彩图选中心附近96个真实格点（包含两个基元），不是全样品；格子按真实x、y坐标排序，x向右、y向上，空位灰色。方格用于显示分量，并不将三角晶格改成方格物理模型。全场模长统计仍使用全部格点。</p>')
        if name=='bauer': page.append('<p>三维场及色彩图显示中央12个格点（50点链的零起算索引19–30），色彩图中每个方格对应一个磁矩，不是全链图。全场模长统计仍包含全部50个磁矩。</p>')
        strong=out/'cold_strong'
        if (strong/'report.json').exists():
            sr=json.loads((strong/'report.json').read_text())
            with h5py.File(strong/'trajectory.h5') as h: ss=h['spins'][:,0]; st=h['time_ps'][:]; se=h['energy_per_spin'][:]
            page.append('<h3>重点：显著扰动初态后的零温确定性演化</h3><p>三个体系均施加绕y轴的平滑空间变角旋转：θ(r)=35°+15°∏ₖcos(2πrₖ/Lₖ)，旋转角20°–50°，同一单元内基元使用相同旋转。旋转严格保持自旋模长；不是演化过程施加噪声。它是专门设计的非线性弛豫诊断，不冒充原文初态，也不用于自动生成正式训练集。</p>')
            movie=strong/'spin3d.gif'; spin_movie(ss,st,name,movie,name+' | T=0 | STRONG initial perturbation')
            page.append(media(movie))
            diagnostic={}
            if name=='miltat':
                from tests.physical_audit_20260917.gallery import triangular_charge
                q,qi=triangular_charge(ss[[0,-1],None]); diagnostic['interior_Q_initial_final']=qi[:,0].tolist()
            if name=='hirst':
                signed=ss*np.array([1,-1,-1,1])[None,None,None,None,:,None]
                neel=signed.reshape(len(ss),-1,3).mean(1)
                diagnostic['neel_initial_final']=neel[[0,-1]].tolist()
            diagnostic['spatial_mean_initial_final']=ss.reshape(len(ss),-1,3).mean(1)[[0,-1]].tolist()
            diagnostic['energy_change_per_spin']=float(se[-1,0]-se[0,0])
            (strong/'derived.json').write_text(json.dumps(diagnostic,indent=2))
            page.append('<pre>'+html.escape(json.dumps({**sr,**diagnostic},indent=2,ensure_ascii=False))+'</pre>')
            fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True); ax.plot(st,se[:,0]-se[0,0])
            ax.set(xlabel='Time (ps)',ylabel='Energy change per spin / E0',title=name+': strong-perturbation T=0 dissipation')
            fig.savefig(strong/'energy.png',dpi=150); plt.close(fig); page.append(media(strong/'energy.png'))
        else: page.append('<p class="warning">显著扰动零温轨迹尚未有完成报告；暂不能展示为已完成。</p>')
        cold=out/'cold'
        if (cold/'report.json').exists():
            report=json.loads((cold/'report.json').read_text())
            with h5py.File(cold/'trajectory.h5') as h: cs=h['spins'][:]; ct=h['time_ps'][:]; ce=h['energy_per_spin'][:]
            page.append('<details><summary>辅助对照：原态与10°弱扰动的零温轨迹（非主要展示）</summary><p>同一哈密顿量、相同阻尼和边界，T=0且每一步Wiener增量严格为0；使用预测/校正都归一化的Heun。保留阻尼，所以能量可以下降，不应要求能量守恒。初态静止是有效结果，但不替代上面的显著扰动实验。</p>')
            page.append('<pre>'+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+'</pre>')
            for b,label in enumerate(report['labels']):
                movie=cold/f'spin3d_{b}.gif'; spin_movie(cs[:,b],ct,name,movie,name+' | T=0 | '+('same initial' if b==0 else '10-degree tilted diagnostic'))
                page.append('<h4>'+('与热参考相同初态' if b==0 else '额外诊断：整体绕y轴倾斜10°，不是原文协议')+'</h4>'+media(movie))
            fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
            for b in range(len(report['labels'])): ax.plot(ct,ce[:,b]-ce[0,b],label='cold '+str(b))
            ax.set(xlabel='Time (ps)',ylabel='Energy change per spin / E0',title=name+': deterministic dissipation'); ax.legend()
            fig.savefig(cold/'energy.png',dpi=150); plt.close(fig); page.append(media(cold/'energy.png'))
            page.append('</details>')
        else: page.append('<p class="warning">零温任务尚未生成完成报告，不计为已完成。</p>')
        page.append('<details><summary>上一轮物理核验与尚未通过项</summary>')
        for filename in ('physical_diagnostics.png','independent_checks.png','texture_stability.png','paired_heun.png'):
            if (out/filename).exists(): page.append(media(out/filename))
        if (out/'zero_T_relaxation.gif').exists(): page.append(media(out/'zero_T_relaxation.gif'))
        page.append('<pre>'+html.escape((out/'checks.json').read_text())+'</pre></details>')
        page.append('<p>原始热数据：<code>'+str(PILOT/name/'trajectory.h5')+'</code><br>新增零温数据：<code>'+str(cold/'trajectory.h5')+'</code></p></section>')
    page.append('<section>'+model_section())
    metrics=[]
    for name in titles:
        summary=json.loads((SMOKE/name/'summary.json').read_text()); c=summary['checks']
        metrics.append([name,str(summary['train_ids'])+' / '+str(summary['dev_id']),f'{summary["heldout_fixed_loss_before"]:.4f} → {summary["heldout_fixed_loss_after"]:.4f}',f'{c["unit_norm_error"]:.2e}',f'{c["rotation_generated_path"]:.2e}'])
    page.append(table(['体系','训练轨迹ID / 开发ID','固定开发样本FM loss 前→后','模长误差','联合旋转误差'],metrics))
    page.append('<p class="warning">仅有一个固定噪声实现的开发loss，不是完整验证分布上的均值；单温度数据不能证明温度条件有效。几何约束通过不等于路径分布学对。当前生成涨落偏大，不能宣称小体系学习或尺寸泛化成功。</p>')
    for name in titles:
        data=np.load(SMOKE/name/'generated.npz'); time=data['time_ps']
        page.append('<h3>'+name+'：同初态开发参考与生成结果</h3>'+media(SMOKE/name/'overview.png'))
        for label,path in [('reference',data['reference']),('generated',data['spins'][0])]:
            movie=ROOT/name/('model_'+label+'_spin3d.gif'); spin_movie(path,time,name,movie,name+' | '+label+' | short smoke window')
            page.append('<details><summary>'+('开发参考' if label=='reference' else '生成 seed 71')+'：三维磁矩动画（只有3帧）</summary>'+media(movie)+'</details>')
    page.append('<p><b>后续完整设计：</b>局域模块 → 可验证的层级/远程多尺度模块 → 更完整的时间建模 → 球面路径生成；初态生成和长时间续写另行验收。完整设计不能用当前最小原型的通过结果代替。</p></section></html>')
    target=ROOT/'index.html'
    if target.exists() and not (ROOT/'index_before_dashboard.html').exists():
        (ROOT/'index_before_dashboard.html').write_bytes(target.read_bytes())
    target.write_text('\n'.join(page))
    print(target,flush=True)


if __name__=='__main__': main()
