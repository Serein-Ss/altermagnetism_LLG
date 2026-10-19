"""Larger independent references distinguish noise, discretization, and model bias."""
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.bauer_p2_20260923.common import comparison
from tests.bauer_comprehensive_20260923.run import diagnostic_comparison
from tests.transport_audit.report import table
from .run import HERE,ROOT,parents,gpu_tasks,dump


def compare(x,r):
    return dict(physical=comparison(x,r,100,93277),diagnostic=diagnostic_comparison(x,r,100,93277))


def brief(value):
    return [value['diagnostic'][k]['rmse'] for k in ('increment_variance','two_time_spin_correlation')]+[value['physical']['energy_mean']['rmse']]


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    refs={};reference_rows=[];results=[];errors=[];view=[];sampling=[]
    for i,row in enumerate(parents()):
        try:
            assert (HERE/f'reference/{i:02d}/complete.json').exists()
            with h5py.File(HERE/f'reference/{i:02d}/paths.h5') as h:
                assert h.attrs['complete'];base=h['base/spins'][:];fine=h['fine/spins'][:]
            assert len(base)==len(fine)==128
            refs[i]=(base,fine)
            with h5py.File(ROOT/row['path']) as h:old=h['spins'][:,:11]
            numerical=compare(base,fine);historical=compare(old,base)
            subsets=[]
            # Four nonoverlapping pairs; all observations are reused elsewhere only descriptively.
            for start in range(0,128,32):
                a=base[start:start+16];b=base[start+16:start+32]
                subsets.append(dict(physical=comparison(a,b),diagnostic=diagnostic_comparison(a,b,0,0)))
            reference_rows.append(dict(index=i,parent=row,old16_vs_new128=historical,base128_vs_fine128=numerical,disjoint16_vs16=subsets))
        except Exception as exc:errors.append(dict(reference=i,error=repr(exc)))
    for index,task in enumerate(gpu_tasks()):
        try:
            assert (HERE/f'gpu/{index:02d}/complete.json').exists()
            with h5py.File(HERE/f'gpu/{index:02d}/paths.h5') as h:
                assert h.attrs['complete']
                for name,g in h.items():
                    i=int(name.removeprefix('parent'));base,fine=refs[i];row=parents()[i]
                    arrays={k:g[k][:] for k in ('euler128','euler512','midpoint128','midpoint256')}
                    for method,x in arrays.items():
                        norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-base[:1,0]).max())
                        assert len(x)==256 and norm<2e-6 and anchor<2e-6
                        cb=compare(x,base);cf=compare(x,fine)
                        results.append(dict(task=task,parent_index=i,parent=row,sampler=method,base=cb,fine=cf,norm_error=norm,anchor_error=anchor))
                        view.append([task['seed'],row['L'],row['family'],row['parent_id'],method,*brief(cb),*brief(cf)])
                    for a,b in (('euler128','euler512'),('midpoint128','midpoint256'),('euler512','midpoint256')):
                        # Paired same-latent solver disagreement; not model-to-LLG trajectory error.
                        sampling.append(dict(task=task,parent=row,pair=[a,b],paired_path_rms=float(np.sqrt(np.mean((arrays[a]-arrays[b])**2)))))
        except Exception as exc:errors.append(dict(gpu=index,error=repr(exc)))
    dump(assets/'summary.json',dict(reference=reference_rows,results=results,sampler_disagreement=sampling,errors=errors,scientific_certification=False))
    body=f'<h1>独立参考扩充与采样修复验证</h1><p>参考{len(reference_rows)}/18；模型-采样比较{len(results)}/216；错误{html.escape(str(errors))}。模型冻结，不新增训练。</p>'
    body+='<p>原16条参考之外，每个父状态新增128条0.25fs参考与128条0.125fs参考，均为1ps、0.1ps输出；两组独立噪声，不是逐路径耦合。三个step18000对照模型，每父状态256条生成；Euler128/512和投影中点128/256使用相同潜变量。中点每步两次网络调用，128步与Euler256次调用相当。</p>'
    rv=[]
    for r in reference_rows:
        p=r['parent'];rv.append([p['L'],p['family'],p['parent_id'],*brief(r['old16_vs_new128']),*brief(r['base128_vs_fine128'])])
    body+='<h2>参考样本量与步长核验</h2>'+table(['L','类别','父状态','旧16/新128增量RMSE','旧/新关联RMSE','旧/新能源RMSE','粗/细增量RMSE','粗/细关联RMSE','粗/细能量RMSE'],rv)
    body+='<h2>模型与两组独立参考分别比较</h2>'+table(['seed','L','类别','父状态','采样器','base增量RMSE','base关联RMSE','base能量RMSE','fine增量RMSE','fine关联RMSE','fine能量RMSE'],view)
    body+='<p>参考—参考差异不是通过阈值。100次固定父状态整分支bootstrap包含双方抽样误差，逐点区间不是同时区间；包含零不证明等价。重复使用的开发父状态不构成新最终留出集。没有自动挑选最小误差采样器或晋升正式代码。</p>'
    body+='<details><summary>配对潜变量的求解器差异</summary><pre>'+html.escape(json.dumps(sampling,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<details><summary>完整指标和不确定区间</summary><pre>'+html.escape(json.dumps(dict(reference=reference_rows,results=results),ensure_ascii=False,indent=2))+'</pre></details>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- reference-resolution:start -->';end='<!-- reference-resolution:end -->'
        block=start+'<section id="reference-resolution">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if errors:raise RuntimeError('Incomplete diagnostic, see summary')


if __name__=='__main__':main()
