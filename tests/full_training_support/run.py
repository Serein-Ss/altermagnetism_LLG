import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import h5py
import numpy as np
from tests.bauer_p2_20260923.common import cfg, dump
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.branch_coverage_repair.run import parents
from .search import ExactSearch, eligible_frames, patch_batch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
PREVIOUS=ROOT/'tests/local_support_diagnosis/assets/summary.json'


def queries():
    rows=[];states=[]
    with h5py.File(ROOT/'tests/deployment_moment_repair/train/00/intermediate.h5') as h:
        assert h.attrs['complete']
        for name, g in h.items():
            meta=json.loads(g.attrs['metadata'])
            assert meta['origin']=='llg' and meta['parent']['split']=='dev'
            rows.append(meta);states.append(g['spins'][0,0])
    assert len(rows)==72 and len({r['parent']['parent_id'] for r in rows})==18
    return rows, states


def native_inventory():
    config=cfg()
    records=[r for size in config['sizes'] for r in PathWindows([ROOT/config['data']/f'L{size}'], 'train', 1., stride_frames=10).records]
    assert len(records)==144
    return records


def prepare():
    assert not (HERE/'source_lock.json').exists()
    records=native_inventory();rows,_=queries()
    training={r[1]['parent_id'] for r in records}
    assert not training & {r['parent']['parent_id'] for r in rows}
    files=[p for p,_,_,_ in records]+[PREVIOUS,ROOT/'tests/deployment_moment_repair/train/00/intermediate.h5']
    files+=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))
    for directory in ('training_restart_repair','model_state_refresh_repair'):
        files += [ROOT/f'tests/{directory}/manifest.json',ROOT/f'tests/{directory}/states.h5']
    files += [ROOT/f'tests/branch_coverage_repair/reference/{i:02d}/train.h5' for i in range(24)]
    dump(HERE/'inventory.json',dict(native_parents=144,eligible_states=sum(row['branches']*len(eligible_frames(starts,steps,row['frames'])) for _,row,starts,steps in records),
         queries=72,query_sources=18,extra_branch_states=24*64*10,extra_snapshot_states=96,
         native_sizes=[16,25,32],reference_lag_ps=.1,
         scope='All currently eligible states, not an exposure trace of all historical optimizers. Native weights remain unequal; nearest distance is not probability or physical certification.'))
    files+=[HERE/'inventory.json']
    dump(HERE/'source_lock.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def scan(radius,smoke=False):
    verify();tick=time.perf_counter();meta,states=queries()
    if smoke:meta,states=meta[:4],states[:4]
    q=[];m=[];query_rows=[]
    for row,state in zip(meta,states):
        p,mask=patch_batch(state[None],radius);q.append(p);m.append(mask)
        query_rows += [dict(state=row['index'],parent=row['parent']['parent_id'],family=row['parent']['family'],L=len(state),site=i) for i in range(len(state))]
    search=ExactSearch(np.concatenate(q),np.concatenate(m));counter=0;patches_count=0

    def add(values,info,branch,frames):
        nonlocal counter,patches_count
        # Each input chunk includes all sites, never a random site subset.
        for start in range(0,len(values),128):
            x=values[start:start+128]
            assert np.isfinite(x).all() and np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6
            p,mask=patch_batch(x,radius);size=x.shape[1]
            def location(index):
                state,site=divmod(index,size);frame=int(frames[start+state])
                return dict(info,branch=branch,frame=frame,site=site,time_ps=float(frame*.1))
            search.update(p,mask,location);counter+=len(x);patches_count+=len(p)

    records=native_inventory()
    for path,row,starts,steps in (records[:1] if smoke else records):
        assert row['split']=='train'
        frames=eligible_frames(starts,steps,row['frames'])
        with h5py.File(path) as h:
            assert h.attrs['complete'] and h.attrs['arm']=='base'
            assert json.loads(h.attrs['parent_json'])['parent_id']==row['parent_id']
            for branch in range(row['branches']):
                add(h['spins'][branch,frames],dict(bank='native',parent=row['parent_id'],path=str(path.relative_to(ROOT))),branch,frames)
    native=search.distances().copy();native_locations=list(search.locations)
    if not smoke:
        for i,row in enumerate(parents()):
            path=ROOT/f'tests/branch_coverage_repair/reference/{i:02d}/train.h5'
            with h5py.File(path) as h:
                assert h.attrs['complete'] and h.attrs['role']=='train' and h.attrs['split']=='train'
                assert json.loads(h.attrs['parent'])==row
                frames=np.arange(10)
                for branch in range(64):
                    add(h['spins'][branch,:10],dict(bank='expanded_branches',parent=row['parent_id'],path=str(path.relative_to(ROOT))),branch,frames)
        for directory in ('training_restart_repair','model_state_refresh_repair'):
            base=ROOT/'tests'/directory;rows=json.loads((base/'manifest.json').read_text())
            with h5py.File(base/'states.h5') as h:
                assert h.attrs['complete']
                for row in rows:
                    assert row['split']==row['parent']['split']=='train'
                    add(h[f"state{row['index']:03d}"][:][None],dict(bank=directory,parent=row['parent']['parent_id'],path=str((base/'states.h5').relative_to(ROOT)),key=f"state{row['index']:03d}"),row['branch'],np.array([row['frame']]))
    distance=search.distances()
    assert np.all(distance<=native+1e-12)
    out=HERE/('smoke' if smoke else 'scan');out.mkdir(exist_ok=True)
    dump(out/f'r{radius}.json',dict(radius=radius,smoke=smoke,seconds=time.perf_counter()-tick,states_scanned=counter,patches_scanned=patches_count,
         rows=[dict(row,native_distance=float(n),full_distance=float(f),native_nearest=a,full_nearest=b) for row,n,f,a,b in zip(query_rows,native,distance,native_locations,search.locations)],
         source_lock_sha256=hashlib.sha256((HERE/'source_lock.json').read_bytes()).hexdigest(),complete=True,scientific_certification=False))
    print(json.dumps(dict(radius=radius,smoke=smoke,seconds=time.perf_counter()-tick,states=counter,patches=patches_count)),flush=True)


def aggregate():
    verify();inventory=json.loads((HERE/'inventory.json').read_text());old=json.loads(PREVIOUS.read_text())
    reports=[json.loads((HERE/f'scan/r{r}.json').read_text()) for r in (1,2,4)]
    assert all(r['complete'] and not r['smoke'] and r['states_scanned']==inventory['eligible_states']+inventory['extra_branch_states']+96 for r in reports)
    rows=[]
    for report in reports:
        for row in report['rows']:
            matching=[x for x in old['sites'] if x['state']==row['state'] and x['site']==row['site'] and x['arm']=='control']
            assert len(matching)==3
            rows.append(dict(row,radius=report['radius'],mean_rmse_across_seeds=float(np.mean([x['site_rmse'] for x in matching])),extra_snapshot_distance=matching[0][f"patch_r{report['radius']}_distance"]))
    states=[]
    for radius in (1,2,4):
        for state in sorted({x['state'] for x in rows}):
            subset=[x for x in rows if x['radius']==radius and x['state']==state]
            states.append(dict(state=state,radius=radius,family=subset[0]['family'],L=subset[0]['L'],
                              **{k:float(np.mean([x[k] for x in subset])) for k in ('native_distance','full_distance','extra_snapshot_distance','mean_rmse_across_seeds')}))
    assets=HERE/'assets';assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(inventory=inventory,states=states,sites=rows,scan_seconds=[r['seconds'] for r in reports],scientific_certification=False))
    body='<h1>完整可采样训练状态的局域覆盖诊断</h1><p>本轮修正上一轮只检查96增强快照的覆盖范围。扫描所有144个TRAIN父来源的native窗口、24来源各64条扩充分支和96增强快照；不是把格点/窗口当独立样本。72个开发快照来自18个父来源，不是72个独立准备。</p><p>三个半径精确搜索，开放边缘掩码一致，整块绕登记z轴对齐，不镜像、不逐自旋对齐。最近邻构型类似不证明其条件转移相同，距离不作为验收阈值。当前native训练尺寸包含16/25/32，不能把L25/32称作未见尺寸迁移。</p>'
    body+='<p>覆盖统计表示当前可被抽中的状态，不表示历史每个状态实际被优化器看过，也不表示训练权重或自然占据概率。0.1ps物理lag、原LLG积分步与flow步数均未改变。所有生成误差复用三seed FM控制组；开发集仍是开发，不是最终测试。</p>'
    body+='<table border="1" cellpadding="7"><tr><th>半径</th><th>native平均最近距离</th><th>完整库平均最近距离</th><th>旧快照平均最近距离</th><th>状态距离—误差描述性相关</th></tr>'
    for radius in (1,2,4):
        subset=[x for x in states if x['radius']==radius]
        correlation=np.corrcoef([x['full_distance'] for x in subset],[x['mean_rmse_across_seeds'] for x in subset])[0,1]
        body+=f"<tr><td>{radius}</td>"+''.join(f'<td>{np.mean([x[k] for x in subset]):.6f}</td>' for k in ('native_distance','full_distance','extra_snapshot_distance'))+f'<td>{correlation:.3f}</td></tr>'
    body+='</table><p>相关系数只有描述性含义，无伪独立p值。完整库更近但模型仍有偏差时，不能继续以旧快照距离证明需要补数据；相似局域patch也不能排除全构型上下文影响。最近邻来源和逐格点数据保存在JSON，下一步改动必须依据结果，而不是自动增加预算。</p><a href="summary.json">完整数值和最近邻来源</a> · <a href="../../local_support_diagnosis/assets/index.html">上一轮误差分解</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- full-training-support:start -->';end='<!-- full-training-support:end -->'
    block=start+'<section id="full-training-support"><h2>完整训练状态局域覆盖诊断</h2><a href="full_training_support/assets/index.html">完整native/扩充分支/增强快照对照</a></section>'+end
    txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
    tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','scan','aggregate']);parser.add_argument('--radius',type=int,choices=[1,2,4],default=2);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='scan':scan(args.radius,args.smoke)
    else:aggregate()
