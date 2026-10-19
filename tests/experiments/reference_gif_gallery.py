"""Offline gallery of every independent numerical reference, without resimulation."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.workflow.bauer_campaign import digest, save

ROOT = Path('tests/data/independent_candidate_reference')
REPORT = Path('tests/results/independent_candidate_reference')
WEB = ROOT / 'trajectory_gallery_1pct'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
COLORS = ['#0072B2', '#B34D00', '#006B50']


def frame(spins, times, k, title):
    """All three components, fixed [-1,1]; no smoothing or temporal subsampling."""
    im = Image.new('RGB', (720, 570), 'white')
    d = ImageDraw.Draw(im)
    d.text((12, 8), title, fill='black', font=FONT)
    d.text((12, 29), f'Reduced time = {times[k]:.2f} | frame {k+1}/{len(times)} | open chain', fill='black', font=FONT)
    for c in range(3):
        top = 77 + c * 106
        d.text((12, top-20), 's'+'xyz'[c]+' by site (single trajectory)', fill=COLORS[c], font=FONT)
        for value in [-1, 0, 1]:
            y = top+35-35*value
            d.line((56, y, 695, y), fill='#dddddd')
            d.text((25, y-8), str(value), fill='black', font=FONT)
        points = [(56+i*639/(spins.shape[1]-1), top+35-35*float(v)) for i,v in enumerate(spins[k,:,c])]
        d.line(points, fill=COLORS[c], width=2)
        for x,y in points:
            d.ellipse((x-2,y-2,x+2,y+2), fill=COLORS[c])
        d.text((56, top+72), '1', fill='black', font=FONT)
        d.text((675, top+72), str(spins.shape[1]), fill='black', font=FONT)
    d.text((12, 385), 'Chain-mean magnetization: x / y / z (same single trajectory)', fill='black', font=FONT)
    means = spins.mean(axis=1)
    for value in [-1,0,1]:
        y=465-48*value
        d.line((56,y,695,y), fill='#dddddd')
        d.text((25,y-8),str(value),fill='black',font=FONT)
    for c in range(3):
        points=[(56+639*float(t-times[0])/(times[-1]-times[0]),465-48*float(v)) for t,v in zip(times[:k+1],means[:k+1,c])]
        if len(points)>1:
            d.line(points,fill=COLORS[c],width=2)
        x,y=points[-1]
        d.ellipse((x-3,y-3,x+3,y+3), fill=COLORS[c])
        d.text((100+c*180,535),f'M{"xyz"[c]} = {means[k,c]:+.4f}',fill=COLORS[c],font=FONT)
    d.text((56,515),f'{times[0]:.2f}',fill='black',font=FONT)
    d.text((630,515),f'{times[-1]:.2f} time',fill='black',font=FONT)
    return im


def render(path, spins, times, title):
    images=[frame(spins,times,k,title) for k in range(len(times))]
    tmp=path.with_suffix('.tmp')
    images[0].save(tmp,format='GIF',save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
    with Image.open(tmp) as check:
        assert check.n_frames==len(times) and check.size==(720,570)
        for k in range(check.n_frames):
            check.seek(k)
            check.load()
    tmp.replace(path)


def selected_indices():
    rng = np.random.default_rng(2026091401)
    extra = set(rng.choice(72, size=9, replace=False).tolist())
    return [sorted(rng.choice(512, size=5+int(g in extra), replace=False).tolist()) for g in range(72)]


def prepare():
    report=json.loads((REPORT/'summary.json').read_text())
    manifest=json.loads((REPORT/'manifest.json').read_text())
    assert report['status']=='COMPLETE_INDEPENDENT_CANDIDATE_REFERENCE_NOT_CERTIFIED'
    groups=[]
    for r in sorted(report['rows'],key=lambda r:(r['case'],r['anchor'])):
        a=manifest['anchors'][r['anchor']]
        groups.append(dict(case=r['case'],anchor=r['anchor'],state=a,source=r['path'],sha256=r['sha256'],
                           seeds=r['numerics']['noise_seeds'],method=r['numerics']['method'],dt=r['numerics']['dt']))
    assert len(groups)==72 and all(len(g['seeds'])==512 for g in groups)
    for g, indices in zip(groups, selected_indices()):
        g['selected_indices'] = indices
    WEB.mkdir(exist_ok=True)
    save(WEB/'catalog.json',dict(groups=groups,source_report_sha256=digest(REPORT/'summary.json'),
        code_sha256=digest(__file__),frames=81,frame_interval=0.01,playback_ms=100,
        sampling_seed=2026091401, selected_count=369,
        scope='Stratified random 369/36864 paths: 5 per group plus 1 for 9 randomly chosen groups. No outcome selection, smoothing or resimulation.'))
    template=Path(__file__).with_name('reference_gif_gallery.html').read_text()
    (ROOT/'index.html').write_text(template.replace('__CATALOG__',json.dumps(groups)),encoding='utf-8')


def run(group, limit):
    if limit is None and not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Full rendering requires CPU Slurm allocation')
    catalog=json.loads((WEB/'catalog.json').read_text())
    assert digest(__file__)==catalog['code_sha256']
    g=catalog['groups'][group]
    assert digest(g['source'])==g['sha256']
    with np.load(g['source']) as z:
        spins=z['spins']; times=z['reduced_time']; seeds=z['seeds']
    assert spins.shape==(512,81,25,3) and np.isfinite(spins).all()
    np.testing.assert_allclose(times,np.arange(81)*.01,atol=1e-12)
    np.testing.assert_array_equal(seeds,g['seeds'])
    np.testing.assert_allclose(np.linalg.norm(spins,axis=-1),1,atol=1e-5,rtol=0)
    dest=WEB/f"case{g['case']}_anchor{g['anchor']}"
    dest.mkdir(exist_ok=True)
    started=time.monotonic(); rows=[]
    indices=g['selected_indices'] if limit is None else g['selected_indices'][:limit]
    for i in indices:
        p=dest/f'trajectory_{i:04d}.gif'
        title=f"case {g['case']} | {g['state']['kind']} | theta {g['state']['theta']} | path {i} | seed {seeds[i]}"
        render(p,spins[i],times,title)
        rows.append(dict(index=i,seed=int(seeds[i]),file=p.name,bytes=p.stat().st_size,sha256=digest(p)))
        if i==indices[0]:
            frame(spins[i],times,80,title).save(dest/'preview.png')
    save(dest/('complete.json' if limit is None else 'preview_manifest.json'),dict(
        status='COMPLETE' if limit is None else 'PREVIEW',source_sha256=g['sha256'],code_sha256=digest(__file__),
        elapsed_seconds=time.monotonic()-started,rows=rows))
    print(json.dumps(dict(group=group,count=len(rows),seconds=time.monotonic()-started,bytes=sum(r['bytes'] for r in rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true')
    p.add_argument('--group',type=int,choices=range(72)); p.add_argument('--limit',type=int,choices=range(1,513))
    args=p.parse_args()
    if args.prepare: prepare()
    else:
        if args.group is None: p.error('--group is required')
        run(args.group,args.limit)
