import json
import numpy as np
from PIL import Image
from tests.experiments.reference_gif_gallery import ROOT, WEB, frame, render, selected_indices


def test_render_preserves_frames_and_raw_values(tmp_path):
    spins=np.zeros((3,25,3)); spins[:,:,2]=1
    times=np.array([0.,.01,.02]); original=spins.copy()
    p=tmp_path/'sample.gif'
    render(p,spins,times,'synthetic test')
    np.testing.assert_array_equal(spins,original)
    with Image.open(p) as im:
        assert im.n_frames==3 and im.size==(720,570)
        for k in range(3):
            im.seek(k)
            assert im.info['duration']==100
    assert frame(spins,times,2,'synthetic test').size==(720,570)


def test_catalog_covers_every_reference_without_duplicates():
    c=json.loads((WEB/'catalog.json').read_text())
    groups=c['groups']
    assert {(g['case'],g['anchor']) for g in groups}=={(a,b) for a in range(3) for b in range(24)}
    assert len(groups)==72 and sum(len(g['seeds']) for g in groups)==36864
    assert all((ROOT/g['source'].split('/')[-1]).is_file() for g in groups)
    html=(ROOT/'index.html').read_text()
    assert '__CATALOG__' not in html and 'fetch(' not in html
    selected=selected_indices()
    assert selected==selected_indices() and sum(map(len,selected))==369
    assert all(len(set(v))==len(v) and all(0<=i<512 for i in v) for v in selected)
    assert [g['selected_indices'] for g in groups]==selected
