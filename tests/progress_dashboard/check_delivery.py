"""Validate the delivered ZIP and render a Python-only visual QA contact sheet."""
import io
import json
import posixpath
from pathlib import Path
import zipfile
from urllib.parse import urlsplit,unquote
import html
from PIL import Image,ImageOps,ImageDraw
from .offline import ATTR

HERE=Path(__file__).resolve().parent


def main():
    original=json.loads((HERE.parent/'bauer_closed_loop_20260919/assets/report.json').read_text())
    styled=json.loads((HERE/'styled/report.json').read_text())
    assert original==styled,'Style-only render must not change numerical report'
    archive=HERE/'research_progress_offline.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=set(z.namelist());checked=0
        assert not any(n.endswith(('.h5','.pt','.npz')) for n in names)
        for name in names:
            if not name.endswith('.html'):continue
            for _,_,raw in ATTR.findall(z.read(name).decode()):
                value=html.unescape(raw);u=urlsplit(value)
                if u.scheme or value.startswith(('#','//')) or not value:continue
                target=posixpath.normpath(posixpath.join(posixpath.dirname(name),unquote(u.path)))
                assert target in names,(name,target)
                checked+=1
    files=['metrics.png','paths_L16.png','paths_L25.png','paths_L32.png','endpoint_distributions.png','sampling_steps.png']
    sheet=Image.new('RGB',(1800,1900),'white');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(files):
        with Image.open(HERE/'styled'/name) as im:
            assert im.format=='PNG' and min(im.info.get('dpi',(0,0)))>=299
            preview=ImageOps.contain(im.convert('RGB'),(880,595))
            x=(i%2)*900;y=(i//2)*630
            draw.text((x+10,y+5),name,fill='black');sheet.paste(preview,(x+10,y+25))
    sheet.save(HERE/'logs/visual_qa.png')
    result=dict(statistics_unchanged=True,zip_crc_pass=True,local_links_checked=checked,
                raw_datasets_or_checkpoints_included=False,png_dpi=300,
                vector_pdf_audit='Not applicable: explicit user PNG-only instruction',
                graphical_scope='Web/report figure preview, not final-size journal submission')
    (HERE/'logs/delivery_qa.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
