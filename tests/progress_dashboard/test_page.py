"""Check real source alignment, offline images, navigation and relative links."""
import base64
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import unittest
from PIL import Image

HERE=Path(__file__).resolve().parent
TESTS=HERE.parent


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.images=[]; self.scripts=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a':self.links.append(a.get('href',''))
        if tag=='img':self.images.append(a)
        if tag=='script':self.scripts.append(a)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text=(TESTS/'index.html').read_text()
        cls.page=Page();cls.page.feed(cls.text)
    def test_links(self):
        self.assertEqual(len(self.page.ids),len(set(self.page.ids)))
        for href in self.page.links:
            if href.startswith('#'):self.assertIn(href[1:],self.page.ids)
            elif not href.startswith(('https://','http://')):self.assertTrue((TESTS/href).exists(),href)
    def test_embedded_images(self):
        self.assertGreaterEqual(len(self.page.images),9)
        for img in self.page.images:
            self.assertTrue(img.get('alt'))
            self.assertTrue(img['src'].startswith(('data:image/png;base64,','data:image/gif;base64,')))
            with Image.open(io.BytesIO(base64.b64decode(img['src'].split(',',1)[1]))) as im:
                self.assertIn(im.format,('PNG','GIF'))
                if im.format=='GIF':self.assertEqual(im.n_frames,int(img.get('data-frames',51)))
                im.verify()
        self.assertFalse(self.page.scripts)
        for text in ('两阶段生成','独立路径噪声','四组分离检验','初态数据合同'):
            self.assertIn(text,self.text)
        for anchor in ('architecture','novelty','requirements','systems'):
            self.assertIn(anchor,self.page.ids)
    def test_source_and_boundaries(self):
        snapshot=json.loads((HERE/'snapshot.json').read_text())
        source=json.loads((TESTS/'bauer_closed_loop_20260919/assets/report.json').read_text())
        self.assertEqual(snapshot['completed_runs'],len(source['runs']))
        self.assertEqual(snapshot['displayed_model'],source['selected_for_display'])
        self.assertFalse(snapshot['scientific_certification'])
        for text in ['不是实时任务监控','不是完整路径分布认证','0.1 ps','初态时间重叠','历史探索归档']:
            self.assertIn(text,self.text)
        for r in source['runs']:
            self.assertIn(r['name'],self.text)
            self.assertIn(f'{r["test"]["score"]:.4f}',self.text)


if __name__=='__main__':unittest.main(verbosity=2)
