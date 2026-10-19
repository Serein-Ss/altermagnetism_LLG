"""Validate the delivered offline artifact and reported counts."""
import base64
import csv
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import unittest
from PIL import Image

OUT=Path(__file__).resolve().parent/'assets'


class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.images=[];self.links=[];self.ids=[]

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='img':self.images.append(a)
        if tag=='a':self.links.append(a.get('href',''))


class ReviewTests(unittest.TestCase):
    def test_offline_images_and_navigation(self):
        page=Page();page.feed((OUT/'index.html').read_text())
        self.assertGreaterEqual(len(page.images),35)
        for image in page.images:
            self.assertTrue(image['alt'])
            self.assertTrue(image['src'].startswith('data:image/png;base64,'))
            with Image.open(BytesIO(base64.b64decode(image['src'].split(',',1)[1]))) as im:
                self.assertEqual(im.format,'PNG');im.verify()
        for link in page.links:
            if link.startswith('#'):self.assertIn(link[1:],page.ids)

    def test_inventory_and_independent_holdout(self):
        summary=json.loads((OUT/'review_summary.json').read_text())
        self.assertEqual(summary['file_count'],252)
        self.assertEqual(sum(row[3]+row[4] for row in summary['inventory']),2880)
        self.assertEqual(sum(row[2] for row in summary['inventory']),216)
        with (OUT/'development_endpoints.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual({r['split'] for r in rows},{'train','dev'})
        self.assertEqual(len(rows),(384+192)*3*3)

    def test_complete_figure_inventory(self):
        text=(OUT/'index.html').read_text()
        for number in range(1,15):self.assertIn(f'图{number}｜',text)


if __name__=='__main__':unittest.main()
