"""Bundle linked reports, rewriting local links; never copy datasets/checkpoints."""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit, quote
import zipfile
from .build import HERE, TESTS, ROOT

ATTR = re.compile(r'''\b(href|src)\s*=\s*(["'])(.*?)\2''', re.I | re.S)
ALLOWED = {'.html','.png','.gif','.jpg','.jpeg','.css','.js','.json','.csv','.md','.yaml','.yml','.txt','.pdf'}


def main():
    package = HERE/'offline_site'
    package.mkdir(exist_ok=True)
    pending, mapped, excluded, external = [], {}, [], set()
    redirects = {(TESTS/'bauer_closed_loop_20260919/assets/index.html').resolve():HERE/'styled/index.html'}
    def register(path):
        path=redirects.get(path.resolve(),path.resolve())
        if path not in mapped:
            if len(mapped)>600:raise RuntimeError('Unexpectedly broad link graph; inspect before packaging')
            name='index.html' if path==TESTS/'index.html' else 'pages/'+hashlib.sha256(str(path).encode()).hexdigest()[:20]+('.html' if path.is_dir() or path.suffix in {'.md','.yaml','.yml','.txt'} or path.suffix not in ALLOWED or not path.exists() else path.suffix)
            mapped[path]=package/name;pending.append(path)
        return mapped[path]
    register(TESTS/'index.html')
    def shell(title,body):
        return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{max-width:1100px;margin:35px auto;padding:20px;font:17px/1.7 sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#075985}td{padding:7px;border-bottom:1px solid #ddd}</style><p><a href="../index.html">返回测试主网页</a></p><h1>'+html.escape(title)+'</h1>'+body+'</html>'
    while pending:
        path=pending.pop(0);dest=mapped[path];dest.parent.mkdir(parents=True,exist_ok=True)
        safe=path.is_relative_to(ROOT) and not any(p.startswith('.') for p in path.relative_to(ROOT).parts)
        # The main report embeds real GIFs; its legitimate size can exceed 64 MiB.
        limit = (256 if path.suffix=='.html' else 64)*1024**2
        allowed=safe and path.exists() and (path.is_dir() or (path.suffix in ALLOWED and path.stat().st_size<=limit))
        if path==TESTS/'index.html' and not allowed:
            raise RuntimeError('Main report cannot be replaced by an omitted-file placeholder')
        if not allowed:
            excluded.append(str(path))
            dest.write_text(shell('此文件未随离线报告打包','<p>原始大数据、checkpoint、不支持的文件或缺失文件不包含在报告包中。该入口是说明页，不是下载失败。</p><pre>'+html.escape(str(path.relative_to(ROOT)) if safe else '外部路径未复制')+'</pre>'))
        elif path.is_dir():
            items=[]
            for p in sorted(path.iterdir()):
                if p.name.startswith('.') or p.name in {'__pycache__','browser_profile'}:continue
                if p.is_file() and p.suffix in {'.json','.csv','.md','.yaml','.yml','.txt'} and p.stat().st_size<8*1024**2:
                    target=register(p);label=f'<a href="{html.escape(os.path.relpath(target,dest.parent))}">{html.escape(p.name)}</a>'
                elif p.is_dir() and path.name=='runs':
                    target=register(p);label=f'<a href="{html.escape(os.path.relpath(target,dest.parent))}">{html.escape(p.name)}</a>'
                else:label=html.escape(p.name)+'（仅目录记录，未打包）'
                items.append('<li>'+label+'</li>')
            dest.write_text(shell(path.name,'<p>轻量报告目录：只复制小型统计与配置；不包含原始轨迹、模型权重或整个历史目录。</p><ul>'+''.join(items)+'</ul>'))
        elif path.suffix=='.html':
            content=path.read_text()
            def rewrite(match):
                attr, mark, raw=match.groups();value=html.unescape(raw)
                if value.startswith(('data:','#','mailto:','javascript:')) or not value:return match.group(0)
                url=urlsplit(value)
                if url.scheme in {'http','https'} or value.startswith('//'):
                    external.add(value);return match.group(0)
                if url.scheme and url.scheme!='file':return match.group(0)
                target=Path(unquote(url.path)) if url.scheme=='file' else path.parent/unquote(url.path)
                target=register(target)
                href=quote(os.path.relpath(target,dest.parent),safe='/._-')
                if url.fragment:href+='#'+url.fragment
                return f'{attr}={mark}{html.escape(href,quote=True)}{mark}'
            content=ATTR.sub(rewrite,content)
            back=os.path.relpath(package/'index.html',dest.parent)
            banner=f'<div style="padding:12px;background:#fff5df;color:#333">离线报告包：<a href="{back}">返回主网页</a>。站内详情已打包；论文/外部站点仍需联网。数据集与checkpoint未包含。</div>'
            content=re.sub(r'(<body\b[^>]*>)',lambda m:m.group(0)+banner,content,count=1,flags=re.I) if re.search(r'<body\b',content,re.I) else banner+content
            dest.write_text(content)
        elif path.suffix in {'.md','.yaml','.yml','.txt'}:
            dest.write_text(shell(path.name,'<pre>'+html.escape(path.read_text())+'</pre>'))
        else:shutil.copyfile(path,dest)
    manifest=dict(files={str(p.relative_to(d)):str(q.relative_to(package)) for p,q in mapped.items() for d in [ROOT] if p.is_relative_to(ROOT)},
                  omitted=excluded,external_links=sorted(external),raw_data_included=False)
    (package/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Check all local HTML links in the exact delivered set, not stale build files.
    for dest in mapped.values():
        if dest.suffix!='.html':continue
        for _,_,value in ATTR.findall(dest.read_text()):
            value=html.unescape(value);url=urlsplit(value)
            if url.scheme or value.startswith(('#','//')) or not value:continue
            target=(dest.parent/unquote(url.path)).resolve()
            assert target.is_relative_to(package.resolve()) and target.exists(),(dest,value)
    archive=HERE/'research_progress_offline.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for dest in sorted(set(mapped.values())|{package/'manifest.json'}):
            z.write(dest,arcname='research_progress/'+str(dest.relative_to(package)))
    print(json.dumps(dict(zip=str(archive),files=len(mapped),bytes=archive.stat().st_size,external_links=len(external),omitted=len(excluded)),ensure_ascii=False))


if __name__=='__main__':main()
