"""Filled, side-by-side densities using every existing endpoint CSV observation."""
import base64
import csv
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'bauer_2d_review_20260918/assets'


def main():
    with (SOURCE/'development_endpoints.csv').open() as stream: rows=list(csv.DictReader(stream))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    bins=np.linspace(-1,1,31); centers=(bins[:-1]+bins[1:])/2; width=(bins[1]-bins[0])*.44
    counts=[]
    for i,length in enumerate([16,25,32]):
        for j,time in enumerate([1,10,100]):
            ax=axes[i,j]
            for split,offset,color,hatch in [('train',-.25,'#0072B2',None),('dev',.25,'#D55E00','//')]:
                values=np.array([float(r['mz']) for r in rows if int(r['L'])==length and float(r['time_ps'])==time and r['split']==split])
                density,_=np.histogram(values,bins=bins,density=True)
                assert np.isfinite(values).all() and len(values)>0 and np.isclose(np.sum(density*np.diff(bins)),1.)
                ax.bar(centers+offset*np.diff(bins),density,width=width,color=color,alpha=.8,
                       edgecolor='#333333',linewidth=.45,hatch=hatch,label=f'{split.capitalize()} (n={len(values)})')
                counts.append(dict(L=length,time_ps=time,split=split,n=len(values)))
            ax.set(title=f'L={length}, t={time} ps',xlabel='Mean sz',ylabel='Probability density',xlim=(-1,1),ylim=(0,None))
            ax.legend(fontsize=9)
    fig.suptitle('Prepared-family mixture, not equilibrium occupancy; unchanged bins and observations',fontsize=12)
    dest=HERE/'styled';dest.mkdir(exist_ok=True)
    output=dest/'endpoint_distributions.png';fig.savefig(output,dpi=300,facecolor='white');plt.close(fig)
    old=base64.b64encode((SOURCE/'endpoint_distributions.png').read_bytes()).decode()
    new=base64.b64encode(output.read_bytes()).decode()
    content=(SOURCE/'index.html').read_text()
    assert old in content
    content=content.replace(old,new)
    def rebase(m):
        value=html.unescape(m.group(2));u=urlsplit(value)
        if u.scheme or value.startswith(('#','//')):return m.group(0)
        return 'href='+m.group(1)+html.escape(os.path.relpath(SOURCE/u.path,dest)+(('#'+u.fragment) if u.fragment else ''))+m.group(1)
    content=re.sub(r'''href=(["'])(.*?)\1''',rebase,content)
    (dest/'data_review.html').write_text(content)
    (dest/'histogram_source_counts.json').write_text(json.dumps(counts,indent=2))


if __name__=='__main__':main()
