"""Bounded-memory raw trajectory animation for all reduced literature models."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from scripts.literature.workflow import require_complete


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trajectory',type=int,default=0); parser.add_argument('--max-frames',type=int,default=100)
    parser.add_argument('--fps',type=int,default=10); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.max_frames<2 or args.fps<1: raise ValueError('positive fps and at least two frames required')
    with h5py.File(args.input) as h:
        require_complete(h); ds=h['spins']
        if not 0<=args.trajectory<ds.shape[1]: raise ValueError('trajectory index out of range')
        ids=np.unique(np.linspace(0,len(ds)-1,min(args.max_frames,len(ds)),dtype=int))
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
        first=ds[0,args.trajectory].reshape(-1,3)
        lines=[]
        for c,ax in enumerate(axes):
            line,=ax.plot(first[:,c],lw=.8); lines.append(line)
            ax.set(xlabel='Flattened site index (basis retained)',ylabel=f's_{"xyz"[c]}',ylim=(-1.05,1.05))
        title=fig.suptitle('')
        def update(i):
            flat=ds[i,args.trajectory].reshape(-1,3)
            for c,line in enumerate(lines): line.set_ydata(flat[:,c])
            title.set_text(f'Trajectory {args.trajectory}, reduced time {h["time"][i]:.5g}')
            return *lines,title
        animation=FuncAnimation(fig,update,frames=ids,blit=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        animation.save(args.output,writer=PillowWriter(fps=args.fps)); plt.close(fig)


if __name__=='__main__': main()
