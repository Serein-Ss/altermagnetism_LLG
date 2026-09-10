"""Plot sublattice AFMR/order parameters, not the compensated total moment."""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.literature.workflow import analysis_arguments,require_complete


def main():
    args=analysis_arguments()
    if args.output.exists(): raise FileExistsError(args.output)
    with h5py.File(args.input) as h:
        require_complete(h); t=h['time'][:]; a=h['m_a'][:]; b=h['m_b'][:]
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        axes[0].plot(t,np.linalg.norm(a,axis=-1),label='A')
        axes[0].plot(t,np.linalg.norm(b,axis=-1),'--',label='B')
        axes[0].set(xlabel='Reduced time',ylabel='Sublattice magnetization length'); axes[0].legend()
        axes[1].plot(t,a[...,2]); axes[1].set(xlabel='Reduced time',ylabel='Sublattice A: m_z')
        axes[2].plot(t,np.linalg.norm((a-b)/2,axis=-1)); axes[2].set(xlabel='Reduced time',ylabel='Neel vector length')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output,dpi=160); plt.close(fig)


if __name__=='__main__': main()
