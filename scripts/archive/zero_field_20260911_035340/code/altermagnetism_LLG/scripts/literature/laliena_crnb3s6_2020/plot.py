"""Corrected BVP profile or continuation plot; no old result is overwritten."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.literature.workflow import analysis_arguments,plot_trajectory


def main():
    args=analysis_arguments()
    if args.input.suffix!='.npz': return plot_trajectory(args.input,args.output)
    with np.load(args.input) as data:
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        if 'profiles' in data:
            ax.plot(data['gamma'],data['profiles'][:,0,0],'.-')
            ax.set(xlabel='Gamma',ylabel='Central polar angle (rad)',title='Pseudo-arclength branch (uncertified)')
        else:
            for i,label in enumerate(('n_x','n_y','n_z')): ax.plot(data['x'],data['spins'][:,i],label=label)
            ax.set(xlabel='Reduced position q0 z',ylabel='Spin component',title=f"Corrected BVP, Gamma={float(data['gamma']):.3f}")
            ax.legend()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists(): raise FileExistsError(args.output)
        fig.savefig(args.output,dpi=160); plt.close(fig)


if __name__=='__main__': main()
