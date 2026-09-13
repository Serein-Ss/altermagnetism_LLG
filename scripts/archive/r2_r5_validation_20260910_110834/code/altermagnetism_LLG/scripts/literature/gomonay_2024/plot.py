"""Plot signed-frequency BZ spectra; trajectory input gives raw diagnostics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.literature.workflow import analysis_arguments,plot_trajectory


def main():
    args=analysis_arguments()
    if args.input.suffix!='.npz':
        return plot_trajectory(args.input,args.output)
    with np.load(args.input) as data:
        omega=data['omega']; order=np.argsort(omega)
        power=data['power'].mean(1).sum(-1)
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
        img=ax.imshow(np.log10(np.maximum(power[order],1e-30)),origin='lower',aspect='auto',
                      extent=(-.5,power.shape[1]-.5,omega[order][0],omega[order][-1]),cmap='viridis')
        ax.set(xlabel='BZ sample index (kx,ky in NPZ)',ylabel='Signed reduced angular frequency')
        fig.colorbar(img,ax=ax,label='log10 spectral power')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists(): raise FileExistsError(args.output)
        fig.savefig(args.output,dpi=160); plt.close(fig)


if __name__=='__main__': main()
