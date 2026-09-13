"""Animate fixed, unselected real-reference paths; never alter source trajectories."""
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/research/runs/bauer_first_loop/20260913_v1'
DEST = ROOT/'assets/research/bauer_L25_first_loop/20260913_v1/reference_data'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'Real simulator reference, not model output',
                'selection': 'Initial group 0, path indices 0,1,2,3; no event selection. Initials differ across temperatures.',
                'display': 'Arrows are x-z projections, color is s_z; y component is not displayed.',
                'time': 'tau=tJ/hbar; all 201 saved frames, no interpolation; playback is not physical real time.',
                'fps': 12, 'runs': []}
    for theta in (0.11, 0.13):
        source = SOURCE/f'reference_T{theta:.2f}.h5'
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        with h5py.File(source, 'r') as handle:
            assert handle.attrs['complete']
            spins = handle['spins'][0, :4]
            initial = handle['initial'][0]
            times = np.arange(spins.shape[1])*float(handle.attrs['save_dt'])
        assert spins.shape == (4, 201, 25, 3) and np.isfinite(spins).all()
        assert np.allclose(spins[:, 0], initial, atol=1e-6)
        mz = spins[..., 2].mean(-1)
        fig, axes = plt.subplots(4, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [1.8, 1]})
        fig.subplots_adjust(left=.065, right=.91, bottom=.10, top=.87, hspace=.75, wspace=.24)
        heading = fig.suptitle('', fontsize=13)
        fig.text(.5, .93, 'Same initial state, independent noise | arrows: x-z projection | red: +z, blue: -z', ha='center', fontsize=9)
        fig.text(.5, .025, 'All saved frames shown; no interpolation. Playback speed is illustrative, not physical real time.', ha='center', fontsize=9)
        arrows, lines, points, cursors, labels = [], [], [], [], []
        sites = np.arange(1, 26)
        for row in range(4):
            ax, curve = axes[row]
            ax.axhline(0, color='.8', lw=1)
            ax.scatter(sites, np.zeros(25), s=7, color='black')
            arrows.append(ax.quiver(sites, np.zeros(25), spins[row, 0, :, 0], spins[row, 0, :, 2],
                spins[row, 0, :, 2], cmap='RdBu_r', clim=(-1, 1), angles='xy',
                scale_units='xy', scale=1, pivot='middle', width=.006))
            ax.set(xlim=(0, 26), ylim=(-.75, .75), yticks=[], xlabel='Site (open chain)', ylabel=f'Path {row}')
            curve.axhline(0, color='.7', lw=.6, ls='--')
            line, = curve.plot([], [], color='#0072B2', lw=1.1)
            point, = curve.plot([], [], 'o', color='#D55E00', ms=4)
            cursor = curve.axvline(0, color='.6', lw=.6)
            labels.append(curve.set_title('', fontsize=9))
            curve.set(xlim=(0, times[-1]), ylim=(-1.05, 1.05), xlabel='Reduced time', ylabel='$M_z$')
            lines.append(line); points.append(point); cursors.append(cursor)
        cax = fig.add_axes([.935, .2, .015, .55])
        fig.colorbar(arrows[0], cax=cax, label='$s_z$')

        def update(frame):
            heading.set_text(f'Bauer L=25 REAL reference | theta={theta:.2f} | time={times[frame]:.0f} / {times[-1]:.0f}')
            for row in range(4):
                arrows[row].set_UVC(spins[row, frame, :, 0], spins[row, frame, :, 2], spins[row, frame, :, 2])
                lines[row].set_data(times[:frame+1], mz[row, :frame+1])
                points[row].set_data([times[frame]], [mz[row, frame]])
                cursors[row].set_xdata([times[frame], times[frame]])
                labels[row].set_text(f'Mz={mz[row,frame]:+.3f}; spatial std(z)={spins[row,frame,:,2].std():.3f}')
            return arrows+lines+points+cursors+labels+[heading]

        target = DEST/f'real_trajectories_theta_{theta:.2f}.gif'
        animation = FuncAnimation(fig, update, frames=len(times), interval=1000/12, blit=False)
        animation.save(target, writer=PillowWriter(fps=12), dpi=85)
        update(100)
        fig.savefig(DEST/f'real_trajectories_theta_{theta:.2f}_preview.png', dpi=85)
        plt.close(fig)
        with Image.open(target) as gif:
            assert gif.n_frames == len(times)
            for frame in range(gif.n_frames):
                gif.seek(frame); gif.load()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
        manifest['runs'].append({'theta': theta, 'source': str(source.relative_to(ROOT)),
                                'source_sha256': checksum, 'gif': target.name, 'frames': len(times),
                                'bytes': target.stat().st_size, 'initial_group': 0, 'paths': [0, 1, 2, 3]})
        print('Verified', target.name, '201 frames; source unchanged', flush=True)
    (DEST/'animation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
