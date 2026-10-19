"""Read existing 1 ns data, diagnose and visualize without rerunning dynamics."""
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'miltat_1ns_20260919/results'
OUT=HERE/'miltat'
A=.251


def triangular_graph(nrows,ncols):
    ids=np.arange(nrows*ncols).reshape(nrows,ncols)
    src=[ids[:,:-1].ravel()];dst=[ids[:,1:].ravel()]
    for r in range(nrows-1):
        src.append(ids[r]);dst.append(ids[r+1])
        if r%2==0:
            src.append(ids[r,1:]);dst.append(ids[r+1,:-1])
        else:
            src.append(ids[r,:-1]);dst.append(ids[r+1,1:])
    a,b=np.concatenate(src),np.concatenate(dst)
    return coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(nrows*ncols,)*2).tocsr()


def raster(s):
    return s.transpose(1,2,0,3).reshape(s.shape[1]*2,s.shape[0],3)


def core_measure(s,pos,graph):
    ids=np.flatnonzero(s[:,2]>=.5)
    if not len(ids):return np.array([np.nan,np.nan]),0,0,0.
    count,labels=connected_components(graph[ids][:,ids],directed=False)
    sizes=np.bincount(labels);chosen=ids[labels==sizes.argmax()]
    return pos[chosen,:2].mean(0),len(chosen),int(count),len(chosen)/len(ids)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.frameon':False})
    data={};frames={};norms={};energy={};reports={}
    animation_ids=np.arange(0,1001,20)
    for phase in ['preparation','thermal']:
        with h5py.File(SOURCE/f'{phase}.h5') as h:
            assert h.attrs['complete'] and h.attrs['frames_written']==len(h['spins'])
            pos=raster(h['positions_nm'][:]).reshape(-1,3)
            nx,ny=h['spins'].shape[2:4];graph=triangular_graph(2*ny,nx)
            energy[phase]=h['energy_reduced'][:]/len(pos)
            norms[phase]=h['norm_error_float64'][:]
            times=h['time_ps'][:]
            result=[];maximum=0.
            for j,t in enumerate(times):
                s=raster(h['spins'][j,0]).reshape(-1,3).astype('f8')
                assert np.isfinite(s).all()
                maximum=max(maximum,float(abs(np.linalg.norm(s,axis=-1)-1).max()))
                center,ncore,ncomponents,fraction=core_measure(s,pos,graph)
                radius=np.sqrt(ncore*(np.sqrt(3)/2)*A*A/np.pi)
                result.append([t,*center,ncore,radius,ncomponents,fraction,s[:,2].mean()])
                if phase=='thermal' and j in animation_ids:frames[j]=s.astype('f4')
                if phase=='preparation' and j in [0,50]:frames[f'prep{j}']=s.astype('f4')
            data[phase]=np.asarray(result)
            reports[phase]=dict(frames=len(times),time_ps=[float(times[0]),float(times[-1])],
                                max_stored_norm_error=maximum,max_integrator_saved_norm_error=float(norms[phase].max()),
                                zero_core_frames=int(np.sum(data[phase][:,3]==0)),
                                min_largest_core_fraction=float(data[phase][:,6].min()))
            with (OUT/f'{phase}_diagnostics.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['time_ps','core_x_nm','core_y_nm','core_sites','equivalent_core_radius_nm','components','largest_fraction','mean_sz'])
                w.writerows(result)
    tr=data['thermal'];centers=tr[:,1:3];valid=np.isfinite(centers).all(1)
    assert valid.any(),'Cannot define a visual core region: no core detected'
    low=centers[valid].min(0)-4;high=centers[valid].max(0)+4
    crop=((pos[:,:2]>=low)&(pos[:,:2]<=high)).all(1)
    rc=np.indices((2*ny,nx)).reshape(2,-1).T
    arrows=crop&(rc[:,0]%4==0)&(rc[:,1]%4==0)
    reports['thermal'].update(initial_core_center_nm=centers[0].tolist(),final_core_center_nm=centers[-1].tolist(),
                              net_displacement_nm=float(np.linalg.norm(centers[-1]-centers[0])),
                              min_core_radius_nm=float(tr[:,4].min()),max_core_radius_nm=float(tr[:,4].max()))
    edge=np.minimum(centers-pos[:,:2].min(0),pos[:,:2].max(0)-centers).min(1)-tr[:,4]
    reports['thermal']['min_radius_based_edge_clearance_nm']=float(np.nanmin(edge))
    # Static full field + fixed zoom: no recentering or spatial smoothing.
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(key,label) in zip(axes.flat,[(0,'Thermal 0 ps: full field'),(1000,'Thermal 1000 ps: full field'),('prep0','Perturbed T=0 start: full field'),(0,'Thermal 0 ps: fixed zoom'),(500 if 500 in frames else 480,'Thermal 500 ps: fixed zoom'),(1000,'Thermal 1000 ps: fixed zoom')]):
        s=frames[key]; zoom='zoom' in label; mask=crop if zoom else np.ones(len(pos),bool)
        im=ax.scatter(pos[mask,0],pos[mask,1],c=s[mask,2],s=6 if zoom else 1.3,marker='s',cmap='RdBu_r',vmin=-1,vmax=1,linewidths=0)
        ax.set(title=label,xlabel='x (nm)',ylabel='y (nm)',aspect='equal')
        if zoom:ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
        fig.colorbar(im,ax=ax,label='sz')
    fig.savefig(OUT/'textures.png',dpi=300,facecolor='white');plt.close(fig)
    # Every frame contributes to diagnostics; GIF below is only temporally subsampled.
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    t=tr[:,0]/1000
    im=axes[0,0].scatter(centers[:,0],centers[:,1],c=t,s=5,cmap='viridis')
    axes[0,0].plot(centers[:,0],centers[:,1],color='#666666',lw=.4,alpha=.5)
    axes[0,0].set(title='Largest contiguous core center',xlabel='x (nm)',ylabel='y (nm)',aspect='equal');fig.colorbar(im,ax=axes[0,0],label='Time (ns)')
    for k,label in enumerate(['x','y']):axes[0,1].plot(t,centers[:,k]-centers[0,k],label=label,ls='-' if k==0 else '--')
    axes[0,1].set(title='Displacement from start',xlabel='Time (ns)',ylabel='Displacement (nm)');axes[0,1].legend()
    axes[0,2].plot(t,tr[:,4]);axes[0,2].set(title='Core area-equivalent radius (sz >= 0.5)',xlabel='Time (ns)',ylabel='Radius (nm)')
    axes[1,0].plot(t,energy['thermal']);axes[1,0].set(title='Thermal energy per spin',xlabel='Time (ns)',ylabel='E / (N J)')
    axes[1,1].plot(t,norms['thermal'],label='Float64 integrator, saved frames');axes[1,1].set(title='Unit norm error: NOT energy conservation',xlabel='Time (ns)',ylabel='max |norm(s) - 1|');axes[1,1].legend(fontsize=8)
    msd=[]
    for lag in range(1,251):
        delta=centers[lag:]-centers[:-lag];keep=np.isfinite(delta).all(1)
        msd.append(np.mean((delta[keep]**2).sum(1)) if keep.any() else np.nan)
    axes[1,2].plot(np.arange(1,251)/1000,msd);axes[1,2].set(title='Single-path time-averaged MSD (no fit)',xlabel='Lag (ns)',ylabel='MSD (nm²)')
    fig.savefig(OUT/'diagnostics.png',dpi=300,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    pr=data['preparation'];axes[0].plot(pr[:,0],energy['preparation']);axes[0].set(xlabel='Time (ps)',ylabel='E / (N J)',title='Zero-T preparation energy')
    axes[1].plot(pr[:,0],pr[:,4]);axes[1].set(xlabel='Time (ps)',ylabel='Core radius (nm)',title='Preparation core size')
    axes[2].plot(pr[:,0],norms['preparation']);axes[2].set(xlabel='Time (ps)',ylabel='max |norm(s) - 1|',title='Saved-step unit norm error')
    fig.savefig(OUT/'preparation.png',dpi=300,facecolor='white');plt.close(fig)
    # Large left 3D view and three scalar component maps, sharing a fixed physical ROI.
    fig=plt.figure(figsize=(14,8),layout='constrained');gs=fig.add_gridspec(3,2,width_ratios=[1.4,1.])
    view=fig.add_subplot(gs[:,0],projection='3d');maps=[fig.add_subplot(gs[k,1]) for k in range(3)]
    view.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(-1,1),xlabel='x (nm)',ylabel='y (nm)',zlabel='Spin arrow height (display only)')
    view.set_box_aspect((high[0]-low[0],high[1]-low[1],5));view.view_init(elev=43,azim=-64)
    artists=[]
    for k,ax in enumerate(maps):
        im=ax.scatter(pos[crop,0],pos[crop,1],c=frames[0][crop,k],s=9,marker='s',linewidths=0,cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x (nm)',ylabel='y (nm)',aspect='equal',title=f's{"xyz"[k]} on the physical film')
        fig.colorbar(im,ax=ax,label=f's{"xyz"[k]}');artists.append(im)
    quiver=[None]
    def update(index):
        s=frames[int(index)]
        if quiver[0] is not None:quiver[0].remove()
        c=plt.get_cmap('RdBu_r')((s[arrows,2]+1)/2)
        quiver[0]=view.quiver(pos[arrows,0],pos[arrows,1],np.zeros(arrows.sum()),s[arrows,0],s[arrows,1],s[arrows,2],length=.65,normalize=False,colors=c,linewidth=.65)
        view.set_title(f'12.5 K LLG | t={index:g} ps\nFixed ROI; arrows every 4 rows/columns')
        for k,artist in enumerate(artists):artist.set_array(s[crop,k])
        return artists+[quiver[0]]
    update(0);fig.savefig(OUT/'spin_view.png',dpi=300,facecolor='white')
    movie=FuncAnimation(fig,update,frames=animation_ids,interval=125,blit=False)
    movie.save(OUT/'thermal.gif',writer=PillowWriter(fps=8),dpi=80);plt.close(fig)
    with (OUT/'msd.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['lag_ps','single_path_time_averaged_msd_nm2']);w.writerows(zip(range(1,251),msd))
    reports.update(scope='Existing single trajectory visualization, not diffusion or topology certification',
                   core_definition='Largest six-neighbor triangular connected component with sz>=0.5; ties use first component; no smoothing',
                   radius_definition='sqrt(Ncore * sqrt(3)/2 * a² / pi); threshold area radius, not fitted wall radius',
                   motion_warning='Largest-component tracking may fail on splitting; no topological-charge or diffusion-coefficient claim',
                   gif=dict(frames=len(animation_ids),source_stride_ps=20,fps=8,spatial_maps='all sites in fixed ROI at true positions',arrow_stride=4,all_statistics_frames=1001),
                   source_files={name:dict(bytes=(SOURCE/name).stat().st_size) for name in ['thermal.h5','preparation.h5']},
                   parent_report_sha256=hashlib.sha256((SOURCE/'report.json').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
