import json
import h5py
import numpy as np
import pytest
from tests.miltat_library_20260920.data import validate_manifest,MiltatWindows


def fixture(root):
    records=[]
    for i,split in enumerate(['train','dev','test']):
        row=dict(parent_id=f'parent{i}',split=split,preparation_group=f'prep{i}',file=f'{split}.h5')
        records.append(row)
        with h5py.File(root/row['file'],'w') as h:
            h.attrs['schema']='miltat_parent_v1';h.attrs['complete']=True
            for k in ['parent_id','split','preparation_group']:h.attrs[k]=row[k]
            h.attrs['physical_parameters_json']=json.dumps(dict(boundary='open',temperature_K=12.5,J_meV=29,
                K_meV=.4,D_meV=-1.5,moment_muB=2.1,a_angstrom=2.51,gamma_per_second_tesla=1.76085963023e11,alpha=.5))
            x=np.zeros((2,11,4,3,2,3),dtype='f4');x[...,2]=1
            h.create_dataset('spins',data=x);h.create_dataset('initial',data=x[0,0])
            xx,yy=np.meshgrid(np.arange(4),np.arange(3),indexing='ij')
            positions=np.stack((xx,yy*np.sqrt(3),np.zeros_like(xx)),-1)[...,None,:]
            positions=(positions+np.array([[0,0,0],[.5,np.sqrt(3)/2,0]]))*.251
            h.create_dataset('positions_nm',data=positions)
            h.create_dataset('time_ps',data=np.arange(11)*.1)
            h.create_dataset('noise_seeds',data=np.array([10+i*2,11+i*2]))
    (root/'manifest.json').write_text(json.dumps(dict(schema='miltat_parent_v1',parents=records)))
    return records


def test_lazy_window_shape_and_physical_time(tmp_path):
    fixture(tmp_path);ds=MiltatWindows(tmp_path,'train',.2)
    assert len(ds)==18 and ds[0]['spins'].shape==(3,4,3,2,3)
    assert ds[1]['origin_ps']==pytest.approx(.1)
    assert ds[-1]['branch']==1 and ds[-1]['split']=='train'
    assert np.allclose(ds[-1]['time_ps'],[0,.1,.2])


@pytest.mark.parametrize('failure',['ancestry','seed','diagnostic','initial','geometry'])
def test_reject_leakage_and_invalid_parent_files(tmp_path,failure):
    rows=fixture(tmp_path)
    with h5py.File(tmp_path/'test.h5','r+') as h:
        if failure=='ancestry':
            rows[2]['preparation_group']=rows[0]['preparation_group']
            h.attrs['preparation_group']=rows[0]['preparation_group']
            (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='miltat_parent_v1',parents=rows)))
        elif failure=='seed':h['noise_seeds'][0]=10
        elif failure=='diagnostic':h.attrs['schema']='coupled_dt_diagnostic'
        elif failure=='geometry':h['positions_nm'][:]=0
        else:h['spins'][1,0,...,2]=-1
    with pytest.raises(ValueError):validate_manifest(tmp_path)
