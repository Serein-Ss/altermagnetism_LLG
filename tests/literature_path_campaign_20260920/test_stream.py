from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import pytest
import torch
from . import long


def test_complete_frames_resume_and_overwrite_guard(tmp_path):
    checkpoint = tmp_path/'checkpoint'; checkpoint.write_bytes(b'test checkpoint identity')
    initial = torch.zeros(2,5,3); initial[...,2] = 1
    def sampler(s, seed):
        return torch.stack((s,s.roll(1,-1)),1)
    setup_result = (None,initial,sampler,3,1,tmp_path/'reference',checkpoint)
    with patch.object(long,'HERE',tmp_path), patch.object(long,'allocation'), patch.object(long,'setup',return_value=setup_result), patch('sys.argv',['long','--device','cuda','--system','bauer']):
        long.main()
        path = tmp_path/'literature/bauer/generated.h5'
        with h5py.File(path,'r+') as h:
            expected = h['spins'][:]
            assert h.attrs['complete'] and h.attrs['completed_blocks']==3
            assert expected.shape==(2,4,5,3)
            np.testing.assert_allclose(h['time_ps'][:],[0,.1,.2,.3])
            h.attrs['complete'] = False; h.attrs['completed_blocks'] = 1
            h['spins'][:,2:] = 9
        long.main()
        with h5py.File(path) as h: np.testing.assert_array_equal(h['spins'][:],expected)
        with pytest.raises(RuntimeError,match='already complete'): long.main()
