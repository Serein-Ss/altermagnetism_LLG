"""Quantify empirical label variability on TRAIN states only."""
from pathlib import Path
import torch
from torch.nn import functional as F
from tests.training_restart_repair.data import load_train
from tests.temporal_source_repair.source import draw
from tests.bauer_p2_20260923.common import dump
from .target import labels

torch.set_num_threads(1);bank,_=load_train();rows=[]
for sid,data in enumerate(bank):
    target=F.normalize(torch.tensor(data),dim=-1);y=target[:16]
    for tv in (0.,.1,.3,.5,.7,.9,.95):
        gen=torch.Generator().manual_seed(955000+sid)
        z=draw(y[:,0],2,gen,0.)
        x,v,rb,d=labels(z,y,torch.full((16,),tv),target,gen)
        rows.append(dict(state=sid,tau=tv,**d,sampled_label_power=float(v[:,1:].square().sum(-1).mean()),
            conditional_label_power=float(rb[:,1:].square().sum(-1).mean())))
dump(Path(__file__).parent/'teacher_audit.json',dict(rows=rows,scope='48TRAINstates from24sourceparents,first16TRAINbranches,descriptive only',physical_certification=False))
