import torch
from .run import HERE,source,dump,verify
from .model import initialize,SCALES,remote_probe


def main():
    verify();torch.set_num_threads(1);rows=[]
    for seed in (11,22,33):
        ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
        for arm,scale in SCALES.items():
            net,_=initialize(ck,scale,seed)
            rows.append(dict(seed=seed,arm=arm,scale=scale,probe=remote_probe(net.eval())))
    dump(HERE/'audit.json',dict(rows=rows,scope='Synthetic remote perturbation of a single velocity call, not physical nonlocality or a root-cause proof. Local radius4; repeated flow updates can propagate farther.',scientific_certification=False))


if __name__=='__main__':main()
