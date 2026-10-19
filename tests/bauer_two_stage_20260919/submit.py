"""One-shot dependency graph with an after-any report; never poll training jobs."""
import hashlib
import subprocess
from .common import HERE,ROOT,dump


def commands(ids):
    # No artificial array concurrency cap. Slurm enforces resource availability.
    return [
      ('prepare','fat',16,'08:00:00','0-4',[],False),
      ('calibrate','fat',2,'01:00:00',None,['prepare'],False),
      ('smoke','rtx4090',2,'01:00:00',None,['prepare'],False),
      ('train','rtx4090',2,'12:00:00','0-8',['calibrate','smoke'],False),
      ('generate','rtx4090',2,'24:00:00',None,['train'],False),
      ('reference','fat',16,'08:00:00','0-4',['generate'],False),
      ('assess','fat',2,'02:00:00',None,['reference'],False),
      ('long','rtx4090',2,'24:00:00',None,['assess'],False),
      ('report','fat',2,'02:00:00',None,['prepare','calibrate','smoke','train','generate','reference','assess','long'],True)]


def main():
    target=HERE/'submission.json'
    if target.exists():raise RuntimeError('Existing submission record: inspect before resubmitting')
    (HERE/'logs').mkdir(exist_ok=True)
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+[HERE/'job.sh']
    for folder in ['bauer_closed_loop_20260919','bauer_active_data_20260918','bauer_fig2_10ns_20260917','mdgen_path_refactor_20260918']:
        sources+=list((ROOT/'tests'/folder).glob('*.py'))+list((ROOT/'tests'/folder).glob('*.yaml'))
    sources+=[ROOT/'scripts/model/sphere.py',ROOT/'scripts/model/gate_f.py']
    sources += [ROOT/f'tests/bauer_closed_loop_20260919/runs/C_{seed}/best.pt' for seed in [11,22,33]]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(set(sources))))
    ids={}
    for phase,partition,cpus,wall,array,deps,always in commands(ids):
        command=['sbatch','--parsable','--job-name=zrs-mag',f'--partition={partition}',
                 '--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}','--mem=16G',f'--time={wall}',f'--chdir={ROOT}',
                 f'--output={HERE}/logs/{phase}_%A_%a.out',f'--error={HERE}/logs/{phase}_%A_%a.err','--kill-on-invalid-dep=yes']
        if partition=='rtx4090':command+=['--gres=gpu:1']
        if array:command+=[f'--array={array}']
        if deps:command+=['--dependency='+('afterany:' if always else 'afterok:')+':'.join(ids[d] for d in deps)]
        command+=[str(HERE/'job.sh'),phase]
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        ids[phase]=result.stdout.strip().split(';')[0]
        dump(target,dict(jobs=ids,monitor=False,train_array=['initial_11','initial_22','initial_33','uniform_11','uniform_22','uniform_33','anchored_11','anchored_22','anchored_33'],size_array=[16,25,32,50,100],long_generation_requires_predeclared_screens=True))
        print(phase,ids[phase],flush=True)


if __name__=='__main__':main()
