"""Submit the bounded campaign once, persist every accepted job immediately."""
import hashlib
import json
from pathlib import Path
import subprocess
from tests.bauer_two_stage_20260919.common import dump

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    target = HERE/'submission.json'
    if target.exists(): raise RuntimeError('Submission record exists; refusing duplicate campaign')
    ledger = dict(jobs=[], scientific_certification=False,
        policy='No monitoring for >10 minute jobs. afterok for computations, afterany for honest reporting.',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))})
    dump(target,ledger)
    script = str((HERE/'job.sh').relative_to(ROOT))
    def submit(name, estimate, note, args):
        if any(arg.startswith('--dependency=') for arg in args):
            args = ['--kill-on-invalid-dep=yes',*args]
        result = subprocess.run(['sbatch','--parsable',*args],cwd=ROOT,text=True,capture_output=True,check=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError('Unrecognized Slurm response: '+result.stdout)
        ledger['jobs'].append(dict(name=name,id=job,estimate=estimate,note=note,command=['sbatch','--parsable',*args]))
        dump(target,ledger); print(name,job,flush=True); return job
    b = submit('一维old/fixed配对训练×3 seed','每任务15–45分钟','已提交；数组6任务，不监控',
               ['--array=0-5','--time=02:00:00',script,'train_bauer'])
    m = submit('二维local/full配对训练×3 seed','每任务4–12分钟','已提交；数组6任务，按上界视为长任务',
               ['--array=0-5','--time=01:00:00',script,'train_miltat'])
    eb = submit('一维修正版L16/25/32/50/100评估','40–90分钟','依赖一维全部训练成功，诊断1ps和10ps',
                [f'--dependency=afterok:{b}','--time=03:00:00',script,'evaluate_bauer'])
    em = submit('二维选定模型开发/测试及积分收敛','10–25分钟','依赖二维训练成功；含16/32/64/128步诊断',
                [f'--dependency=afterok:{m}','--time=01:30:00',script,'evaluate'])
    lb = submit('Bauer L100三条10ns模型生成','20–24小时','64 flow步、0.1ps块；预留32h，可恢复；尚不具备加速优势',
                [f'--dependency=afterok:{b}','--time=32:00:00',script,'long','--system','bauer'])
    lm = submit('二维完整模型文献几何1ns生成','65–90分钟','200×116原胞、单条；固定开放窗口与原文有差异',
                [f'--dependency=afterok:{m}','--time=03:00:00',script,'long','--system','miltat','--kind','full'])
    ll = submit('二维局部模型文献几何1ns生成','10–20分钟','与完整模型相同初态、seed和输出时间',
                [f'--dependency=afterok:{m}','--time=01:00:00',script,'long','--system','miltat','--kind','local'])
    report = str((HERE/'report.sh').relative_to(ROOT))
    early = submit('训练/短时评估完成后刷新网页','2–8分钟','afterany：失败时仍保留真实状态、不伪造结果',
                   [f'--dependency=afterany:{eb}:{em}',report])
    submit('全部生成结束后刷新网页与离线ZIP','5–15分钟','afterany；包含1D完整轨迹、2D动态图，未完成部分明确标记',
           [f'--dependency=afterany:{lb}:{lm}:{ll}:{early}',report])


if __name__ == '__main__': main()
