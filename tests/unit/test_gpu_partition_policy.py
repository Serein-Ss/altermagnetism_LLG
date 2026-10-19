"""Current GPU batch entry points must use the server's designated partition."""
from pathlib import Path
import re


def test_current_gpu_batch_partition():
    root = Path(__file__).resolve().parents[2]
    checked = []
    for path in [*(root / 'slurm').glob('*.sbatch'), *(root / 'tests/slurm').glob('*.sbatch')]:
        text = path.read_text()
        if re.search(r'^#SBATCH\s+--(?:gres[= ]gpu|gpus)', text, re.MULTILINE):
            partitions = re.findall(r'^#SBATCH\s+--partition[= ](\S+)', text, re.MULTILINE)
            assert partitions == ['rtx4090'], str(path)
            checked.append(path)
    assert checked, 'no GPU entry points found'
