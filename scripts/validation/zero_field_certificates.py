"""Hash-bound numerical certificates. A pass boolean alone is never accepted."""
import json
import math
from pathlib import Path
from scripts.core.literature_config import sha256


def verify(path,scope,expected_bindings):
    c=json.loads(Path(path).read_text())
    if c['scope']!=scope or c['bindings']!=expected_bindings: raise ValueError('certificate scope/bindings mismatch')
    if c['status']!='pass' or not c.get('metrics') or not c.get('evidence_sha256'):
        raise ValueError('certificate is not a measured pass')
    for filename,digest in c['evidence_sha256'].items():
        if sha256(filename)!=digest: raise ValueError('certificate evidence changed: '+filename)
    for metric in c['metrics']:
        value=metric['value']
        if not math.isfinite(value) or not metric['lower']<=value<=metric['upper']:
            raise ValueError('certificate measurement outside registered bounds')
    return c
