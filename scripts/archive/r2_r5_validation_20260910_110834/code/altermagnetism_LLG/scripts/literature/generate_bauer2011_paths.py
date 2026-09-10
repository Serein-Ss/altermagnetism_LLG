"""Deprecated import/CLI compatibility; implementation is explicitly legacy."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from altermagnetism_LLG.scripts.literature.bauer_2011.legacy_run import *
if __name__=='__main__':
    import warnings
    warnings.warn('Legacy SI/pilot entry: not the strict reduced reproduction',RuntimeWarning)
    main()
