"""Reuse the existing exhaustive sample verifier on the five existing GTs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT.parent/'laion-six-cell-20261111/scripts'))
import common
import gt_audit
common.ROOT=Path(sys.argv[1])
gt_audit.main()
