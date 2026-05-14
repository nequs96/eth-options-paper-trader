from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from execution.institutional_recorder import record_full_cycle_outputs

if __name__ == '__main__':
    record_full_cycle_outputs()
