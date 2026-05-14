from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from execution.clean_reporting import print_clean_cycle_report
from execution.paper_trader import PaperTraderConfig


if __name__ == '__main__':
    print_clean_cycle_report(PaperTraderConfig())
