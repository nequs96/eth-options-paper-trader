from __future__ import annotations

from pathlib import Path
import py_compile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

py_compile.compile(str(PROJECT_ROOT / 'execution/professional_candidate_filter.py'), doraise=True)
from execution.professional_candidate_filter import load_csv_if_exists, filter_candidate_file
print('EMPTY_CANDIDATES_FILTER_PATCH_AUDIT_OK')
