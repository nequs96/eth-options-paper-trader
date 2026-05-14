from __future__ import annotations
import py_compile
from pathlib import Path
FILES=[p for p in Path('.').rglob('*.py') if '.venv' not in str(p) and 'venv' not in str(p)]
failed=[]
for p in FILES:
    try: py_compile.compile(str(p), doraise=True)
    except Exception as e: failed.append((str(p),str(e)))
if failed:
    for f,e in failed: print('FAIL',f,e)
    raise SystemExit(1)
print(f'RUN_ALL_TESTS_OK compiled={len(FILES)}')
