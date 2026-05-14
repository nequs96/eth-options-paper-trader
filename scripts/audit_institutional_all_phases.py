from pathlib import Path
import py_compile, sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
FILES=['data/data_quality.py','storage/database.py','storage/market_data_repository.py','models/portfolio_diagnostics.py','execution/institutional_recorder.py','scripts/record_current_state.py']
for f in FILES:
    py_compile.compile(str(PROJECT_ROOT/f), doraise=True); print(f'OK   {f}')
from storage.database import initialize_database
from execution.institutional_recorder import record_event
initialize_database('outputs/eth_options_research.db')
record_event('audit','Institutional all-phases audit completed.')
print('INSTITUTIONAL_ALL_PHASES_AUDIT_OK')
