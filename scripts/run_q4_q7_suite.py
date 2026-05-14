from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from execution.walk_forward_validation import run_walk_forward_validation
from execution.portfolio_risk_limits import check_portfolio_risk_limits
from execution.smart_exit_engine import generate_smart_exit_recommendations
from visualization.enhanced_research_dashboard import generate_enhanced_research_dashboard
if __name__=='__main__':
    run_walk_forward_validation(); check_portfolio_risk_limits(); generate_smart_exit_recommendations(); generate_enhanced_research_dashboard(); print('Q4_Q7_SUITE_COMPLETE')
