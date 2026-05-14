from __future__ import annotations
import argparse
from backtesting.full_strategy_replay import ReplayConfig, run_full_strategy_replay
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input-file', required=True); ap.add_argument('--output-dir', default='outputs/backtests/12m')
    args=ap.parse_args(); run_full_strategy_replay(ReplayConfig(input_file=args.input_file, output_dir=args.output_dir))
