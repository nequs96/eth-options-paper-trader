from __future__ import annotations
import argparse
from backtesting.historical_data_adapters import HistoricalFetchConfig, fetch_historical_options

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--provider', default='local_csv', choices=['local_csv','laevitas','tardis','coinapi','amberdata','cryptodatadownload'])
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--currency', default='ETH')
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--output-dir', default='data/historical_options')
    args=ap.parse_args()
    fetch_historical_options(HistoricalFetchConfig(provider=args.provider,start=args.start,end=args.end,currency=args.currency,interval=args.interval,output_dir=args.output_dir))
