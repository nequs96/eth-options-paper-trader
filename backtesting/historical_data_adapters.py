from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gzip, json, os, urllib.parse, urllib.request
import pandas as pd
from backtesting.historical_options_schema import normalize_options_dataframe

@dataclass(frozen=True)
class HistoricalFetchConfig:
    provider: str
    start: str
    end: str
    currency: str = 'ETH'
    market: str = 'DERIBIT'
    interval: str = '1h'
    output_dir: str = 'data/historical_options'
    api_key: str | None = None

class HistoricalDataAdapter:
    name='base'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        raise NotImplementedError

class LocalCSVAdapter(HistoricalDataAdapter):
    name = "local_csv"

    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        root = Path(cfg.output_dir)

        files = sorted(
            p for p in root.glob("*.csv")
            if not p.name.startswith("normalized_")
            and not p.name.startswith("backtest_")
            and not p.name.startswith("summary_")
            and p.stat().st_size > 0
        )

        if not files:
            raise RuntimeError(
                f"No raw provider CSV files found in {root}. "
                "Download/export historical Deribit ETH options data first, "
                "put the raw CSV files in data/historical_options/, then rerun normalization."
            )

        frames = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    df["source_file"] = file.name
                    frames.append(df)
            except Exception as error:
                print(f"Warning: could not read {file}: {error}")

        if not frames:
            raise RuntimeError(
                f"CSV files were found in {root}, but all were empty or unreadable."
            )

        combined = pd.concat(frames, ignore_index=True)
        normalized = normalize_options_dataframe(combined)

        if normalized.empty:
            raise RuntimeError(
                "Raw provider CSVs were loaded, but normalization produced 0 usable rows. "
                f"Loaded files: {[p.name for p in files]}. "
                f"Raw columns: {list(combined.columns)}"
            )

        return normalized

class LaevitasAdapter(HistoricalDataAdapter):
    name='laevitas'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        key=cfg.api_key or os.getenv('LAEVITAS_API_KEY')
        if not key: raise RuntimeError('Set LAEVITAS_API_KEY or pass api_key')
        base=f'https://api.laevitas.ch/historical/options/snapshot/{cfg.market}/{cfg.currency}'
        params=urllib.parse.urlencode({'start':cfg.start,'end':cfg.end,'frequency':cfg.interval})
        req=urllib.request.Request(base+'?'+params, headers={'apiKey':key, 'accept':'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data=json.loads(r.read().decode('utf-8'))
        items=data.get('items', data if isinstance(data,list) else [])
        return normalize_options_dataframe(pd.DataFrame(items))

class TardisAdapter(HistoricalDataAdapter):
    name='tardis'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        key=cfg.api_key or os.getenv('TARDIS_API_KEY')
        if not key: raise RuntimeError('Set TARDIS_API_KEY or pass api_key')
        # Tardis low-level endpoint returns exchange-native NDJSON. This adapter stores raw messages
        # and normalizes ticker/options_chain-like messages when fields are present.
        filters=json.dumps([{'channel':'ticker'}])
        params=urllib.parse.urlencode({'from':cfg.start,'filters':filters,'sliceSize':10})
        url=f'https://api.tardis.dev/v1/data-feeds/deribit?{params}'
        req=urllib.request.Request(url, headers={'Authorization':'Bearer '+key,'Accept-Encoding':'gzip'})
        rows=[]
        with urllib.request.urlopen(req, timeout=120) as r:
            raw=r.read()
            try: text=gzip.decompress(raw).decode('utf-8')
            except Exception: text=raw.decode('utf-8', errors='ignore')
        for line in text.splitlines():
            if not line.strip(): continue
            try:
                obj=json.loads(line.split(' ',1)[-1])
                params=obj.get('params',{}); data=params.get('data',obj.get('data',{}))
                if isinstance(data,dict): rows.append(data)
            except Exception: pass
        return normalize_options_dataframe(pd.DataFrame(rows))

class CoinAPIAdapter(HistoricalDataAdapter):
    name='coinapi'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        raise NotImplementedError('CoinAPI symbol discovery is provider-plan specific. Use export/download to CSV, then provider=local_csv.')

class AmberdataAdapter(HistoricalDataAdapter):
    name='amberdata'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        raise NotImplementedError('Amberdata endpoints vary by subscription. Export/download to CSV, then provider=local_csv, or extend this adapter with your endpoint path.')

class CryptoDataDownloadAdapter(HistoricalDataAdapter):
    name='cryptodatadownload'
    def fetch(self, cfg: HistoricalFetchConfig) -> pd.DataFrame:
        raise NotImplementedError('CryptoDataDownload files require manual/Plus download. Put CSV files in data/historical_options and use provider=local_csv.')

ADAPTERS={a.name:a for a in [LocalCSVAdapter(), LaevitasAdapter(), TardisAdapter(), CoinAPIAdapter(), AmberdataAdapter(), CryptoDataDownloadAdapter()]}

def fetch_historical_options(cfg: HistoricalFetchConfig) -> pd.DataFrame:
    adapter=ADAPTERS.get(cfg.provider.lower())
    if not adapter: raise ValueError(f'Unknown provider {cfg.provider}. Choose {list(ADAPTERS)}')
    df=adapter.fetch(cfg)
    outdir=Path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    out=outdir/f'normalized_{cfg.provider}_{cfg.currency}_{cfg.start}_{cfg.end}.csv'.replace(':','-')
    df.to_csv(out,index=False)
    print(f'Saved normalized historical options: {out} rows={len(df)}')
    return df
