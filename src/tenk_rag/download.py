"""Download 10-K filings from SEC EDGAR.

EDGAR rules: send a descriptive User-Agent (name + email) and stay under
10 requests/second. See https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


class EdgarClient:
    def __init__(self, user_agent: str, min_interval: float = 0.15):
        if "example.com" in user_agent:
            raise ValueError("Set corpus.user_agent in config.yaml to your name and email.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=60)
        self._last = time.time()
        resp.raise_for_status()
        return resp

    def ticker_to_cik(self) -> dict[str, int]:
        data = self.get(TICKERS_URL).json()
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def list_10ks(self, cik: int, limit: int) -> list[dict]:
        sub = self.get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = sub["filings"]["recent"]
        out = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":  # skip 10-K/A amendments
                continue
            out.append({
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_doc": recent["primaryDocument"][i],
                "company": sub.get("name", ""),
            })
            if len(out) >= limit:
                break
        return out


def download_filings(tickers: list[str], per_company: int, out_dir: str | Path,
                     user_agent: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(user_agent)
    cik_map = client.ticker_to_cik()
    saved = []
    for ticker in tickers:
        ticker = ticker.upper()
        cik = cik_map.get(ticker)
        if cik is None:
            print(f"[skip] {ticker}: not found in EDGAR ticker list")
            continue
        for f in client.list_10ks(cik, per_company):
            fiscal_year = int(f["report_date"][:4]) if f["report_date"] else int(f["filing_date"][:4])
            stem = f"{ticker}_{fiscal_year}"
            html_path = out_dir / f"{stem}.html"
            if html_path.exists():
                saved.append(html_path)
                continue
            url = ARCHIVE_URL.format(cik=cik, acc=f["accession"].replace("-", ""), doc=f["primary_doc"])
            html_path.write_bytes(client.get(url).content)
            meta = {"ticker": ticker, "cik": cik, "fiscal_year": fiscal_year, "url": url, **f}
            (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
            print(f"[ok] {stem}  ({f['filing_date']})")
            saved.append(html_path)
    return saved
