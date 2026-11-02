"""
Pengambil data otomatis dari Index Alpha API (https://indexalpha.id).

Endpoint yang dipakai (sesuai dokumentasi resmi):
  GET /stocks/broker-summary?ticker&from&to&investor   -> broker summary (agregat rentang tanggal)
  GET /foreign-flow?ticker&from&to                     -> foreign buy/sell/net (agregat rentang tanggal)
  GET /stocks/ohlcv?ticker&from&to                     -> OHLCV harian (maks 366 hari per request)
  GET /usage                                           -> sisa kuota (gratis, tidak memotong kuota)

Data end-of-day, diperbarui setiap hari bursa pukul 19:00 WIB. Riwayat mulai 2025-01-01.
API key dibaca dari st.secrets["INDEXALPHA_API_KEY"] atau variabel lingkungan INDEXALPHA_API_KEY.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = os.environ.get("INDEXALPHA_BASE_URL", "https://api.indexalpha.id")
MIN_DATE = date(2025, 1, 1)


class IndexAlphaError(RuntimeError):
    pass


class IndexAlphaClient:
    def __init__(self, token: str, base_url: str = BASE_URL, timeout: float = 20.0):
        if not token:
            raise IndexAlphaError("API key Index Alpha belum diisi.")
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json"})

    # ------------------------------------------------------------------
    def _get(self, path: str, **params):
        r = self.s.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code != 200 or not body.get("success", False):
            detail = body.get("detail") or body.get("error") or r.text[:200]
            hint = {
                401: "API key tidak terkirim.",
                403: "API key tidak aktif, langganan habis, atau kuota habis.",
                429: "Terlalu banyak request per menit; coba lagi sebentar.",
            }.get(r.status_code, "")
            raise IndexAlphaError(f"Index Alpha {r.status_code}: {detail} {hint}".strip())
        return body["data"]

    # ------------------------------------------------------------------
    def usage(self) -> dict:
        return self._get("/usage")

    def broker_summary(self, ticker: str, d_from: date, d_to: date, investor: str = "all",
                       market: str = "RG") -> pd.DataFrame:
        rows = self._get("/stocks/broker-summary", ticker=ticker, investor=investor, market=market,
                         **{"from": str(d_from), "to": str(d_to)})
        df = pd.DataFrame(rows or [])
        if df.empty:
            return df
        df = df.rename(columns={"code": "broker"})
        for c in ("buy_freq", "buy_volume", "buy_value", "sell_freq", "sell_volume", "sell_value",
                  "buy_avg", "sell_avg"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # volume dari API dalam lembar -> lot (1 lot = 100 lembar)
        df["buy_lot"] = df["buy_volume"] / 100
        df["sell_lot"] = df["sell_volume"] / 100
        return df

    def foreign_flow(self, ticker: str, d_from: date, d_to: date) -> dict:
        return self._get("/foreign-flow", ticker=ticker, **{"from": str(d_from), "to": str(d_to)})

    def ohlcv(self, ticker: str, d_from: date, d_to: date) -> pd.DataFrame:
        d_from = max(d_from, MIN_DATE)
        parts, start = [], d_from
        while start <= d_to:
            end = min(start + timedelta(days=365), d_to)
            rows = self._get("/stocks/ohlcv", ticker=ticker, **{"from": str(start), "to": str(end)})
            parts.append(pd.DataFrame(rows or []))
            start = end + timedelta(days=1)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            raise IndexAlphaError(f"OHLCV {ticker} kosong dari Index Alpha.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").set_index("date").sort_index()
        df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df[df["Volume"] > 0]

    # ------------------------------------------------------------------
    # Rangkaian harian (1 request per hari bursa)
    # ------------------------------------------------------------------
    def foreign_flow_daily(self, ticker: str, days: list[date], workers: int = 4) -> pd.DataFrame:
        def one(d):
            try:
                x = self.foreign_flow(ticker, d, d)
                return {"date": pd.Timestamp(d), "foreign_buy": float(x.get("foreign_buy") or 0),
                        "foreign_sell": float(x.get("foreign_sell") or 0),
                        "net_foreign": float(x.get("net_foreign") or 0)}
            except IndexAlphaError as e:
                return {"date": pd.Timestamp(d), "error": str(e)}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(one, days))
        df = pd.DataFrame(rows)
        if "error" in df.columns:
            errs = df["error"].dropna()
            if len(errs) == len(df):
                raise IndexAlphaError(errs.iloc[0])
            df = df[df["error"].isna()].drop(columns="error")
        return df.sort_values("date").reset_index(drop=True)

    def broker_summary_daily(self, ticker: str, days: list[date], workers: int = 4) -> pd.DataFrame:
        def one(d):
            try:
                df = self.broker_summary(ticker, d, d)
                if not df.empty:
                    df["date"] = pd.Timestamp(d)
                return df
            except IndexAlphaError:
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = [p for p in ex.map(one, days) if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bandar_timeline(bs_daily: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Akumulasi/distribusi harian: (net lot top-N pembeli + net lot top-N penjual) / total lot."""
    out = []
    for d, g in bs_daily.groupby("date"):
        net = g["buy_lot"] - g["sell_lot"]
        tot = g["buy_lot"].sum()
        if tot <= 0:
            continue
        val = (net[net > 0].nlargest(top_n).sum() + net[net < 0].nsmallest(top_n).sum()) / tot
        out.append({"date": d, "top_net_ratio": val})
    return pd.DataFrame(out)


def get_api_key() -> str:
    try:
        import streamlit as st

        key = st.secrets.get("INDEXALPHA_API_KEY", "")
        if key:
            return key
    except Exception:  # noqa: BLE001  (secrets.toml belum ada)
        pass
    return os.environ.get("INDEXALPHA_API_KEY", "")
