"""Pengambil data Yahoo Finance untuk seri non-saham (IHSG, EIDO, USD/IDR, komoditas)."""
from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_close(symbol: str, period: str = "5y") -> pd.Series | None:
    """Harga penutupan harian; None bila gagal (tidak menghentikan dashboard)."""
    try:
        import yfinance as yf

        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = symbol
        return s if len(s) > 30 else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    """OHLCV harian saham IDX (KODE.JK) dari Yahoo; None bila gagal."""
    try:
        import yfinance as yf

        sym = ticker.upper().strip()
        sym = sym if sym.endswith(".JK") else sym + ".JK"
        df = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[df["Volume"] > 0]
    except Exception:  # noqa: BLE001
        return None
