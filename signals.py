"""
Sinyal Foreign Flow & Bandarmology untuk dashboard Monte Carlo IDX.

Semua skor dinormalisasi ke rentang [-1, +1]:
  +1 = akumulasi kuat (bullish), 0 = netral, -1 = distribusi kuat (bearish).

Sumber data:
  - Foreign Flow  : CSV (data riil dari broker/aplikasi Anda) atau input manual.
                    Tidak ada proxy, karena foreign flow tidak bisa ditebak dari OHLCV.
  - Bandarmology  : CSV broker summary (data riil) atau PROXY dari OHLCV
                    (CMF, A/D line, OBV) bila broker summary tidak tersedia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def _clip_score(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip(x, -1.0, 1.0))


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _slope_norm(series: pd.Series, scale: pd.Series | float) -> float:
    """Kemiringan regresi linier dari series, dinormalisasi oleh skala."""
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 5:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    s = float(np.nanmean(scale)) if isinstance(scale, pd.Series) else float(scale)
    if s == 0 or not np.isfinite(s):
        return 0.0
    return slope / s


# ----------------------------------------------------------------------------
# FOREIGN FLOW
# ----------------------------------------------------------------------------
def parse_foreign_flow_csv(file) -> pd.DataFrame:
    """
    Format yang diterima (header tidak case-sensitive):
      date, net_foreign                      (Rupiah, + = net buy asing)
    atau
      date, foreign_buy, foreign_sell        (Rupiah)
    Kolom 'net_foreign' boleh juga bernama 'net' / 'nbsa'.
    """
    df = _norm_cols(pd.read_csv(file))
    if "date" not in df.columns:
        raise ValueError("CSV foreign flow wajib punya kolom 'date'.")
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for alias in ("net", "nbsa", "net_foreign_value", "foreign_net"):
        if alias in df.columns and "net_foreign" not in df.columns:
            df = df.rename(columns={alias: "net_foreign"})
    if "net_foreign" not in df.columns:
        if {"foreign_buy", "foreign_sell"} <= set(df.columns):
            df["net_foreign"] = pd.to_numeric(df["foreign_buy"], errors="coerce") - pd.to_numeric(
                df["foreign_sell"], errors="coerce"
            )
        else:
            raise ValueError(
                "CSV foreign flow butuh kolom 'net_foreign' atau 'foreign_buy' + 'foreign_sell'."
            )
    df["net_foreign"] = pd.to_numeric(df["net_foreign"], errors="coerce").fillna(0.0)
    return df[["date", "net_foreign"] + [c for c in ("foreign_buy", "foreign_sell") if c in df.columns]]


def foreign_flow_score(ff: pd.DataFrame, prices: pd.DataFrame, window: int = 20) -> dict:
    """
    Skor foreign flow dari 3 komponen:
      1. Intensitas : total net foreign N hari / total nilai transaksi N hari
      2. Konsistensi: (hari net buy - hari net sell) / N
      3. Tren       : kemiringan kumulatif net foreign (dinormalisasi)
    """
    ff = ff.sort_values("date").tail(window)
    if ff.empty:
        return {"score": 0.0, "detail": {}}

    px = prices.copy()
    px["value"] = px["Close"] * px["Volume"]
    px = px.set_index(px.index.normalize())
    traded = px["value"].reindex(pd.to_datetime(ff["date"]).dt.normalize()).astype(float)
    total_value = float(np.nansum(traded)) if np.isfinite(traded).any() else np.nan
    if not np.isfinite(total_value) or total_value <= 0:
        # fallback: pakai rata-rata nilai transaksi N hari terakhir di data harga
        total_value = float(px["value"].tail(window).sum())

    net_sum = float(ff["net_foreign"].sum())
    intensity = net_sum / total_value if total_value > 0 else 0.0
    # 10% dari nilai transaksi sebagai net asing = sangat signifikan
    intensity_s = np.tanh(intensity / 0.10)

    consistency = float((np.sign(ff["net_foreign"]) ).mean())

    cum = ff["net_foreign"].cumsum().reset_index(drop=True)
    avg_daily_val = total_value / max(len(ff), 1)
    trend_s = np.tanh(_slope_norm(cum, avg_daily_val) / 0.05)

    score = 0.5 * intensity_s + 0.25 * consistency + 0.25 * trend_s
    return {
        "score": _clip_score(score),
        "detail": {
            "Net asing (Rp, N hari)": net_sum,
            "Net asing / nilai transaksi": intensity,
            "Konsistensi (hari beli vs jual)": consistency,
            "Skor intensitas": intensity_s,
            "Skor tren kumulatif": trend_s,
        },
    }


def foreign_flow_score_aggregate(net_foreign: float, prices: pd.DataFrame, window: int = 20) -> dict:
    """Skor foreign flow dari SATU angka agregat (mode hemat): hanya komponen intensitas."""
    total_value = float((prices["Close"] * prices["Volume"]).tail(window).sum())
    intensity = net_foreign / total_value if total_value > 0 else 0.0
    return {
        "score": _clip_score(np.tanh(intensity / 0.10)),
        "detail": {"Net asing (Rp, N hari)": net_foreign, "Net asing / nilai transaksi": intensity},
    }


# ----------------------------------------------------------------------------
# BANDARMOLOGY - dari broker summary (data riil)
# ----------------------------------------------------------------------------
def parse_broker_summary_csv(file) -> pd.DataFrame:
    """
    Format (satu baris per broker per hari; 'date' opsional bila hanya 1 periode):
      date, broker, buy_lot, sell_lot [, buy_value, sell_value]
    Alias diterima: buy_vol/sell_vol, bvol/svol, code/broker_code.
    """
    df = _norm_cols(pd.read_csv(file))
    ren = {
        "broker_code": "broker", "code": "broker", "kode": "broker",
        "buy_vol": "buy_lot", "bvol": "buy_lot", "b_lot": "buy_lot",
        "sell_vol": "sell_lot", "svol": "sell_lot", "s_lot": "sell_lot",
        "bval": "buy_value", "sval": "sell_value",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    need = {"broker", "buy_lot", "sell_lot"}
    if not need <= set(df.columns):
        raise ValueError("CSV broker summary butuh kolom: broker, buy_lot, sell_lot (+ date opsional).")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("buy_lot", "sell_lot", "buy_value", "sell_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["broker"] = df["broker"].astype(str).str.upper().str.strip()
    return df


def bandar_score_broker(bs: pd.DataFrame, last_price: float, window: int = 20) -> dict:
    """
    Metode 'Bandar Detector' klasik:
      Top-N accumulation = (Σ net lot N pembeli bersih terbesar
                            − Σ |net lot| N penjual bersih terbesar) / total lot transaksi
    Ditambah:
      - Avg price akumulator vs harga terakhir (bandar masih 'di bawah' = ruang naik)
      - Jumlah broker pembeli vs penjual (sedikit pembeli + banyak penjual = akumulasi)
    """
    df = bs.copy()
    if "date" in df.columns and df["date"].notna().any():
        dates = sorted(df["date"].dropna().unique())[-window:]
        df = df[df["date"].isin(dates)]
    agg = df.groupby("broker").agg(
        buy_lot=("buy_lot", "sum"), sell_lot=("sell_lot", "sum"),
        **({"buy_value": ("buy_value", "sum")} if "buy_value" in df.columns else {}),
        **({"sell_value": ("sell_value", "sum")} if "sell_value" in df.columns else {}),
    )
    agg["net_lot"] = agg["buy_lot"] - agg["sell_lot"]
    total_lot = float(agg["buy_lot"].sum())
    if total_lot <= 0:
        return {"score": 0.0, "detail": {}, "table": agg}

    buyers = agg[agg["net_lot"] > 0].sort_values("net_lot", ascending=False)
    sellers = agg[agg["net_lot"] < 0].sort_values("net_lot")
    res = {}
    for n in (1, 3, 5):
        res[n] = (buyers["net_lot"].head(n).sum() + sellers["net_lot"].head(n).sum()) / total_lot
    top5 = res[5]
    top5_s = np.tanh(top5 / 0.15)  # 15% konsentrasi net = sinyal kuat
    top1_s = np.tanh(res[1] / 0.10)

    # broker count: sedikit pembeli bersih, banyak penjual bersih -> akumulasi
    nb, ns = len(buyers), len(sellers)
    count_s = (ns - nb) / max(ns + nb, 1)

    avg_s = 0.0
    avg_price_top = np.nan
    if "buy_value" in agg.columns and len(buyers):
        top_b = buyers.head(5)
        lots = top_b["buy_lot"].sum()
        if lots > 0:
            # nilai dalam Rupiah, 1 lot = 100 lembar
            avg_price_top = top_b["buy_value"].sum() / (lots * 100)
            gap = (last_price - avg_price_top) / avg_price_top
            # harga di bawah/dekat avg bandar = positif; sudah jauh di atas = berkurang
            avg_s = float(np.tanh(-gap / 0.10)) * 0.5

    score = 0.55 * top5_s + 0.2 * top1_s + 0.15 * count_s + 0.10 * (avg_s * 2)
    return {
        "score": _clip_score(score),
        "detail": {
            "Top1 net / total lot": res[1],
            "Top3 net / total lot": res[3],
            "Top5 net / total lot": top5,
            "Broker net buy": nb,
            "Broker net sell": ns,
            "Avg harga top-5 akumulator": avg_price_top,
        },
        "table": agg.sort_values("net_lot", ascending=False),
    }


# ----------------------------------------------------------------------------
# BANDARMOLOGY - PROXY dari OHLCV (bukan data broker riil)
# ----------------------------------------------------------------------------
def bandar_proxy_indicators(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"]
    df["AD"] = mfv.cumsum()
    df["CMF20"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    return df


def bandar_score_proxy(px: pd.DataFrame, window: int = 20) -> dict:
    """
    Proxy akumulasi/distribusi:
      - CMF(20)                         : tekanan beli/jual berbobot volume
      - Kemiringan A/D line N hari      : akumulasi diam-diam
      - Divergensi OBV vs harga         : OBV naik saat harga datar/turun = akumulasi
    """
    df = bandar_proxy_indicators(px)
    tail = df.tail(window)
    vol_scale = float(df["Volume"].tail(window).mean())

    cmf = float(df["CMF20"].iloc[-1]) if np.isfinite(df["CMF20"].iloc[-1]) else 0.0
    cmf_s = np.tanh(cmf / 0.15)

    ad_s = np.tanh(_slope_norm(tail["AD"], vol_scale) / 0.15)
    obv_slope = _slope_norm(tail["OBV"], vol_scale)
    px_slope = _slope_norm(tail["Close"], float(tail["Close"].mean())) * window
    obv_s = np.tanh(obv_slope / 0.3)
    # divergensi: OBV positif tapi harga turun -> akumulasi (bonus), sebaliknya distribusi
    div_s = np.tanh((obv_s - np.tanh(px_slope / 0.10)) / 1.0)

    score = 0.4 * cmf_s + 0.3 * ad_s + 0.15 * obv_s + 0.15 * div_s
    return {
        "score": _clip_score(score),
        "detail": {
            "CMF(20)": cmf,
            "Skor CMF": cmf_s,
            "Skor slope A/D": ad_s,
            "Skor slope OBV": obv_s,
            "Skor divergensi OBV-harga": div_s,
        },
        "frame": df,
    }
