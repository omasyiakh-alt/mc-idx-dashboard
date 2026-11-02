"""
Screener LQ45: simulasi Monte Carlo untuk banyak saham sekaligus, diurutkan berdasarkan
peluang TP tersentuh lebih dulu. Hanya memakai data open source (Yahoo Finance via yfinance).

Sinyal yang dipakai:
  - Bandarmology PROXY dari OHLCV (CMF, A/D line, OBV)
  - Foreign flow TIDAK tersedia dari sumber terbuka -> dianggap netral (0)
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import factors as fc
import market_data as md
import mc_engine as mc

# Perkiraan konstituen LQ45. Daftar resmi berganti tiap Februari & Agustus:
# cek idx.co.id dan sunting daftar ini di layar bila ada perubahan.
LQ45_DEFAULT = [
    "ACES", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARTO", "ASII", "BBCA",
    "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRPT", "CPIN", "CTRA", "ESSA", "EXCL",
    "GOTO", "ICBP", "INCO", "INDF", "INKP", "ISAT", "ITMG", "JPFA", "JSMR", "KLBF",
    "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "PGAS", "PGEO", "PTBA", "SIDO", "SMGR",
    "SMRA", "TLKM", "TOWR", "UNTR", "UNVR",
]


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def download_many(tickers: tuple, period: str) -> dict:
    import yfinance as yf

    syms = [t + ".JK" for t in tickers]
    raw = yf.download(syms, period=period, interval="1d", auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    out = {}
    for t, s in zip(tickers, syms):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[s] if s in raw.columns.get_level_values(0) else raw.xs(s, axis=1, level=1)
            else:
                df = raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[df["Volume"] > 0]
            if len(df) >= 60:
                out[t] = df
        except Exception:  # noqa: BLE001
            continue
    return out


def load_aux() -> dict:
    return {"ihsg": md.load_close(fc.IHSG), "eido": md.load_close(fc.EIDO), "fx": md.load_close(fc.USDIDR),
            "comm": {k: md.load_close(sym) for k, (sym, _) in fc.COMMODITIES.items()}}


def screen_one(t: str, px: pd.DataFrame, days: int, rr: tuple, cfg: dict, objective: str,
               sig_window: int, weights: dict) -> dict:
    sigs = fc.yahoo_signals(t, px, load_aux(), weights, sig_window)
    c = mc.SimConfig(**{**cfg, "horizon_days": days, "ff_score": 0.0, "bd_score": 0.0,
                        "w_ff": 0.0, "w_bd": 0.0, "extra": sigs["extra"]})
    res = mc.simulate(px["Close"], c)
    best = mc.suggest_tp_sl_rr(res, days, rr[0], rr[1], objective=objective)
    lr = np.log(px["Close"]).diff().dropna()
    ret_end = res.paths[:, days] / res.s0 - 1
    return {
        "Kode": t,
        "Harga": res.s0,
        "TP": best["tp"], "SL": best["sl"], "R:R": best["rr"],
        "P(TP dulu)": best["p_tp"], "P(SL dulu)": best["p_sl"],
        "Win rate": best["win_rate"], "Edge": best["edge"], "Expectancy": best["expectancy"],
        "Median hari ke TP": best["days_to_tp"],
        "P(naik)": float((ret_end > 0).mean()),
        "VaR95": float(-np.percentile(ret_end, 5)),
        "Skor gabungan": res.combined,
        "Skor bandar (proxy)": sigs["bd"], "Momentum & RS": sigs["mom"], "Proxy asing pasar": sigs["mkt"],
        "Komoditas": sigs["comm"],
        "Return 3B": float(px["Close"].iloc[-1] / px["Close"].iloc[-min(63, len(px) - 1)] - 1),
        "Vol tahunan": float(lr.tail(126).std() * np.sqrt(mc.TRADING_DAYS)),
    }


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def screen_cached(t: str, tickers: tuple, period: str, days: int, rr: tuple, cfg_items: tuple, objective: str,
                  sig_window: int, w_items: tuple, _day: date):
    data = download_many(tickers, period)
    if t not in data:
        return None
    return screen_one(t, data[t], days, rr, dict(cfg_items), objective, sig_window, dict(w_items))


def run_screen(tickers: tuple, period: str, days: int, rr: tuple, cfg_items: tuple, objective: str,
               sig_window: int, w_items: tuple) -> tuple[pd.DataFrame, list]:
    rows, failed = [], []
    bar = st.progress(0.0, text="Menjalankan simulasi...")
    for i, t in enumerate(tickers):
        try:
            r = screen_cached(t, tickers, period, days, rr, cfg_items, objective, sig_window, w_items, date.today())
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            failed.append(t)
        else:
            rows.append(r)
        bar.progress((i + 1) / len(tickers), text=f"Simulasi {t} ({i + 1}/{len(tickers)})")
    bar.empty()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["P(TP dulu)", "Edge"], ascending=False).reset_index(drop=True)
        df.index = df.index + 1
    return df, failed


def render(fmt_pct, colors: dict):
    st.title("Screener LQ45 · peluang TP lebih dulu")
    st.caption("Sumber data: Yahoo Finance (gratis). Sinyal: bandarmology PROXY dari harga & volume, momentum & "
               "kekuatan relatif vs IHSG, proxy asing pasar (EIDO + USD/IDR), dan komoditas terkait. Volatilitas "
               "GARCH(1,1). Foreign flow per saham tidak tersedia dari sumber terbuka.")

    with st.sidebar:
        st.header("Pengaturan screener")
        hname = st.selectbox("Timeframe", list(mc.HORIZONS), index=1)
        days = mc.HORIZONS[hname]
        lo, hi = mc.DEFAULT_RR[hname]
        rr = st.slider(f"Rentang R:R {hname}", 0.5, 5.0, (float(lo), float(hi)), 0.25)
        objective = st.selectbox(
            "TP/SL tiap saham dipilih berdasarkan", ["p_tp", "edge", "expectancy"],
            format_func={"p_tp": "Peluang TP tertinggi", "edge": "Edge peluang",
                         "expectancy": "Expectancy tertinggi"}.get)
        n_sims = st.number_input("Simulasi per saham", 10_000, 50_000, 10_000, 5_000)
        period = st.selectbox("Data historis", ["1y", "2y", "3y"], index=1)
        drift_mode = st.radio("Drift dasar", ["historis", "nol"], horizontal=True)
        strength = st.slider("Kekuatan geser drift (× σ harian)", 0.0, 0.5, 0.10, 0.01)
        sig_window = st.slider("Jendela sinyal (hari bursa)", 5, 60, 20)
        vol_model = st.selectbox("Model volatilitas", ["garch", "ewma"],
                                 format_func={"garch": "GARCH(1,1)", "ewma": "Historis + EWMA"}.get)
        st.caption("Bobot sinyal (0 = tidak dipakai)")
        w_items = tuple(sorted({
            "bd": st.slider("Bandarmology proxy", 0.0, 1.0, fc.DEFAULT_WEIGHTS["bd"], 0.05),
            "mom": st.slider("Momentum & RS vs IHSG", 0.0, 1.0, fc.DEFAULT_WEIGHTS["mom"], 0.05),
            "mkt": st.slider("Proxy asing pasar (EIDO+USD/IDR)", 0.0, 1.0, fc.DEFAULT_WEIGHTS["mkt"], 0.05),
            "comm": st.slider("Komoditas", 0.0, 1.0, fc.DEFAULT_WEIGHTS["comm"], 0.05),
        }.items()))
        min_p = st.slider("Tampilkan P(TP dulu) minimal", 0.0, 0.9, 0.0, 0.05)

    with st.expander(f"Daftar saham ({len(LQ45_DEFAULT)} emiten LQ45, bisa disunting)"):
        st.caption("Konstituen LQ45 berganti tiap Februari dan Agustus. Cek daftar resmi di idx.co.id "
                   "lalu sunting di sini bila perlu.")
        txt = st.text_area("Kode emiten, pisahkan dengan koma atau spasi", ", ".join(LQ45_DEFAULT), height=120)
    tickers = tuple(dict.fromkeys(t.strip().upper().removesuffix(".JK")
                                  for t in txt.replace("\n", ",").replace(" ", ",").split(",") if t.strip()))

    cfg_items = tuple(sorted({
        "n_sims": int(n_sims), "innovation": "bootstrap", "use_ewma_vol": True, "strength": strength,
        "vol_boost": 0.25, "half_life": 21.0, "apply_auto_reject": True, "arb": 0.15,
        "drift_mode": drift_mode, "seed": 42, "vol_model": vol_model,
    }.items()))

    if not st.button(f"Jalankan screener ({len(tickers)} saham)", type="primary") and \
            "screen_ran" not in st.session_state:
        st.info("Klik tombol di atas untuk mulai. Satu kali screening 45 saham butuh sekitar 1 menit; "
                "hasilnya disimpan 3 jam.")
        return
    st.session_state["screen_ran"] = True

    with st.spinner(f"Mengunduh harga {len(tickers)} saham dari Yahoo Finance..."):
        download_many(tickers, period)
    df, failed = run_screen(tickers, period, days, tuple(rr), cfg_items, objective, sig_window, w_items)
    if failed:
        st.warning(f"Tidak ada data / gagal diproses: {', '.join(failed)}")
    if df.empty:
        st.error("Tidak ada hasil.")
        return

    view = df[df["P(TP dulu)"] >= min_p]
    be = 1 / (1 + (rr[0] + rr[1]) / 2)
    n_over = int((df["P(TP dulu)"] > 0.5).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saham diproses", len(df))
    c2.metric("P(TP dulu) > 50%", n_over)
    c3.metric("Edge positif", int((df["Edge"] > 0).sum()))
    c4.metric(f"Peluang acak di R:R tengah", fmt_pct(be),
              help="Tanpa drift, peluang TP lebih dulu ≈ 1/(1+R). Saham di atas angka ini didorong drift positif.")

    top = view.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["P(TP dulu)"], y=top["Kode"], orientation="h",
        marker_color=np.where(top["Edge"] > 0, colors["tp"], colors["gray"]),
        customdata=np.stack([top["TP"], top["SL"], top["R:R"], top["Edge"]], axis=1),
        hovertemplate="%{y}: %{x:.1%}<br>TP %{customdata[0]:,.0f} · SL %{customdata[1]:,.0f} · "
                      "R:R 1:%{customdata[2]:.2f}<br>Edge %{customdata[3]:+.1%}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line=dict(color=colors["sl"], width=1, dash="dash"),
                  annotation_text="50%", annotation_position="top")
    fig.update_layout(title=f"Top 20 peluang TP lebih dulu · {hname}", height=560, xaxis_tickformat=".0%",
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Hijau = edge positif (peluang TP lebih dulu di atas win rate impas). Abu-abu = belum ada edge.")

    pct = ["P(TP dulu)", "P(SL dulu)", "Win rate", "Edge", "Expectancy", "P(naik)", "VaR95", "Return 3B",
           "Vol tahunan"]
    st.dataframe(
        view.style.format({**{c: "{:.1%}" for c in pct}, "Edge": "{:+.1%}", "Expectancy": "{:+.2%}",
                           "Return 3B": "{:+.1%}", "Harga": "{:,.0f}", "TP": "{:,.0f}", "SL": "{:,.0f}",
                           "R:R": "1:{:.2f}", "Median hari ke TP": "{:.0f}", "Skor bandar (proxy)": "{:+.2f}",
                           "Skor gabungan": "{:+.2f}", "Momentum & RS": "{:+.2f}", "Proxy asing pasar": "{:+.2f}",
                           "Komoditas": "{:+.2f}"},
                          na_rep="-"),
        width="stretch", height=620,
    )
    st.download_button("Unduh hasil (CSV)", view.to_csv().encode(), file_name=f"screener_lq45_{hname}.csv",
                       mime="text/csv")
    st.caption("Untuk analisis lengkap satu saham (fan chart, VaR, data broker), pindah ke halaman "
               "Analisis emiten di sidebar. Hasil screener bergantung pada asumsi drift; gunakan untuk "
               "membandingkan antar saham, bukan sebagai jaminan peluang.")
