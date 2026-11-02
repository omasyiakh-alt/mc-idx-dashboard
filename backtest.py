"""
Backtest walk-forward: apakah peluang dari simulasi Monte Carlo jujur?

Di setiap tanggal uji di masa lalu, model dijalankan HANYA dengan data sampai tanggal itu,
lalu prediksinya dibandingkan dengan harga yang benar-benar terjadi.

Ukuran yang dipakai:
  - Cakupan pita 90% dan 50%      : harga aktual harus jatuh di dalam pita ~90% / ~50% dari waktu
  - Pelanggaran VaR 95%           : rugi melewati VaR harus terjadi ~5% dari waktu
  - Brier arah (P naik)           : makin kecil makin baik
  - Brier & kalibrasi P(TP dulu)  : prediksi 60% harus benar ~60% dari waktu
  - Skill vs model dasar          : model dengan sinyal harus lebih baik dari random walk polos
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import factors as fc
import market_data as md
import mc_engine as mc

DEFAULT_STEP = {"1D": 5, "1W": 5, "1M": 10, "3M": 10, "6M": 10}
EST_WINDOW = 500  # hari bursa untuk estimasi di setiap tanggal (≈ 2 tahun, sama dengan default dashboard)


def realized_touch(future: pd.DataFrame, tp: float, sl: float) -> tuple[str, float]:
    """Mana yang tersentuh lebih dulu pada data High/Low aktual. Hari yang sama -> SL (konservatif)."""
    for _, row in future.iterrows():
        hit_sl = row["Low"] <= sl
        hit_tp = row["High"] >= tp
        if hit_sl:
            return "SL", sl
        if hit_tp:
            return "TP", tp
    return "Tidak tersentuh", float(future["Close"].iloc[-1])


def model_specs(strengths: list[float], weights: dict) -> dict:
    specs = {"Dasar (random walk, tanpa sinyal)": dict(drift_mode="nol", vol_model="ewma", use_signals=False,
                                                        strength=0.0)}
    specs["GARCH + drift historis (sinyal 0)"] = dict(drift_mode="historis", vol_model="garch", use_signals=False,
                                                      strength=0.0)
    for s in strengths:
        specs[f"Lengkap · kekuatan {s:.2f}"] = dict(drift_mode="historis", vol_model="garch", use_signals=True,
                                                    strength=s)
    return specs


def _aux() -> dict:
    return {"ihsg": md.load_close(fc.IHSG), "eido": md.load_close(fc.EIDO), "fx": md.load_close(fc.USDIDR),
            "comm": {k: md.load_close(sym) for k, (sym, _) in fc.COMMODITIES.items()}}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _rows_for_date(ticker: str, hname: str, i: int, j: int, n_sims: int, strengths: tuple,
                   w_items: tuple) -> list[dict]:
    days = mc.HORIZONS[hname]
    px = md.load_ohlcv(ticker, "5y")
    weights = dict(w_items)
    specs = model_specs(list(strengths), weights)
    lo, hi = mc.DEFAULT_RR[hname]
    hist = px.iloc[max(0, i + 1 - EST_WINDOW): i + 1]
    future = px.iloc[i + 1: i + 1 + days]
    s0, actual = float(hist["Close"].iloc[-1]), float(future["Close"].iloc[-1])
    sigs = fc.yahoo_signals(ticker, hist, _aux(), weights)
    g = fc.garch_fit(hist["Close"])
    rows = []
    for name, sp in specs.items():
        cfg = mc.SimConfig(n_sims=n_sims, horizon_days=days, drift_mode=sp["drift_mode"],
                           vol_model=sp["vol_model"], strength=sp["strength"], ff_score=0.0, bd_score=0.0,
                           w_ff=0.0, w_bd=0.0, extra=sigs["extra"] if sp["use_signals"] else (), seed=7 + j)
        res = mc.simulate(hist["Close"], cfg, garch=g if sp["vol_model"] == "garch" else None)
        end = res.paths[:, days]
        ret = end / s0 - 1
        var95 = float(-np.percentile(ret, 5))
        pit = float((end <= actual).mean())
        best = mc.suggest_tp_sl_rr(res, days, lo, hi, objective="edge")
        outcome, exit_px = ("-", np.nan)
        if np.isfinite(best["tp"]):
            outcome, exit_px = realized_touch(future, best["tp"], best["sl"])
        rows.append({
            "Tanggal": hist.index[-1], "Target": future.index[-1], "Model": name,
            "Harga awal": s0, "Harga aktual": actual, "Return aktual": actual / s0 - 1,
            "P5": float(np.percentile(end, 5)), "P25": float(np.percentile(end, 25)),
            "Median": float(np.median(end)), "P75": float(np.percentile(end, 75)),
            "P95": float(np.percentile(end, 95)), "PIT": pit,
            "Dalam pita 90%": 0.05 <= pit <= 0.95, "Dalam pita 50%": 0.25 <= pit <= 0.75,
            "VaR95": var95, "Langgar VaR": (actual / s0 - 1) < -var95,
            "P(naik)": float((end > s0).mean()), "Naik": actual > s0,
            "TP": best["tp"], "SL": best["sl"], "R:R": best["rr"], "P(TP dulu)": best["p_tp"],
            "Hasil": outcome, "TP dulu": outcome == "TP",
            "Return strategi": (exit_px / s0 - 1) if np.isfinite(exit_px) else np.nan,
            "Skor gabungan": res.combined,
        })
    return rows


def run_backtest(ticker: str, hname: str, n_dates: int, step: int, n_sims: int, strengths: tuple,
                 w_items: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = mc.HORIZONS[hname]
    px = md.load_ohlcv(ticker, "5y")
    if px is None or len(px) < 260 + days:
        raise ValueError("Data historis kurang (butuh minimal ±1 tahun + horizon).")
    specs = model_specs(list(strengths), dict(w_items))
    last_start = len(px) - 1 - days
    starts = list(range(last_start, 250, -step))[:n_dates][::-1]
    rows = []
    bar = st.progress(0.0, text="Backtest berjalan...")
    for j, i in enumerate(starts):
        rows.extend(_rows_for_date(ticker, hname, i, j, n_sims, strengths, w_items))
        bar.progress((j + 1) / len(starts), text=f"Tanggal uji {j + 1}/{len(starts)} · {px.index[i]:%d %b %Y}")
    bar.empty()
    det = pd.DataFrame(rows)

    summ = []
    base = det[det["Model"] == next(iter(specs))]
    b_dir = float(((base["P(naik)"] - base["Naik"]) ** 2).mean())
    b_tp = float(((base["P(TP dulu)"] - base["TP dulu"]) ** 2).mean())
    for name in specs:
        d = det[det["Model"] == name]
        n = len(d)
        bdir = float(((d["P(naik)"] - d["Naik"]) ** 2).mean())
        btp = float(((d["P(TP dulu)"] - d["TP dulu"]) ** 2).mean())
        summ.append({
            "Model": name, "Jumlah uji": n,
            "Cakupan pita 90%": d["Dalam pita 90%"].mean(), "Cakupan pita 50%": d["Dalam pita 50%"].mean(),
            "Pelanggaran VaR95": d["Langgar VaR"].mean(),
            "Brier arah": bdir, "Skill arah vs dasar": 1 - bdir / b_dir if b_dir > 0 else np.nan,
            "Rata-rata P(TP dulu)": d["P(TP dulu)"].mean(), "TP dulu aktual": d["TP dulu"].mean(),
            "Brier TP": btp, "Skill TP vs dasar": 1 - btp / b_tp if b_tp > 0 else np.nan,
            "Rata-rata return strategi": d["Return strategi"].mean(),
        })
    return pd.DataFrame(summ), det


def _verdict(row: pd.Series, n: int) -> list[str]:
    tol90 = max(0.05, 1.96 * np.sqrt(0.9 * 0.1 / max(n, 1)))
    tol50 = max(0.08, 1.96 * np.sqrt(0.25 / max(n, 1)))
    tolv = max(0.03, 1.96 * np.sqrt(0.05 * 0.95 / max(n, 1)))
    out = []
    out.append("✓" if abs(row["Cakupan pita 90%"] - 0.9) <= tol90 else "✗")
    out.append("✓" if abs(row["Cakupan pita 50%"] - 0.5) <= tol50 else "✗")
    out.append("✓" if abs(row["Pelanggaran VaR95"] - 0.05) <= tolv else "✗")
    out.append("✓" if abs(row["Rata-rata P(TP dulu)"] - row["TP dulu aktual"]) <= tol50 else "✗")
    return out


def render(fmt_pct, colors: dict):
    st.title("Backtest walk-forward · seberapa jujur peluangnya?")
    st.caption("Di setiap tanggal uji di masa lalu, model hanya melihat data sampai tanggal itu, lalu prediksinya "
               "dibandingkan dengan harga yang benar-benar terjadi. Data: Yahoo Finance. Foreign flow & broker "
               "summary riil tidak ikut diuji karena riwayatnya tidak tersedia gratis; bandarmology memakai proxy.")
    with st.sidebar:
        st.header("Pengaturan backtest")
        ticker = st.text_input("Kode emiten", "BBCA").upper().strip()
        hname = st.selectbox("Timeframe", list(mc.HORIZONS), index=2)
        n_dates = st.slider("Jumlah tanggal uji", 10, 80, 40, 5)
        step = st.slider("Jarak antar tanggal uji (hari bursa)", 1, 30, DEFAULT_STEP[hname])
        n_sims = st.number_input("Simulasi per tanggal", 10_000, 30_000, 10_000, 5_000)
        strengths = st.multiselect("Kekuatan geser drift yang diuji", [0.05, 0.10, 0.20, 0.30], default=[0.10],
                                   help="Uji beberapa nilai untuk mencari kekuatan yang paling terkalibrasi.")
        st.caption("Bobot sinyal")
        w_items = tuple(sorted({
            "bd": st.slider("Bandarmology proxy", 0.0, 1.0, fc.DEFAULT_WEIGHTS["bd"], 0.05, key="bt_bd"),
            "mom": st.slider("Momentum & RS", 0.0, 1.0, fc.DEFAULT_WEIGHTS["mom"], 0.05, key="bt_mom"),
            "mkt": st.slider("Proxy asing pasar", 0.0, 1.0, fc.DEFAULT_WEIGHTS["mkt"], 0.05, key="bt_mkt"),
            "comm": st.slider("Komoditas", 0.0, 1.0, fc.DEFAULT_WEIGHTS["comm"], 0.05, key="bt_comm"),
        }.items()))

    n_models = 2 + len(strengths)
    if not st.button(f"Jalankan backtest {ticker} · {hname} ({n_dates} tanggal × {n_models} model)", type="primary") \
            and st.session_state.get("bt_key") != (ticker, hname):
        st.info("Klik tombol untuk mulai. Timeframe 1D–1M butuh sekitar 1 menit; 3M–6M bisa 2–4 menit. "
                "Hasil disimpan 6 jam.")
        return
    st.session_state["bt_key"] = (ticker, hname)

    try:
        summ, det = run_backtest(ticker, hname, n_dates, step, int(n_sims), tuple(sorted(strengths)), w_items)
    except Exception as e:  # noqa: BLE001
        st.error(f"Backtest gagal: {e}")
        return

    st.subheader("Rapor model")
    n = int(summ["Jumlah uji"].iloc[0])
    checks = summ.apply(lambda r: _verdict(r, n), axis=1, result_type="expand")
    checks.columns = ["Pita 90%", "Pita 50%", "VaR 95%", "Kalibrasi TP"]
    rapor = pd.concat([summ[["Model"]], checks], axis=1).set_index("Model")
    st.dataframe(rapor, width="stretch")
    st.caption("✓ = dalam batas wajar untuk jumlah uji ini. Target: pita 90% ≈ 90%, pita 50% ≈ 50%, "
               "pelanggaran VaR ≈ 5%, rata-rata P(TP dulu) ≈ frekuensi TP dulu yang terjadi.")

    pct_cols = ["Cakupan pita 90%", "Cakupan pita 50%", "Pelanggaran VaR95", "Skill arah vs dasar",
                "Rata-rata P(TP dulu)", "TP dulu aktual", "Skill TP vs dasar", "Rata-rata return strategi"]
    st.dataframe(summ.set_index("Model").style.format({**{c: "{:.1%}" for c in pct_cols}, "Brier arah": "{:.4f}",
                                                       "Brier TP": "{:.4f}"}, na_rep="-"), width="stretch")
    st.caption("Skill > 0% berarti model lebih baik daripada random walk polos dalam menebak arah / TP lebih dulu. "
               "Skill ≤ 0% berarti sinyal belum menambah nilai pada saham & timeframe ini.")

    main = st.selectbox("Model untuk grafik", list(summ["Model"]), index=len(summ) - 1)
    d = det[det["Model"] == main].sort_values("Target")

    f = go.Figure()
    f.add_trace(go.Scatter(x=d["Target"], y=d["P95"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    f.add_trace(go.Scatter(x=d["Target"], y=d["P5"], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                           line=dict(width=0), name="Pita prediksi 5–95%"))
    f.add_trace(go.Scatter(x=d["Target"], y=d["Median"], mode="lines+markers", name="Median prediksi",
                           line=dict(color=colors["blue"], width=2), marker=dict(size=6)))
    f.add_trace(go.Scatter(x=d["Target"], y=d["Harga aktual"], mode="markers", name="Harga aktual",
                           marker=dict(size=9, color=np.where(d["Dalam pita 90%"], colors["tp"], colors["sl"]),
                                       line=dict(width=1, color="white"))))
    f.update_layout(title=f"Prediksi {hname} vs harga aktual pada tanggal target", height=420,
                    hovermode="x unified", legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(f, width="stretch")
    st.caption("Titik hijau = harga aktual di dalam pita 90%, merah = di luar.")

    c1, c2 = st.columns(2)
    with c1:
        hist, edges = np.histogram(d["PIT"], bins=10, range=(0, 1))
        h = go.Figure(go.Bar(x=[f"{int(e * 100)}–{int(e * 100) + 10}%" for e in edges[:-1]], y=hist / max(len(d), 1),
                             marker_color=colors["blue"], hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        h.add_hline(y=0.1, line=dict(color=colors["gray"], dash="dash"), annotation_text="ideal 10%")
        h.update_layout(title="Posisi harga aktual di dalam sebaran prediksi", yaxis_tickformat=".0%", height=360,
                        xaxis_title="Persentil", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(h, width="stretch")
        st.caption("Rata = terkalibrasi. Menumpuk di ujung = model terlalu yakin (pita terlalu sempit). "
                   "Menumpuk di kanan = model terlalu pesimis.")
    with c2:
        b = pd.cut(d["P(TP dulu)"], bins=[0, 0.2, 0.35, 0.5, 0.65, 1.0], include_lowest=True)
        rel = d.groupby(b, observed=True).agg(pred=("P(TP dulu)", "mean"), obs=("TP dulu", "mean"),
                                               n=("TP dulu", "size")).dropna()
        r = go.Figure()
        r.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=colors["gray"], dash="dash"),
                               name="Sempurna"))
        r.add_trace(go.Scatter(x=rel["pred"], y=rel["obs"], mode="markers+lines", name="Model",
                               marker=dict(size=np.clip(rel["n"] * 2, 8, 30), color=colors["blue"]),
                               customdata=rel["n"],
                               hovertemplate="Prediksi %{x:.0%} → terjadi %{y:.0%} (n=%{customdata})<extra></extra>"))
        r.update_layout(title="Kalibrasi P(TP lebih dulu)", xaxis_title="Peluang yang diprediksi",
                        yaxis_title="Frekuensi aktual", xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=360,
                        xaxis_range=[0, 1], yaxis_range=[0, 1], legend=dict(orientation="h", y=-0.25),
                        margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(r, width="stretch")
        st.caption("Titik dekat garis putus-putus = peluang TP yang ditampilkan dashboard bisa dipercaya.")

    with st.expander("Detail setiap tanggal uji"):
        show = d[["Tanggal", "Target", "Harga awal", "Harga aktual", "P5", "Median", "P95", "PIT", "VaR95",
                  "Langgar VaR", "TP", "SL", "P(TP dulu)", "Hasil", "Return strategi", "Skor gabungan"]]
        st.dataframe(show.style.format({"Harga awal": "{:,.0f}", "Harga aktual": "{:,.0f}", "P5": "{:,.0f}",
                                        "Median": "{:,.0f}", "P95": "{:,.0f}", "PIT": "{:.0%}", "VaR95": "{:.1%}",
                                        "TP": "{:,.0f}", "SL": "{:,.0f}", "P(TP dulu)": "{:.0%}",
                                        "Return strategi": "{:+.2%}", "Skor gabungan": "{:+.2f}",
                                        "Tanggal": "{:%d %b %Y}", "Target": "{:%d %b %Y}"}, na_rep="-"),
                     width="stretch", hide_index=True)
    st.download_button("Unduh detail (CSV)", det.to_csv(index=False).encode(),
                       file_name=f"backtest_{ticker}_{hname}.csv", mime="text/csv")
    st.info(f"Catatan: tanggal uji berjarak {step} hari bursa, sedangkan horizon {mc.HORIZONS[hname]} hari, sehingga "
            "periode uji bisa tumpang tindih dan tidak sepenuhnya independen. Untuk keyakinan lebih, ulangi di "
            "beberapa saham. Satu saham yang lolos belum membuktikan model berlaku untuk semua saham.")
