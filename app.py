"""
Dashboard Simulasi Monte Carlo Saham IDX
dengan penyesuaian Foreign Flow & Bandarmology.

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import hmac
import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_provider as dp
import mc_engine as mc
import signals as sg

st.set_page_config(page_title="Monte Carlo IDX", page_icon="📈", layout="wide")

# Palet (tervalidasi untuk buta warna): biru = simulasi, aqua = TP, merah = SL
C_BLUE = "#2a78d6"
C_TP = "#1baf7a"
C_SL = "#e34948"
C_GRAY = "#8a8985"
C_ORANGE = "#eb6834"


# ----------------------------------------------------------------------------
# Data harga
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_yf(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    sym = ticker.upper().strip()
    if not sym.endswith(".JK"):
        sym += ".JK"
    df = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Data {sym} kosong. Cek kode emiten.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # buang hari tanpa transaksi (suspensi) supaya volatilitas tidak bias
    df = df[df["Volume"] > 0]
    return df


def load_price_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().capitalize() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError("CSV harga wajib punya kolom Date, Open, High, Low, Close, Volume.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


@st.cache_data(show_spinner=False)
def run_sim(close_values: tuple, close_index: tuple, cfg_dict: dict):
    close = pd.Series(close_values, index=pd.to_datetime(list(close_index)))
    return mc.simulate(close, mc.SimConfig(**cfg_dict))


@st.cache_data(ttl=60, show_spinner=False)
def ia_usage(key: str) -> dict:
    return dp.IndexAlphaClient(key).usage()


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def ia_ohlcv(key: str, ticker: str, d_from: date, d_to: date) -> pd.DataFrame:
    return dp.IndexAlphaClient(key).ohlcv(ticker, d_from, d_to)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def ia_foreign(key: str, ticker: str, days: tuple, lengkap: bool):
    c = dp.IndexAlphaClient(key)
    if lengkap:
        df = c.foreign_flow_daily(ticker, list(days))
        return df[(df["foreign_buy"] + df["foreign_sell"]) > 0].reset_index(drop=True)
    return c.foreign_flow(ticker, days[0], days[-1])


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def ia_broker(key: str, ticker: str, days: tuple, lengkap: bool) -> pd.DataFrame:
    c = dp.IndexAlphaClient(key)
    if lengkap:
        return c.broker_summary_daily(ticker, list(days))
    return c.broker_summary(ticker, days[0], days[-1])


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001
        return ""


def detail_df(d: dict) -> pd.DataFrame:
    def f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        if not np.isfinite(v):
            return "-"
        return f"{v:,.0f}" if abs(v) >= 1000 or v == int(v) else f"{v:.4f}"
    return pd.DataFrame({"Nilai": [f(v) for v in d.values()]}, index=list(d.keys()))


def fmt_rp(x: float) -> str:
    return "-" if not np.isfinite(x) else f"Rp {x:,.0f}".replace(",", ".")


def fmt_pct(x: float, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "-"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def score_label(s: float) -> str:
    if s >= 0.5:
        return "Akumulasi kuat"
    if s >= 0.15:
        return "Akumulasi"
    if s > -0.15:
        return "Netral"
    if s > -0.5:
        return "Distribusi"
    return "Distribusi kuat"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
_app_pw = _secret("APP_PASSWORD")
if _app_pw and not st.session_state.get("auth_ok"):
    st.title("Monte Carlo IDX")
    pw = st.text_input("Kata sandi", type="password")
    if pw and hmac.compare_digest(pw, _app_pw):
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("Kata sandi salah.")
    st.stop()

with st.sidebar:
    st.header("Emiten")
    ticker = st.text_input("Kode emiten", value="BBCA", help="Tanpa .JK, contoh: BBCA, TLKM, GOTO").upper().strip()
    lookback = st.selectbox("Data historis untuk estimasi", ["6mo", "1y", "2y", "3y", "5y"], index=2)
    price_csv = st.file_uploader("Atau unggah CSV harga (opsional)", type="csv",
                                 help="Kolom: Date, Open, High, Low, Close, Volume. Dipakai bila Yahoo gagal.")

    st.header("Data otomatis")
    ia_key = dp.get_api_key()
    if not ia_key:
        ia_key = st.text_input("API key Index Alpha", type="password",
                               help="Lebih aman disimpan di Secrets Streamlit sebagai INDEXALPHA_API_KEY.")
    ia_plan = None
    if ia_key:
        try:
            _u = ia_usage(ia_key)["account"]
            ia_plan = _u.get("plan")
            st.caption(f"Index Alpha · paket {ia_plan} · sisa kuota {_u.get('remaining', 0):,} request")
        except Exception as e:  # noqa: BLE001
            st.caption(f"Index Alpha: {e}")
        ia_mode = st.radio(
            "Mode pengambilan", ["Lengkap (per hari)", "Hemat (2 request)"],
            index=1 if (ia_plan or "").lower() == "free" else 0,
            help="Lengkap: 1 request per hari bursa untuk foreign flow dan broker summary (≈2 × jendela sinyal), "
                 "memberi grafik harian. Hemat: 1 request agregat untuk masing-masing (cocok untuk paket gratis).",
        )
    else:
        ia_mode = "Hemat (2 request)"
        st.caption("Isi API key [Index Alpha](https://indexalpha.id) supaya foreign flow & broker summary "
                   "terambil otomatis.")
    ia_lengkap = ia_mode.startswith("Lengkap")

    st.header("Simulasi")
    n_sims = st.number_input("Jumlah simulasi", min_value=10_000, max_value=200_000, value=20_000, step=5_000)
    innovation = st.selectbox(
        "Distribusi guncangan harian",
        ["bootstrap", "student_t", "normal"],
        format_func={"bootstrap": "Bootstrap return historis (fat-tail riil)",
                     "student_t": "Student-t (fat-tail)", "normal": "Normal (GBM klasik)"}.get,
    )
    t_df = st.slider("Derajat bebas Student-t", 3.0, 30.0, 4.0, 0.5) if innovation == "student_t" else 4.0
    drift_mode = st.radio("Drift dasar", ["historis", "nol"], horizontal=True,
                          help="'nol' = tanpa asumsi tren masa lalu; hanya sinyal FF/Bandar yang menggeser drift.")
    use_ewma = st.checkbox("Volatilitas campuran EWMA (lebih responsif)", value=True)
    auto_reject = st.checkbox("Terapkan batas ARA/ARB IDX", value=True)
    arb = st.slider("Batas ARB", 0.05, 0.35, 0.15, 0.01, disabled=not auto_reject)
    seed = st.number_input("Seed acak (0 = acak tiap run)", value=42, step=1)

    st.header("Foreign Flow")
    _ff_opts = (["Otomatis (Index Alpha)"] if ia_key else []) + ["CSV data riil", "Input manual", "Abaikan (netral)"]
    ff_source = st.radio("Sumber", _ff_opts, index=0 if ia_key else 2, key="ff_src")
    ff_file = None
    ff_manual = 0.0
    if ff_source == "CSV data riil":
        ff_file = st.file_uploader("CSV foreign flow", type="csv", key="ff",
                                   help="Kolom: date, net_foreign (Rp)  atau  date, foreign_buy, foreign_sell")
    elif ff_source == "Input manual":
        ff_manual = st.slider("Skor foreign flow manual", -1.0, 1.0, 0.0, 0.05,
                              help="-1 = net sell asing besar & konsisten, +1 = net buy asing besar & konsisten")

    st.header("Bandarmology")
    _bd_opts = (["Otomatis (Index Alpha)"] if ia_key else []) + [
        "Proxy OHLCV (CMF, A/D, OBV)", "CSV broker summary", "Input manual", "Abaikan (netral)"]
    bd_source = st.radio("Sumber", _bd_opts, index=0, key="bd_src")
    bd_file = None
    bd_manual = 0.0
    if bd_source == "CSV broker summary":
        bd_file = st.file_uploader("CSV broker summary", type="csv", key="bd",
                                   help="Kolom: date, broker, buy_lot, sell_lot [, buy_value, sell_value]")
    elif bd_source == "Input manual":
        bd_manual = st.slider("Skor bandarmology manual", -1.0, 1.0, 0.0, 0.05)

    sig_window = st.slider("Jendela sinyal (hari bursa)", 5, 60, 20)

    st.header("Pengaruh sinyal")
    w_ff = st.slider("Bobot Foreign Flow", 0.0, 1.0, 0.5, 0.05)
    w_bd = st.slider("Bobot Bandarmology", 0.0, 1.0, 0.5, 0.05)
    strength = st.slider("Kekuatan geser drift (× σ harian)", 0.0, 0.5, 0.10, 0.01,
                         help="Skor gabungan +1 menambah drift harian sebesar nilai ini × volatilitas harian.")
    vol_boost = st.slider("Tambahan volatilitas saat distribusi", 0.0, 1.0, 0.25, 0.05)
    half_life = st.slider("Waktu paruh sinyal (hari)", 5, 126, 21,
                          help="Pengaruh sinyal memudar setengahnya tiap N hari bursa.")

    st.header("Target TP / SL")
    p_tp = st.slider("Peluang TP tersentuh", 0.10, 0.90, 0.40, 0.05)
    p_sl = st.slider("Peluang SL tersentuh", 0.05, 0.50, 0.20, 0.05)

# ----------------------------------------------------------------------------
# Ambil data
# ----------------------------------------------------------------------------
st.title(f"Simulasi Monte Carlo · {ticker}")

try:
    if price_csv is not None:
        px = load_price_csv(price_csv)
        price_src = "CSV unggahan"
    else:
        try:
            with st.spinner(f"Mengambil data {ticker}.JK dari Yahoo Finance..."):
                px = load_yf(ticker, lookback)
            price_src = "Yahoo Finance (yfinance)"
        except Exception:  # noqa: BLE001
            if not ia_key:
                raise
            with st.spinner(f"Yahoo gagal, mengambil OHLCV {ticker} dari Index Alpha..."):
                px = ia_ohlcv(ia_key, ticker, dp.MIN_DATE, date.today())
            price_src = "Index Alpha (data sejak 2025)"
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal mengambil data harga: {e}\n\nCoba unggah CSV harga di sidebar.")
    st.stop()

if len(px) < 60:
    st.error("Data historis kurang dari 60 hari bursa; hasil simulasi tidak akan andal.")
    st.stop()

last_price = float(px["Close"].iloc[-1])

# ----------------------------------------------------------------------------
# Hitung sinyal
# ----------------------------------------------------------------------------
sig_days = tuple(d.date() for d in px.index[-sig_window:] if d.date() >= dp.MIN_DATE)
bs_daily = None

ff_res = {"score": 0.0, "detail": {}}
ff_df = None
ff_tag = "Netral (diabaikan)"
if ff_source.startswith("Otomatis"):
    try:
        with st.spinner("Mengambil foreign flow dari Index Alpha..."):
            ff_raw = ia_foreign(ia_key, ticker, sig_days, ia_lengkap)
        if ia_lengkap:
            if ff_raw.empty:
                raise dp.IndexAlphaError("data foreign flow kosong untuk rentang ini")
            ff_df = ff_raw
            ff_res = sg.foreign_flow_score(ff_df, px, sig_window)
        else:
            ff_res = sg.foreign_flow_score_aggregate(float(ff_raw.get("net_foreign") or 0), px, sig_window)
        ff_tag = f"Data riil otomatis (Index Alpha, {sig_days[0]:%d %b} – {sig_days[-1]:%d %b %Y})"
    except Exception as e:  # noqa: BLE001
        st.warning(f"Foreign flow otomatis gagal: {e}")
elif ff_source == "CSV data riil":
    if ff_file is not None:
        try:
            ff_df = sg.parse_foreign_flow_csv(ff_file)
            ff_res = sg.foreign_flow_score(ff_df, px, sig_window)
            ff_tag = "Data riil (CSV)"
        except Exception as e:  # noqa: BLE001
            st.warning(f"CSV foreign flow tidak terbaca: {e}")
    else:
        st.info("Unggah CSV foreign flow di sidebar; sementara skor asing = 0.")
elif ff_source == "Input manual":
    ff_res = {"score": ff_manual, "detail": {}}
    ff_tag = "Input manual"

bd_res = {"score": 0.0, "detail": {}}
bd_tag = "Netral (diabaikan)"
if bd_source.startswith("Otomatis"):
    try:
        with st.spinner("Mengambil broker summary dari Index Alpha..."):
            bs_raw = ia_broker(ia_key, ticker, sig_days, ia_lengkap)
        if bs_raw.empty:
            raise dp.IndexAlphaError("broker summary kosong untuk rentang ini")
        if ia_lengkap:
            bs_daily = bs_raw
        bd_res = sg.bandar_score_broker(bs_raw, last_price, sig_window)
        bd_tag = f"Data riil otomatis (Index Alpha broker summary, {len(sig_days)} hari bursa)"
    except Exception as e:  # noqa: BLE001
        st.warning(f"Broker summary otomatis gagal: {e}. Memakai proxy OHLCV.")
        bd_res = sg.bandar_score_proxy(px, sig_window)
        bd_tag = "PROXY dari OHLCV (broker summary gagal diambil)"
elif bd_source.startswith("Proxy"):
    bd_res = sg.bandar_score_proxy(px, sig_window)
    bd_tag = "PROXY dari OHLCV"
elif bd_source == "CSV broker summary":
    if bd_file is not None:
        try:
            bd_res = sg.bandar_score_broker(sg.parse_broker_summary_csv(bd_file), last_price, sig_window)
            bd_tag = "Data riil (broker summary)"
        except Exception as e:  # noqa: BLE001
            st.warning(f"CSV broker summary tidak terbaca: {e}")
    else:
        st.info("Unggah CSV broker summary di sidebar; sementara skor bandar = 0.")
elif bd_source == "Input manual":
    bd_res = {"score": bd_manual, "detail": {}}
    bd_tag = "Input manual"

# ----------------------------------------------------------------------------
# Jalankan simulasi
# ----------------------------------------------------------------------------
cfg = dict(
    n_sims=int(n_sims), horizon_days=126, innovation=innovation, t_df=float(t_df),
    use_ewma_vol=use_ewma, ff_score=ff_res["score"], bd_score=bd_res["score"],
    w_ff=w_ff, w_bd=w_bd, strength=strength, vol_boost=vol_boost, half_life=float(half_life),
    apply_auto_reject=auto_reject, arb=arb, drift_mode=drift_mode,
    seed=None if int(seed) == 0 else int(seed),
)
with st.spinner(f"Menjalankan {int(n_sims):,} simulasi × 126 hari bursa..."):
    close = px["Close"]
    res = run_sim(tuple(close.values), tuple(close.index.astype(str)), cfg)

# ----------------------------------------------------------------------------
# Ringkasan atas
# ----------------------------------------------------------------------------
st.caption(f"Data harga: {price_src} · {len(px)} hari bursa · terakhir {px.index[-1]:%d %b %Y} · "
           f"{int(n_sims):,} jalur simulasi")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Harga terakhir", fmt_rp(last_price))
c2.metric("Volatilitas tahunan", fmt_pct(res.sigma_used * np.sqrt(mc.TRADING_DAYS)),
          help=f"σ harian dipakai {res.sigma_used:.2%} (historis+EWMA {res.sigma_hist:.2%})")
c3.metric("Skor Foreign Flow", f"{ff_res['score']:+.2f}", score_label(ff_res["score"]), delta_color="off", delta_arrow="off",
          help=ff_tag)
c4.metric("Skor Bandarmology", f"{bd_res['score']:+.2f}", score_label(bd_res["score"]), delta_color="off", delta_arrow="off",
          help=bd_tag)
c5.metric("Skor gabungan", f"{res.combined:+.2f}",
          f"drift {res.tilt * res.sigma_used * 245:+.1%}/thn awal", delta_color="off", delta_arrow="off",
          help="Tambahan drift tahunan di awal periode sebelum memudar sesuai waktu paruh.")

tag_note = []
if "PROXY" in bd_tag:
    tag_note.append("**Bandarmology memakai PROXY** dari harga & volume (bukan data broker). "
                    "Unggah CSV broker summary untuk analisis bandar yang sesungguhnya.")
if ff_tag.startswith("Netral"):
    tag_note.append("**Foreign flow tidak dipakai.** Unggah CSV net foreign atau isi skor manual.")
if tag_note:
    st.warning("  \n".join(tag_note))

# ----------------------------------------------------------------------------
# Tab per horizon
# ----------------------------------------------------------------------------
tabs = st.tabs([f"{h} · {d} hari" for h, d in mc.HORIZONS.items()] + ["Ringkasan", "Bandarmology & Asing"])
all_stats = {}
all_tpsl = {}

for (hname, days), tab in zip(mc.HORIZONS.items(), tabs):
    stats = mc.horizon_stats(res, days)
    sug = mc.suggest_tp_sl(res, days, p_tp, p_sl)
    all_stats[hname] = stats
    all_tpsl[hname] = sug
    with tab:
        # --- kartu TP/SL
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Take Profit", fmt_rp(sug["tp"]), fmt_pct(sug["tp"] / res.s0 - 1, True))
        k2.metric("Stop Loss", fmt_rp(sug["sl"]), fmt_pct(sug["sl"] / res.s0 - 1, True))
        k3.metric("Risk : Reward", f"1 : {sug['rr']:.2f}" if np.isfinite(sug["rr"]) else "-")
        k4.metric("P(TP lebih dulu)", fmt_pct(sug["p_tp"]))
        k5.metric("P(SL lebih dulu)", fmt_pct(sug["p_sl"]))
        k6.metric("Expectancy", fmt_pct(sug["expectancy"], True),
                  help="Return rata-rata bila keluar di TP/SL, atau di akhir periode bila tidak tersentuh.")

        left, right = st.columns([3, 2])

        # --- fan chart
        with left:
            x = np.arange(days + 1)
            seg = res.paths[:, : days + 1]
            pct = np.percentile(seg, [5, 25, 50, 75, 95], axis=0)
            fig = go.Figure()
            rng = np.random.default_rng(0)
            idx = rng.choice(seg.shape[0], size=min(60, seg.shape[0]), replace=False)
            if days > 1:
                for i in idx:
                    fig.add_trace(go.Scatter(x=x, y=seg[i], mode="lines", line=dict(width=0.6, color=C_GRAY),
                                             opacity=0.25, hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=pct[4], line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=pct[0], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                     line=dict(width=0), name="Rentang 5–95%",
                                     customdata=pct[4], hovertemplate="P5 %{y:,.0f} · P95 %{customdata:,.0f}"))
            fig.add_trace(go.Scatter(x=x, y=pct[3], line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=pct[1], fill="tonexty", fillcolor="rgba(42,120,214,0.30)",
                                     line=dict(width=0), name="Rentang 25–75%",
                                     customdata=pct[3], hovertemplate="P25 %{y:,.0f} · P75 %{customdata:,.0f}"))
            fig.add_trace(go.Scatter(x=x, y=pct[2], line=dict(width=2, color=C_BLUE), name="Median",
                                     hovertemplate="Median %{y:,.0f}"))
            fig.add_hline(y=sug["tp"], line=dict(color=C_TP, width=2, dash="dash"),
                          annotation_text=f"TP {sug['tp']:,.0f}", annotation_position="top left")
            fig.add_hline(y=sug["sl"], line=dict(color=C_SL, width=2, dash="dash"),
                          annotation_text=f"SL {sug['sl']:,.0f}", annotation_position="bottom left")
            fig.add_hline(y=res.s0, line=dict(color=C_GRAY, width=1, dash="dot"))
            fig.update_layout(
                title=f"Fan chart jalur harga {hname}", height=440, hovermode="x unified",
                xaxis_title="Hari bursa ke-", yaxis_title="Harga (Rp)", margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", y=-0.18),
            )
            if days <= 5:
                fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, width="stretch")

        # --- histogram harga akhir
        with right:
            end = res.paths[:, days]
            h = go.Figure()
            h.add_trace(go.Histogram(x=end, nbinsx=80, marker_color=C_BLUE, opacity=0.85, name="Harga akhir",
                                     hovertemplate="Rp %{x:,.0f}<br>%{y} jalur<extra></extra>"))
            var_px = res.s0 * (1 - stats["VaR 95%"])
            for val, col, txt in [(res.s0, C_GRAY, "Harga kini"), (var_px, C_ORANGE, "Batas VaR 95%"),
                                  (sug["tp"], C_TP, "TP"), (sug["sl"], C_SL, "SL")]:
                h.add_vline(x=val, line=dict(color=col, width=2, dash="dash"))
                h.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{txt} {val:,.0f}",
                                       line=dict(color=col, width=2, dash="dash")))
            h.update_layout(title=f"Distribusi harga akhir {hname}", height=440, bargap=0.02,
                            xaxis_title="Harga (Rp)", yaxis_title="Jumlah jalur",
                            legend=dict(orientation="h", y=-0.18),
                            margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(h, width="stretch")

        # --- tabel statistik
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Distribusi harga**")
            st.dataframe(pd.DataFrame({
                "Nilai": [fmt_rp(stats[k]) for k in ("P5", "P10", "P25", "Harga median", "Harga rata-rata",
                                                    "P75", "P90", "P95")],
            }, index=["Persentil 5", "Persentil 10", "Persentil 25", "Median", "Rata-rata",
                      "Persentil 75", "Persentil 90", "Persentil 95"]), width="stretch")
        with s2:
            st.markdown("**Risiko**")
            st.dataframe(pd.DataFrame({
                "Nilai": [fmt_pct(stats["Return harapan"], True), fmt_pct(stats["Prob. naik"]),
                          fmt_pct(stats["VaR 95%"]), fmt_pct(stats["CVaR 95%"]),
                          fmt_pct(stats["VaR 99%"]), fmt_pct(stats["CVaR 99%"]),
                          fmt_pct(stats["Rata-rata penurunan terdalam"]),
                          fmt_pct(stats["Prob. turun >10% intra-periode"]),
                          fmt_pct(stats["Prob. naik >10% intra-periode"])],
            }, index=["Return harapan", "Peluang harga naik", "VaR 95% (rugi)", "CVaR 95% (rugi rata-rata ekor)",
                      "VaR 99%", "CVaR 99%", "Rata-rata titik terendah vs harga kini",
                      "Peluang sempat turun >10%", "Peluang sempat naik >10%"]), width="stretch")

        # --- peta TP/SL
        if days >= 5:
            with st.expander("Peta optimasi TP/SL (expectancy setiap kombinasi)"):
                vol_h = res.sigma_used * np.sqrt(days)
                tp_grid = np.unique(np.round(np.linspace(0.5, 3.0, 8) * vol_h, 3))
                sl_grid = np.unique(np.round(np.linspace(0.3, 2.0, 8) * vol_h, 3))
                tp_grid, sl_grid = tp_grid[tp_grid > 0], sl_grid[(sl_grid > 0) & (sl_grid < 0.9)]
                grid = mc.tp_sl_grid(res, days, tp_grid, sl_grid)
                zmax = float(np.abs(grid.values).max()) or 1e-6
                hm = go.Figure(go.Heatmap(
                    z=grid.values, x=grid.columns, y=grid.index, zmid=0, zmin=-zmax, zmax=zmax,
                    colorscale=[[0, C_SL], [0.5, "#f0efec"], [1, C_TP]],
                    text=np.vectorize(lambda v: f"{v:+.1%}")(grid.values), texttemplate="%{text}",
                    hovertemplate="TP +%{x} · SL −%{y}<br>Expectancy %{z:+.2%}<extra></extra>",
                    colorbar=dict(title="Expectancy", tickformat="+.0%"),
                ))
                hm.update_layout(height=420, xaxis_title="Take Profit (+%)", yaxis_title="Stop Loss (−%)",
                                 margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(hm, width="stretch")
                st.caption("Hijau = kombinasi yang secara rata-rata menguntungkan di seluruh jalur simulasi. "
                           "Expectancy dekat nol berarti TP/SL hanya menggeser bentuk risiko, bukan menambah edge.")

# ----------------------------------------------------------------------------
# Tab ringkasan
# ----------------------------------------------------------------------------
with tabs[len(mc.HORIZONS)]:
    rows = []
    for hname, stt in all_stats.items():
        sug = all_tpsl[hname]
        rows.append({
            "Horizon": hname, "Hari": stt["Hari bursa"],
            "Median": round(stt["Harga median"]), "P5": round(stt["P5"]), "P95": round(stt["P95"]),
            "Return harapan": stt["Return harapan"], "P(naik)": stt["Prob. naik"],
            "VaR95": stt["VaR 95%"], "CVaR95": stt["CVaR 95%"],
            "TP": sug["tp"], "SL": sug["sl"], "R:R": sug["rr"],
            "P(TP dulu)": sug["p_tp"], "P(SL dulu)": sug["p_sl"], "Expectancy": sug["expectancy"],
        })
    summary = pd.DataFrame(rows).set_index("Horizon")
    pct_cols = ["Return harapan", "P(naik)", "VaR95", "CVaR95", "P(TP dulu)", "P(SL dulu)", "Expectancy"]
    st.dataframe(summary.style.format({**{c: "{:.2%}" for c in pct_cols}, "R:R": "{:.2f}",
                                       "Median": "{:,.0f}", "P5": "{:,.0f}", "P95": "{:,.0f}",
                                       "TP": "{:,.0f}", "SL": "{:,.0f}"}),
                 width="stretch")

    # VaR/CVaR antar horizon (bar)
    vb = go.Figure()
    vb.add_trace(go.Bar(x=summary.index, y=summary["VaR95"], name="VaR 95%", marker_color=C_ORANGE,
                        hovertemplate="%{x}: %{y:.2%}<extra>VaR 95%</extra>"))
    vb.add_trace(go.Bar(x=summary.index, y=summary["CVaR95"], name="CVaR 95%", marker_color=C_SL,
                        hovertemplate="%{x}: %{y:.2%}<extra>CVaR 95%</extra>"))
    vb.update_layout(title="Potensi rugi per horizon", yaxis_tickformat=".0%", barmode="group", height=360,
                     bargap=0.3, bargroupgap=0.08, legend=dict(orientation="h", y=-0.2),
                     margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(vb, width="stretch")

    csv_buf = io.StringIO()
    summary.to_csv(csv_buf)
    st.download_button("Unduh ringkasan (CSV)", csv_buf.getvalue(), file_name=f"mc_{ticker}_ringkasan.csv",
                       mime="text/csv")

# ----------------------------------------------------------------------------
# Tab sinyal
# ----------------------------------------------------------------------------
with tabs[len(mc.HORIZONS) + 1]:
    a, b = st.columns(2)
    with a:
        st.subheader("Foreign Flow")
        st.caption(f"Sumber: {ff_tag}")
        if ff_df is not None and not ff_df.empty:
            ffp = ff_df.tail(max(sig_window * 3, 60))
            fg = go.Figure()
            fg.add_trace(go.Bar(x=ffp["date"], y=ffp["net_foreign"] / 1e9, name="Net asing harian",
                                marker_color=np.where(ffp["net_foreign"] >= 0, C_TP, C_SL),
                                hovertemplate="%{x|%d %b}: Rp %{y:,.2f} M<extra></extra>"))
            fg.update_layout(title="Net foreign harian (Rp miliar)", height=320, showlegend=False,
                             margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fg, width="stretch")
            fc = go.Figure(go.Scatter(x=ffp["date"], y=ffp["net_foreign"].cumsum() / 1e9,
                                      line=dict(color=C_BLUE, width=2),
                                      hovertemplate="%{x|%d %b}: Rp %{y:,.2f} M<extra></extra>"))
            fc.update_layout(title="Kumulatif net foreign (Rp miliar)", height=280,
                             margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fc, width="stretch")
        if ff_res["detail"]:
            st.dataframe(detail_df(ff_res["detail"]),
                         width="stretch")
    with b:
        st.subheader("Bandarmology")
        st.caption(f"Sumber: {bd_tag}")
        if "frame" in bd_res:
            fr = bd_res["frame"].tail(120)
            cm = go.Figure(go.Scatter(x=fr.index, y=fr["CMF20"], line=dict(color=C_BLUE, width=2),
                                      hovertemplate="%{x|%d %b}: %{y:.3f}<extra>CMF20</extra>"))
            cm.add_hline(y=0, line=dict(color=C_GRAY, width=1))
            cm.update_layout(title="Chaikin Money Flow 20 hari (proxy akumulasi)", height=300,
                             margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(cm, width="stretch")
            ad = go.Figure(go.Scatter(x=fr.index, y=fr["AD"] / 1e6, line=dict(color=C_BLUE, width=2),
                                      hovertemplate="%{x|%d %b}: %{y:,.1f} jt<extra>A/D</extra>"))
            ad.update_layout(title="Accumulation/Distribution line (juta lembar)", height=280,
                             margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(ad, width="stretch")
        if "table" in bd_res:
            t = bd_res["table"]
            top = pd.concat([t.head(5), t.tail(5)])
            bg = go.Figure(go.Bar(x=top["net_lot"], y=top.index, orientation="h",
                                  marker_color=np.where(top["net_lot"] >= 0, C_TP, C_SL),
                                  hovertemplate="%{y}: %{x:,.0f} lot<extra></extra>"))
            bg.update_layout(title="Top 5 net buyer & net seller (lot)", height=360,
                             yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(bg, width="stretch")
        if bd_res["detail"]:
            st.dataframe(detail_df(bd_res["detail"]),
                         width="stretch")

    if "table" in bd_res and not bd_res["table"].empty:
        st.markdown("---")
        st.subheader("Broker summary")
        t = bd_res["table"].copy()
        tab_bs = pd.DataFrame(index=t.index)
        tab_bs["Beli (lot)"] = t["buy_lot"]
        tab_bs["Jual (lot)"] = t["sell_lot"]
        tab_bs["Net (lot)"] = t["net_lot"]
        if {"buy_value", "sell_value"} <= set(t.columns):
            tab_bs["Net (Rp miliar)"] = (t["buy_value"] - t["sell_value"]) / 1e9
            tab_bs["Avg beli"] = (t["buy_value"] / (t["buy_lot"] * 100)).where(t["buy_lot"] > 0)
            tab_bs["Avg jual"] = (t["sell_value"] / (t["sell_lot"] * 100)).where(t["sell_lot"] > 0)
        tab_bs["% volume"] = t["buy_lot"].add(t["sell_lot"]) / (2 * t["buy_lot"].sum())
        st.dataframe(
            tab_bs.style.format({"Beli (lot)": "{:,.0f}", "Jual (lot)": "{:,.0f}", "Net (lot)": "{:+,.0f}",
                                 "Net (Rp miliar)": "{:+,.2f}", "Avg beli": "{:,.0f}", "Avg jual": "{:,.0f}",
                                 "% volume": "{:.1%}"}, na_rep="-"),
            width="stretch", height=380,
        )

    if bs_daily is not None and not bs_daily.empty:
        tl = dp.bandar_timeline(bs_daily)
        if not tl.empty:
            from plotly.subplots import make_subplots

            pxw = px.loc[px.index >= tl["date"].min()]
            sp = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06,
                               subplot_titles=("Harga penutupan", "Akumulasi (+) / distribusi (−) top-5 broker"))
            sp.add_trace(go.Scatter(x=pxw.index, y=pxw["Close"], line=dict(color=C_BLUE, width=2), name="Close",
                                    hovertemplate="%{x|%d %b}: Rp %{y:,.0f}<extra></extra>"), row=1, col=1)
            sp.add_trace(go.Bar(x=tl["date"], y=tl["top_net_ratio"], name="Top-5 net / total lot",
                                marker_color=np.where(tl["top_net_ratio"] >= 0, C_TP, C_SL),
                                hovertemplate="%{x|%d %b}: %{y:+.1%}<extra></extra>"), row=2, col=1)
            sp.update_yaxes(tickformat="+.0%", row=2, col=1)
            sp.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(sp, width="stretch")

    st.markdown("---")
    st.markdown(
        f"""
**Cara sinyal masuk ke simulasi**

- Skor gabungan = rata-rata berbobot FF dan Bandar = **{res.combined:+.2f}**.
- Drift harian ditambah `skor × {strength:.2f} × σ harian`, lalu memudar setengahnya tiap **{half_life}** hari bursa.
- Bila skor gabungan negatif (distribusi), volatilitas dinaikkan hingga **{vol_boost:.0%}** × |skor|.
- Setiap langkah dipotong batas ARA/ARB dan harga dibulatkan ke fraksi harga IDX saat menentukan TP/SL.
"""
    )

st.caption("Simulasi ini alat bantu manajemen risiko, bukan rekomendasi beli/jual. Hasil sangat bergantung "
           "pada asumsi distribusi return dan kualitas data sinyal.")
