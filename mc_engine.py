"""
Engine simulasi Monte Carlo untuk saham IDX.

Model: log-return harian
    r_t = mu_t + sigma_t * z_t
dengan
    mu_t    = mu_hist + tilt * sigma * decay(t)
    tilt    = strength * (w_ff * FF + w_bd * BD) / (w_ff + w_bd)
    sigma_t = sigma_hist * (1 + vol_boost * max(0, -combined))   # distribusi -> volatilitas naik
    decay   = 0.5 ** (t / half_life)                              # sinyal memudar
    z_t     : bootstrap residual historis / Student-t / Normal

Tiap langkah dipotong batas Auto Reject (ARA/ARB) IDX dan harga minimum Rp 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZONS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126}
TRADING_DAYS = 245  # perkiraan hari bursa IDX per tahun


# ----------------------------------------------------------------------------
# Aturan pasar IDX
# ----------------------------------------------------------------------------
def tick_size(price: float) -> int:
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


def round_tick(price: float, mode: str = "nearest") -> float:
    if not np.isfinite(price):
        return price
    t = tick_size(price)
    f = {"nearest": np.round, "down": np.floor, "up": np.ceil}[mode]
    return float(max(1, f(price / t) * t))


def ara_limit(price: np.ndarray) -> np.ndarray:
    """Batas atas auto reject (proporsi) berdasarkan fraksi harga."""
    return np.where(price <= 200, 0.35, np.where(price <= 5000, 0.25, 0.20))


def arb_limit(price: np.ndarray, arb: float) -> np.ndarray:
    return np.full_like(price, arb, dtype=float)


# ----------------------------------------------------------------------------
# Konfigurasi & hasil
# ----------------------------------------------------------------------------
@dataclass
class SimConfig:
    n_sims: int = 10_000
    horizon_days: int = 126
    innovation: str = "bootstrap"   # bootstrap | student_t | normal
    t_df: float = 4.0
    use_ewma_vol: bool = True
    ewma_lambda: float = 0.94
    vol_blend: float = 0.5          # bobot EWMA vs vol historis
    ff_score: float = 0.0
    bd_score: float = 0.0
    w_ff: float = 0.5
    w_bd: float = 0.5
    strength: float = 0.10          # tilt drift maksimum, satuan sigma harian
    vol_boost: float = 0.25
    half_life: float = 21.0
    apply_auto_reject: bool = True
    arb: float = 0.15
    drift_mode: str = "historis"    # historis | nol
    seed: int | None = 42
    vol_model: str = "garch"        # garch | ewma
    extra: tuple = ()               # sinyal tambahan: ((skor, bobot), ...)


@dataclass
class SimResult:
    paths: np.ndarray              # (n_sims, horizon_days+1), kolom 0 = harga awal
    s0: float
    mu_hist: float
    sigma_hist: float
    sigma_used: float
    combined: float
    tilt: float
    params: dict = field(default_factory=dict)
    sigma_path: np.ndarray | None = None   # volatilitas harian per langkah


# ----------------------------------------------------------------------------
# Estimasi parameter
# ----------------------------------------------------------------------------
def estimate_params(close: pd.Series, cfg: SimConfig) -> tuple[float, float, np.ndarray]:
    lr = np.log(close / close.shift(1)).dropna()
    lr = lr[np.isfinite(lr)]
    mu = float(lr.mean())
    sig_hist = float(lr.std(ddof=1))
    sigma = sig_hist
    if cfg.use_ewma_vol and len(lr) > 20:
        lam = cfg.ewma_lambda
        var = lr.iloc[:20].var()
        for r in lr.iloc[20:]:
            var = lam * var + (1 - lam) * r * r
        sigma = cfg.vol_blend * float(np.sqrt(var)) + (1 - cfg.vol_blend) * sig_hist
    std_resid = ((lr - mu) / sig_hist).to_numpy()
    return mu, sigma, std_resid


# ----------------------------------------------------------------------------
# Simulasi
# ----------------------------------------------------------------------------
def combine_scores(cfg: SimConfig) -> float:
    pairs = [(cfg.ff_score, cfg.w_ff), (cfg.bd_score, cfg.w_bd), *cfg.extra]
    wsum = sum(w for _, w in pairs)
    return sum(s * w for s, w in pairs) / wsum if wsum > 0 else 0.0


def simulate(close: pd.Series, cfg: SimConfig, garch: dict | None = None) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    s0 = float(close.iloc[-1])
    mu_hist, sigma, resid = estimate_params(close, cfg)
    mu_base = mu_hist if cfg.drift_mode == "historis" else 0.0
    n, T = cfg.n_sims, cfg.horizon_days

    combined = combine_scores(cfg)
    tilt = cfg.strength * combined
    boost = 1.0 + cfg.vol_boost * max(0.0, -combined)

    sig_path = np.full(T, sigma)
    if cfg.vol_model == "garch" and len(close) > 120:
        from factors import garch_fit, garch_sigma_path

        g = garch if garch is not None else garch_fit(close)
        sig_path = garch_sigma_path(g, T)
        resid = g["std_resid"]
    sig_path = sig_path * boost
    sigma_used = float(sig_path[0])

    if cfg.innovation == "bootstrap" and len(resid) > 30:
        z = rng.choice(resid, size=(n, T), replace=True)
        z = (z - z.mean()) / z.std()
    elif cfg.innovation == "student_t":
        nu = max(cfg.t_df, 2.1)
        z = rng.standard_t(nu, size=(n, T)) / np.sqrt(nu / (nu - 2))
    else:
        z = rng.standard_normal(size=(n, T))

    t = np.arange(1, T + 1)
    decay = 0.5 ** (t / cfg.half_life) if cfg.half_life > 0 else np.ones(T)
    mu_t = mu_base + tilt * sig_path * decay
    lr = mu_t[None, :] + sig_path[None, :] * z

    paths = np.empty((n, T + 1))
    paths[:, 0] = s0
    price = np.full(n, s0)
    for k in range(T):
        ret = np.expm1(lr[:, k])
        if cfg.apply_auto_reject:
            ret = np.clip(ret, -arb_limit(price, cfg.arb), ara_limit(price))
        price = np.maximum(price * (1.0 + ret), 1.0)
        paths[:, k + 1] = price

    return SimResult(
        paths=paths, s0=s0, mu_hist=mu_hist, sigma_hist=sigma, sigma_used=sigma_used,
        combined=combined, tilt=tilt,
        params={"mu_base": mu_base, "decay_half_life": cfg.half_life}, sigma_path=sig_path,
    )


# ----------------------------------------------------------------------------
# Statistik risiko
# ----------------------------------------------------------------------------
def horizon_stats(res: SimResult, days: int) -> dict:
    p = res.paths[:, days]
    ret = p / res.s0 - 1.0
    q = np.percentile(p, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    var95 = -np.percentile(ret, 5)
    var99 = -np.percentile(ret, 1)
    cvar95 = -ret[ret <= np.percentile(ret, 5)].mean()
    cvar99 = -ret[ret <= np.percentile(ret, 1)].mean()
    path_min = res.paths[:, 1 : days + 1].min(axis=1)
    path_max = res.paths[:, 1 : days + 1].max(axis=1)
    mdd = 1 - path_min / res.s0
    return {
        "Hari bursa": days,
        "Harga rata-rata": float(p.mean()),
        "Harga median": float(q[4]),
        "P5": float(q[1]), "P10": float(q[2]), "P25": float(q[3]),
        "P75": float(q[5]), "P90": float(q[6]), "P95": float(q[7]),
        "Return harapan": float(ret.mean()),
        "Prob. naik": float((ret > 0).mean()),
        "VaR 95%": float(var95), "CVaR 95%": float(cvar95),
        "VaR 99%": float(var99), "CVaR 99%": float(cvar99),
        "Rata-rata penurunan terdalam": float(mdd.mean()),
        "Prob. turun >10% intra-periode": float((path_min <= res.s0 * 0.9).mean()),
        "Prob. naik >10% intra-periode": float((path_max >= res.s0 * 1.1).mean()),
    }


def first_touch(paths: np.ndarray, s0: float, days: int, tp: float, sl: float) -> dict:
    """Probabilitas TP tersentuh lebih dulu vs SL lebih dulu (berbasis jalur, bukan harga akhir)."""
    seg = paths[:, 1 : days + 1]
    hit_tp = seg >= tp
    hit_sl = seg <= sl
    big = days + 10
    i_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), big)
    i_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), big)
    tp_first = i_tp < i_sl
    sl_first = i_sl < i_tp
    # hari yang sama (gap besar) -> anggap SL dulu (konservatif)
    same = (i_tp == i_sl) & (i_tp < big)
    sl_first |= same
    tp_first &= ~same
    neither = ~(tp_first | sl_first)
    end = paths[:, days]
    pnl = np.where(tp_first, tp / s0 - 1, np.where(sl_first, sl / s0 - 1, end / s0 - 1))
    return {
        "p_tp": float(tp_first.mean()),
        "p_sl": float(sl_first.mean()),
        "p_none": float(neither.mean()),
        "expectancy": float(pnl.mean()),
        "pnl": pnl,
        "days_to_tp": float(np.median(i_tp[tp_first]) + 1) if tp_first.any() else np.nan,
        "days_to_sl": float(np.median(i_sl[sl_first]) + 1) if sl_first.any() else np.nan,
    }


def suggest_tp_sl(res: SimResult, days: int, p_tp_touch: float, p_sl_touch: float) -> dict:
    """
    TP = harga yang disentuh (high intra-periode) oleh p_tp_touch dari seluruh jalur.
    SL = harga yang disentuh (low intra-periode) oleh p_sl_touch dari seluruh jalur.
    Dibulatkan ke fraksi harga IDX.
    """
    seg = res.paths[:, 1 : days + 1]
    pmax, pmin = seg.max(axis=1), seg.min(axis=1)
    tp_raw = float(np.percentile(pmax, 100 * (1 - p_tp_touch)))
    sl_raw = float(np.percentile(pmin, 100 * p_sl_touch))
    tp = round_tick(max(tp_raw, res.s0 + tick_size(res.s0)), "down")
    sl = round_tick(min(sl_raw, res.s0 - tick_size(res.s0)), "up")
    ft = first_touch(res.paths, res.s0, days, tp, sl)
    rr = (tp - res.s0) / (res.s0 - sl) if res.s0 > sl else np.nan
    return {"tp": tp, "sl": sl, "rr": rr, **ft}


def tp_sl_grid(res: SimResult, days: int, tp_pcts: np.ndarray, sl_pcts: np.ndarray) -> pd.DataFrame:
    """Expectancy (return rata-rata strategi) untuk kombinasi TP% dan SL%."""
    seg = res.paths[:, 1 : days + 1]
    run_max = np.maximum.accumulate(seg, axis=1)
    run_min = np.minimum.accumulate(seg, axis=1)
    end = res.paths[:, days] / res.s0 - 1
    big = days + 10
    out = np.empty((len(sl_pcts), len(tp_pcts)))
    i_tp_all = []
    for tpp in tp_pcts:
        lvl = res.s0 * (1 + tpp)
        hit = run_max >= lvl
        i_tp_all.append(np.where(hit[:, -1], hit.argmax(axis=1), big))
    for a, slp in enumerate(sl_pcts):
        lvl = res.s0 * (1 - slp)
        hit = run_min <= lvl
        i_sl = np.where(hit[:, -1], hit.argmax(axis=1), big)
        for b, tpp in enumerate(tp_pcts):
            i_tp = i_tp_all[b]
            tp_first = i_tp < i_sl
            sl_first = (i_sl <= i_tp) & (i_sl < big)
            pnl = np.where(tp_first, tpp, np.where(sl_first, -slp, end))
            out[a, b] = pnl.mean()
    return pd.DataFrame(out, index=[f"{s:.0%}" for s in sl_pcts], columns=[f"{t:.0%}" for t in tp_pcts])


# ----------------------------------------------------------------------------
# TP/SL berbasis target Risk:Reward per timeframe
# ----------------------------------------------------------------------------
# Kaidah umum: scalping (1D) 1:1–1:1,5; swing (1M–3M) 1:2–1:3. 1W dan 6M diinterpolasi.
DEFAULT_RR = {
    "1D": (1.0, 1.5),
    "1W": (1.5, 2.0),
    "1M": (2.0, 3.0),
    "3M": (2.0, 3.0),
    "6M": (2.5, 3.5),
}


def suggest_tp_sl_rr(res: SimResult, days: int, rr_lo: float, rr_hi: float, rr_step: float = 0.25,
                     sl_probs: np.ndarray | None = None, objective: str = "edge",
                     min_resolve: float = 0.5) -> dict:
    """
    Cari pasangan TP/SL yang R:R-nya berada di rentang [rr_lo, rr_hi] dan paling baik menurut simulasi.

    1. Kandidat SL = harga terendah intra-periode yang disentuh oleh 5%..50% jalur
       (SL dekat = sering tersentuh, SL jauh = jarang tersentuh).
    2. Untuk tiap SL, TP = harga masuk + R × (harga masuk − SL), R dari rr_lo s/d rr_hi.
    3. Setiap kombinasi diuji pada SEMUA jalur: mana yang tersentuh lebih dulu (TP atau SL),
       sisanya keluar di harga akhir periode.
    4. Pemilihan (objective):
       - "edge"       : win rate (TP lebih dulu ÷ yang tersentuh) dikurangi win rate impas 1/(1+R).
                        Hanya kombinasi yang selesai (TP/SL tersentuh) di ≥ min_resolve jalur.
                        Bila tidak ada edge positif, dipilih peluang TP tertinggi (paling realistis).
       - "expectancy" : return rata-rata strategi tertinggi.
       - "p_tp"       : peluang TP lebih dulu tertinggi.
    """
    s0 = res.s0
    seg = res.paths[:, 1 : days + 1]
    run_max = np.maximum.accumulate(seg, axis=1)
    run_min = np.minimum.accumulate(seg, axis=1)
    end_ret = res.paths[:, days] / s0 - 1
    big = days + 10
    if sl_probs is None:
        sl_probs = np.arange(0.05, 0.501, 0.05)
    rr_vals = np.round(np.arange(rr_lo, rr_hi + 1e-9, rr_step), 4)
    if len(rr_vals) == 0:
        rr_vals = np.array([rr_lo])

    path_min = run_min[:, -1]
    tick = tick_size(s0)
    rows, seen = [], set()
    for ps in sl_probs:
        sl = round_tick(min(float(np.percentile(path_min, 100 * ps)), s0 - tick), "up")
        if sl >= s0 or sl < 1:
            continue
        hit_sl = run_min <= sl
        i_sl = np.where(hit_sl[:, -1], hit_sl.argmax(axis=1), big)
        risk = s0 - sl
        tps = set()
        for rr in rr_vals:
            for mode in ("down", "up"):
                tp = round_tick(s0 + rr * risk, mode)
                if tp > s0 and rr_lo - 0.05 <= (tp - s0) / risk <= rr_hi + 0.05:
                    tps.add(tp)
        for tp in sorted(tps):
            if (sl, tp) in seen:
                continue
            seen.add((sl, tp))
            hit_tp = run_max >= tp
            i_tp = np.where(hit_tp[:, -1], hit_tp.argmax(axis=1), big)
            tp_first = i_tp < i_sl
            sl_first = (i_sl <= i_tp) & (i_sl < big)  # hari sama -> SL dulu (konservatif)
            pnl = np.where(tp_first, tp / s0 - 1, np.where(sl_first, sl / s0 - 1, end_ret))
            p_tp, p_sl = float(tp_first.mean()), float(sl_first.mean())
            decided = p_tp + p_sl
            rr_real = (tp - s0) / risk
            rows.append({
                "tp": tp, "sl": sl, "rr": rr_real,
                "p_tp": p_tp, "p_sl": p_sl, "p_none": 1 - decided,
                "win_rate": p_tp / decided if decided > 0 else np.nan,
                "breakeven_wr": 1 / (1 + rr_real),
                "expectancy": float(pnl.mean()),
                "days_to_tp": float(np.median(i_tp[tp_first]) + 1) if tp_first.any() else np.nan,
                "days_to_sl": float(np.median(i_sl[sl_first]) + 1) if sl_first.any() else np.nan,
            })
    cand = pd.DataFrame(rows)
    if cand.empty:
        return {"tp": np.nan, "sl": np.nan, "rr": np.nan, "p_tp": np.nan, "p_sl": np.nan, "p_none": np.nan,
                "win_rate": np.nan, "breakeven_wr": np.nan, "expectancy": np.nan, "days_to_tp": np.nan,
                "days_to_sl": np.nan, "edge": np.nan, "resolve": np.nan, "candidates": cand,
                "rr_band": (rr_lo, rr_hi)}
    cand["edge"] = cand["win_rate"] - cand["breakeven_wr"]
    cand["resolve"] = cand["p_tp"] + cand["p_sl"]
    key = {"p_tp": ["p_tp", "expectancy"], "expectancy": ["expectancy", "p_tp"]}.get(objective, ["edge", "p_tp"])
    ok = cand["resolve"] >= min_resolve
    if objective == "edge" and not (cand["edge"] > 0).any():
        # tidak ada R:R yang menguntungkan: tampilkan yang paling realistis (peluang TP tertinggi)
        cand = cand.sort_values(["p_tp", "edge"], ascending=False)
    elif objective == "edge" and ok.any():
        cand = pd.concat([cand[ok].sort_values(key, ascending=False),
                          cand[~ok].sort_values(key, ascending=False)])
    else:
        cand = cand.sort_values(key, ascending=False)
    cand = cand.reset_index(drop=True)
    best = cand.iloc[0].to_dict()
    best["candidates"] = cand
    best["rr_band"] = (rr_lo, rr_hi)
    return best
