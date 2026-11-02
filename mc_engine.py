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
def simulate(close: pd.Series, cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    s0 = float(close.iloc[-1])
    mu_hist, sigma, resid = estimate_params(close, cfg)
    mu_base = mu_hist if cfg.drift_mode == "historis" else 0.0

    wsum = cfg.w_ff + cfg.w_bd
    combined = (cfg.w_ff * cfg.ff_score + cfg.w_bd * cfg.bd_score) / wsum if wsum > 0 else 0.0
    tilt = cfg.strength * combined
    sigma_used = sigma * (1.0 + cfg.vol_boost * max(0.0, -combined))

    n, T = cfg.n_sims, cfg.horizon_days
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
    mu_t = mu_base + tilt * sigma_used * decay
    lr = mu_t[None, :] + sigma_used * z

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
        params={"mu_base": mu_base, "decay_half_life": cfg.half_life},
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
