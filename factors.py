"""
Faktor tambahan yang datanya gratis dari Yahoo Finance:

1. GARCH(1,1)            -> jalur volatilitas ke depan + rezim volatilitas
2. Momentum & RS         -> tren MA, momentum 6 bulan, kekuatan relatif vs IHSG (^JKSE)
3. Proxy asing pasar     -> ETF EIDO (iShares MSCI Indonesia) + kurs USD/IDR
4. Komoditas             -> tren komoditas terkait x korelasi saham-komoditas

Semua skor sinyal berada di rentang [-1, +1]. Setiap fungsi hanya memakai data
sampai indeks terakhir series yang diberikan, sehingga aman untuk backtest walk-forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

# ----------------------------------------------------------------------------
# Simbol Yahoo
# ----------------------------------------------------------------------------
IHSG = "^JKSE"
EIDO = "EIDO"
USDIDR = "IDR=X"

COMMODITIES = {
    "coal": ("MTF=F", "Batu bara (API2 Rotterdam)"),
    "gold": ("GC=F", "Emas"),
    "copper": ("HG=F", "Tembaga"),
    "oil": ("BZ=F", "Minyak Brent"),
}
# Nikel (LME) dan CPO tidak tersedia andal di Yahoo Finance, jadi tidak dipetakan.
COMMODITY_MAP = {
    **{t: ["coal"] for t in ("ADRO", "AADI", "PTBA", "ITMG", "BUMI", "INDY", "HRUM", "BSSR", "GEMS", "ADMR",
                             "DOID", "ABMM", "TOBA")},
    **{t: ["gold"] for t in ("BRMS", "PSAB", "ARCI", "ANTM")},
    "MDKA": ["gold", "copper"],
    "AMMN": ["copper", "gold"],
    **{t: ["oil"] for t in ("MEDC", "ENRG", "ELSA", "AKRA", "PGAS", "RAJA")},
}
NOT_AVAILABLE = {
    "nikel": ("INCO", "MBMA", "NCKL", "ANTM", "HRUM"),
    "CPO": ("AALI", "LSIP", "SSMS", "TAPG", "DSNG", "SIMP"),
}


def _clip(x: float) -> float:
    return float(np.clip(x, -1, 1)) if np.isfinite(x) else 0.0


def _ret(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def _vol(s: pd.Series, n: int = 126) -> float:
    lr = np.log(s.dropna()).diff().dropna().tail(n)
    return float(lr.std()) if len(lr) > 10 else np.nan


def weekly_corr(a: pd.Series, b: pd.Series, weeks: int = 52) -> float:
    """Korelasi return mingguan (meredam beda jam bursa IDX vs AS)."""
    ra = a.dropna().resample("W-FRI").last().pct_change()
    rb = b.dropna().resample("W-FRI").last().pct_change()
    df = pd.concat([ra, rb], axis=1).dropna().tail(weeks)
    if len(df) < 12:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


# ----------------------------------------------------------------------------
# 1. GARCH(1,1)
# ----------------------------------------------------------------------------
def _garch_var(params, r2, var0):
    omega, alpha, beta = params
    # sigma2_t = omega + alpha*r2_{t-1} + beta*sigma2_{t-1}
    x = np.concatenate([[var0], omega + alpha * r2[:-1]])
    return lfilter([1.0], [1.0, -beta], x)


def garch_fit(close: pd.Series, max_obs: int = 1000) -> dict:
    """Estimasi GARCH(1,1) Gaussian QMLE pada log-return (dalam %)."""
    lr = np.log(close.dropna()).diff().dropna().tail(max_obs)
    r = (lr - lr.mean()).to_numpy() * 100.0
    r2 = r * r
    var0 = float(r.var())

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.995:
            return 1e12
        v = _garch_var(p, r2, var0)
        if np.any(v <= 0):
            return 1e12
        return 0.5 * np.sum(np.log(v) + r2 / v)

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95)):
        x0 = (var0 * (1 - a0 - b0), a0, b0)
        sol = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or sol.fun < best.fun:
            best = sol
    omega, alpha, beta = best.x
    ok = best.fun < 1e11 and omega > 0 and 0 <= alpha and 0 <= beta and alpha + beta < 0.995
    if not ok:  # fallback: vol konstan
        omega, alpha, beta = var0 * 0.05, 0.05, 0.90
    v = _garch_var((omega, alpha, beta), r2, var0)
    next_var = omega + alpha * r2[-1] + beta * v[-1]
    pers = alpha + beta
    long_var = omega / (1 - pers)
    cond_vol = pd.Series(np.sqrt(v) / 100.0, index=lr.index)
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta), "persistence": float(pers),
        "next_var": float(next_var), "long_var": float(long_var), "cond_vol": cond_vol,
        "std_resid": r / np.sqrt(v), "converged": bool(ok),
    }


def garch_sigma_path(g: dict, horizon: int) -> np.ndarray:
    """Volatilitas harian (desimal) untuk h = 1..horizon."""
    h = np.arange(horizon)
    var = g["long_var"] + (g["persistence"] ** h) * (g["next_var"] - g["long_var"])
    return np.sqrt(np.maximum(var, 1e-8)) / 100.0


def vol_regime(g: dict) -> dict:
    cv = g["cond_vol"]
    now = float(np.sqrt(g["next_var"]) / 100.0)
    hist = cv.tail(500)
    pct = float((hist < now).mean()) if len(hist) else 0.5
    label = "Tinggi" if pct >= 0.8 else "Rendah" if pct <= 0.2 else "Normal"
    return {"vol_now": now, "vol_long": float(np.sqrt(g["long_var"]) / 100.0), "percentile": pct, "label": label}


# ----------------------------------------------------------------------------
# 2. Momentum & kekuatan relatif
# ----------------------------------------------------------------------------
def momentum_score(px: pd.DataFrame, ihsg: pd.Series | None = None) -> dict:
    c = px["Close"].dropna()
    last = float(c.iloc[-1])
    checks = []
    mas = {}
    for n in (20, 50, 200):
        if len(c) >= n:
            mas[n] = float(c.tail(n).mean())
            checks.append(last > mas[n])
    if 50 in mas and 200 in mas:
        checks.append(mas[50] > mas[200])
    trend = (np.mean(checks) * 2 - 1) if checks else 0.0

    sig = _vol(c)
    mom = np.nan
    if len(c) > 130 and np.isfinite(sig) and sig > 0:
        r = float(c.iloc[-22] / c.iloc[-127] - 1)  # 6 bulan, tanpa 1 bulan terakhir
        mom = np.tanh(r / (sig * np.sqrt(105)))

    rs = np.nan
    rs_raw = np.nan
    if ihsg is not None and len(ihsg.dropna()) > 70 and np.isfinite(sig) and sig > 0:
        i = ihsg.dropna()
        i = i[i.index <= c.index[-1]]
        rs_raw = _ret(c, 63) - _ret(i, 63)
        rs = np.tanh(rs_raw / (sig * np.sqrt(63)))

    parts = [(trend, 0.35), (mom, 0.35), (rs, 0.30)]
    num = sum(v * w for v, w in parts if np.isfinite(v))
    den = sum(w for v, w in parts if np.isfinite(v))
    return {
        "score": _clip(num / den if den else 0.0),
        "detail": {
            "Skor tren MA": trend, "Skor momentum 6 bulan": mom, "Skor kekuatan relatif vs IHSG": rs,
            "Return 3 bulan − IHSG": rs_raw,
            **{f"MA{n}": v for n, v in mas.items()},
        },
    }


# ----------------------------------------------------------------------------
# 3. Proxy asing tingkat pasar
# ----------------------------------------------------------------------------
def market_foreign_proxy(px: pd.DataFrame, eido: pd.Series | None, usdidr: pd.Series | None) -> dict:
    c = px["Close"].dropna()
    asof = c.index[-1]
    parts, detail = [], {}
    rel = np.nan
    if eido is not None and len(eido.dropna()) > 70:
        e = eido.dropna()
        e = e[e.index <= asof]
        sig = _vol(e)
        r20 = _ret(e, 20)
        e_s = np.tanh(r20 / (sig * np.sqrt(20))) if sig and np.isfinite(r20) else np.nan
        trend = 1.0 if e.iloc[-1] > e.tail(50).mean() else -1.0
        e_score = 0.7 * e_s + 0.3 * trend if np.isfinite(e_s) else trend
        parts.append((e_score, 0.6))
        rel = weekly_corr(c, e)
        detail.update({"EIDO return 20 hari": r20, "Skor EIDO": e_score, "Korelasi mingguan saham-EIDO": rel})
    if usdidr is not None and len(usdidr.dropna()) > 70:
        u = usdidr.dropna()
        u = u[u.index <= asof]
        sig = _vol(u)
        r20 = _ret(u, 20)
        fx = np.tanh(-r20 / (sig * np.sqrt(20))) if sig and np.isfinite(r20) else np.nan  # rupiah menguat = +
        if np.isfinite(fx):
            parts.append((fx, 0.4))
        detail.update({"USD/IDR perubahan 20 hari": r20, "Skor rupiah": fx})
    if not parts:
        return {"score": 0.0, "detail": {"Catatan": "Data EIDO / USD-IDR tidak tersedia"}}
    raw = sum(v * w for v, w in parts) / sum(w for _, w in parts)
    relevance = float(np.clip(rel, 0, 1)) if np.isfinite(rel) else 0.5
    detail["Bobot relevansi (korelasi)"] = relevance
    return {"score": _clip(raw * relevance), "detail": detail}


# ----------------------------------------------------------------------------
# 4. Komoditas
# ----------------------------------------------------------------------------
def commodity_keys(ticker: str) -> list[str]:
    return COMMODITY_MAP.get(ticker.upper(), [])


def commodity_score(px: pd.DataFrame, comm: dict[str, pd.Series]) -> dict:
    """comm: {key: series harga komoditas}."""
    c = px["Close"].dropna()
    asof = c.index[-1]
    scores, detail = [], {}
    for key, s in comm.items():
        if s is None or len(s.dropna()) < 70:
            continue
        s = s.dropna()
        s = s[s.index <= asof]
        sig = _vol(s)
        if not sig or not np.isfinite(sig):
            continue
        r20, r60 = _ret(s, 20), _ret(s, 60)
        trend = 0.5 * np.tanh(r20 / (sig * np.sqrt(20))) + 0.5 * np.tanh(r60 / (sig * np.sqrt(60)))
        corr = weekly_corr(c, s)
        w = float(np.clip(corr, 0, 1)) if np.isfinite(corr) else 0.0
        scores.append(trend * w)
        label = COMMODITIES[key][1]
        detail.update({f"{label}: return 20 hari": r20, f"{label}: return 60 hari": r60,
                       f"{label}: korelasi mingguan": corr, f"{label}: skor": trend * w})
    if not scores:
        return {"score": 0.0, "detail": detail or {"Catatan": "Tidak ada komoditas terkait / data tidak tersedia"},
                "active": False}
    return {"score": _clip(float(np.mean(scores))), "detail": detail, "active": True}


# ----------------------------------------------------------------------------
# Gabungan (dipakai Screener & Backtest)
# ----------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"bd": 0.5, "mom": 0.5, "mkt": 0.3, "comm": 0.4}


def yahoo_signals(ticker: str, px: pd.DataFrame, aux: dict, weights: dict | None = None,
                  sig_window: int = 20) -> dict:
    """
    Semua sinyal yang bisa dihitung dari data gratis Yahoo, as-of tanggal terakhir px.
    aux = {"ihsg": Series|None, "eido": Series|None, "fx": Series|None, "comm": {key: Series}}
    Mengembalikan skor per faktor + tuple `extra` untuk SimConfig (bandar proxy ikut di extra).
    """
    import signals as sg

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    asof = px.index[-1]

    def cut(s):
        return None if s is None else s[s.index <= asof]

    bd = sg.bandar_score_proxy(px, sig_window)["score"] if w["bd"] > 0 else 0.0
    mom = momentum_score(px, cut(aux.get("ihsg")))["score"] if w["mom"] > 0 else 0.0
    mkt = market_foreign_proxy(px, cut(aux.get("eido")), cut(aux.get("fx")))["score"] if w["mkt"] > 0 else 0.0
    comm_series = {k: cut(s) for k, s in (aux.get("comm") or {}).items() if k in commodity_keys(ticker)}
    cres = commodity_score(px, comm_series) if (comm_series and w["comm"] > 0) else {"score": 0.0, "active": False}
    extra = [(bd, w["bd"]), (mom, w["mom"]), (mkt, w["mkt"])]
    if cres.get("active"):
        extra.append((cres["score"], w["comm"]))
    extra = tuple((float(s), float(x)) for s, x in extra if x > 0)
    return {"bd": bd, "mom": mom, "mkt": mkt, "comm": cres["score"] if cres.get("active") else np.nan,
            "extra": extra}
