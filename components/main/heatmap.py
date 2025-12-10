import pandas as pd
from datetime import datetime, timedelta
from dash import html
import re, unicodedata
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go

# =========================
# Config
# =========================
VALORES_MAP = {
    "PS_RRC":  ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RRC":  ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB":  ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB":  ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
    "PS_S1":   ("ps_s1_ia_percent", "ps_s1_fail"),
}
SEV_COLORS = {
    "excelente": "#2ecc71",  # verde
    "bueno":     "#f1c40f",  # amarillo
    "regular":   "#e67e22",  # naranja
    "critico":   "#e74c3c",  # rojo
}
SEV_ORDER = ["excelente", "bueno", "regular", "critico"]
# =========================
# Helpers
# =========================
def _infer_networks(df_long: pd.DataFrame) -> list[str]:
    if df_long is None or df_long.empty or "network" not in df_long.columns:
        return []
    return sorted(df_long["network"].dropna().unique().tolist())

def _day_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _safe_hour_to_idx(hhmmss) -> int | None:
    try:
        hh = int(str(hhmmss).split(":")[0])
        if 0 <= hh <= 23:
            return hh
    except Exception:
        pass
    return None

def _empty24():
    return [None]*24

def _max_date_str(series: pd.Series) -> str | None:
    try:
        return max(pd.to_datetime(series).dt.date).strftime("%Y-%m-%d")
    except Exception:
        return None

def _normalize_profile_cfg(cfg: dict, profile: str = "main") -> dict:
    """
    Si cfg viene con estructura completa (version, profiles, colors),
    devuelve cfg['profiles'][profile]. Si ya es un perfil, lo regresa tal cual.
    """
    if not isinstance(cfg, dict):
        return {}

    # Si ya parece un perfil (tiene 'severity' o 'progress'), lo dejamos.
    if "severity" in cfg or "progress" in cfg:
        return cfg

    # Si viene con 'profiles', bajamos al perfil
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict):
        return profiles.get(profile) or {}

    return cfg or {}


def _sev_cfg(metric: str, net: str | None, cfg: dict):
    """Obtiene thresholds y orientación para métricas de % (severity)."""
    cfg = _normalize_profile_cfg(cfg, profile="main")
    s = (cfg.get("severity") or {}).get(metric) or {}
    # soporta tanto {"orientation":..., "thresholds":{...}} como {"default":{...}, "per_network":{...}}
    if "thresholds" in s:
        thresholds = s.get("thresholds") or {}
        orient = s.get("orientation", s.get("default", {}).get("orientation", "lower_is_better"))
    else:
        orient = (s.get("default") or {}).get("orientation", "lower_is_better")
        pern = (s.get("per_network") or {})
        if net and net in pern and "thresholds" in pern[net]:
            thresholds = pern[net]["thresholds"]
        else:
            thresholds = (s.get("default") or {}).get("thresholds") or {}
    # Asegura orden y float
    thr = {k: float(thresholds.get(k)) for k in SEV_ORDER if k in thresholds}
    # rellena por si faltan
    for k in SEV_ORDER:
        thr.setdefault(k, thr.get("regular", 0.0))
    return orient, thr

def _prog_cfg(metric: str, net: str | None, cfg: dict):
    """Obtiene min/max para métricas UNIT (progress), con per_network si existe."""
    cfg = _normalize_profile_cfg(cfg, profile="main")
    p = (cfg.get("progress") or {}).get(metric) or {}
    if "default" in p or "per_network" in p:
        d = p.get("default") or {}
        mn = d.get("min", 0.0); mx = d.get("max", 1.0)
        pern = p.get("per_network") or {}
        if net and net in pern:
            mn = pern[net].get("min", mn)
            mx = pern[net].get("max", mx)
        return float(mn), float(mx)
    # forma plana {min,max}
    return float(p.get("min", 0.0)), float(p.get("max", 1.0))

def _normalize(v: float | None, vmin: float, vmax: float) -> float | None:
    if v is None:
        return None
    if vmax <= vmin:
        return 0.0
    x = (float(v) - vmin) / (vmax - vmin)
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def _vendor_key(raw: str) -> str:
    if not raw:
        return ""
    s = unicodedata.normalize("NFKD", str(raw)).encode("ascii","ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[\s_/|-]+", " ", s)
    if "eric" in s:   return "ericsson"
    if "nokia" in s or "alcatel" in s or "lucent" in s: return "nokia"
    if "huawei" in s: return "huawei"
    if "samsung" in s:return "samsung"
    # fallback: primera palabra
    return s.split(" ",1)[0] if s else ""

def _vendor_initial(vendor: str) -> str:
    key = _vendor_key(vendor)
    m = {
        "ericsson": "E",
        "huawei":   "H",
        "nokia":    "N",
        "samsung":  "S",
    }
    if key in m:
        return m[key]
    # fallback: primera letra del texto “limpio”
    s = (vendor or "").strip()
    return s[:1].upper() if s else ""

def _vendor_badge(vendor_key: str, *, title: str = "", size: int = 20, asset_url_getter=None):
    key = (vendor_key or "").strip().lower()
    if not key:
        return html.Span("", className="vendor-icon vendor-empty", title=title)
    src = asset_url_getter(f"vendor/{key}.svg") if asset_url_getter else f"/assets/vendor/{key}.svg"
    return html.Img(
        src=src, alt=title or key, title=title or key,
        className="vendor-icon",
        style={"width": f"{size}px", "height": f"{size}px", "objectFit": "contain"}
    )

def _only_time(s: str) -> str:
    if not s:
        return ""
    s = str(s).replace("T", " ")
    parts = s.split()
    hhmm = parts[-1] if parts else s
    return hhmm[:5] if len(hhmm) >= 5 else hhmm

def _fmt(v, dec):
    try:
        f = float(v)
        if not np.isfinite(f): return ""
        return f"{f:,.{dec}f}"
    except Exception:
        return ""

def _last_numeric(seq):
    if not seq:
        return None
    for v in reversed(seq):
        try:
            f = float(v)
            if np.isfinite(f):
                return f
        except Exception:
            pass
    return None
# === Alturas para alinear fila-a-fila ===
ROW_H = 26           # 👈 Debe coincidir con tu CSS
MARG_TOP = 0
MARG_BOTTOM = 124    # Igual a 'b' en fig.update_layout(margin=...)
EXTRA = 0      # Ajuste fino opcional

def _hm_height(nrows: int) -> int:
    content = nrows * ROW_H                 # alto EXACTO de filas
    total   = MARG_TOP + content + MARG_BOTTOM + EXTRA
    return int(round(total))


def _sev_score(value: float | None, orient: str, thr: dict) -> float | None:
    if value is None:
        return None

    v = float(value)
    exc = float(thr["excelente"])
    cri = float(thr["critico"])
    if cri == exc:
        return 0.0  # evita división por cero

    if orient == "higher_is_better":
        # invertimos: valores altos son buenos
        r = (cri - v) / (cri - exc)  # 0 en crítico, 1 en excelente
    else:
        # lower_is_better: valores bajos son buenos
        r = (v - exc) / (cri - exc)  # 0 en excelente, 1 en crítico

    # clamp 0..1
    r = max(0.0, min(1.0, r))

    # escala a 0..3 (como antes), pero continuo
    return 3.0 * r

def _sev_score_continuo(value: float | None, orient: str, thr: dict, max_ratio: float = 2.0) -> float | None:
    """
    Devuelve un score continuo basado en thresholds:
      - 0   ~ excelente
      - 1   ~ crítico
      - >1  ~ peor que crítico (hasta max_ratio)
    Se usa como z en el heatmap de %.
    """
    if value is None:
        return None

    v = float(value)
    exc = float(thr.get("excelente", 0.0))
    cri = float(thr.get("critico", exc))

    if cri == exc:  # evita división por cero
        return 0.0

    if orient == "higher_is_better":
        # valores altos son buenos → peor cuando se baja
        r = (cri - v) / (cri - exc)   # 0 en crítico, 1 en excelente
        r = 1.0 - r                   # 0 en excelente, 1 en crítico
    else:
        # lower_is_better (tu caso): valores altos son peores
        r = (v - exc) / (cri - exc)   # 0 en excelente, 1 en crítico

    # dejamos que se pase un poco de 1 para ver más “rojez”
    r = max(0.0, min(r, max_ratio))   # 0..max_ratio

    return r

# =========================
# Payloads de heatmap (48 columnas: Ayer 0–23 | Hoy 24–47) con paginado
# =========================
def build_heatmap_payloads_fast(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    UMBRAL_CFG: dict,
    networks=None,
    valores_order=("PS_RRC","CS_RRC","PS_S1", "PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    today=None,
    yday=None,
    alarm_keys=None,
    alarm_only=False,
    offset=0,
    limit=5,
    order_by="unit"
):
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # --- redes a usar ---
    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # --- fechas hoy/ayer ---
    if today is None:
        if df_ts is not None and "fecha" in df_ts.columns:
            today = _max_date_str(df_ts["fecha"]) or _day_str(datetime.now())
        else:
            today = _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- métricas requeridas (pct + unit) ---
    metrics_needed = {
        m
        for v in valores_order
        for m in VALORES_MAP.get(v, (None, None))
        if m
    }
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # --- cartesian (meta x networks) ---
    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].reset_index(drop=True)
    rows_full = base.assign(_tmp=1).merge(
        pd.DataFrame({"network": networks, "_tmp": 1}),
        on="_tmp"
    ).drop(columns="_tmp")

    rows_all_list = []
    for v in valores_order:
        rf = rows_full.copy()
        rf["valores"] = v
        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            mask = list(zip(rf["technology"], rf["vendor"], rf["noc_cluster"], rf["network"]))
            rf = rf[[m in keys_ok for m in mask]]
        rows_all_list.append(rf)

    if not rows_all_list:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    rows_all = pd.concat(rows_all_list, ignore_index=True)

    # ---- Ordenamiento por criterio ----
    order_by = (order_by or "unit").lower()

    if df_ts is not None and not df_ts.empty:

        # métricas disponibles
        pct_metrics = [pm for pm, _ in VALORES_MAP.values() if pm and pm in df_ts.columns]
        unit_metrics = [um for _, um in VALORES_MAP.values() if um and um in df_ts.columns]

        # filtra ayer/hoy
        base_cols = ["technology", "vendor", "noc_cluster", "network", "fecha", "hora"]
        used_cols = base_cols + [c for c in (pct_metrics + unit_metrics) if c in df_ts.columns]
        df2 = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), used_cols].copy()

        # ===========================
        # alarm_hours SOLO %
        # ===========================
        order_by = (order_by or "unit").lower()

        if df_ts is not None and not df_ts.empty:

            pct_metrics = [pm for pm, _ in VALORES_MAP.values() if pm and pm in df_ts.columns]
            unit_metrics = [um for _, um in VALORES_MAP.values() if um and um in df_ts.columns]

            base_cols = ["technology", "vendor", "noc_cluster", "network", "fecha", "hora"]
            used_cols = base_cols + [c for c in (pct_metrics + unit_metrics) if c in df_ts.columns]
            df2 = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), used_cols].copy()

            # --- alarm_hours % (actual) ---
            if order_by in ("alarm", "alarm_hours", "hours", "alarm_hours_pct"):

                out_col = "alarm_hours"

                if not pct_metrics:
                    rows_all[out_col] = np.nan
                else:
                    COL_TO_VAL_PCT = {pm: name for name, (pm, _) in VALORES_MAP.items() if pm}

                    df_long = df2.melt(
                        id_vars=base_cols,
                        value_vars=pct_metrics,
                        var_name="metric",
                        value_name="value",
                    )
                    df_long["valores"] = df_long["metric"].map(COL_TO_VAL_PCT)

                    thr_cache = {}
                    for pm in pct_metrics:
                        for net in networks:
                            try:
                                orient, thr = _sev_cfg(pm, net, UMBRAL_CFG)
                            except Exception:
                                orient, thr = ("lower_is_better",
                                               {"excelente": 0, "bueno": 0, "regular": 0, "critico": 0})
                            thr_cache[(pm, net)] = (orient, thr)

                    def _is_alarm_pct(metric, net, v):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return False
                        orient, thr = thr_cache.get((metric, net), ("lower_is_better", None))
                        if not thr:
                            return False
                        crit = float(thr.get("critico", 0.0))
                        fv = float(v)
                        return (fv <= crit) if orient == "higher_is_better" else (fv >= crit)

                    df_long["alarm"] = [
                        _is_alarm_pct(m, n, v)
                        for m, n, v in zip(df_long["metric"], df_long["network"], df_long["value"])
                    ]

                    df_hour = (
                        df_long.dropna(subset=["valores"])
                        .groupby(
                            ["technology", "vendor", "noc_cluster", "network", "valores", "fecha", "hora"],
                            as_index=False
                        )["alarm"]
                        .max()
                    )

                    df_alarm = (
                        df_hour.groupby(
                            ["technology", "vendor", "noc_cluster", "network", "valores"],
                            as_index=False
                        )["alarm"]
                        .sum()
                        .rename(columns={"alarm": out_col})
                    )

                    rows_all = rows_all.merge(
                        df_alarm,
                        on=["technology", "vendor", "noc_cluster", "network", "valores"],
                        how="left"
                    )

            # --- alarm_hours UNIT (nuevo) ---
            elif order_by in ("alarm_hours_unit", "alarm_hours_fail"):

                out_col = "alarm_hours_unit"

                if not unit_metrics:
                    rows_all[out_col] = np.nan
                else:
                    COL_TO_VAL_UNIT = {um: name for name, (_, um) in VALORES_MAP.items() if um}

                    df_long = df2.melt(
                        id_vars=base_cols,
                        value_vars=unit_metrics,
                        var_name="metric",
                        value_name="value",
                    )
                    df_long["valores"] = df_long["metric"].map(COL_TO_VAL_UNIT)

                    # cache min/max por (metric, net) usando tu config progress
                    prog_cache = {}
                    for um in unit_metrics:
                        for net in networks:
                            try:
                                mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
                            except Exception:
                                mn, mx = (0.0, 1.0)
                            prog_cache[(um, net)] = (mn, mx)

                    # criterio de alarma UNIT:
                    # "alarmado" si value >= max configurado (por red si aplica)
                    def _is_alarm_unit(metric, net, v):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return False
                        mn, mx = prog_cache.get((metric, net), (0.0, None))
                        if mx is None:
                            return False
                        try:
                            return float(v) >= float(mx)
                        except Exception:
                            return False

                    df_long["alarm"] = [
                        _is_alarm_unit(m, n, v)
                        for m, n, v in zip(df_long["metric"], df_long["network"], df_long["value"])
                    ]

                    df_hour = (
                        df_long.dropna(subset=["valores"])
                        .groupby(
                            ["technology", "vendor", "noc_cluster", "network", "valores", "fecha", "hora"],
                            as_index=False
                        )["alarm"]
                        .max()
                    )

                    df_alarm = (
                        df_hour.groupby(
                            ["technology", "vendor", "noc_cluster", "network", "valores"],
                            as_index=False
                        )["alarm"]
                        .sum()
                        .rename(columns={"alarm": out_col})
                    )

                    rows_all = rows_all.merge(
                        df_alarm,
                        on=["technology", "vendor", "noc_cluster", "network", "valores"],
                        how="left"
                    )

        # ===========================
        # Fallback original: max_pct / max_unit
        # ===========================
        else:
            if order_by == "pct":
                cols = pct_metrics
                COL_TO_VAL2 = {pm: name for name, (pm, _) in VALORES_MAP.items() if pm}
                out_col = "max_pct"
            else:
                cols = unit_metrics
                COL_TO_VAL2 = {um: name for name, (_, um) in VALORES_MAP.items() if um}
                out_col = "max_unit"

            if cols:
                df_long = df2.melt(
                    id_vars=["technology", "vendor", "noc_cluster", "network"],
                    value_vars=cols,
                    var_name="metric",
                    value_name="value",
                )
                df_long["valores"] = df_long["metric"].map(COL_TO_VAL2)

                df_max = (
                    df_long.dropna(subset=["valores"])
                    .groupby(["technology", "vendor", "noc_cluster", "network", "valores"], as_index=False)["value"]
                    .max()
                    .rename(columns={"value": out_col})
                )

                rows_all = rows_all.merge(
                    df_max,
                    on=["technology", "vendor", "noc_cluster", "network", "valores"],
                    how="left"
                )
            else:
                rows_all[out_col] = np.nan

    else:
        out_col = "alarm_hours" if order_by in ("alarm", "alarm_hours", "hours") else \
            ("max_pct" if order_by == "pct" else "max_unit")
        rows_all[out_col] = np.nan

    # --- orden final ---
    ord_col = "__ord__"
    rows_all[ord_col] = rows_all[out_col].astype(float).fillna(float("-inf"))
    rows_all = rows_all.sort_values(ord_col, ascending=False, kind="stable")

    # --- paginado ---
    total_rows = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- df_small reducido (TS solo de las keys visibles) ---
    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df))
    else:
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df))

        df_small = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today])].merge(
            keys_df,
            on=["technology","vendor","noc_cluster","network"]
        )
        hh = df_small["hora"].astype(str).str.split(":", n=1).str[0]
        df_small["h"] = pd.to_numeric(hh, errors="coerce").where(lambda s: (s >= 0) & (s <= 23))
        df_small["offset48"] = df_small["h"] + np.where(df_small["fecha"].astype(str) == today, 24, 0)
        df_small = df_small.dropna(subset=["offset48"])
        df_small["offset48"] = df_small["offset48"].astype(int)

        rows_page = rows_page.merge(
            keys_df,
            on=["technology", "vendor", "noc_cluster", "network"],
            how="left",
        )

    # --- diccionarios (rid, offset48) → valor por métrica ---
    metric_maps = {}
    if not df_small.empty:
        for m in metrics_needed:
            if m in df_small.columns:
                sub = df_small[["rid","offset48",m]].dropna()
                metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset48"]), sub[m]))
            else:
                metric_maps[m] = {}
    else:
        metric_maps = {m: {} for m in metrics_needed}

    def _row48_raw(metric, rid):
        mp = metric_maps.get(metric)
        if not mp:
            return [None] * 48
        return [mp.get((rid, off)) for off in range(48)]

    # --- matrices y stats ---
    x_dt = [
        f"{yday}T{h:02d}:00:00" for h in range(24)
    ] + [
        f"{today}T{h:02d}:00:00" for h in range(24)
    ]

    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []

    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    all_scores_pct = []
    all_scores_unit = []

    for r in rows_page.itertuples(index=False):
        tech, vend, clus, net, valores, rid = (
            r.technology,
            r.vendor,
            r.noc_cluster,
            r.network,
            r.valores,
            r.rid,
        )
        pm, um = VALORES_MAP.get(valores, (None, None))

        y_id = f"{tech}/{vend}/{clus}/{net}/{valores}"

        y_labels.append(y_id)
        row_detail.append(y_id)

        row_raw = _row48_raw(pm, rid) if pm else [None] * 48
        row_raw_u = _row48_raw(um, rid) if um else [None] * 48

        # --- % (severity) → score continuo ---
        if pm:
            orient, thr = _sev_cfg(pm, net, UMBRAL_CFG)
            row_color = [
                _sev_score_continuo(v, orient, thr, max_ratio=2.0) if v is not None else None
                for v in row_raw
            ]
            z_pct.append(row_color)
            z_pct_raw.append(row_raw)

            for s in row_color:
                if s is not None:
                    all_scores_pct.append(s)
        else:
            z_pct.append([None] * 48)
            z_pct_raw.append(row_raw)

        # --- UNIT (progress) → normalizado ---
        if um:
            mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
            row_norm = [
                _normalize(v, mn, mx) if v is not None else None
                for v in row_raw_u
            ]
            z_unit.append(row_norm)
            z_unit_raw.append(row_raw_u)

            for s in row_norm:
                if s is not None:
                    all_scores_unit.append(s)
        else:
            z_unit.append([None] * 48)
            z_unit_raw.append(row_raw_u)

        # --- stats por fila ---
        arr_u = np.array(
            [v if isinstance(v, (int, float)) else np.nan for v in row_raw_u],
            float
        )
        arr_p = np.array(
            [v if isinstance(v, (int, float)) else np.nan for v in row_raw],
            float
        )

        if np.isfinite(arr_u).any():
            rmax_u = np.nanmax(arr_u)
            valid_idx = np.where(np.isfinite(arr_u))[0]
        else:
            rmax_u = np.nan
            valid_idx = np.where(np.isfinite(arr_p))[0]

        rmax_p = np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan

        if valid_idx.size:
            last_label = str(x_dt[int(valid_idx[-1])]).replace("T", " ")[:16]
        else:
            last_label = ""

        row_last_ts.append(last_label)
        row_max_pct.append(rmax_p)
        row_max_unit.append(rmax_u)

    # --- rangos dinámicos para color (% y UNIT) ---
    if all_scores_pct:
        zmin_pct = 0.0
        zmax_pct = 1.0
    else:
        zmin_pct, zmax_pct = 0.0, 1.0

    if all_scores_unit:
        zmin_unit = min(all_scores_unit)
        zmax_unit = max(all_scores_unit)
    else:
        zmin_unit, zmax_unit = 0.0, 1.0

    # --- payloads ---
    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "severity",
        "zmin": zmin_pct,
        "zmax": zmax_pct,
        "title": "% IA / % DC",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }

    unit_payload = {
        "z": z_unit,
        "z_raw": z_unit_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",
        "zmin": zmin_unit,
        "zmax": zmax_unit,
        "title": "Unidades",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }

    page_info = {
        "total_rows": total_rows,
        "offset": start,
        "limit": limit,
        "showing": len(rows_page),
    }

    return pct_payload, unit_payload, page_info

# =========================
# Figura de Heatmap (Plotly) — detalle en hover, eje Y ligero
# =========================
def build_heatmap_figure(
    payload,
    *,
    height=750,
    decimals=2,
):

    if not payload:
        return go.Figure()

    z       = payload["z"]                 # z para COLOR (clase 0..3 o 0..1)
    z_raw   = payload.get("z_raw") or z    # valores reales para hover
    x       = payload.get("x_dt") or payload.get("x")  # usamos datetime si existe
    y       = payload["y"]
    zmin    = payload["zmin"]
    zmax    = payload["zmax"]
    mode    = payload.get("color_mode", "severity")
    detail  = payload.get("row_detail") or y

    # ----- Colores según modo -----
    if mode == "severity":
        colorscale = [
            [0/3, SEV_COLORS["excelente"]],
            [1/3, SEV_COLORS["bueno"]],
            [2/3, SEV_COLORS["regular"]],
            [3/3, SEV_COLORS["critico"]],
        ]
    else:  # progress
        colorscale = [
            [0.0, "#9ec5fe"],
            [1.0, "#0d6efd"],
        ]

    # ----- customdata por celda (DETALLE COMPLETO; sin cambios) -----
    customdata = []
    for i, row in enumerate(z_raw):
        arr = np.array([v if isinstance(v, (int, float)) else np.nan for v in row], dtype=float)
        if np.isfinite(arr).any():
            rmax = np.nanmax(arr); rmin = np.nanmin(arr)
            valid_idx = np.where(np.isfinite(arr))[0]
            last_idx = int(valid_idx[-1])
            last_label = (x[last_idx] if isinstance(x[last_idx], str) else str(x[last_idx]))
            last_label = last_label.replace("T", " ")[:16]
        else:
            rmax = np.nan; rmin = np.nan; last_label = "—"

        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 4)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        clus   = parts[2] if len(parts) > 2 else ""
        net    = parts[3] if len(parts) > 3 else ""
        valor  = parts[4] if len(parts) > 4 else ""

        def _fmt(v):
            if not np.isfinite(v): return ""
            return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"
        rmax_s = _fmt(rmax); rmin_s = _fmt(rmin)

        row_cd = []
        for j in range(len(x)):
            raw_cell = arr[j] if j < len(arr) else np.nan
            raw_s = _fmt(raw_cell)
            row_cd.append([tech, vendor, clus, net, valor, last_label, rmax_s, rmin_s, raw_s])
        customdata.append(row_cd)

    hover_tmpl = (
        "<span style='font-size:120%; font-weight:700'>%{customdata[8]}</span><br>"
        "<span style='opacity:0.85'>%{x|%Y-%m-%d %H:%M}</span><br>"
        "──────────<br>"
        "DETALLE<br>"
        "<b>Net:</b> %{customdata[3]}<br>"
        "<b>Última hora con registro:</b> %{customdata[5]}<br>"
        "<b>Máx:</b> %{customdata[6]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        showscale=False,             # sin barra de color
        customdata=customdata,
        hovertemplate=hover_tmpl,
        hoverongaps=False,
        xgap=0.5, ygap=0.5,          # divisiones remarcadas
    ))

    # Eje X: SIN labels/ticks para no “robar” altura
    THREE_H_MS = 3 * 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=THREE_H_MS,
        showticklabels=False,  #
        ticks="",              #
        showgrid=False,
        ticklabelmode="instant",
        fixedrange=True,
        automargin=False,
    )

    # Eje Y oculto pero preservando el orden
    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        title="",
        automargin=False,
        fixedrange=True,
        categoryorder="array",
        categoryarray=y,
        autorange="reversed",
    )

    # Línea de corte entre días (opcional)
    if isinstance(x, (list, tuple)) and len(x) >= 25:
        fig.add_vline(
            x=x[24],
            line_width=3,  # más gruesa
            line_color="rgba(0,0,0,0.75)",  # oscuro
            line_dash="solid",
            layer="above"  # por encima del heatmap
        )

    # Márgenes ultra compactos; MARG_BOTTOM debe coincidir con _hm_height
    fig.update_layout(
        autosize=False,
        height=height,
        margin=dict(l=4, r=4, t=MARG_TOP, b=MARG_BOTTOM),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff", size=13)),
        uirevision="keep",
    )

    return fig


def render_heatmap_summary_table(pct_payload, unit_payload, *, pct_decimals=2, unit_decimals=0, asset_url_getter=None, active_y=None):
    src = pct_payload or unit_payload
    if not src:
        return dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

    y = src.get("y") or []
    detail = src.get("row_detail") or y
    row_last_ts = (unit_payload or pct_payload).get("row_last_ts") or []
    z_raw_pct = (pct_payload or {}).get("z_raw")
    z_raw_unit = (unit_payload or {}).get("z_raw")

    cols = [
        ("Cluster", "w-cluster"),
        ("Tech", "w-tech"),
        ("Vendor", "w-vendor"),
        ("Valor", "w-valor"),
        ("Última hora", "w-ultima"),
        ("Valor de la última muestra (%)", "w-num"),
        ("Valor de la última muestra (UNIT)", "w-num"),
    ]
    thead = html.Thead(html.Tr([html.Th(n, className=c) for n, c in cols]), className="table-dark")

    # más precisión para el tooltip (sin cambiar lo visible)
    def _fmt_full(v):
        try:
            f = float(v)
            if not np.isfinite(f): return ""
            return f"{f:,.2f}"  # muestra más decimales en el hover
        except Exception:
            return ""

    body_rows = []
    for i in range(len(y)):
        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 4)
        tech = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        cluster = parts[2] if len(parts) > 2 else ""
        valor = parts[4] if len(parts) > 4 else ""

        last_pct = _last_numeric(z_raw_pct[i]) if z_raw_pct and i < len(z_raw_pct) else None
        last_unit = _last_numeric(z_raw_unit[i]) if z_raw_unit and i < len(z_raw_unit) else None

        ultima_txt = _only_time(row_last_ts[i] if i < len(row_last_ts) else "")
        pct_txt    = _fmt(last_pct, pct_decimals)
        unit_txt   = _fmt(last_unit, unit_decimals)

        body_rows.append(
            html.Tr([
                html.Td(
                    html.Span(
                        html.Span(cluster, className="unflip"),
                        className="ellipsis-left"
                    ),
                    className="w-cluster",
                    title=cluster
                ),
                html.Td(tech, className="w-tech", title=tech),
                html.Td(
                    html.Span(_vendor_initial(vendor), className="vendor-initial"),
                    className="td-vendor w-vendor",
                    title=vendor
                ),

                html.Td(valor, className="w-valor", title=valor),
                html.Td(ultima_txt, className="w-ultima ta-center", title=ultima_txt),
                html.Td(pct_txt, className="w-num ta-right",  title=_fmt_full(last_pct)),
                html.Td(unit_txt, className="w-num ta-right", title=_fmt_full(last_unit)),
            ])
        )

    return dbc.Table([thead, html.Tbody(body_rows)],
                     striped=True, bordered=False, hover=True, size="sm",
                     className="mb-0 table-dark kpi-table kpi-table-summary compact")



def _build_time_header_children(x_dt):
    """
    x_dt: lista de 48 timestamps (str tipo 'YYYY-MM-DDTHH:MM:SS')
    Devuelve (dates_children, hours_children) como listas de Divs.
    """
    if not x_dt or len(x_dt) < 48:
        # fallback vacío (mantiene estructura)
        dates = [html.Div("", className="cell"), html.Div("", className="cell sep")]
        hours = [html.Div("", className="cell muted") for _ in range(48)]
        return dates, hours

    # Fechas "YYYY-MM-DD"
    yday = x_dt[0].split("T", 1)[0]
    today = x_dt[24].split("T", 1)[0]

    # Fila de días (2 celdas de 24 columnas)
    dates = [
        html.Div(yday,  className="cell"),
        html.Div(today, className="cell sep"),
    ]

    # Fila de horas (48 celdas: etiqueta cada 3h; otras atenuadas)
    hours = []
    for i in range(48):
        hh = int(x_dt[i].split("T", 1)[1][:2])  # 00..23
        show = (hh % 3 == 0)
        cls = "cell"
        if i == 24:
            cls += " sep"  # separador entre días
        if not show:
            cls += " muted"
        hours.append(html.Div(f"{hh:02d}", className=cls if show else cls))
    return dates, hours

def _build_time_header_children_by_dates(fecha_str: str):
    from datetime import datetime, timedelta
    from dash import html

    try:
        today_dt = datetime.strptime(fecha_str, "%Y-%m-%d") if fecha_str else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()
    yday_dt = today_dt - timedelta(days=1)
    today_s = today_dt.strftime("%Y-%m-%d")
    yday_s  = yday_dt.strftime("%Y-%m-%d")

    # Fila de días (dos bloques de 24 columnas)
    dates_children = [
        html.Div(yday_s,  className="cell"),
        html.Div(today_s, className="cell sep"),
    ]

    # Fila de horas (48 celdas con ticks; etiqueta cada 3h, fuerte cada 6h)
    hours_children = []
    for i in range(48):
        hh = i if i < 24 else i - 24
        cls = ["cell"]
        if i == 24:
            cls.append("sep")
        if hh % 6 == 0:
            cls.append("tick6")  # tick más alto/visible

        # etiqueta solo cada 3 horas
        label = html.Span(f"{hh:02d}", className="lbl") if (hh % 3 == 0) else None
        hours_children.append(html.Div(label, className=" ".join(cls)))

    return dates_children, hours_children
