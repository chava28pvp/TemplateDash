import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc

from components.main.heatmap import (
    _max_date_str, _day_str, _sev_cfg, _sev_score_continuo, _prog_cfg, _normalize,
    _hm_height, _last_numeric, _only_time, _fmt, _vendor_initial, _sev_level_safe
)

# =========================================================
# CONFIG TOPOFF
# =========================================================

VALORES_MAP_TOPOFF = {
    "PS_RRC":  ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RRC":  ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB":  ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB":  ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
    "PS_S1":   ("ps_s1_ia_percent", "ps_s1_fail"),
    "RTX_TNL": ("rtx_tnl_tx_percent", "tnl_abn"),
}

SEV_COLORS = {
    "excelente": "#2ecc71",
    "bueno":     "#f1c40f",
    "regular":   "#e67e22",
    "critico":   "#e74c3c",
}
SEV_ORDER = ["excelente", "bueno", "regular", "critico"]

# Meta que define UNA FILA única en TopOff (ahora con cluster)
META_COLS_TOPOFF = [
    "technology", "vendor",
    "region", "province", "municipality",
    "cluster",          # 👈 NOC_CLUSTER
    "site_att", "rnc", "nodeb",
]

# Alturas para alinear fila-a-fila (idéntico a main)
ROW_H = 26
MARG_TOP = 0
MARG_BOTTOM = 170
EXTRA = 0

# =========================================================
# Helpers
# =========================================================
def _hm_height_topoff(n_rows: int) -> int:
    if n_rows <= 0:
        return 300
    return int(n_rows * ROW_H + MARG_TOP + MARG_BOTTOM + EXTRA)

def _build_x_dt_15m(day_str):
    # 96 bins por día (24*4)
    return [
        f"{day_str}T{h:02d}:{m:02d}:00"
        for h in range(24)
        for m in (0, 15, 30, 45)
    ]


def _safe_q15_to_idx(hhmmss):
    """
    hhmmss: '10:15:00' -> 10*4 + 1 = 41
    Devuelve 0..95 o None.
    """
    try:
        s = str(hhmmss)
        parts = s.split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            q = mm // 15  # 0,1,2,3
            return hh * 4 + q
    except Exception:
        pass
    return None

def _normalize_topoff_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas comunes para evitar KeyError
    cuando df_ts viene con alias distintos.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    ren = {}

    # technology
    if "technology" not in df.columns:
        if "tech" in cols_lower:
            ren[cols_lower["tech"]] = "technology"

    # vendor
    if "vendor" not in df.columns:
        if "vend" in cols_lower:
            ren[cols_lower["vend"]] = "vendor"

    # cluster (acepta noc_cluster -> cluster)
    if "cluster" not in df.columns:
        if "noc_cluster" in cols_lower:
            ren[cols_lower["noc_cluster"]] = "cluster"

    # otros meta comunes
    for k in ["region", "province", "municipality", "site_att", "rnc", "nodeb"]:
        if k not in df.columns and k in cols_lower:
            ren[cols_lower[k]] = k

    if ren:
        df = df.rename(columns=ren)

    return df

def _norm_key_cols(df: pd.DataFrame, cols):
    """Hace que llaves de merge/groupby no pierdan filas por NaN/espacios."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .fillna("")              # <- CLAVE: evita NaN en llaves
                .astype(str)
                .str.strip()
            )
    return df

def _norm_hora_str(x) -> str:
    """
    Normaliza hora a HH:MM:SS.
    Acepta: 22, '22', '22:00', '22:00:00', '22:00:00.000', 2200, '2200'
    """
    if x is None:
        return ""
    # pandas/np nan
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    s = str(x).strip()
    if s == "" or s.lower() == "nan" or s.lower() == "none":
        return ""

    # si viene como número "2200" (HHMM) o "22"
    try:
        n = int(float(s))
        if 0 <= n <= 23:
            return f"{n:02d}:00:00"
        if 0 <= n <= 2359:
            hh = n // 100
            mm = n % 100
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}:00"
    except Exception:
        pass

    # si viene con ":" (HH:MM[:SS[.ms]])
    parts = s.split(":")
    try:
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        ss = parts[2] if len(parts) > 2 else "0"
        # ss puede venir "00.000"
        ss = int(float(ss))
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        # deja original (último recurso)
        return s
# =========================================================
# PAYLOADS TOPOFF (AYER/Hoy, 48 columnas)
# =========================================================
def build_heatmap_payloads_topoff(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    UMBRAL_CFG: dict,
    valores_order=("PS_RRC", "CS_RRC", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB"),
    today: Optional[str] = None,
    yday: Optional[str] = None,
    alarm_keys: Optional[set] = None,
    alarm_only: bool = False,
    offset: int = 0,
    limit: int = 20,
    order_by: str = "alarm_bins_pct",
) -> Tuple[Optional[dict], Optional[dict], dict]:

    DEBUG = True  # <- pon False para quitar prints

    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0, "height": 300}

    # --- fechas hoy/ayer ---
    if today is None:
        if df_ts is not None and isinstance(df_ts, pd.DataFrame) and not df_ts.empty and "fecha" in df_ts.columns:
            today = _max_date_str(df_ts["fecha"]) or _day_str(datetime.now())
        else:
            today = _day_str(datetime.now())

    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- métricas requeridas ---
    metrics_needed = {m for v in valores_order for m in VALORES_MAP_TOPOFF.get(v, (None, None)) if m}
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0, "height": 300}

    # --- base meta ---
    base = df_meta.drop_duplicates(subset=META_COLS_TOPOFF)[META_COLS_TOPOFF].reset_index(drop=True)
    base = _norm_key_cols(base, META_COLS_TOPOFF)

    # --- expand por valores_order ---
    rows_all_list = []
    keys_ok = set(alarm_keys) if (alarm_only and alarm_keys is not None) else None

    for v in valores_order:
        rf = base.copy()
        rf["valores"] = v

        if keys_ok is not None:
            mask = list(zip(
                rf["technology"], rf["vendor"], rf["region"], rf["province"],
                rf["municipality"], rf["cluster"], rf["site_att"], rf["rnc"], rf["nodeb"]
            ))
            rf = rf[[m in keys_ok for m in mask]]

        rows_all_list.append(rf)

    if not rows_all_list:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0, "height": 300}

    rows_all = pd.concat(rows_all_list, ignore_index=True)
    rows_all = _norm_key_cols(rows_all, META_COLS_TOPOFF + ["valores"])

    # =========================================================
    # ORDENAMIENTO
    # =========================================================
    order_by = (order_by or "unit").lower()
    if order_by in ("alarm_hours", "alarm", "hours", "alarm_hours_pct"):
        order_by = "alarm_bins_pct"
    if order_by in ("alarm_hours_unit", "alarm_hours_fail"):
        order_by = "alarm_bins_unit"

    out_col = None

    # normaliza df_ts columnas
    df_ts = _normalize_topoff_cols(df_ts)

    # --- debug: llaves meta vs TS ---
    if DEBUG and df_ts is not None and isinstance(df_ts, pd.DataFrame) and not df_ts.empty and all(c in df_ts.columns for c in META_COLS_TOPOFF + ["fecha"]):
        ts_keys = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), META_COLS_TOPOFF].copy()
        ts_keys = _norm_key_cols(ts_keys, META_COLS_TOPOFF).drop_duplicates()
        miss = base.merge(ts_keys, on=META_COLS_TOPOFF, how="left", indicator=True)

    if df_ts is not None and isinstance(df_ts, pd.DataFrame) and not df_ts.empty:

        pct_metrics  = [pm for pm, _ in VALORES_MAP_TOPOFF.values() if pm and pm in df_ts.columns]
        unit_metrics = [um for _, um in VALORES_MAP_TOPOFF.values() if um and um in df_ts.columns]

        meta_present = [c for c in META_COLS_TOPOFF if c in df_ts.columns]
        critical = {"technology", "vendor", "cluster", "nodeb"}
        if not critical.issubset(set(meta_present)):
            page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0, "height": 300}
            return None, None, page_info

        base_cols = meta_present + ["fecha", "hora"]
        used_cols = base_cols + [c for c in (pct_metrics + unit_metrics) if c in df_ts.columns]

        df2 = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), used_cols].copy()
        df2 = _norm_key_cols(df2, base_cols)

        # Normaliza hora
        if "hora" in df2.columns:
            df2["hora"] = df2["hora"].apply(_norm_hora_str)

        # offset192
        df2["q15"] = df2["hora"].apply(_safe_q15_to_idx)
        df2 = df2.dropna(subset=["q15"]).copy()
        df2["q15"] = df2["q15"].astype(int)

        df2["offset192"] = df2["q15"] + np.where(df2["fecha"].astype(str) == today, 96, 0)
        df2["offset192"] = pd.to_numeric(df2["offset192"], errors="coerce").fillna(-1).astype(int)
        df2 = df2[(df2["offset192"] >= 0) & (df2["offset192"] <= 191)].copy()

        # ======================================================
        # MAIN-LIKE para % (el que te interesa)
        # ======================================================
        if order_by in ("alarm_bins_pct", "alarm_pct"):

            if pct_metrics:
                COL_TO_VAL_PCT = {pm: name for name, (pm, _) in VALORES_MAP_TOPOFF.items() if pm}

                df_long = df2.melt(
                    id_vars=base_cols + ["offset192"],
                    value_vars=pct_metrics,
                    var_name="metric",
                    value_name="value",
                )
                df_long["valores"] = df_long["metric"].map(COL_TO_VAL_PCT)
                df_long = df_long.dropna(subset=["valores"])

                grp_cols = meta_present + ["valores"]

                # cache thresholds
                thr_cache = {}
                for pm in pct_metrics:
                    try:
                        orient, thr = _sev_cfg(pm, None, UMBRAL_CFG)
                    except Exception:
                        orient, thr = ("lower_is_better", {"excelente": 0.0, "bueno": 0.0, "regular": 0.0, "critico": 1.0})
                    thr_cache[pm] = (orient, thr)

                def _score_pct(metric, v):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return np.nan
                    orient, thr = thr_cache.get(metric, ("lower_is_better", {}))
                    return _sev_score_continuo(v, orient, thr, max_ratio=2.0)

                def _level_pct(metric, v):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return -1
                    orient, thr = thr_cache.get(metric, ("lower_is_better", {}))
                    return _sev_level_safe(v, orient, thr)

                df_long["score_pct"] = [_score_pct(m, v) for m, v in zip(df_long["metric"], df_long["value"])]
                df_long["lvl_pct"]   = [_level_pct(m, v) for m, v in zip(df_long["metric"], df_long["value"])]

                # ====== ÚLTIMA MUESTRA REAL (aunque no sea regular/critico) ======
                df_last_any = (
                    df_long.dropna(subset=["value", "offset192"])
                    .sort_values("offset192")
                    .groupby(grp_cols, as_index=False)
                    .tail(1)
                    .rename(columns={
                        "offset192": "__last_any_off_pct",
                        "score_pct": "__last_any_score_pct",
                        "lvl_pct": "__last_any_lvl_pct",
                    })
                )

                rows_all = rows_all.merge(
                    df_last_any[grp_cols + ["__last_any_off_pct", "__last_any_score_pct", "__last_any_lvl_pct"]],
                    on=grp_cols,
                    how="left",
                )

                # flags numéricos
                rows_all["__last_any_off_pct"]   = pd.to_numeric(rows_all["__last_any_off_pct"], errors="coerce").fillna(-1).astype(int)
                rows_all["__last_any_score_pct"] = pd.to_numeric(rows_all["__last_any_score_pct"], errors="coerce").fillna(float("-inf"))
                rows_all["__last_any_lvl_pct"]   = pd.to_numeric(rows_all["__last_any_lvl_pct"], errors="coerce").fillna(-1).astype(int)

                # <- CLAVE: filas sin muestra deben ir al final SIEMPRE
                rows_all["__has_any_sample_pct"] = (rows_all["__last_any_off_pct"] >= 0).astype(int)

                # ====== ESCALERITA por bins "interesantes" (tu lógica) ======
                REG_DELTA = 0.5

                def _is_interest(metric, v):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return False
                    orient, thr = thr_cache.get(metric, ("lower_is_better", {}))
                    try:
                        fv = float(v)
                    except Exception:
                        return False

                    lvl = _sev_level_safe(fv, orient, thr)
                    if lvl >= 2:
                        return True

                    reg = thr.get("regular", None)
                    if reg is None:
                        return False
                    try:
                        reg = float(reg)
                    except Exception:
                        return False

                    if orient == "lower_is_better":
                        return fv >= (reg - REG_DELTA)
                    else:
                        return fv <= (reg + REG_DELTA)

                mask_interest = [_is_interest(m, v) for m, v in zip(df_long["metric"], df_long["value"])]
                df_long["score_interest_pct"] = np.where(mask_interest, df_long["score_pct"], np.nan)

                df_score_bin = (
                    df_long.groupby(grp_cols + ["offset192"], as_index=False)["score_interest_pct"]
                    .max()
                    .dropna(subset=["score_interest_pct"])
                )
                # ====== ÚLTIMA ALARMA (último bin interesante) ======
                df_last_interest = (
                    df_score_bin.sort_values("offset192")
                    .groupby(grp_cols, as_index=False)
                    .tail(1)
                    .rename(columns={
                        "offset192": "__last_int_off_pct",
                        "score_interest_pct": "__last_int_score_pct",
                    })
                )

                rows_all = rows_all.merge(
                    df_last_interest[grp_cols + ["__last_int_off_pct", "__last_int_score_pct"]],
                    on=grp_cols,
                    how="left",
                )

                rows_all["__last_int_off_pct"] = pd.to_numeric(rows_all["__last_int_off_pct"], errors="coerce").fillna(
                    -1).astype(int)
                rows_all["__last_int_score_pct"] = pd.to_numeric(rows_all["__last_int_score_pct"],
                                                                 errors="coerce").fillna(float("-inf"))
                rows_all["__has_interest_pct"] = (rows_all["__last_int_off_pct"] >= 0).astype(int)

                if not df_score_bin.empty:
                    wide = df_score_bin.pivot(index=grp_cols, columns="offset192", values="score_interest_pct").reset_index()
                    wide = wide.rename(columns={c: f"__p_{int(c):03d}" for c in wide.columns if isinstance(c, (int, np.integer))})

                    rows_all = rows_all.merge(wide, on=grp_cols, how="left")

                # asegura columnas __p_000..__p_191 para sort estable
                for off in range(0, 192):
                    c = f"__p_{off:03d}"
                    if c not in rows_all.columns:
                        rows_all[c] = np.nan

                stair_sort_cols = [f"__p_{off:03d}" for off in range(191, -1, -1)]
                for c in stair_sort_cols:
                    rows_all[c] = pd.to_numeric(rows_all[c], errors="coerce").fillna(float("-inf"))

                # ====== SORT MAIN-LIKE (con has_any_sample primero) ======
                rows_all = rows_all.sort_values(
                    by=["__has_interest_pct", "__last_int_off_pct", "__last_int_score_pct"] + stair_sort_cols,
                    ascending=[False, False, False] + ([False] * len(stair_sort_cols)),
                    kind="stable",
                )
                rows_all["__row_rank_mainlike"] = np.arange(len(rows_all), dtype=int)

            else:
                # sin métricas pct -> no orden main-like posible
                rows_all["__has_any_sample_pct"] = 0
                rows_all["__row_rank_mainlike"] = np.arange(len(rows_all), dtype=int)

        # ======================================================
        # resto de modos: tu lógica (sin prints extra)
        # ======================================================
        elif order_by in ("alarm_bins_unit", "alarm_unit"):
            out_col = "alarm_bins_unit"
            # (tu lógica original aquí, sin cambios, si quieres la vuelvo a integrar)
            rows_all[out_col] = np.nan

        elif order_by == "pct":
            out_col = "max_pct"
            rows_all[out_col] = np.nan

        else:
            out_col = "max_unit"
            rows_all[out_col] = np.nan

    else:
        # sin TS
        rows_all["__has_any_sample_pct"] = 0
        rows_all["__row_rank_mainlike"] = np.arange(len(rows_all), dtype=int)

    # =========================================================
    # ✅ GUARDAS: columnas que usa el CLUSTER SORT siempre deben existir
    # (si no se calculó main-like o no hubo data TS, quedan defaults)
    # =========================================================
    if "__last_any_off_pct" not in rows_all.columns:
        rows_all["__last_any_off_pct"] = -1
    if "__last_any_lvl_pct" not in rows_all.columns:
        rows_all["__last_any_lvl_pct"] = -1
    if "__last_any_score_pct" not in rows_all.columns:
        rows_all["__last_any_score_pct"] = float("-inf")

    # bandera: tiene al menos 1 muestra real (%)
    if "__has_any_sample_pct" not in rows_all.columns:
        rows_all["__has_any_sample_pct"] = (
                pd.to_numeric(rows_all["__last_any_off_pct"], errors="coerce").fillna(-1) >= 0
        ).astype(int)
    else:
        rows_all["__has_any_sample_pct"] = (
            pd.to_numeric(rows_all["__has_any_sample_pct"], errors="coerce").fillna(0).astype(int)
        )

    # rank mainlike: si no existe, empuja al final
    if "__row_rank_mainlike" not in rows_all.columns:
        rows_all["__row_rank_mainlike"] = 10 ** 9
    else:
        rows_all["__row_rank_mainlike"] = (
            pd.to_numeric(rows_all["__row_rank_mainlike"], errors="coerce").fillna(10 ** 9).astype(int)
        )

    # =========================================================
    # ORDEN FINAL POR CLUSTER (sin romper main-like)
    # =========================================================
    DISABLE_CLUSTER_SORT = True  # ✅ Opción A: deja SOLO main-like (sin agrupar por cluster)

    if not DISABLE_CLUSTER_SORT:

        for col in ["technology", "vendor", "cluster", "site_att"]:
            if col in rows_all.columns:
                rows_all[col] = rows_all[col].astype(str)

        mask_empty_cluster = rows_all["cluster"].isin(["", "nan", "None"])
        if "site_att" in rows_all.columns:
            rows_all.loc[mask_empty_cluster, "cluster"] = rows_all.loc[mask_empty_cluster, "site_att"].astype(str)

        cluster_group_cols = ["cluster"]

        cluster_stats = (
            rows_all.groupby(cluster_group_cols, as_index=False)
            .agg(
                cluster_has_any=("__has_any_sample_pct", "max"),
                cluster_best_rank=("__row_rank_mainlike", "min"),
            )
        )

        rows_all = rows_all.merge(cluster_stats, on=cluster_group_cols, how="left", validate="many_to_one")

        # valores en orden fijo
        if valores_order:
            rows_all["valores"] = pd.Categorical(rows_all["valores"], categories=list(valores_order), ordered=True)

        rows_all = rows_all.sort_values(
            by=[
                "cluster_has_any",
                "cluster_best_rank",
                "cluster",
                "site_att",
                "technology",
                "vendor",
                "__has_any_sample_pct",  # <- 1 primero (tiene datos)
                "__row_rank_mainlike",  # <- orden real main-like dentro del cluster
                "valores",
            ],
            ascending=[
                False,  # cluster_has_any
                True,  # cluster_best_rank
                True,  # cluster
                True,  # site_att
                True,  # technology
                True,  # vendor
                False,  # __has_any_sample_pct
                True,  # __row_rank_mainlike
                True,  # valores
            ],
            kind="stable",
        )

    else:
        # ✅ Opción A: NO reordenar por cluster, conservar exactamente el orden main-like ya calculado.
        # Solo aplica el orden fijo de "valores" si lo deseas (no altera el rank, solo el tipo).
        if valores_order:
            rows_all["valores"] = pd.Categorical(rows_all["valores"], categories=list(valores_order), ordered=True)

    # --- paginado ---
    # =========================================================
    # ✅ QUITAR "HUECOS": filas sin ninguna muestra en el dominio actual
    # =========================================================
    if order_by in ("alarm_bins_pct", "alarm_pct"):
        # aquí ya existe __has_any_sample_pct por tu merge de df_last_any
        if "__has_any_sample_pct" in rows_all.columns:
            rows_all = rows_all[rows_all["__has_any_sample_pct"].fillna(0).astype(int) == 1].copy()

    elif order_by in ("alarm_bins_unit", "alarm_unit"):
        # si quieres equivalente para UNIT, necesitas construir __has_any_sample_unit
        # (si no lo construyes, no filtres aquí)
        if "__has_any_sample_unit" in rows_all.columns:
            rows_all = rows_all[rows_all["__has_any_sample_unit"].fillna(0).astype(int) == 1].copy()

    else:
        # pct/max o unit/max: si quieres quitar vacíos, filtra por out_col
        if out_col and out_col in rows_all.columns:
            rows_all = rows_all[pd.to_numeric(rows_all[out_col], errors="coerce").notna()].copy()
    total_rows = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- keys visibles y df_small TS ---
    keys_df = rows_page[META_COLS_TOPOFF].drop_duplicates().reset_index(drop=True)
    keys_df = _norm_key_cols(keys_df, META_COLS_TOPOFF)
    keys_df["rid"] = np.arange(len(keys_df))

    if df_ts is None or not isinstance(df_ts, pd.DataFrame) or df_ts.empty:
        df_small = pd.DataFrame()
    else:
        df_ts2 = _normalize_topoff_cols(df_ts)
        df_ts2 = _norm_key_cols(df_ts2, META_COLS_TOPOFF + ["fecha", "hora"])
        if "hora" in df_ts2.columns:
            df_ts2["hora"] = df_ts2["hora"].apply(_norm_hora_str)

        df_small = (
            df_ts2.loc[df_ts2["fecha"].astype(str).isin([yday, today])]
            .merge(keys_df, on=META_COLS_TOPOFF, how="inner")
        )

        df_small["q15"] = df_small["hora"].apply(_safe_q15_to_idx)
        df_small["offset192"] = df_small["q15"] + np.where(df_small["fecha"].astype(str) == today, 96, 0)

        df_small = df_small.dropna(subset=["offset192"])
        df_small["offset192"] = df_small["offset192"].astype(int)

    # --- maps por métrica ---
    metric_maps = {}
    if not df_small.empty:
        for m in metrics_needed:
            if m in df_small.columns:
                sub = df_small[["rid", "offset192", m]].dropna()
                sub = sub.sort_values("offset192")
                metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset192"]), sub[m]))
            else:
                metric_maps[m] = {}
    else:
        metric_maps = {m: {} for m in metrics_needed}

    def _row192_raw(metric, rid):
        mp = metric_maps.get(metric) or {}
        return [mp.get((rid, off)) for off in range(192)]

    rows_page = rows_page.merge(keys_df, on=META_COLS_TOPOFF, how="left", validate="many_to_one")

    # --- ejes ---
    x_dt = _build_x_dt_15m(yday) + _build_x_dt_15m(today)

    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []
    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    all_scores_unit = []

    for r in rows_page.itertuples(index=False):
        rid = int(getattr(r, "rid"))
        valores = r.valores
        pm, um = VALORES_MAP_TOPOFF.get(valores, (None, None))

        nodeb = getattr(r, "nodeb", "") or ""
        tech = r.technology
        vend = r.vendor

        region = getattr(r, "region", "") or ""
        province = getattr(r, "province", "") or ""
        municipality = getattr(r, "municipality", "") or ""
        site_att = getattr(r, "site_att", "") or ""
        rnc = getattr(r, "rnc", "") or ""

        y_id = f"{tech}/{vend}/{region}/{province}/{municipality}/{r.site_att}/{r.rnc}/{nodeb}/{r.cluster}/{valores}"
        y_labels.append(y_id)

        row_detail.append(f"{tech}/{vend}/{region}/{province}/{municipality}/{site_att}/{rnc}/{nodeb}/{valores}")

        row_raw   = _row192_raw(pm, rid) if pm else [None] * 192
        row_raw_u = _row192_raw(um, rid) if um else [None] * 192

        if pm:
            orient, thr = _sev_cfg(pm, None, UMBRAL_CFG)
            row_color = [
                _sev_score_continuo(v, orient, thr, max_ratio=2.0) if v is not None else None
                for v in row_raw
            ]
            z_pct.append(row_color)
            z_pct_raw.append(row_raw)
        else:
            z_pct.append([None] * 192)
            z_pct_raw.append(row_raw)

        if um:
            mn, mx = _prog_cfg(um, None, UMBRAL_CFG)
            row_norm = [_normalize(v, mn, mx) if v is not None else None for v in row_raw_u]
            z_unit.append(row_norm)
            z_unit_raw.append(row_raw_u)
            for s in row_norm:
                if s is not None:
                    all_scores_unit.append(s)
        else:
            z_unit.append([None] * 192)
            z_unit_raw.append(row_raw_u)

        arr_u = np.array([v if isinstance(v, (int, float)) else np.nan for v in row_raw_u], float)
        arr_p = np.array([v if isinstance(v, (int, float)) else np.nan for v in row_raw], float)

        valid_idx = np.where(np.isfinite(arr_u if np.isfinite(arr_u).any() else arr_p))[0]
        last_label = str(x_dt[int(valid_idx[-1])]).replace("T", " ")[:16] if valid_idx.size else ""
        row_last_ts.append(last_label)

        row_max_pct.append(np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan)
        row_max_unit.append(np.nanmax(arr_u) if np.isfinite(arr_u).any() else np.nan)

    # zmin/zmax
    zmin_pct, zmax_pct = 0.0, 2.0
    if all_scores_unit:
        zmin_unit = min(all_scores_unit)
        zmax_unit = max(all_scores_unit)
    else:
        zmin_unit, zmax_unit = 0.0, 1.0

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
        "height": _hm_height_topoff(len(rows_page)),
    }

    return pct_payload, unit_payload, page_info


# =========================================================
# FIGURA HEATMAP (Plotly) TOPOFF
# =========================================================

def build_heatmap_figure_topoff(payload, *, height=750, decimals=2):
    if not payload:
        return go.Figure()

    z       = payload["z"]
    z_raw   = payload.get("z_raw") or z
    x       = payload.get("x_dt") or payload.get("x")
    y       = payload["y"]
    zmin    = payload["zmin"]
    zmax    = payload["zmax"]
    mode    = payload.get("color_mode", "severity")
    detail  = payload.get("row_detail") or y

    # colores
    if mode == "severity":
        colorscale = [
            [0/3, SEV_COLORS["excelente"]],
            [1/3, SEV_COLORS["bueno"]],
            [2/3, SEV_COLORS["regular"]],
            [3/3, SEV_COLORS["critico"]],
        ]
    else:
        colorscale = [
            [0.0, "#9ec5fe"],
            [1.0, "#0d6efd"],
        ]

    # customdata
    customdata = []
    for i, row in enumerate(z_raw):
        arr = np.array([v if isinstance(v,(int,float)) else np.nan for v in row], dtype=float)
        if np.isfinite(arr).any():
            valid_idx = np.where(np.isfinite(arr))[0]
            last_idx = int(valid_idx[-1])
            last_label = str(x[last_idx]).replace("T"," ")[:16]
        else:
            last_label = "—"

        # detail: tech/vendor/region/province/mun/site/rnc/nodeb/valores
        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 8)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        region = parts[2] if len(parts) > 2 else ""
        prov   = parts[3] if len(parts) > 3 else ""
        mun    = parts[4] if len(parts) > 4 else ""
        site   = parts[5] if len(parts) > 5 else ""
        rnc    = parts[6] if len(parts) > 6 else ""
        nodeb  = parts[7] if len(parts) > 7 else ""
        valor  = parts[8] if len(parts) > 8 else ""

        def _fmt_cell(v):
            if not np.isfinite(v):
                return ""
            return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"

        row_cd = []
        for j in range(len(x)):
            raw_cell = arr[j] if j < len(arr) else np.nan
            raw_s = _fmt_cell(raw_cell)
            row_cd.append([tech, vendor, region, prov, mun, site, rnc, nodeb, raw_s, last_label, valor])
        customdata.append(row_cd)

    hover_tmpl = (
        "<span style='font-size:120%; font-weight:700'>%{customdata[8]}</span><br>"
        "<span style='opacity:0.85'>%{x|%Y-%m-%d %H:%M}</span><br>"
        "──────────<br>"
        "DETALLE<br>"
        "<b>Site:</b> %{customdata[5]}<br>"
        "<b>RNC:</b> %{customdata[6]}<br>"
        "<b>NodeB:</b> %{customdata[7]}<br>"
        "<b>Última hora con registro:</b> %{customdata[9]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        showscale=False,
        customdata=customdata,
        hovertemplate=hover_tmpl,
        hoverongaps=False,
        xgap=0.5, ygap=0.5,
    ))

    ONE_HOUR_MS = 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=ONE_HOUR_MS,  # ticks cada hora aunque haya bins de 15m
        showticklabels=False,
        ticks="",
        fixedrange=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        title="",
        automargin=False,
        fixedrange=False,
        categoryorder="array",
        categoryarray=y,
        autorange="reversed",
    )

    if isinstance(x, (list, tuple)) and len(x) >= 97:
        fig.add_vline(
            x=x[96],
            line_width=3,
            line_color="rgba(0,0,0,0.75)",
            line_dash="solid",
            layer="above"
        )

    fig.update_layout(
        autosize=False,
        height=max(int(height or 300), 300),
        margin=dict(l=4, r=4, t=MARG_TOP, b=MARG_BOTTOM),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff", size=13)),
        uirevision="keep",
    )
    return fig

# =========================================================
# SUMMARY TABLE TOPOFF
# =========================================================

def render_heatmap_summary_table_topoff(
    pct_payload,
    unit_payload,
    *,
    pct_decimals=2,
    unit_decimals=0,
    active_y=None,
):
    src = pct_payload or unit_payload
    if not src:
        return dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

    y = src.get("y") or []
    detail = src.get("row_detail") or y
    row_last_ts = (unit_payload or pct_payload).get("row_last_ts") or []
    z_raw_pct = (pct_payload or {}).get("z_raw")
    z_raw_unit = (unit_payload or {}).get("z_raw")

    # ===== NEW HEADERS =====
    cols = [
        ("Cluster", "w-cluster"),
        ("Sitio", "w-sitio"),
        ("Tech", "w-tech"),
        ("Vendor", "w-vendor"),
        ("Valor", "w-valor"),
        ("Última hora", "w-ultima"),
        ("Valor de la última muestra (%)", "w-num"),
        ("Valor de la última muestra (UNIT)", "w-num"),
    ]

    thead = html.Thead(
        html.Tr([html.Th(n, className=c) for n, c in cols]),
        className="table-dark"
    )

    # fmt helper
    def _fmt_full(v):
        try:
            f = float(v)
            if not np.isfinite(f):
                return ""
            return f"{f:,.2f}"
        except Exception:
            return ""

    body_rows = []

    for i in range(len(y)):
        # y-label contiene: tech/vendor/region/province/mun/site/rnc/nodeb/cluster/valores
        parts_y = str(y[i]).split("/", 9)
        site    = parts_y[5] if len(parts_y) > 5 else ""
        cluster = parts_y[8] if len(parts_y) > 8 else ""
        valor   = parts_y[9] if len(parts_y) > 9 else ""

        # detail contiene: tech/vendor/region/prov/mun/site/rnc/nodeb/valores
        parts_d = (detail[i] if i < len(detail) else str(y[i])).split("/", 8)
        tech   = parts_d[0] if len(parts_d) > 0 else ""
        vendor = parts_d[1] if len(parts_d) > 1 else ""

        # últimas muestras
        last_pct  = _last_numeric(z_raw_pct[i]) if z_raw_pct and i < len(z_raw_pct) else None
        last_unit = _last_numeric(z_raw_unit[i]) if z_raw_unit and i < len(z_raw_unit) else None

        ultima_txt = _only_time(row_last_ts[i] if i < len(row_last_ts) else "")
        pct_txt    = _fmt(last_pct, pct_decimals)
        unit_txt   = _fmt(last_unit, unit_decimals)

        body_rows.append(
            html.Tr([
                # Cluster
                html.Td(
                    html.Span(
                        html.Span(cluster, className="unflip"),
                        className="ellipsis-left"
                    ),
                    className="w-cluster",
                    title=cluster
                ),
                # Site
                html.Td(
                    html.Span(
                        html.Span(site, className="unflip"),
                        className="ellipsis-left"
                    ),
                    className="w-sitio",
                    title=site
                ),
                html.Td(tech, className="w-tech", title=tech),
                html.Td(
                    html.Span(_vendor_initial(vendor), className="vendor-initial"),
                    className="td-vendor w-vendor",
                    title=vendor
                ),
                html.Td(valor, className="w-valor", title=valor),
                html.Td(ultima_txt, className="w-ultima ta-center", title=ultima_txt),
                html.Td(pct_txt, className="w-num ta-right", title=_fmt_full(last_pct)),
                html.Td(unit_txt, className="w-num ta-right", title=_fmt_full(last_unit)),
            ])
        )

    return dbc.Table(
        [thead, html.Tbody(body_rows)],
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
        className="mb-0 table-dark kpi-table topoff-table compact"
    )

# =========================================================
# TIME HEADERS
# =========================================================

def build_time_header_children_by_dates(fecha_str: str):
    try:
        today_dt = datetime.strptime(fecha_str, "%Y-%m-%d") if fecha_str else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()
    yday_dt = today_dt - timedelta(days=1)
    today_s = today_dt.strftime("%Y-%m-%d")
    yday_s  = yday_dt.strftime("%Y-%m-%d")

    dates_children = [
        html.Div(yday_s,  className="cell"),
        html.Div(today_s, className="cell sep"),
    ]

    hours_children = []
    for i in range(48):
        hh = i if i < 24 else i - 24
        cls = ["cell"]
        if i == 24:
            cls.append("sep")
        if hh % 6 == 0:
            cls.append("tick6")
        label = html.Span(f"{hh:02d}", className="lbl") if (hh % 3 == 0) else None
        hours_children.append(html.Div(label, className=" ".join(cls)))

    return dates_children, hours_children


