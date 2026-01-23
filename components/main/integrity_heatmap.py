import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from dash import html
import dash_bootstrap_components as dbc
from components.main.heatmap import _infer_networks, _max_date_str, _day_str, _normalize

def vendor_disp(v):
    s = "" if v is None or pd.isna(v) else str(v).strip()
    return (s[0].upper()) if s else ""

def _safe_dt_col(df: pd.DataFrame) -> pd.Series:
    if "fecha" not in df.columns or "hora" not in df.columns:
        return pd.to_datetime(pd.NaT)
    return pd.to_datetime(df["fecha"].astype(str).str.strip() + " " + df["hora"].astype(str).str.strip(), errors="coerce")

def _fmt_last_ts(ts):
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).strftime("%H:%M")  # <- solo hora
    except Exception:
        s = str(ts)
        # fallback: intenta cortar lo último tipo "HH:MM"
        return s[-5:] if len(s) >= 5 else s

def build_integrity_heatmap_payloads_fast(
        df_meta: pd.DataFrame,
        df_ts: pd.DataFrame,
        *,
        networks=None,
        today=None,
        yday=None,
        min_pct_ok: float = 80.0,
        offset=0,
        limit=50,
        sort_by_degrade: bool = True,
        degrade_thr: float = 80.0,
        streak_cap: int = 3,
):
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    if df_ts is None or df_ts.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # redes
    if not networks:
        networks = _infer_networks(df_ts)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # fechas hoy/ayer
    if today is None:
        today = _max_date_str(df_ts["fecha"]) if "fecha" in df_ts.columns else None
        today = today or _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # columnas requeridas
    pct_col = "integrity_deg_pct"
    unit_col = "integrity"
    needed_cols = {"technology", "vendor", "noc_cluster", "network", "fecha", "hora", pct_col, unit_col}
    missing = [c for c in [pct_col, unit_col] if c not in df_ts.columns]
    if missing:
        # si no existen, no podemos armar el heatmap
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # base meta (technology/vendor/noc_cluster) * networks
    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].copy()
    # orden estable "simple"
    base = base.sort_values(["noc_cluster", "vendor", "technology"], kind="stable").reset_index(drop=True)

    rows_full = base.assign(_tmp=1).merge(
        pd.DataFrame({"network": networks, "_tmp": 1}),
        on="_tmp"
    ).drop(columns="_tmp")

    # RID global por fila (trío x network)
    rows_full = rows_full.reset_index(drop=True)
    rows_full["rid"] = np.arange(len(rows_full), dtype=int)

    # keys_df global (NO por página)
    keys_df = rows_full[["technology", "vendor", "noc_cluster", "network", "rid"]].copy()

    # filtra TS a ayer/hoy y keys visibles
    base_cols = ["technology", "vendor", "noc_cluster", "network", "fecha", "hora"]
    used_cols = base_cols + [pct_col, unit_col]
    df2 = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), used_cols].copy()

    df_small = df2.merge(keys_df, on=["technology", "vendor", "noc_cluster", "network"], how="inner")

    # offset48 0..47
    hh = df_small["hora"].astype(str).str.split(":", n=1).str[0]
    df_small["h"] = pd.to_numeric(hh, errors="coerce")
    df_small = df_small[(df_small["h"] >= 0) & (df_small["h"] <= 23)].copy()
    df_small["h"] = df_small["h"].astype(int)
    df_small["offset48"] = df_small["h"] + np.where(df_small["fecha"].astype(str) == today, 24, 0)
    df_small["offset48"] = pd.to_numeric(df_small["offset48"], errors="coerce").fillna(-1).astype(int)
    df_small = df_small[(df_small["offset48"] >= 0) & (df_small["offset48"] <= 47)].copy()

    # map (rid, offset48)->value
    metric_maps = {}
    for m in (pct_col, unit_col):
        sub = df_small[["rid", "offset48", m]].dropna()
        metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset48"]), sub[m]))
    # ===== ORDENAR POR DEGRADE RECIENTE + RACHA =====
    if sort_by_degrade:
        mp_pct = metric_maps.get(pct_col) or {}

        def _last_off(rid: int):
            # última muestra disponible (offset más reciente con % numérico)
            for off in range(47, -1, -1):
                v = mp_pct.get((rid, off))
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        return off
                except Exception:
                    pass
            return None

        def _sort_key(rid: int):
            lo = _last_off(rid)
            if lo is None:
                # sin datos: al final
                return (10 ** 9, 0, 10 ** 9)

            # busca el degrade más cercano desde la última muestra hacia atrás
            found_off = None
            found_pct = None
            recency = 10 ** 9

            for d in range(0, lo + 1):
                off = lo - d
                v = mp_pct.get((rid, off))
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv) and fv < float(degrade_thr):
                        recency = d  # 0 = última muestra ya degradada
                        found_off = off
                        found_pct = fv
                        break
                except Exception:
                    pass

            if found_off is None:
                # nunca degradó (<thr): al final
                return (10 ** 9, 0, 10 ** 9)

            # racha consecutiva hacia atrás desde found_off
            streak = 0
            off = found_off
            while off >= 0:
                v = mp_pct.get((rid, off))
                if v is None:
                    break
                try:
                    fv = float(v)
                    if np.isfinite(fv) and fv < float(degrade_thr):
                        streak += 1
                        off -= 1
                    else:
                        break
                except Exception:
                    break

            streak = min(int(streak), int(streak_cap or 3))

            # Orden: recency ASC (0 primero), streak DESC (por eso negativo), pct peor primero (ASC)
            return (recency, -streak, float(found_pct) if found_pct is not None else 10 ** 9)

        # Calcula claves y ordena estable
        keys = [_sort_key(rid) for rid in rows_full["rid"].to_list()]
        rows_full["_recency"] = [k[0] for k in keys]
        rows_full["_streak"] = [k[1] for k in keys]
        rows_full["_pctbad"] = [k[2] for k in keys]

        rows_full = rows_full.sort_values(
            ["_recency", "_streak", "_pctbad", "noc_cluster", "vendor", "technology", "network"],
            kind="stable"
        ).reset_index(drop=True)
        # ===== PAGINACIÓN (DESPUÉS DEL SORT) =====
        total_rows = len(rows_full)
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        rows_page = rows_full.iloc[start:end].reset_index(drop=True)

    def _row48(metric, rid):
        mp = metric_maps.get(metric) or {}
        return [mp.get((rid, off)) for off in range(48)]

    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]

    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []
    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    all_unit_norm = []

    # primero construimos raws + máscara por pct>=80
    tmp_unit_raws = []
    tmp_pct_raws = []

    for r in rows_page.itertuples(index=False):
        tech, vend, clus, net = r.technology, r.vendor, r.noc_cluster, r.network
        rid = int(r.rid)

        y_id = f"{tech}/{vend}/{clus}/{net}/INTEGRITY"
        y_labels.append(y_id)
        row_detail.append(y_id)

        raw_pct = _row48(pct_col, rid)
        raw_unit = _row48(unit_col, rid)

        # máscara:
        # - % SIEMPRE visible (si es numérico) y lo clampeamos a 0..100
        # - UNIT solo visible si % >= min_pct_ok
        pct_masked = []
        unit_masked = []
        for p, u in zip(raw_pct, raw_unit):
            try:
                fp = float(p)
                pct_ok_num = np.isfinite(fp)
            except Exception:
                fp = None
                pct_ok_num = False

            if pct_ok_num:
                # clamp 0..100
                fp = max(0.0, min(100.0, fp))
                pct_masked.append(fp)
            else:
                pct_masked.append(None)

            ok = pct_ok_num and fp >= float(min_pct_ok)
            unit_masked.append(u if ok else None)

        # MUY IMPORTANTE: guardar filas en los temporales
        tmp_pct_raws.append(pct_masked)
        tmp_unit_raws.append(unit_masked)

        # stats por fila
        arr_u = np.array([v if isinstance(v, (int, float)) else np.nan for v in unit_masked], float)
        arr_p = np.array([v if isinstance(v, (int, float)) else np.nan for v in pct_masked], float)

        rmax_u = np.nanmax(arr_u) if np.isfinite(arr_u).any() else np.nan
        rmax_p = np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan

        valid_idx = np.where(np.isfinite(arr_p) | np.isfinite(arr_u))[0]
        last_label = str(x_dt[int(valid_idx[-1])]).replace("T", " ")[:16] if valid_idx.size else ""

        row_last_ts.append(last_label)
        row_max_pct.append(rmax_p)
        row_max_unit.append(rmax_u)

    z_pct_raw = tmp_pct_raws
    z_pct = z_pct_raw

    # normalización UNIT: min/max dinámico de lo visible
    # recolecta valores
    vals_unit = []
    for row in tmp_unit_raws:
        for v in row:
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals_unit.append(float(v))
    if vals_unit:
        umin, umax = min(vals_unit), max(vals_unit)
    else:
        umin, umax = 0.0, 1.0

    z_unit_raw = tmp_unit_raws
    z_unit = [[_normalize(v, umin, umax) if v is not None else None for v in row] for row in z_unit_raw]

    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",   # usamos gradiente
        "zmin": 0.0,
        "zmax": 100.0,
        "color_theme": "pct_rg_80",
        "title": "Integridad (%)",
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
        "color_theme": "blue",
        "zmin": 0.0,
        "zmax": 1.0,
        "title": "Integridad (UNIT)",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }

    page_info = {"total_rows": total_rows, "offset": start, "limit": limit, "showing": len(rows_page)}
    return pct_payload, unit_payload, page_info

def render_integrity_summary_table(df_ts: pd.DataFrame, pct_payload: dict, nets_heat: list):
    if df_ts is None or df_ts.empty or not pct_payload:
        return dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

    df = df_ts.copy()
    for c in ["network", "vendor", "noc_cluster", "technology"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["_ts"] = _safe_dt_col(df)

    # filas EXACTAS (y orden) del heatmap
    detail = pct_payload.get("row_detail") or pct_payload.get("y") or []

    rows = []
    for y_id in detail:
        # tech/vendor/cluster/network/INTEGRITY
        parts = str(y_id).split("/", 4)
        tech = parts[0] if len(parts) > 0 else ""
        vend = parts[1] if len(parts) > 1 else ""
        clus = parts[2] if len(parts) > 2 else ""
        net  = parts[3] if len(parts) > 3 else ""

        m = (
            (df.get("noc_cluster") == clus) &
            (df.get("technology") == tech) &
            (df.get("vendor") == vend) &
            (df.get("network") == net)
        )
        sub = df.loc[m]

        if sub.empty:
            last_str = last_pct = last_unit = ""
        else:
            last_ts = sub["_ts"].max()
            sub_last = sub[sub["_ts"] == last_ts]
            last_str = _fmt_last_ts(last_ts)  # ya solo HH:MM

            v_pct = pd.to_numeric(sub_last.get("integrity_deg_pct"), errors="coerce").mean()
            last_pct = "" if pd.isna(v_pct) else f"{v_pct:.2f}"

            v_u = pd.to_numeric(sub_last.get("integrity"), errors="coerce").mean()
            last_unit = "" if pd.isna(v_u) else f"{v_u:.0f}"

        rows.append(html.Tr([
            html.Td(clus, className="w-cluster"),
            html.Td(tech, className="w-tech"),
            html.Td(vendor_disp(vend), title=vend, className="w-vendor"),
            html.Td(last_str, className="w-ultima"),
            html.Td(last_pct, className="w-num ta-right"),
            html.Td(last_unit, className="w-num ta-right"),
        ]))

    return dbc.Table(
        [html.Tbody(rows)],
        striped=True, bordered=False, hover=True, size="sm",
        className="mb-0 table table-dark table-hover kpi-table kpi-table-summary compact",
    )

