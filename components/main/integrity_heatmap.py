import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from dash import html
import dash_bootstrap_components as dbc
from components.main.heatmap import _infer_networks, _max_date_str, _day_str, _normalize


def vendor_disp(v):
    """Devuelve una “sigla” simple del vendor (primera letra en mayúscula)."""
    s = "" if v is None or pd.isna(v) else str(v).strip()
    return (s[0].upper()) if s else ""


def _safe_dt_col(df: pd.DataFrame) -> pd.Series:
    """Construye una columna datetime segura usando fecha+hora (si no existe, devuelve NaT)."""
    if "fecha" not in df.columns or "hora" not in df.columns:
        return pd.to_datetime(pd.NaT)
    return pd.to_datetime(
        df["fecha"].astype(str).str.strip() + " " + df["hora"].astype(str).str.strip(),
        errors="coerce"
    )


def _fmt_last_ts(ts):
    """Formatea el timestamp para mostrarlo como HH:MM (fallback simple si falla)."""
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).strftime("%H:%M")  # solo hora
    except Exception:
        s = str(ts)
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
    """
    Builder principal: arma payloads (pct y unit) para un heatmap de INTEGRITY (48 horas: ayer+hoy).
    - pct_payload: integra % (0..100) con máscara de “missing” para celdas sin dato.
    - unit_payload: integra UNIT normalizado 0..1 (min/max dinámico de lo visible).
    - page_info: paginación sobre filas (cluster+vendor+tech+network).
    """

    # Sin meta o sin TS no se puede construir nada
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}
    if df_ts is None or df_ts.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Redes (si no se pasan, se infieren del TS)
    if not networks:
        networks = _infer_networks(df_ts)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Fechas hoy/ayer (para crear el eje de 48 horas)
    if today is None:
        today = _max_date_str(df_ts["fecha"]) if "fecha" in df_ts.columns else None
        today = today or _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Columnas esperadas para este heatmap
    pct_col = "integrity_deg_pct"
    unit_col = "integrity"

    # Si faltan columnas, no se puede armar el heatmap
    missing = [c for c in [pct_col, unit_col] if c not in df_ts.columns]
    if missing:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Base meta única (technology/vendor/noc_cluster) y orden estable “simple”
    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].copy()
    base = base.sort_values(["noc_cluster", "vendor", "technology"], kind="stable").reset_index(drop=True)

    # Expande filas: (tech/vendor/cluster) x network
    rows_full = base.assign(_tmp=1).merge(
        pd.DataFrame({"network": networks, "_tmp": 1}),
        on="_tmp"
    ).drop(columns="_tmp")

    # RID global: id entero por fila para indexar rápido matrices/lookup
    rows_full = rows_full.reset_index(drop=True)
    rows_full["rid"] = np.arange(len(rows_full), dtype=int)

    # keys_df: mapa global (tech/vendor/cluster/network -> rid)
    keys_df = rows_full[["technology", "vendor", "noc_cluster", "network", "rid"]].copy()

    # Filtra TS a ayer/hoy y deja solo columnas requeridas
    base_cols = ["technology", "vendor", "noc_cluster", "network", "fecha", "hora"]
    used_cols = base_cols + [pct_col, unit_col]
    df2 = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]), used_cols].copy()

    # Cruza TS con las filas válidas (keys_df) para asegurar consistencia
    df_small = df2.merge(keys_df, on=["technology", "vendor", "noc_cluster", "network"], how="inner")

    # Convierte hora en h (0..23) y crea offset48 (0..47) para indexar ayer/hoy
    hh = df_small["hora"].astype(str).str.split(":", n=1).str[0]
    df_small["h"] = pd.to_numeric(hh, errors="coerce")
    df_small = df_small[(df_small["h"] >= 0) & (df_small["h"] <= 23)].copy()
    df_small["h"] = df_small["h"].astype(int)

    df_small["offset48"] = df_small["h"] + np.where(df_small["fecha"].astype(str) == today, 24, 0)
    df_small["offset48"] = pd.to_numeric(df_small["offset48"], errors="coerce").fillna(-1).astype(int)
    df_small = df_small[(df_small["offset48"] >= 0) & (df_small["offset48"] <= 47)].copy()

    # Crea mapas rápidos (rid, offset48) -> valor para cada métrica
    metric_maps = {}
    for m in (pct_col, unit_col):
        sub = df_small[["rid", "offset48", m]].dropna()
        metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset48"]), sub[m]))

    # ===== Orden opcional: degradación reciente + racha consecutiva =====
    if sort_by_degrade:
        mp_pct = metric_maps.get(pct_col) or {}

        def _last_off(rid: int):
            """Encuentra el offset48 más reciente que tenga % numérico."""
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
            """
            Construye clave de orden:
            1) qué tan reciente fue la última degradación (<degrade_thr)
            2) longitud de racha degradada
            3) qué tan “malo” fue el % (más bajo primero)
            """
            lo = _last_off(rid)
            if lo is None:
                return (10**9, 0, 10**9)  # sin datos: al final

            found_off = None
            found_pct = None
            recency = 10**9

            # Busca la degradación más cercana desde la última muestra hacia atrás
            for d in range(0, lo + 1):
                off = lo - d
                v = mp_pct.get((rid, off))
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv) and fv < float(degrade_thr):
                        recency = d
                        found_off = off
                        found_pct = fv
                        break
                except Exception:
                    pass

            if found_off is None:
                return (10**9, 0, 10**9)  # nunca degradó: al final

            # Calcula racha consecutiva hacia atrás desde found_off
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

            # Orden: más reciente primero, racha más larga primero, % más bajo primero
            return (recency, -streak, float(found_pct) if found_pct is not None else 10**9)

        # Aplica claves de orden y sort estable
        keys = [_sort_key(rid) for rid in rows_full["rid"].to_list()]
        rows_full["_recency"] = [k[0] for k in keys]
        rows_full["_streak"] = [k[1] for k in keys]
        rows_full["_pctbad"] = [k[2] for k in keys]

        rows_full = rows_full.sort_values(
            ["_recency", "_streak", "_pctbad", "noc_cluster", "vendor", "technology", "network"],
            kind="stable"
        ).reset_index(drop=True)

        # ===== Paginación (después del sort) =====
        total_rows = len(rows_full)
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        rows_page = rows_full.iloc[start:end].reset_index(drop=True)

    def _row48(metric, rid):
        """Devuelve la fila de 48 puntos para una métrica dada (usando el mapa rid/offset48)."""
        mp = metric_maps.get(metric) or {}
        return [mp.get((rid, off)) for off in range(48)]

    # Eje X: 48 timestamps (ayer 0..23 + hoy 24..47)
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]

    # Metadatos por fila (para hover/tablas)
    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    # Buffers temporales para construir matrices crudas y máscara de missing
    tmp_unit_raws = []
    tmp_pct_raws = []
    missing_mask = []

    def _is_num(x):
        """True si x se puede convertir a float finito."""
        try:
            fx = float(x)
            return np.isfinite(fx)
        except Exception:
            return False

    def _is_finite_num(x):
        """Alias simple: numérico finito."""
        try:
            fx = float(x)
            return np.isfinite(fx)
        except Exception:
            return False

    # Detecta qué offsets (0..47) tienen “alguna muestra” global (para marcar missing real)
    present_off = set()
    if not df_small.empty and "offset48" in df_small.columns:
        m_present = df_small[pct_col].map(_is_finite_num) | df_small[unit_col].map(_is_finite_num)
        if m_present.any():
            present_off = set(df_small.loc[m_present, "offset48"].astype(int).tolist())
    global_last_off = max(present_off) if present_off else -1

    # Construye filas visibles
    for r in rows_page.itertuples(index=False):
        tech, vend, clus, net = r.technology, r.vendor, r.noc_cluster, r.network
        rid = int(r.rid)

        # ID de fila consistente con el heatmap
        y_id = f"{tech}/{vend}/{clus}/{net}/INTEGRITY"
        y_labels.append(y_id)
        row_detail.append(y_id)

        raw_pct = _row48(pct_col, rid)
        raw_unit = _row48(unit_col, rid)

        # Último offset con alguna muestra (pct o unit)
        last_obs_off = -1
        for off, (p, u) in enumerate(zip(raw_pct, raw_unit)):
            if _is_num(p) or _is_num(u):
                last_obs_off = off

        # Máscaras:
        # - pct: se muestra si es numérico (0..100), si no -> None
        # - missing_mask: marca “missing” cuando esa hora existe en el sistema pero esta fila no reportó
        # - unit: se muestra solo si pct >= min_pct_ok (o si pct es NaN, se permite)
        pct_masked = []
        unit_masked = []
        miss_row = []

        for off, (p, u) in enumerate(zip(raw_pct, raw_unit)):
            try:
                fp = float(p)
                pct_ok_num = np.isfinite(fp)
            except Exception:
                fp = None
                pct_ok_num = False

            if pct_ok_num:
                fp = max(0.0, min(100.0, fp))
                pct_masked.append(fp)
                miss_row.append(None)
            else:
                pct_masked.append(None)
                # Missing “real”: el offset existe globalmente y no es futuro
                miss_row.append(
                    1 if (off in present_off and (global_last_off < 0 or off <= global_last_off)) else None
                )

            # Regla para UNIT: solo mostrar si el % pasa el umbral (o si no hay %)
            if pct_ok_num and fp is not None:
                ok_unit = fp >= float(min_pct_ok)
            else:
                ok_unit = True

            unit_masked.append(u if ok_unit else None)

        tmp_pct_raws.append(pct_masked)
        tmp_unit_raws.append(unit_masked)
        missing_mask.append(miss_row)

        # Stats por fila (para tabla/resumen)
        arr_u = np.array([v if isinstance(v, (int, float)) else np.nan for v in unit_masked], float)
        arr_p = np.array([v if isinstance(v, (int, float)) else np.nan for v in pct_masked], float)

        rmax_u = np.nanmax(arr_u) if np.isfinite(arr_u).any() else np.nan
        rmax_p = np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan

        valid_idx = np.where(np.isfinite(arr_p) | np.isfinite(arr_u))[0]
        last_label = str(x_dt[int(valid_idx[-1])]).replace("T", " ")[:16] if valid_idx.size else ""

        row_last_ts.append(last_label)
        row_max_pct.append(rmax_p)
        row_max_unit.append(rmax_u)

    # PCT: se usa crudo (0..100) y color_mode progress (gradiente)
    z_pct_raw = tmp_pct_raws
    z_pct = z_pct_raw

    # UNIT: normaliza dinámicamente con min/max de lo visible en esta página
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

    # Payload de % (integridad en porcentaje)
    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",
        "zmin": 0.0,
        "zmax": 100.0,
        "color_theme": "pct_rg_80",
        "title": "Integridad (%)",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
        "missing_mask": missing_mask,
    }

    # Payload de UNIT (integridad en unidades normalizadas 0..1)
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


def render_integrity_summary_table(
    df_ts: pd.DataFrame,
    pct_payload: dict,
    nets_heat: list,
    integrity_baseline_map: dict | None = None,
):
    """
    Render UI: construye una tabla resumen (Dash/DBC) usando el mismo orden de filas del heatmap.
    Muestra: cluster, tech, vendor, última hora, % último, baseline (trend), y UNIT último.
    """
    if df_ts is None or df_ts.empty or not pct_payload:
        return dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

    # Normaliza strings clave para evitar mismatches en filtros
    df = df_ts.copy()
    for c in ["network", "vendor", "noc_cluster", "technology"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Columna datetime segura para calcular el último registro por fila
    df["_ts"] = _safe_dt_col(df)

    # Usa EXACTAMENTE las filas del heatmap (mismo orden) para armar la tabla
    detail = pct_payload.get("row_detail") or pct_payload.get("y") or []

    rows = []
    for y_id in detail:
        # y_id: tech/vendor/cluster/network/INTEGRITY
        parts = str(y_id).split("/", 4)
        tech = parts[0] if len(parts) > 0 else ""
        vend = parts[1] if len(parts) > 1 else ""
        clus = parts[2] if len(parts) > 2 else ""
        net  = parts[3] if len(parts) > 3 else ""

        # Filtra TS a esa fila específica
        m = (
            (df.get("noc_cluster") == clus) &
            (df.get("technology") == tech) &
            (df.get("vendor") == vend) &
            (df.get("network") == net)
        )
        sub = df.loc[m]

        # Calcula último timestamp y promedios del último bloque
        if sub.empty:
            last_str = last_pct = last_unit = ""
        else:
            last_ts = sub["_ts"].max()
            sub_last = sub[sub["_ts"] == last_ts]
            last_str = _fmt_last_ts(last_ts)

            v_pct = pd.to_numeric(sub_last.get("integrity_deg_pct"), errors="coerce").mean()
            last_pct = "" if pd.isna(v_pct) else f"{v_pct:.2f}"

            v_u = pd.to_numeric(sub_last.get("integrity"), errors="coerce").mean()
            last_unit = "" if pd.isna(v_u) else f"{v_u:.0f}"

        # Baseline/trend (si se proporciona mapa de referencia)
        integrity_baseline_map = integrity_baseline_map or {}
        base_val = integrity_baseline_map.get(
            (str(net).strip(), str(vend).strip(), str(clus).strip(), str(tech).strip())
        )
        trend = "" if base_val is None or pd.isna(base_val) else f"{float(base_val):.0f}"

        # Arma fila HTML (Dash)
        rows.append(html.Tr([
            html.Td(clus, className="w-cluster"),
            html.Td(tech, className="w-tech"),
            html.Td(vendor_disp(vend), title=vend, className="w-vendor"),
            html.Td(last_str, className="w-ultima"),
            html.Td(last_pct, className="w-num ta-right"),
            html.Td(trend, className="w-num ta-right"),  # baseline (trend)
            html.Td(last_unit, className="w-num ta-right"),
        ]))

    # Tabla final con estilos (dark/compact)
    return dbc.Table(
        [html.Tbody(rows)],
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
        className="mb-0 table table-dark table-hover kpi-table kpi-table-summary compact",
    )
