import math
import pandas as pd
from dash import Input, Output, State, ALL, no_update, ctx
import time
from datetime import timedelta
import numpy as np
from components.main.main_table import (
    pivot_by_network,
    render_kpi_table_multinet,
    strip_net, prefixed_progress_cols,
)
import dash_bootstrap_components as dbc

from src.Utils.alarmados import add_alarm_streak
from src.dataAccess.data_access import fetch_kpis, COLMAP, fetch_kpis_paginated_severity_global_sort, \
    fetch_kpis_paginated_severity_sort, fetch_integrity_baseline_week, fetch_kpis_by_keys
from src.config import REFRESH_INTERVAL_MS

from src.Utils.utils_time import now_local
from src.dataAccess.data_acess_topoff import fetch_topoff_distinct
from dash.exceptions import PreventUpdate

_DFTS_CACHE = {}
_DFTS_TTL = 300

_MAIN_CTX_CACHE = {}
_MAIN_CTX_TTL = 120
MOCK_INTEGRITY_BASELINE = False
MOCK_BASELINE_MULT = 1.25
MOCK_ONLY_NETWORKS = {"NET", "ATT", "TEF"}
PREFERRED_NET_ORDER = ["NET", "ATT", "TEF"]
_LAST_HEATMAP_KEY = None
_LAST_HI_KEY = None

HOLD_SECONDS = 600  # 10 minutos (ajústalo)

def order_networks(nets):
    first = [n for n in PREFERRED_NET_ORDER if n in nets]
    rest = sorted([n for n in nets if n not in PREFERRED_NET_ORDER])
    return first + rest
def _ensure_df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def round_down_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def _keep_valid(selected, valid_values):
    if not valid_values:
        return []
    if selected is None:
        return []
    if not isinstance(selected, (list, tuple)):
        selected = [selected]
    valid_set = set(valid_values)
    filtered = [v for v in selected if v in valid_set]
    return filtered

def _compute_progress_max_for_filters(fecha, hora, networks, technologies, vendors, clusters):
    """
    Calcula el máximo de cada columna de progress usando TODAS las filas
    filtradas (sin paginar) para que las barras no dependan de la página actual.
    """
    networks = _as_list(networks)
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    clusters = _as_list(clusters)

    # Dataset completo filtrado por fecha/hora + filtros, sin paginación
    df_full = fetch_kpis(
        fecha=fecha,
        hora=hora,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=None,
    )
    df_full = _ensure_df(df_full)

    if df_full.empty:
        return {}

    # Inferir redes efectivas si no vienen fijas por filtro
    if networks:
        nets = networks
    else:
        nets = (
            sorted(df_full["network"].dropna().unique().tolist())
            if "network" in df_full.columns
            else []
        )

    if not nets:
        return {}

    # Pasamos a formato wide para tener columnas tipo NET__metric
    df_wide_full = pivot_by_network(df_full, networks=nets)
    if df_wide_full is None or df_wide_full.empty:
        return {}

    progress_cols = prefixed_progress_cols(nets)
    max_dict = {}

    for col in progress_cols:
        if col in df_wide_full.columns:
            serie = df_wide_full[col]
            # ignorar NaN / inf
            valid = serie.replace([np.inf, -np.inf], np.nan).dropna()
            max_dict[col] = float(valid.max()) if not valid.empty else None
        else:
            max_dict[col] = None

    return max_dict

def _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters):
    def _norm(x):
        x = _as_list(x) or []
        return tuple(sorted(x))
    return (
        "main_ctx",
        fecha, hora,
        _norm(networks),
        _norm(technologies),
        _norm(vendors),
        _norm(clusters),
    )

def _get_main_context_cached(fecha, hora, networks, technologies, vendors, clusters):
    key = _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters)
    now = time.time()
    hit = _MAIN_CTX_CACHE.get(key)
    if hit and (now - hit["ts"] < _MAIN_CTX_TTL):
        return hit["data"]
    return None

def _set_main_context_cached(fecha, hora, networks, technologies, vendors, clusters, data):
    key = _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters)
    _MAIN_CTX_CACHE[key] = {"ts": time.time(), "data": data}

def _is_integrity_pct_sort(sort_state: dict | None) -> bool:
    col = (sort_state or {}).get("column")
    return bool(col) and (str(col).endswith("__integrity_deg_pct") or col == "integrity_deg_pct")

def _norm_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s

def _build_global_order_keys_by_integrity_pct(
    *,
    fecha, hora, networks, technologies, vendors, clusters,
    sort_state, integrity_baseline_map,
):
    """
    Devuelve (ordered_keys, sort_net) donde ordered_keys es list de tuples:
      (fecha,hora,vendor,noc_cluster,technology)
    ordenadas globalmente por % integridad (asc/desc) para la red clickeada.
    """
    col = (sort_state or {}).get("column")
    asc = bool((sort_state or {}).get("ascending", True))

    sort_net = None
    if col and "__" in col:
        sort_net = col.split("__", 1)[0]  # NET/ATT/TEF

    MIN_COLS = ["fecha", "hora", "vendor", "noc_cluster", "technology", "network", "integrity"]

    df_min = fetch_kpis(
        fecha=fecha,
        hora=hora,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=None,
        columns=MIN_COLS,
    )
    df_min = _ensure_df(df_min)
    if df_min.empty:
        return [], sort_net

    # normaliza fecha/hora para que el match por keys sea estable
    for c in ["fecha", "hora", "vendor", "noc_cluster", "technology", "network"]:
        if c in df_min.columns:
            df_min[c] = df_min[c].astype(str).str.strip()

    if sort_net and "network" in df_min.columns:
        df_min = df_min[df_min["network"] == sort_net]

    if df_min.empty or "integrity" not in df_min.columns:
        return [], sort_net

    def _health_pct_row(r):
        key = (r.get("network"), r.get("vendor"), r.get("noc_cluster"), r.get("technology"))
        baseline = integrity_baseline_map.get(key)
        integ = r.get("integrity")
        if baseline is None or baseline <= 0:
            return np.nan
        try:
            if integ is None or pd.isna(integ):
                return np.nan
            pct = (float(integ) / float(baseline)) * 100.0
            return max(0.0, min(100.0, pct))
        except Exception:
            return np.nan

    df_min = df_min.copy()
    df_min["health_pct"] = df_min.apply(_health_pct_row, axis=1)

    KEY_COLS = ["fecha", "hora", "vendor", "noc_cluster", "technology"]

    # una fila por key (si hay duplicados, toma el primero)
    g = df_min.groupby(KEY_COLS, dropna=False)["health_pct"].first().reset_index()

    # orden global (NaN al final)
    g = g.sort_values("health_pct", ascending=asc, na_position="last", kind="mergesort")

    ordered_keys = list(map(tuple, g[KEY_COLS].to_numpy()))
    return ordered_keys, sort_net

def register_callbacks(app):

    # -------------------------------------------------
    # 0) Actualiza opciones de Network y Technology
    # -------------------------------------------------
    @app.callback(
        Output("f-network", "options"),
        Output("f-network", "value"),
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),
        Input("refresh-timer", "n_intervals"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
    )
    def update_all_filters(_tick, fecha, hora,
                           net_val_current, tech_val_current,
                           ven_val_current, clu_val_current):
        # -------- 1) Traer DF principal para esa fecha/hora --------
        df_main = fetch_kpis(fecha=fecha, hora=None, limit=None)
        df_main = _ensure_df(df_main)

        # Catálogos base
        networks_all = order_networks(df_main["network"].dropna().unique().tolist()) \
            if "network" in df_main.columns else []
        techs_all = sorted(df_main["technology"].dropna().unique().tolist()) \
            if "technology" in df_main.columns else []

        # Normalizamos seleccionados para filtrar vendors/clusters
        nets_sel = _as_list(net_val_current)
        techs_sel = _as_list(tech_val_current)

        df_filtered = df_main.copy()
        if nets_sel and "network" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["network"].isin(nets_sel)]
        if techs_sel and "technology" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["technology"].isin(techs_sel)]

        vendors_main = sorted(df_filtered["vendor"].dropna().unique().tolist()) \
            if "vendor" in df_filtered.columns else []
        clusters_main = sorted(df_filtered["noc_cluster"].dropna().unique().tolist()) \
            if "noc_cluster" in df_filtered.columns else []

        # -------- 2) Merge con TOPOFF (para tech y vendors) --------
        top_opts = fetch_topoff_distinct(
            fecha=fecha,
            technologies=techs_sel or None,
            vendors=None,
        ) or {}

        techs_top = top_opts.get("technologies", []) or []
        vendors_top = top_opts.get("vendors", []) or []

        techs_all = sorted(set(techs_all) | set(techs_top))
        vendors_all = sorted(set(vendors_main) | set(vendors_top))

        # -------- 3) Construir opciones (sin ifs que devuelvan no_update) --------
        net_opts = [{"label": n, "value": n} for n in networks_all]
        tech_opts = [{"label": t, "value": t} for t in techs_all]
        ven_opts = [{"label": v, "value": v} for v in vendors_all]
        clu_opts = [{"label": c, "value": c} for c in clusters_main]

        # -------- 4) Mantener selección previa válida --------
        new_net_value = _keep_valid(net_val_current, networks_all)
        new_tech_value = _keep_valid(tech_val_current, techs_all)
        new_ven_value = _keep_valid(ven_val_current, vendors_all)
        new_clu_value = _keep_valid(clu_val_current, clusters_main)

        return (
            net_opts, new_net_value,
            tech_opts, new_tech_value,
            ven_opts, new_ven_value,
            clu_opts, new_clu_value,
        )
    # -------------------------------------------------
    # Botones de sort en headers
    # -------------------------------------------------
    @app.callback(
        Output("sort-state", "data"),
        Input({"type": "sort-btn", "col": ALL}, "n_clicks"),
        State("sort-state", "data"),
        prevent_initial_call=True,
    )
    def on_click_sort(n_clicks_list, sort_state):
        sort_state = sort_state or {"column": None, "ascending": True}

        # ✅ Bloquea disparos “fantasma” por re-render
        if not ctx.triggered:
            raise PreventUpdate

        trig = ctx.triggered[0]
        # Cuando se recrea el header, puede disparar con 0 o None
        if trig.get("value") in (None, 0):
            raise PreventUpdate

        trig_id = ctx.triggered_id
        if not trig_id or "col" not in trig_id:
            raise PreventUpdate

        clicked_col = trig_id["col"]

        if sort_state.get("column") in (clicked_col, strip_net(clicked_col)):
            sort_state["ascending"] = not sort_state.get("ascending", True)
        else:
            sort_state["column"] = clicked_col
            sort_state["ascending"] = True

        return sort_state

    # -------------------------------------------------
    # 2) Paginación: reset page cuando cambian filtros/tamaño
    # -------------------------------------------------
    @app.callback(
        Output("page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("page-size", "value"),
        prevent_initial_call=True,
    )
    def reset_page_on_filters(_fecha, _hora, _net, _tech, _ven, _clu, page_size):
        ps = max(1, int(page_size or 50))
        return {"page": 1, "page_size": ps}

    # -------------------------------------------------
    # 3) Botones Anterior/Siguiente → actualizan page-state
    # -------------------------------------------------
    @app.callback(
        Output("page-state", "data", allow_duplicate=True),
        Input("page-prev", "n_clicks"),
        Input("page-next", "n_clicks"),
        State("page-state", "data"),
        prevent_initial_call=True,
    )
    def paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "page-prev":
            page = max(1, page - 1)
        elif trig == "page-next":
            page = page + 1
        return {"page": page, "page_size": ps}

    # -------------------------------------------------
    # 4) Tabla + Charts + Indicadores de paginación (UN SOLO callback)
    #    ← Este callback es el ÚNICO que escribe en table-container y charts
    # -------------------------------------------------
    # ... (resto de imports iguales)
    @app.callback(
        Output("table-container", "children"),
        Output("page-indicator", "children"),
        Output("total-rows-banner", "children"),
        Output("table-page-data", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("f-sort-mode", "value"),
        Input("refresh-timer", "n_intervals"),
        Input("sort-state", "data"),
        Input("page-state", "data"),
        Input("main-context-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_table(
            fecha, hora, networks, technologies, vendors, clusters,
            sort_mode, _n, sort_state, page_state,
            main_ctx
    ):
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # Defaults de contexto por si aún no llega el store
        main_ctx = main_ctx or {}

        integrity_baseline_list = main_ctx.get("integrity_baseline_map") or []
        alarm_list = main_ctx.get("alarm_map") or []
        progress_max_by_col = main_ctx.get("progress_max_by_col") or {}

        # reconstruimos dict con llaves tupla SOLO para uso interno
        integrity_baseline_map = {
            (d.get("network"), d.get("vendor"), d.get("noc_cluster"), d.get("technology")): d.get("integrity_week_avg")
            for d in integrity_baseline_list
            if d is not None
        }

        alarm_map = {
            (
                d.get("fecha"),
                d.get("hora"),
                d.get("network"),
                d.get("vendor"),
                d.get("noc_cluster"),
                d.get("technology"),
            ): int(d.get("alarmas", 0) or 0)
            for d in alarm_list
            if d is not None
        }

        # ---------- orden explícito para alarmado / global (columna clickeada) ----------
        sort_by = None
        sort_net = None
        ascending = True

        if sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))

            if "__" in col:
                sort_net, base = col.split("__", 1)
                sort_by = base
            else:
                sort_by = col

        # =========================================================
        # ✅ sort GLOBAL por % integridad (integrity_deg_pct) SIN BD
        #    - Construye orden global de keys en Python
        #    - Trae SOLO las keys de la página
        # =========================================================
        df = None
        total = None
        safe_sort_state = None
        explicit_page_key_order = None  # para mantener orden al pivotear

        if _is_integrity_pct_sort(sort_state) and (sort_mode != "alarmado"):
            ordered_keys, _sort_net_clicked = _build_global_order_keys_by_integrity_pct(
                fecha=fecha, hora=hora,
                networks=networks, technologies=technologies,
                vendors=vendors, clusters=clusters,
                sort_state=sort_state,
                integrity_baseline_map=integrity_baseline_map,
            )

            total = len(ordered_keys)
            total_pages = max(1, math.ceil(total / max(1, page_size)))
            page_corrected = min(max(1, page), total_pages)

            start = (page_corrected - 1) * page_size
            end = start + page_size
            page_keys = ordered_keys[start:end]
            explicit_page_key_order = page_keys  # ✅ orden estable para el pivot

            df = fetch_kpis_by_keys(
                fecha=fecha,
                hora=hora,
                vendors=vendors or None,
                clusters=clusters or None,
                networks=networks or None,
                technologies=technologies or None,
                row_keys=page_keys,
            )
            df = _ensure_df(df)

            # para que el header muestre la flecha en % integridad
            safe_sort_state = sort_state

        # ---------- fuente paginada normal (solo si NO venimos del modo especial) ----------
        if df is None:
            if sort_mode == "alarmado":
                safe_sort_state = sort_state  # si quieres permitir reorder visual
                df, total = fetch_kpis_paginated_severity_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=technologies or None,
                    page=page, page_size=page_size,
                )
            else:
                # GLOBAL con sort opcional por columna
                safe_sort_state = sort_state if (sort_state and sort_state.get("column")) else None
                df, total = fetch_kpis_paginated_severity_global_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=technologies or None,
                    page=page, page_size=page_size,
                    sort_by_friendly=sort_by,
                    sort_net=sort_net,
                    ascending=ascending,
                )

        if df is None or df.empty:
            store_payload = {"columns": [], "rows": []}
            empty_alert = dbc.Alert("Sin datos para los filtros seleccionados.", color="warning")
            return empty_alert, "Página 1 de 1", "Sin resultados.", store_payload

        # ---------- Reordenar GLOBAL por bucket de completitud (solo modo global) ----------
        # ⚠️ NO lo apliques cuando el usuario está ordenando por % integridad, porque rompería ese orden.
        if (sort_mode != "alarmado") and (not _is_integrity_pct_sort(sort_state)):
            def _health_pct_row(row):
                net = row.get("network")
                vendor_val = row.get("vendor")
                cluster_val = row.get("noc_cluster")
                tech_val = row.get("technology")
                integ_val = row.get("integrity")

                if not integrity_baseline_map:
                    return None

                key = (net, vendor_val, cluster_val, tech_val)
                baseline = integrity_baseline_map.get(key)

                if (
                        baseline is None
                        or baseline <= 0
                        or not isinstance(integ_val, (int, float))
                        or pd.isna(integ_val)
                ):
                    return None

                ratio = float(integ_val) / float(baseline)
                health_pct = max(0.0, min(100.0, ratio * 100.0))
                return health_pct

            df["integrity_health_pct"] = df.apply(_health_pct_row, axis=1)

            df["complete_bucket"] = df["integrity_health_pct"].apply(
                lambda x: 0 if isinstance(x, (int, float)) and x >= 80.0 else 1
            )

            df = df.sort_values(
                by=["complete_bucket"],
                ascending=[True],
                kind="mergesort",
            )

        # ---------- alarmas (sin apply) ----------
        if "network" in df.columns:
            key_cols_alarm = ["fecha", "hora", "network", "vendor", "noc_cluster", "technology"]
            if all(k in df.columns for k in key_cols_alarm) and alarm_map:
                df["fecha"] = df["fecha"].astype(str).str.strip()
                df["hora"] = df["hora"].astype(str).str.strip()
                keys = list(map(tuple, df[key_cols_alarm].to_numpy()))
                df["alarmas"] = [alarm_map.get(k, 0) for k in keys]
            else:
                df["alarmas"] = 0
        else:
            df["alarmas"] = 0

        # ---------- inferir nets ----------
        if networks:
            nets = networks
        else:
            nets_raw = df["network"].dropna().unique().tolist() if "network" in df.columns else []
            nets = order_networks(nets_raw)

        # ---------- pivot + orden estable visual ----------
        key_cols = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
        use_df = df

        if all(k in df.columns for k in key_cols) and nets:
            # ✅ si venimos del sort global por % integridad, respeta el orden de la página
            tuples_in_order = (
                    explicit_page_key_order
                    or list(dict.fromkeys(map(tuple, df[key_cols].itertuples(index=False, name=None))))
            )
            order_map = {t: i for i, t in enumerate(tuples_in_order)}

            wide = pivot_by_network(df, networks=nets)
            if wide is not None and not wide.empty:
                wide["_ord"] = wide[key_cols].apply(
                    lambda r: order_map.get(tuple(r.values.tolist()), 10 ** 9),
                    axis=1,
                )
                wide = wide.sort_values("_ord").drop(columns=["_ord"])
                use_df = wide

        # ---------- render ----------
        table = render_kpi_table_multinet(
            use_df,
            networks=nets,
            sort_state=safe_sort_state,
            progress_max_by_col=progress_max_by_col,
            integrity_baseline_map=integrity_baseline_map,
        )

        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)

        indicator = f"Página {page_corrected} de {total_pages}"
        banner = (
            "Sin resultados."
            if (total or 0) == 0
            else f"Mostrando {(page_corrected - 1) * page_size + 1}–"
                 f"{min(page_corrected * page_size, total)} de {total} registros"
        )

        store_payload = {
            "columns": list(use_df.columns),
            "rows": use_df.to_dict("records"),
        }

        return table, indicator, banner, store_payload

    # -------------------------------------------------
    # 5) Intervalo global → sincroniza el del card (si aplica)
    # -------------------------------------------------
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False,
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS


    # -------------------------------------------------
    # 7) Tick: actualizar fecha/hora al inicio de cada hora
    # -------------------------------------------------
    @app.callback(
        Output("dt-manual-store", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        prevent_initial_call=True,
    )
    def mark_datetime_manual(_fecha, _hora):
        # Si no hay trigger real, no hagas nada
        if not ctx.triggered_id:
            raise PreventUpdate

        return {"last_manual_ts": time.time()}

    @app.callback(
        Output("f-hora", "value"),
        Output("f-fecha", "date"),
        Input("refresh-timer", "n_intervals"),
        State("f-hora", "value"),
        State("f-fecha", "date"),
        State("f-hora", "options"),
        State("dt-manual-store", "data"),
        prevent_initial_call=False,
    )
    def tick(n, current_hour, current_date, hour_options, manual_store):
        now = now_local()
        floored = round_down_to_hour(now)
        hh = floored.strftime("%H:00:00")
        today = floored.strftime("%Y-%m-%d")

        # Validar que la hora actual exista en options
        opt_values = {(o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])}
        if opt_values and hh not in opt_values:
            return no_update, no_update

        # ✅ 1) Primer arranque: fuerza "ahora"
        if n in (None, 0):
            return hh, today

        # ✅ 2) Hold inteligente
        last_manual_ts = float((manual_store or {}).get("last_manual_ts") or 0)

        # Calcula el timestamp del siguiente cambio de hora local
        next_hour_dt = floored + timedelta(hours=1)
        # Convertimos a epoch "naive"
        next_hour_ts = next_hour_dt.timestamp()
        now_ts = now.timestamp()

        # Hold hasta:
        # - X segundos desde edición manual
        # - o el siguiente cambio de hora (lo que ocurra primero)
        hold_until = min(last_manual_ts + HOLD_SECONDS, next_hour_ts)

        if last_manual_ts > 0 and now_ts < hold_until:
            return no_update, no_update

        # ✅ 3) Si ya pasó el hold, actualiza a "ahora"
        return hh, today

    @app.callback(
        Output("page-state", "data", allow_duplicate=True),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("page-size", "value"),
        Input("f-sort-mode", "value"),  # 👈 nuevo
        prevent_initial_call=True,
    )
    def reset_page_on_filters(_fecha, _hora, _net, _tech, _ven, _clu, page_size, _mode):
        ps = max(1, int(page_size or 50))
        return {"page": 1, "page_size": ps}

    @app.callback(
        Output("filters-collapse", "is_open"),
        Input("filters-toggle", "n_clicks"),
        State("filters-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_filters(n, is_open):
        return not is_open

    @app.callback(
        Output("sort-state", "data", allow_duplicate=True),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("f-sort-mode", "value"),  # si cambia entre 'global' y 'alarmado'
        prevent_initial_call=True,
    )
    def reset_sort_state_on_filters(_fecha, _hora, _net, _tech, _ven, _clu, _mode):
        # Vuelve al estado “sin columna seleccionada”
        return {"column": None, "ascending": True}

    @app.callback(
        Output("topoff-link-state", "data", allow_duplicate=True),
        Input({"type": "main-cluster-link", "cluster": ALL, "vendor": ALL, "technology": ALL}, "n_clicks"),
        State({"type": "main-cluster-link", "cluster": ALL, "vendor": ALL, "technology": ALL}, "id"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def sync_topoff_from_main(n_clicks_list, ids_list, current_state):

        if not ids_list:
            raise PreventUpdate

        safe_clicks = [(c or 0) for c in (n_clicks_list or [])]
        if max(safe_clicks, default=0) == 0:
            # ✅ evita falsos disparos por re-render
            raise PreventUpdate
        if max([(c or 0) for c in n_clicks_list]) == 0: PreventUpdate

        current_state = current_state or {"selected": None}
        current_sel = current_state.get("selected")

        trig = ctx.triggered_id
        if not trig:
            raise PreventUpdate

        new_sel = {
            "cluster": trig.get("cluster"),
            "vendor": trig.get("vendor"),
            "technology": trig.get("technology"),
        }

        if current_sel == new_sel:
            return {"selected": None}

        return {"selected": new_sel}


    @app.callback(
        Output("topoff-link-state", "data", allow_duplicate=True),
        Input("main-cluster-header-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_topoff_link_from_header(n):
        if not n:
            raise PreventUpdate
        # Limpia el link para TODOS los clusters
        return {"selected": None}

        # -------------------------------------------------
        # 1.5) Contexto pesado (baseline + progress max + alarm map)
        # -------------------------------------------------

    @app.callback(
        Output("main-context-store", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        prevent_initial_call=False,
    )
    def build_main_context(fecha, hora, networks, technologies, vendors, clusters):
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        # -----------------------------
        # 0) Cache (recomendación)
        #    - Para evitar el bug de "hora", el baseline NO debe cachearse por hora
        # -----------------------------
        # Puedes cachear todo por (fecha, hora, filtros) si quieres,
        # pero asegúrate de que el baseline se calcule sin usar hora.
        cached = _get_main_context_cached(fecha, hora, networks, technologies, vendors, clusters)
        if cached is not None:
            return cached

        # ============================================================
        # 1) BASELINE semanal de integridad (NO depende de hora)
        # ============================================================
        df_baseline = fetch_integrity_baseline_week(
            fecha=fecha,
            vendors=vendors or None,
            clusters=clusters or None,
            networks=networks or None,
            technologies=technologies or None,
        )
        df_baseline = df_baseline if isinstance(df_baseline, pd.DataFrame) else pd.DataFrame()

        integrity_baseline_list = []

        if not df_baseline.empty:
            # normaliza floats y strings (evita NaN)
            integrity_baseline_list = [
                {
                    "network": (None if pd.isna(r.get("network")) else str(r.get("network")).strip()),
                    "vendor": (None if pd.isna(r.get("vendor")) else str(r.get("vendor")).strip()),
                    "noc_cluster": (None if pd.isna(r.get("noc_cluster")) else str(r.get("noc_cluster")).strip()),
                    "technology": (None if pd.isna(r.get("technology")) else str(r.get("technology")).strip()),
                    "integrity_week_avg": (
                        None if pd.isna(r.get("integrity_week_avg")) else float(r.get("integrity_week_avg"))
                    ),
                }
                for _, r in df_baseline.iterrows()
            ]

        # ============================================================
        # 1.1) MOCK opcional (si NO hay baseline real)
        #      OJO: aquí conviene NO depender de hora, o al menos hacerlo estable
        # ============================================================
        elif MOCK_INTEGRITY_BASELINE:
            # sugerencia: usa hora=None para que no dependa del dropdown
            df_now = fetch_kpis(
                fecha=fecha,
                hora=None,  # <<<<<< clave: no te acoples a "Todos"
                vendors=vendors or None,
                clusters=clusters or None,
                networks=networks or None,
                technologies=technologies or None,
                limit=None,
            )
            df_now = _ensure_df(df_now)

            if not df_now.empty and "integrity" in df_now.columns:
                if "network" in df_now.columns and MOCK_ONLY_NETWORKS:
                    df_now = df_now[df_now["network"].isin(MOCK_ONLY_NETWORKS)]

                gcols = ["network", "vendor", "noc_cluster", "technology"]
                if all(c in df_now.columns for c in gcols):
                    df_g = (
                        df_now
                        .dropna(subset=["integrity"])
                        .groupby(gcols, dropna=False)["integrity"]
                        .mean()
                        .reset_index()
                    )

                    integrity_baseline_list = []
                    for _, r in df_g.iterrows():
                        cur = r.get("integrity")
                        if cur is None or (isinstance(cur, float) and pd.isna(cur)):
                            continue

                        integrity_baseline_list.append({
                            "network": r.get("network"),
                            "vendor": r.get("vendor"),
                            "noc_cluster": r.get("noc_cluster"),
                            "technology": r.get("technology"),
                            "integrity_week_avg": float(cur) * MOCK_BASELINE_MULT,
                        })

        # ============================================================
        # 2) Progress max (puede depender de hora si tu diseño lo requiere)
        # ============================================================
        progress_max_by_col = _compute_progress_max_for_filters(
            fecha=fecha,
            hora=hora,
            networks=networks,
            technologies=technologies,
            vendors=vendors,
            clusters=clusters,
        )

        # ============================================================
        # 3) Alarm map para el día completo (NO depende de hora)
        # ============================================================
        df_day = fetch_kpis(
            fecha=fecha,
            hora=None,  # día completo
            vendors=vendors or None,
            clusters=clusters or None,
            networks=networks or None,
            technologies=technologies or None,
            limit=None,
        )
        df_day = _ensure_df(df_day)

        def _norm_f(x):
            return None if x is None else str(x).strip()

        def _norm_h(x):
            return None if x is None else str(x).strip()

        alarm_list = []
        if not df_day.empty:
            df_day = add_alarm_streak(df_day)
            alarm_list = [
                {
                    "fecha": _norm_f(r.get("fecha")),
                    "hora": _norm_h(r.get("hora")),
                    "network": r.get("network"),
                    "vendor": r.get("vendor"),
                    "noc_cluster": r.get("noc_cluster"),
                    "technology": r.get("technology"),
                    "alarmas": int(r.get("alarmas", 0) or 0),
                }
                for _, r in df_day.iterrows()
            ]

        payload = {
            "integrity_baseline_map": integrity_baseline_list,
            "progress_max_by_col": progress_max_by_col,
            "alarm_map": alarm_list,
            "ts": time.time(),
        }

        _set_main_context_cached(fecha, hora, networks, technologies, vendors, clusters, payload)
        return payload
