import math
import pandas as pd
from dash import Input, Output, State, ALL, no_update, ctx
import time
import numpy as np
from components.main.main_table import (
    pivot_by_network,
    render_kpi_table_multinet,
    strip_net, prefixed_progress_cols,
)
import dash_bootstrap_components as dbc

from src.Utils.alarmados import add_alarm_streak
from src.dataAccess.data_access import fetch_kpis, COLMAP, fetch_kpis_paginated_severity_global_sort, \
    fetch_kpis_paginated_severity_sort, fetch_integrity_baseline_week
from src.config import REFRESH_INTERVAL_MS

from src.Utils.utils_time import now_local
from src.dataAccess.data_acess_topoff import fetch_topoff_distinct
from dash.exceptions import PreventUpdate

#Cach simple en memoria para df_ts
_DFTS_CACHE = {}
_DFTS_TTL = 300  # segundos

# Última clave renderizada para evitar re-render idéntico
_LAST_HEATMAP_KEY = None
_LAST_HI_KEY = None
def _ensure_df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters):
    """Obtiene df_ts = df(today)+df(yday) cacheado por filtros (sin hora)."""
    key = ("df_ts", today_str, yday_str,
           tuple(sorted(networks or [])),
           tuple(sorted(technologies or [])),
           tuple(sorted(vendors or [])),
           tuple(sorted(clusters or [])))
    now = time.time()
    hit = _DFTS_CACHE.get(key)
    if hit and (now - hit["ts"] < _DFTS_TTL):
        return hit["df"]

    df_today = fetch_kpis(fecha=today_str, hora=None,
                          vendors=vendors or None, clusters=clusters or None,
                          networks=networks or None, technologies=technologies or None,
                          limit=None)
    df_today = _ensure_df(df_today)

    df_yday = fetch_kpis(fecha=yday_str, hora=None,
                         vendors=vendors or None, clusters=clusters or None,
                         networks=networks or None, technologies=technologies or None,
                         limit=None)
    df_yday = _ensure_df(df_yday)

    df_ts = pd.concat([df_today, df_yday], ignore_index=True, sort=False)
    _DFTS_CACHE[key] = {"df": df_ts, "ts": now}
    return df_ts

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


def register_callbacks(app):

    # -------------------------------------------------
    # 0) Actualiza opciones de Network y Technology
    # -------------------------------------------------
    @app.callback(
        # Network
        Output("f-network", "options"),
        Output("f-network", "value"),
        # Technology
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        # Vendor
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        # Cluster
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),

        # Disparadores
        Input("refresh-timer", "n_intervals"),  # 👈 fuerza ejecución al cargar
        Input("f-fecha", "date"),
        Input("f-hora", "value"),

        # Estados actuales
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
    )
    def update_all_filters(_tick, fecha, hora,
                           net_val_current, tech_val_current,
                           ven_val_current, clu_val_current):
        # -------- 1) Traer DF principal para esa fecha/hora --------
        df_main = fetch_kpis(fecha=fecha, hora=hora, limit=None)
        df_main = _ensure_df(df_main)

        # Catálogos base
        networks_all = sorted(df_main["network"].dropna().unique().tolist()) \
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
        trig = ctx.triggered_id
        if not trig or "col" not in trig:
            return sort_state

        clicked_col = trig["col"]
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
        prevent_initial_call=False,
    )
    def refresh_table(fecha, hora, networks, technologies, vendors, clusters,
                      sort_mode, _n, sort_state, page_state):
        # ---------- normaliza filtros ----------
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # ---------- orden explícito ----------
        sort_by = None
        sort_net = None
        ascending = True

        # 👇 IMPORTANTE: en modo "global" NO usamos sort_state para la query
        if sort_mode != "global" and sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))
            if "__" in col:
                sort_net, base = col.split("__", 1)
                sort_by = base
            else:
                sort_by = col

        # ---------- fuente de datos paginada ----------
        if sort_mode == "alarmado":
            # en modo alarmado, el orden lo define fetch_kpis_paginated_severity_sort
            safe_sort_state = None
            df, total = fetch_kpis_paginated_severity_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=technologies or None,
                page=page, page_size=page_size,
            )
        else:
            # MODO GLOBAL → siempre global puro por severidad desde SQL
            safe_sort_state = None  # 👈 no reordenar en el render
            df, total = fetch_kpis_paginated_severity_global_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=technologies or None,
                page=page, page_size=page_size,
                sort_by_friendly=None,
                sort_net=None,
                ascending=True,
            )

        # ---------- si no hay df -> alert ----------
        if df is None or df.empty:
            store_payload = {"columns": [], "rows": []}
            empty_alert = dbc.Alert("Sin datos para los filtros seleccionados.", color="warning")
            return empty_alert, "Página 1 de 1", "Sin resultados.", store_payload  # ← 4 valores

        # ---------- baseline semanal de integridad (sin paginar) ----------
        df_baseline = fetch_integrity_baseline_week(
            fecha=fecha,
            vendors=vendors or None,
            clusters=clusters or None,
            networks=networks or None,
            technologies=technologies or None,
        )
        if df_baseline is None:
            df_baseline = pd.DataFrame()

        integrity_baseline_map = {}
        if not df_baseline.empty:
            for _, r in df_baseline.iterrows():
                key = (
                    r.get("network"),
                    r.get("vendor"),
                    r.get("noc_cluster"),
                    r.get("technology"),
                )
                integrity_baseline_map[key] = r.get("integrity_week_avg")

        # ---------- máximos de progress usando TODOS los datos filtrados (sin paginar) ----------
        progress_max_by_col = _compute_progress_max_for_filters(
            fecha=fecha,
            hora=hora,
            networks=networks,
            technologies=technologies,
            vendors=vendors,
            clusters=clusters,
        )

        # ---------- calcular racha de alarmas (día completo) ----------
        df_day = fetch_kpis(
            fecha=fecha,
            hora=None,  # todas las horas del día seleccionado
            vendors=vendors or None,
            clusters=clusters or None,
            networks=networks or None,
            technologies=technologies or None,
            limit=None,
        )
        df_day = _ensure_df(df_day)

        alarm_map = {}
        if not df_day.empty:
            df_day = add_alarm_streak(df_day)

            key_cols = ["fecha", "hora", "network", "vendor", "noc_cluster", "technology"]
            for _, r in df_day.iterrows():
                key = (
                    r.get("fecha"),
                    r.get("hora"),
                    r.get("network"),
                    r.get("vendor"),
                    r.get("noc_cluster"),
                    r.get("technology"),
                )
                alarm_map[key] = r.get("alarmas", 0)

        def _lookup_alarmas(row):
            key = (
                row.get("fecha"),
                row.get("hora"),
                row.get("network"),
                row.get("vendor"),
                row.get("noc_cluster"),
                row.get("technology"),
            )
            return alarm_map.get(key, 0)

        # añade la columna 'alarmas' al DF paginado
        df["alarmas"] = df.apply(_lookup_alarmas, axis=1)

        # ---------- inferir nets ----------
        if networks:
            nets = networks
        else:
            nets = (
                sorted(df["network"].dropna().unique().tolist())
                if "network" in df.columns
                else []
            )

        # ---------- pivot + orden estable (si aplica) ----------
        key_cols = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
        if all(k in df.columns for k in key_cols) and nets:
            tuples_in_order = list(
                dict.fromkeys(
                    map(tuple, df[key_cols].itertuples(index=False, name=None))
                )
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
            else:
                use_df = df
        else:
            use_df = df

        # ---------- render tabla (usando máximos globales filtrados) ----------
        table = render_kpi_table_multinet(
            use_df,
            networks=nets,
            sort_state=safe_sort_state,
            progress_max_by_col=progress_max_by_col,
            integrity_baseline_map=integrity_baseline_map,
        )

        # ---------- banners ----------
        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)
        indicator = f"Página {page_corrected} de {total_pages}"
        banner = (
            "Sin resultados."
            if (total or 0) == 0
            else f"Mostrando {(page_corrected - 1) * page_size + 1}–"
                 f"{min(page_corrected * page_size, total)} de {total} registros"
        )

        # ---------- store ----------
        store_payload = {
            "columns": list(use_df.columns),
            "rows": use_df.to_dict("records"),
        }
        return table, indicator, banner, store_payload

        # añade la columna 'alarmas' al DF paginado
        df["alarmas"] = df.apply(_lookup_alarmas, axis=1)

        # ---------- inferir nets ----------
        if networks:
            nets = networks
        else:
            nets = (
                sorted(df["network"].dropna().unique().tolist())
                if "network" in df.columns
                else []
            )

        # ---------- pivot + orden estable (si aplica) ----------
        key_cols = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
        if all(k in df.columns for k in key_cols) and nets:
            tuples_in_order = list(
                dict.fromkeys(
                    map(tuple, df[key_cols].itertuples(index=False, name=None))
                )
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
            else:
                use_df = df
        else:
            use_df = df

        # ---------- render tabla (usando máximos globales filtrados) ----------
        table = render_kpi_table_multinet(
            use_df,
            networks=nets,
            sort_state=safe_sort_state,
            progress_max_by_col=progress_max_by_col,
            integrity_baseline_map=integrity_baseline_map,
        )

        # ---------- banners ----------
        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)
        indicator = f"Página {page_corrected} de {total_pages}"
        banner = (
            "Sin resultados."
            if (total or 0) == 0
            else f"Mostrando {(page_corrected - 1) * page_size + 1}–"
                 f"{min(page_corrected * page_size, total)} de {total} registros"
        )

        # ---------- store ----------
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
        Output("f-hora", "value"),
        Output("f-fecha", "date"),
        Input("refresh-timer", "n_intervals"),
        State("f-hora", "value"),
        State("f-fecha", "date"),
        State("f-hora", "options"),
        prevent_initial_call=False,
    )
    def tick(_, current_hour, current_date, hour_options):
        now = now_local()
        floored = round_down_to_hour(now)
        hh = floored.strftime("%H:00:00")
        today = floored.strftime("%Y-%m-%d")

        # No sobre/escribas si el usuario fijó manualmente
        if (current_date not in (None, today)) or (current_hour not in (None, hh)):
            return no_update, no_update

        opt_values = {(o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])}
        if opt_values and hh not in opt_values:
            return no_update, no_update

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
        Output("topoff-link-state", "data"),
        Input({"type": "main-cluster-link", "cluster": ALL, "vendor": ALL, "technology": ALL}, "n_clicks"),
        State({"type": "main-cluster-link", "cluster": ALL, "vendor": ALL, "technology": ALL}, "id"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def sync_topoff_from_main(n_clicks_list, ids_list, current_state):
        # no hay clicks válidos
        if not n_clicks_list or not ids_list:
            raise PreventUpdate

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

        # toggle: si ya estaba seleccionado el mismo, limpias
        if current_sel == new_sel:
            return {"selected": None}

        return {"selected": new_sel}