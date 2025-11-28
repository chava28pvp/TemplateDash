import math
import pandas as pd
from dash import Input, Output, State, ALL, no_update, ctx
from hashlib import md5
import json
import time
from components.main.main_table import (
    pivot_by_network,
    render_kpi_table_multinet,
    strip_net,
)
import dash_bootstrap_components as dbc

from src.dataAccess.data_access import fetch_kpis, COLMAP, fetch_kpis_paginated_severity_global_sort, fetch_kpis_paginated_severity_sort
from src.config import REFRESH_INTERVAL_MS

from src.Utils.utils_time import now_local
from src.dataAccess.data_acess_topoff import fetch_topoff_distinct

#Cach simple en memoria para df_ts
_DFTS_CACHE = {}
_DFTS_TTL = 300  # segundos

# Última clave renderizada para evitar re-render idéntico
_LAST_HEATMAP_KEY = None
_LAST_HI_KEY = None


def _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit):
    """Clave estable del estado visible del heatmap (sin hora)."""
    def _norm(x):
        x = x if isinstance(x, (list, tuple)) else ([] if x is None else [x])
        return sorted([str(v) for v in x if v is not None])
    obj = {
        "fecha": fecha,
        "networks": _norm(networks),
        "technologies": _norm(technologies),
        "vendors": _norm(vendors),
        "clusters": _norm(clusters),
        "offset": int(offset),
        "limit": int(limit),
    }
    return md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

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


def register_callbacks(app):

    # -------------------------------------------------
    # 0) Actualiza opciones de Network y Technology
    # -------------------------------------------------
    @app.callback(
        Output("f-network", "options"),
        Output("f-network", "value"),
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
    )
    def update_network_tech(fecha, hora):
        # Para construir opciones, usa consulta no paginada (solo metadata)
        df_main = fetch_kpis(fecha=fecha, hora=hora, limit=None)
        networks_main = sorted(df_main["network"].dropna().unique().tolist()) if "network" in df_main.columns else []
        techs_main = sorted(df_main["technology"].dropna().unique().tolist()) if "technology" in df_main.columns else []

        # TOPOFF (solo para techs fallback/union)
        top_opts = fetch_topoff_distinct(fecha=fecha)
        techs_top = top_opts.get("technologies", [])

        # UNION techs (si main vacío, te quedan los de topoff)
        techs = sorted(set(techs_main) | set(techs_top))

        net_opts = [{"label": n, "value": n} for n in networks_main]
        tech_opts = [{"label": t, "value": t} for t in techs]

        # valores por defecto vacíos (como tenías)
        return net_opts, [], tech_opts, []

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
    # 1) Actualiza opciones de Vendor/Cluster
    # -------------------------------------------------
    @app.callback(
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
    )
    def update_vendor_cluster(fecha, hora, networks, technologies):
        networks = _as_list(networks)
        technologies = _as_list(technologies)

        # ---------- MAIN ----------
        df_main = fetch_kpis(fecha=fecha, hora=hora, limit=None)

        if networks and "network" in df_main.columns:
            df_main = df_main[df_main["network"].isin(networks)]
        if technologies and "technology" in df_main.columns:
            df_main = df_main[df_main["technology"].isin(technologies)]

        vendors_main = sorted(df_main["vendor"].dropna().unique().tolist()) if "vendor" in df_main.columns else []
        clusters_main = sorted(
            df_main["noc_cluster"].dropna().unique().tolist()) if "noc_cluster" in df_main.columns else []

        # ---------- TOPOFF (vendors fallback/union) ----------
        top_opts = fetch_topoff_distinct(
            fecha=fecha,
            technologies=technologies,  # topoff sí tiene technology
            vendors=None,  # no filtres vendors aquí
        )
        vendors_top = top_opts.get("vendors", [])

        # UNION vendors (si main vacío, sobreviven los de topoff)
        vendors = sorted(set(vendors_main) | set(vendors_top))

        vendor_opts = [{"label": v, "value": v} for v in vendors]
        cluster_opts = [{"label": c, "value": c} for c in clusters_main]  # clusters SOLO main

        return vendor_opts, [], cluster_opts, []

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
        if sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))
            if "__" in col:
                sort_net, base = col.split("__", 1)
                sort_by = base
            else:
                sort_by = col

        # ---------- fuente de datos ----------
        if sort_mode == "alarmado":
            safe_sort_state = None
            df, total = fetch_kpis_paginated_severity_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=technologies or None,
                page=page, page_size=page_size,
            )

        else:
            # Nuevo GLOBAL basado en profiles.main.severity
            safe_sort_state = None  # 👈 importante para no reordenar en render
            if sort_by in COLMAP:
                df, total = fetch_kpis_paginated_severity_global_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=technologies or None,
                    page=page, page_size=page_size,
                    sort_by_friendly=sort_by,
                    sort_net=sort_net,
                    ascending=ascending,
                )

            else:
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

        # ---------- inferir nets ----------
        if networks:
            nets = networks
        else:
            nets = sorted(df["network"].dropna().unique().tolist()) if "network" in df.columns else []

        # ---------- pivot + orden estable (si aplica) ----------
        key_cols = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
        if all(k in df.columns for k in key_cols) and nets:
            tuples_in_order = list(dict.fromkeys(map(tuple, df[key_cols].itertuples(index=False, name=None))))
            order_map = {t: i for i, t in enumerate(tuples_in_order)}
            wide = pivot_by_network(df, networks=nets)
            if wide is not None and not wide.empty:
                wide["_ord"] = wide[key_cols].apply(lambda r: order_map.get(tuple(r.values.tolist()), 10 ** 9), axis=1)
                wide = wide.sort_values("_ord").drop(columns=["_ord"])
                use_df = wide
            else:
                use_df = df
        else:
            use_df = df

        # ---------- render tabla ----------
        table = render_kpi_table_multinet(use_df, networks=nets, sort_state=safe_sort_state)

        # ---------- banners ----------
        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)
        indicator = f"Página {page_corrected} de {total_pages}"
        banner = "Sin resultados." if (total or 0) == 0 else \
            f"Mostrando {(page_corrected - 1) * page_size + 1}–{min(page_corrected * page_size, total)} de {total} registros"

        # ---------- store ----------
        store_payload = {"columns": list(use_df.columns), "rows": use_df.to_dict("records")}
        return table, indicator, banner, store_payload  # ← 4 valores

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