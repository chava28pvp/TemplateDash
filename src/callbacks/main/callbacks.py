import math
import pandas as pd
from dash import Input, Output, State, ALL, no_update, ctx
import time
import logging
import numpy as np
from components.main.main_table import (
    pivot_by_network,
    render_kpi_table_multinet,
    strip_net, prefixed_progress_cols,
)
import dash_bootstrap_components as dbc

from src.dataAccess.data_access import fetch_kpis, COLMAP, fetch_kpis_paginated_severity_global_sort, \
    fetch_kpis_paginated_severity_sort, fetch_integrity_baseline_week, fetch_kpis_by_keys, \
    fetch_main_distinct_catalogs, fetch_progress_max_by_network, fetch_main_alarm_state, \
    fetch_latest_available_slot
from src.config import PROFILE_MAIN_CALLBACKS
from src.dataAccess.data_acess_topoff import fetch_topoff_distinct, fetch_latest_available_slot_topoff
from dash.exceptions import PreventUpdate
from src.callbacks.common import paginate_state, reset_page_state, toggle_bool, choose_common_available_slot

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

HOLD_SECONDS = 600  # 10 minutos
logger = logging.getLogger(__name__)


def _perf_log(callback_name, started_at, marks=None, extra=None):
    if not PROFILE_MAIN_CALLBACKS:
        return
    marks = marks or []
    extra = extra or {}
    now = time.perf_counter()
    prev = started_at
    parts = []
    for label, ts in marks:
        parts.append(f"{label}={(ts - prev) * 1000:.1f}ms")
        prev = ts
    parts.append(f"total={(now - started_at) * 1000:.1f}ms")
    if extra:
        parts.append(
            ", ".join([f"{k}={v}" for k, v in extra.items()])
        )
    logger.warning("%s | %s", callback_name, " | ".join(parts))

def order_networks(nets):
    first = [n for n in PREFERRED_NET_ORDER if n in nets]
    rest = sorted([n for n in nets if n not in PREFERRED_NET_ORDER])
    return first + rest
def _ensure_df(x):
    """
       Asegura que el resultado sea un DataFrame.
       - Si ya es DataFrame, lo regresa.
       - Si no, regresa un DataFrame vacío (evita errores aguas abajo).
       """
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _as_list(x):
    """
     Normaliza cualquier input a lista:
     - None -> None (se usa como "sin filtro")
     - lista/tupla -> lista
     - valor simple -> [valor]
     """
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _applied_value(applied_filters, key):
    return (applied_filters or {}).get(key)

def _keep_valid(selected, valid_values):
    """
        Mantiene solo los valores seleccionados que aún existen en el catálogo válido.
        Útil cuando actualizas opciones de un dropdown y quieres conservar la selección
        sin dejar valores "fantasma".
        """
    if not valid_values:
        return []
    if selected is None:
        return []
    if not isinstance(selected, (list, tuple)):
        selected = [selected]
    valid_set = set(valid_values)
    filtered = [v for v in selected if v in valid_set]
    return filtered


def _normalize_hour_to_options(hour_value, hour_options):
    """
    Si la hora no existe exactamente en el dropdown, la baja a la hora completa previa.
    Ejemplo: 10:30:00 -> 10:00:00.
    """
    if not hour_value:
        return None

    opt_values = {(o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])}
    if hour_value in opt_values:
        return hour_value

    try:
        hh = str(hour_value).strip()[:2]
        normalized = f"{int(hh):02d}:00:00"
    except Exception:
        return None

    return normalized if normalized in opt_values else None

def _compute_progress_max_for_filters(fecha, hora, networks, technologies, vendors, clusters):
    """
        Calcula el máximo por columna "progress" para los filtros actuales.

        ¿Para qué sirve?
        - Para que las barras (progress bars) se escalen al máximo real del dataset filtrado.
        - Se calcula con el dataset COMPLETO (sin paginación) para que el máximo sea correcto.
        """
    networks = _as_list(networks)
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    clusters = _as_list(clusters)

    max_dict = fetch_progress_max_by_network(
        fecha=fecha,
        hora=hora,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
    )
    return max_dict


def _build_alarm_list(df_alarm_state: pd.DataFrame):
    if df_alarm_state is None or df_alarm_state.empty:
        return []

    df = df_alarm_state.copy()
    key_cols = ["network", "vendor", "noc_cluster", "technology"]
    sort_cols = key_cols + ["fecha", "hora"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df["has_alarm"] = df["has_alarm"].fillna(0).astype(int).astype(bool)

    grp_id = df.groupby(key_cols, sort=False).ngroup()
    reset_id = (~df["has_alarm"]).groupby(grp_id, sort=False).cumsum()
    df["alarmas"] = (
        df["has_alarm"].astype(int)
        .groupby([grp_id, reset_id], sort=False)
        .cumsum()
        .astype(int)
    )

    return [
        {
            "fecha": None if pd.isna(r.fecha) else str(r.fecha).strip(),
            "hora": None if pd.isna(r.hora) else str(r.hora).strip(),
            "network": r.network,
            "vendor": r.vendor,
            "noc_cluster": r.noc_cluster,
            "technology": r.technology,
            "alarmas": int(r.alarmas or 0),
        }
        for r in df.itertuples(index=False)
    ]

def _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters):
    """
       Crea una llave hashable para cachear el 'contexto' del main.
       Normaliza listas para que el orden no afecte (['NET','ATT'] == ['ATT','NET']).
       """
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
    """
        Lee del cache el contexto si:
        - existe
        - y no ha expirado (TTL).
        """
    key = _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters)
    now = time.time()
    hit = _MAIN_CTX_CACHE.get(key)
    if hit and (now - hit["ts"] < _MAIN_CTX_TTL):
        return hit["data"]
    return None

def _set_main_context_cached(fecha, hora, networks, technologies, vendors, clusters, data):
    """
        Guarda el contexto en cache con timestamp para expiración por TTL.
        """
    key = _make_ctx_key(fecha, hora, networks, technologies, vendors, clusters)
    _MAIN_CTX_CACHE[key] = {"ts": time.time(), "data": data}

def _is_integrity_pct_sort(sort_state: dict | None) -> bool:
    """
        Detecta si el usuario está ordenando por la columna de % integridad.
        Puede venir:
        - "integrity_deg_pct" (sin prefijo)
        - "NET__integrity_deg_pct" (con prefijo de network)
        """
    col = (sort_state or {}).get("column")
    return bool(col) and (str(col).endswith("__integrity_deg_pct") or col == "integrity_deg_pct")

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

    @app.callback(
        Output("data-ready-store", "data"),
        Input("refresh-timer", "n_intervals"),
        State("data-ready-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_data_ready_store(_tick, current_store):
        main_slot = fetch_latest_available_slot()
        topoff_slot = fetch_latest_available_slot_topoff()
        common_slot = choose_common_available_slot(main_slot, topoff_slot)

        if not common_slot:
            raise PreventUpdate

        current_slot = (current_store or {}).get("slot") or {}
        if current_slot == common_slot:
            raise PreventUpdate

        return {
            "slot": common_slot,
            "sources": {
                "main": main_slot,
                "topoff": topoff_slot,
            },
            "updated_at": time.time(),
        }

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
        """
            Objetivo:
            - Refrescar catálogos de filtros (networks/tech/vendor/cluster)
            - Conservar selecciones previas si siguen existiendo
            - Mezclar catálogo principal con catálogo TopOff (tech/vendors)
            """
        # Normalizamos seleccionados para filtrar vendors/clusters
        nets_sel = _as_list(net_val_current)
        techs_sel = _as_list(tech_val_current)

        # -------- 1) Catálogos principales vía DISTINCT (sin traer dataset completo) --------
        main_opts = fetch_main_distinct_catalogs(
            fecha=fecha,
            hora=None,
            networks=nets_sel or None,
            technologies=techs_sel or None,
        ) or {}

        networks_all = order_networks(main_opts.get("networks", []) or [])
        techs_all = main_opts.get("technologies", []) or []
        vendors_main = main_opts.get("vendors", []) or []
        clusters_main = main_opts.get("clusters", []) or []

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

    @app.callback(
        Output("applied-filters-store", "data"),
        Input("apply-filters-btn", "n_clicks"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("f-sort-mode", "value"),
        prevent_initial_call=False,
    )
    def apply_filters(_n, networks, technologies, vendors, clusters, sort_mode):
        return {
            "network": _as_list(networks) or [],
            "technology": _as_list(technologies) or [],
            "vendor": _as_list(vendors) or [],
            "cluster": _as_list(clusters) or [],
            "sort_mode": sort_mode or "alarmado",
        }
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
        """
           Guarda el estado de orden:
           - column: la columna clickeada (puede venir prefijada NET__...)
           - ascending: True/False
           Reglas:
           - Si vuelves a clickear la misma, invierte asc/desc
           - Si clickeas otra, cambia columna y arranca ascendente
           """
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
        Input("applied-filters-store", "data"),
        Input("page-size", "value"),
        prevent_initial_call=True,
    )
    def reset_page_on_filters(_fecha, _hora, _applied_filters, page_size):
        """
            Cada vez que cambian filtros o tamaño de página:
            - resetea la paginación a página 1
            """
        return reset_page_state(page_size, default_size=50)

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
        """
            Botones anterior / siguiente:
            - Solo modifica 'page'
            """
        return paginate_state(state, prev_id="page-prev", next_id="page-next", default_size=50)

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
        Input("data-ready-store", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("applied-filters-store", "data"),
        Input("sort-state", "data"),
        Input("page-state", "data"),
        Input("main-context-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_table(
            _ready, fecha, hora, applied_filters,
            sort_state, page_state,
            main_ctx
    ):
        perf_start = time.perf_counter()
        perf_marks = []
        applied_filters = applied_filters or {}
        networks = _as_list(_applied_value(applied_filters, "network"))
        technologies = _as_list(_applied_value(applied_filters, "technology"))
        vendors = _as_list(_applied_value(applied_filters, "vendor"))
        clusters = _as_list(_applied_value(applied_filters, "cluster"))
        sort_mode = _applied_value(applied_filters, "sort_mode") or "alarmado"

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
        perf_marks.append(("ctx", time.perf_counter()))

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
        # sort GLOBAL por % integridad (integrity_deg_pct) SIN BD
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
            explicit_page_key_order = page_keys  # orden estable para el pivot

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
            perf_marks.append(("special_sort_fetch", time.perf_counter()))

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
            perf_marks.append(("page_fetch", time.perf_counter()))

        if df is None or df.empty:
            _perf_log(
                "refresh_table",
                perf_start,
                perf_marks,
                {"page": page, "page_size": page_size, "rows": 0, "mode": sort_mode},
            )
            store_payload = {"columns": [], "rows": []}
            empty_alert = dbc.Alert("Sin datos para los filtros seleccionados.", color="warning")
            return empty_alert, "Página 1 de 1", "Sin resultados.", store_payload

        # ---------- Reordenar GLOBAL por bucket de completitud (solo modo global) ----------
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
        perf_marks.append(("row_enrichment", time.perf_counter()))

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
            # si venimos del sort global por % integridad, respeta el orden de la página
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
        perf_marks.append(("pivot_order", time.perf_counter()))

        # ---------- render ----------
        table = render_kpi_table_multinet(
            use_df,
            networks=nets,
            sort_state=safe_sort_state,
            progress_max_by_col=progress_max_by_col,
            integrity_baseline_map=integrity_baseline_map,
        )
        perf_marks.append(("render_table", time.perf_counter()))

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
        perf_marks.append(("serialize_store", time.perf_counter()))
        _perf_log(
            "refresh_table",
            perf_start,
            perf_marks,
            {"page": page_corrected, "page_size": page_size, "rows": len(use_df), "mode": sort_mode},
        )

        return table, indicator, banner, store_payload

    # -------------------------------------------------
    # 5) Intervalo global → sincroniza el del card (si aplica)
    # -------------------------------------------------
    # -------------------------------------------------
    # 7) Tick: actualizar fecha/hora al inicio de cada hora
    # -------------------------------------------------
    @app.callback(
        Output("dt-manual-store", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        State("data-ready-store", "data"),
        prevent_initial_call=True,
    )
    def mark_datetime_manual(_fecha, _hora, data_ready):
        if not ctx.triggered_id:
            raise PreventUpdate

        slot = (data_ready or {}).get("slot") or {}
        if _fecha == slot.get("fecha") and _hora == slot.get("hora"):
            raise PreventUpdate

        return {"last_manual_ts": time.time(), "fecha": _fecha, "hora": _hora}

    @app.callback(
        Output("f-hora", "value"),
        Output("f-fecha", "date"),
        Input("data-ready-store", "data"),
        State("f-hora", "value"),
        State("f-fecha", "date"),
        State("f-hora", "options"),
        State("dt-manual-store", "data"),
        prevent_initial_call=False,
    )
    def sync_datetime_from_data_ready(data_ready, current_hour, current_date, hour_options, manual_store):
        """
           Auto-actualiza fecha/hora a la hora actual, pero con "hold" inteligente:
           - Si el usuario cambió manualmente hace poco, NO lo pisamos
           - El hold dura HOLD_SECONDS o hasta el siguiente cambio de hora (lo que ocurra primero)
           """
        slot = (data_ready or {}).get("slot") or {}
        hh = _normalize_hour_to_options(slot.get("hora"), hour_options)
        today = slot.get("fecha")
        if not hh or not today:
            return no_update, no_update

        if current_hour == hh and current_date == today:
            return no_update, no_update

        return hh, today

    @app.callback(
        Output("filters-collapse", "is_open"),
        Input("filters-toggle", "n_clicks"),
        State("filters-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_filters(n, is_open):
        return toggle_bool(n, is_open)

    @app.callback(
        Output("sort-state", "data", allow_duplicate=True),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("applied-filters-store", "data"),
        prevent_initial_call=True,
    )
    def reset_sort_state_on_filters(_fecha, _hora, _applied_filters):
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
        """
            Cuando el usuario da click en el botón del cluster (en la tabla main),
            guardamos la selección para que TopOff se filtre por:
            - cluster
            - vendor
            - technology

            Si el usuario vuelve a clickear el mismo, lo des-seleccionamos (toggle).
            """
        if not ids_list:
            raise PreventUpdate

        safe_clicks = [(c or 0) for c in (n_clicks_list or [])]
        if max(safe_clicks, default=0) == 0:
            # evita falsos disparos por re-render
            raise PreventUpdate
        if max([(c or 0) for c in (n_clicks_list or [])]) == 0:
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
        Input("applied-filters-store", "data"),
        prevent_initial_call=False,
    )
    def build_main_context(fecha, hora, applied_filters):
        """
           Construye un 'contexto' pesado para evitar repetir cálculos en refresh_table:
           - Baseline semanal de integridad (por network/vendor/cluster/tech)
           - Máximos para progress bars (por KPI prefijado: NET__ps_rrc_fail, etc.)
           - Alarm map del día completo (incluye racha)

           Se guarda en dcc.Store (main-context-store).
           """
        perf_start = time.perf_counter()
        perf_marks = []
        applied_filters = applied_filters or {}
        networks = _as_list(_applied_value(applied_filters, "network"))
        technologies = _as_list(_applied_value(applied_filters, "technology"))
        vendors = _as_list(_applied_value(applied_filters, "vendor"))
        clusters = _as_list(_applied_value(applied_filters, "cluster"))

        # -----------------------------
        # 0) Cache
        #    - Para evitar el bug de "hora", el baseline NO debe cachearse por hora
        # -----------------------------
        cached = _get_main_context_cached(fecha, hora, networks, technologies, vendors, clusters)
        if cached is not None:
            _perf_log(
                "build_main_context",
                perf_start,
                [("cache_hit", time.perf_counter())],
                {"cached": True},
            )
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
        perf_marks.append(("baseline", time.perf_counter()))

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
        # 1.1) MOCK (si NO hay baseline real)
        # ============================================================
        elif MOCK_INTEGRITY_BASELINE:
            df_now = fetch_kpis(
                fecha=fecha,
                hora=None,
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
        # 2) Progress max
        # ============================================================
        progress_max_by_col = _compute_progress_max_for_filters(
            fecha=fecha,
            hora=hora,
            networks=networks,
            technologies=technologies,
            vendors=vendors,
            clusters=clusters,
        )
        perf_marks.append(("progress_max", time.perf_counter()))

        # ============================================================
        # 3) Alarm map para el día completo
        # ============================================================
        df_alarm_state = fetch_main_alarm_state(
            fecha=fecha,
            vendors=vendors or None,
            clusters=clusters or None,
            networks=networks or None,
            technologies=technologies or None,
        )
        perf_marks.append(("alarm_fetch", time.perf_counter()))
        alarm_list = _build_alarm_list(df_alarm_state if isinstance(df_alarm_state, pd.DataFrame) else pd.DataFrame())
        perf_marks.append(("alarm_streak", time.perf_counter()))

        payload = {
            "integrity_baseline_map": integrity_baseline_list,
            "progress_max_by_col": progress_max_by_col,
            "alarm_map": alarm_list,
            "ts": time.time(),
        }

        _set_main_context_cached(fecha, hora, networks, technologies, vendors, clusters, payload)
        _perf_log(
            "build_main_context",
            perf_start,
            perf_marks,
            {
                "baseline_rows": len(integrity_baseline_list),
                "progress_cols": len(progress_max_by_col),
                "alarm_rows": len(alarm_list),
            },
        )
        return payload
