import os
import math
import pandas as pd
from dash import Input, Output, State, no_update, ctx
from hashlib import md5
import json
import time
import logging
import threading
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Helpers de Heatmap (figuras + payloads + UI)
from components.main.heatmap import (
    build_heatmap_figure,
    render_heatmap_summary_table,
    build_heatmap_payloads_fast,
    build_heatmap_ranked_rows_fast,
    build_heatmap_payloads_from_ranked_rows,
    _hm_height,
    _build_time_header_children_by_dates
)

# Helpers de Histograma (payloads + overlay waves)
from components.main.histograma import (
    build_histo_payloads_fast,
    build_overlay_waves_figure
)

import dash_bootstrap_components as dbc
from src.callbacks.common import paginate_state, reset_page_state, purge_expired_cache_entries

# Manager de umbrales (config centralizada)
from src.Utils.umbrales.umbrales_manager import UM_MANAGER

# Acceso a datos
from src.dataAccess.data_access import fetch_kpis, fetch_alarm_meta_for_heatmap
from src.dataAccess import main_visuals_api_client
from src.config import (
    DATA_SOURCE,
    MAIN_QUERY_API_HEATMAP_MAX_ROWS,
    PROFILE_MAIN_CALLBACKS,
    PREWARM_MAIN_CACHE,
    PREWARM_MAIN_PAGE_SIZE,
)
from src.Utils.alarmados import load_threshold_cfg
from src.Utils.utils_time import default_date_str


# =========================================================
# CACHES EN MEMORIA
# =========================================================

# Cache simple para df_ts (datos de hoy + ayer) por filtros
_DFTS_CACHE = {}
_DFTS_TTL = 300  # segundos

# Ãšltima clave renderizada para evitar re-render idÃ©ntico (heatmap)
_LAST_HEATMAP_KEY = None

# Ãšltima clave renderizada para evitar re-render idÃ©ntico (histograma PS/CS)
_LAST_HI_KEY = {"PS": None, "CS": None}

# Cache simple en memoria para meta de alarmados (df_meta + keys)
_ALARM_META_CACHE = {}
_ALARM_META_TTL = 300  # seg

# Cache de payloads del heatmap por state_key (para no recalcular z/y/x cada rato)
_HM_PAYLOAD_CACHE = {}
_HM_PAYLOAD_TTL = 120  # seg
_HM_RANK_CACHE = {}
_HM_RANK_TTL = 120  # seg

# Cache de payloads/figuras del histograma
_HI_PAYLOAD_CACHE = {}
_HI_PAYLOAD_TTL = 120  # seg
_HI_FIG_CACHE = {}
_HI_FIG_TTL = 120  # seg
_PREWARM_LOCK = threading.Lock()
_PREWARM_STARTED = False
logger = logging.getLogger(__name__)


def purge_main_heatmap_caches(now_ts=None):
    purge_expired_cache_entries(_DFTS_CACHE, _DFTS_TTL, now_ts=now_ts)
    purge_expired_cache_entries(_ALARM_META_CACHE, _ALARM_META_TTL, now_ts=now_ts)
    purge_expired_cache_entries(_HM_PAYLOAD_CACHE, _HM_PAYLOAD_TTL, now_ts=now_ts)
    purge_expired_cache_entries(_HM_RANK_CACHE, _HM_RANK_TTL, now_ts=now_ts)
    purge_expired_cache_entries(_HI_PAYLOAD_CACHE, _HI_PAYLOAD_TTL, now_ts=now_ts)
    purge_expired_cache_entries(_HI_FIG_CACHE, _HI_FIG_TTL, now_ts=now_ts)
    if DATA_SOURCE == "api":
        main_visuals_api_client.clear_cache()


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


# =========================================================
# MÃ‰TRICAS POR DOMINIO
# =========================================================
PS_VALORES = ("PS_RRC", "PS_S1", "PS_DROP", "PS_RAB")
CS_VALORES = ("CS_RRC", "CS_DROP", "CS_RAB")


# =========================================================
# HELPERS GENERALES
# =========================================================

def _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit, revision=None):
    """
    Genera una "firma" (hash md5) del estado actual del heatmap:
    - filtros
    - paginaciÃ³n (offset/limit)
    Esto permite cachear y tambiÃ©n evitar renders duplicados.
    """
    def _norm(x):
        # Normaliza a lista de strings ordenada (estable)
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
        "revision": revision,
    }
    return md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _ensure_df(x):
    """Garantiza que lo que regrese sea DataFrame."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters, revision=None):
    """
    Trae el dataset de serie de tiempo (TS) del dÃ­a de HOY y AYER:
    - fetch_kpis(... hoy ...)
    - fetch_kpis(... ayer ...)
    - concatena ambos
    Se cachea por filtros para ahorrar consultas/cÃ¡lculo.
    """
    key = (
        "df_ts",
        today_str,
        yday_str,
        tuple(sorted(networks or [])),
        tuple(sorted(technologies or [])),
        tuple(sorted(vendors or [])),
        tuple(sorted(clusters or [])),
        revision,
    )
    now = time.time()
    hit = _DFTS_CACHE.get(key)
    if hit and (now - hit["ts"] < _DFTS_TTL):
        return hit["df"]
    api_limit = None
    if DATA_SOURCE == "api":
        api_limit = max(0, int(MAIN_QUERY_API_HEATMAP_MAX_ROWS or 0))
        if api_limit <= 0:
            empty = pd.DataFrame()
            _DFTS_CACHE[key] = {"df": empty, "ts": now}
            return empty


    # Hoy (dÃ­a completo: hora=None)
    df_today = fetch_kpis(
        fecha=today_str,
        hora=None,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=api_limit
    )
    df_today = _ensure_df(df_today)

    # Ayer (dÃ­a completo: hora=None)
    df_yday = fetch_kpis(
        fecha=yday_str,
        hora=None,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=api_limit
    )
    df_yday = _ensure_df(df_yday)

    # Serie de tiempo final (hoy + ayer)
    df_ts = pd.concat([df_today, df_yday], ignore_index=True, sort=False)

    # Guardar en cache
    _DFTS_CACHE[key] = {"df": df_ts, "ts": now}
    return df_ts


def _as_list(x):
    """Normaliza a lista:
    - None -> None
    - list/tuple -> list
    - valor -> [valor]
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _applied_value(applied_filters, key):
    return (applied_filters or {}).get(key)


def _cache_get(cache: dict, key, ttl: int):
    """Lee un cache con TTL."""
    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit["ts"] < ttl):
        return hit["data"]
    return None


def _cache_set(cache: dict, key, data):
    """Guarda en cache con timestamp."""
    cache[key] = {"data": data, "ts": time.time()}


def _fetch_alarm_meta_cached(today_str, vendors, clusters, networks, technologies):
    """
    Trae (df_meta, keys) para alarmados en el heatmap:
    - df_meta: metadata para filas (sitio/rnc/nodeb/etc)
    - keys: set/keys para saber cuÃ¡les estÃ¡n alarmados
    Cacheado por filtros.
    """
    key = (
        "alarm_meta",
        today_str,
        tuple(sorted(vendors or [])),
        tuple(sorted(clusters or [])),
        tuple(sorted(networks or [])),
        tuple(sorted(technologies or [])),
    )
    hit = _cache_get(_ALARM_META_CACHE, key, _ALARM_META_TTL)
    if hit is not None:
        return hit

    df_meta, keys = fetch_alarm_meta_for_heatmap(
        fecha=today_str,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
    )

    data = (df_meta, keys)
    _cache_set(_ALARM_META_CACHE, key, data)
    return data


def _trigger_revision(trigger_data):
    slot = (trigger_data or {}).get("slot") or {}
    return f"{slot.get('fecha') or ''}|{slot.get('hora') or ''}|{(trigger_data or {}).get('updated_at') or ''}"


def prewarm_main_cache(
    *,
    fecha: str | None = None,
    page_size: int | None = None,
    order_by: str = "alarm_hours",
):
    """
    Calienta en segundo plano los caches del main para la fecha por defecto:
    - df_ts 48h
    - alarm_meta
    - ranking y payload del heatmap principal
    - payloads/figuras de histogramas PS y CS
    """
    if not PREWARM_MAIN_CACHE:
        return

    try:
        target_date = fecha or default_date_str()
        limit = int(page_size or PREWARM_MAIN_PAGE_SIZE or 50)

        perf_start = time.perf_counter()
        perf_marks = []
        try:
            today_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except Exception:
            today_dt = datetime.utcnow()
            target_date = today_dt.strftime("%Y-%m-%d")
        yday_str = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        networks = technologies = vendors = clusters = None

        df_ts = _fetch_df_ts_cached(target_date, yday_str, networks, technologies, vendors, clusters)
        perf_marks.append(("df_ts", time.perf_counter()))
    except Exception as exc:
        logger.warning("prewarm_main_cache omitido por error inicial: %s", exc)
        return

    nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) if (
        df_ts is not None and not df_ts.empty and "network" in df_ts.columns
    ) else []

    df_meta_heat, alarm_keys_set = _fetch_alarm_meta_cached(
        target_date,
        vendors,
        clusters,
        nets_heat,
        technologies,
    )
    perf_marks.append(("alarm_meta", time.perf_counter()))

    cfg = UM_MANAGER.config()

    if df_meta_heat is not None and not df_meta_heat.empty and nets_heat:
        rank_key = _hm_key(target_date, networks, technologies, vendors, clusters, 0, 0) + f"|ord={order_by}|rank"
        ranked_rows, rank_meta = build_heatmap_ranked_rows_fast(
            df_meta=df_meta_heat,
            df_ts=df_ts,
            UMBRAL_CFG=cfg,
            networks=nets_heat,
            valores_order=("PS_RRC", "CS_RRC", "PS_S1", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB"),
            today=target_date,
            yday=yday_str,
            alarm_keys=alarm_keys_set,
            alarm_only=False,
            order_by=order_by,
        )
        _cache_set(_HM_RANK_CACHE, rank_key, (ranked_rows, rank_meta))
        perf_marks.append(("ranking", time.perf_counter()))

        state_key = _hm_key(target_date, networks, technologies, vendors, clusters, 0, limit) + f"|ord={order_by}"
        pct_payload, unit_payload, page_info = build_heatmap_payloads_from_ranked_rows(
            ranked_rows,
            df_ts=df_ts,
            UMBRAL_CFG=cfg,
            rank_meta=rank_meta,
            offset=0,
            limit=limit,
        )
        _cache_set(_HM_PAYLOAD_CACHE, state_key, (pct_payload, unit_payload, page_info))
        perf_marks.append(("heatmap_payload", time.perf_counter()))

        for domain, valores, traffic in [
            ("PS", _valores_by_domain("PS"), "ps_traff_gb"),
            ("CS", _valores_by_domain("CS"), "cs_traff_erl"),
        ]:
            base_state_key = _hm_key(target_date, networks, technologies, vendors, clusters, 0, limit) + f"|dom={domain}"
            pct_p, unit_p, page_info = build_histo_payloads_fast(
                df_meta=df_meta_heat,
                df_ts=df_ts,
                UMBRAL_CFG=cfg,
                networks=nets_heat,
                valores_order=valores,
                today=target_date,
                yday=yday_str,
                alarm_keys=alarm_keys_set,
                alarm_only=False,
                offset=0,
                limit=limit,
                traffic_metric=traffic,
            )
            _cache_set(_HI_PAYLOAD_CACHE, base_state_key, (pct_p, unit_p, page_info))

            state_key = f"{base_state_key}|selw=None"
            fig_pct = build_overlay_waves_figure(
                pct_p,
                UMBRAL_CFG=cfg,
                mode="severity",
                height=420,
                smooth_win=3,
                opacity=0.28,
                line_width=1.2,
                decimals=2,
                show_yaxis_ticks=True,
                selected_wave=None,
                show_traffic_bars=True if domain == "PS" else False,
                traffic_agg="mean",
                traffic_decimals=1,
            ) if pct_p else go.Figure()

            fig_unit = build_overlay_waves_figure(
                unit_p,
                UMBRAL_CFG=cfg,
                mode="progress",
                height=420,
                smooth_win=3,
                opacity=0.25,
                line_width=1.2,
                decimals=0,
                show_yaxis_ticks=True,
                selected_wave=None,
                show_traffic_bars=False,
            ) if unit_p else go.Figure()
            _cache_set(_HI_FIG_CACHE, state_key, (fig_pct, fig_unit))
        perf_marks.append(("histos", time.perf_counter()))

        _perf_log(
            "prewarm_main_cache",
            perf_start,
            perf_marks,
            {
                "date": target_date,
                "rows_ts": 0 if df_ts is None else len(df_ts),
                "rows_ranked": 0 if ranked_rows is None else len(ranked_rows),
                "page_size": limit,
            },
        )


def start_main_prewarm_thread():
    global _PREWARM_STARTED
    if not PREWARM_MAIN_CACHE:
        return
    with _PREWARM_LOCK:
        if _PREWARM_STARTED:
            return
        _PREWARM_STARTED = True

    def _safe_prewarm():
        try:
            prewarm_main_cache(fecha=default_date_str(), page_size=PREWARM_MAIN_PAGE_SIZE)
        except Exception as exc:
            logger.warning("prewarm_main_cache omitido por error: %s", exc)

    t = threading.Thread(
        target=_safe_prewarm,
        daemon=True,
        name="main-cache-prewarm",
    )
    t.start()


def _valores_by_domain(domain: str):
    """Devuelve quÃ© mÃ©tricas se usan segÃºn dominio PS o CS."""
    return CS_VALORES if str(domain).upper() == "CS" else PS_VALORES


def _build_histograma_for_domain(
    domain: str,
    sel_wave,
    fecha,
    networks,
    technologies,
    vendors,
    clusters,
    hm_page_state,
    link_state,
    trigger_data=None,
):
    """
    Construye 2 figuras (pct y unit) del histograma overlay para el dominio:
    - PS: %IA/%DC + Units
    - CS: %IA/%DC + Units
    Aplica:
    - filtros normales
    - paginado (mismo del heatmap/histo)
    - filtro extra de "link_state" (click desde main para fijar cluster/vendor/tech)
    """
    global _LAST_HI_KEY
    perf_start = time.perf_counter()
    perf_marks = []

    # selected_wave viene de click en histograma (series_key)
    selected_wave = (sel_wave or {}).get("series_key")
    trigger_source = (trigger_data or {}).get("source")
    trigger_revision = _trigger_revision(trigger_data)

    # Normaliza filtros
    networks = _as_list(networks)
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    clusters = _as_list(clusters)

    # -------- Filtro extra desde MAIN (link_state) --------
    clusters_effective = clusters
    vendors_effective = vendors
    technologies_effective = technologies

    if link_state and link_state.get("selected"):
        sel = link_state["selected"]
        clus = sel.get("cluster")
        ven = sel.get("vendor")
        tech = sel.get("technology")

        # Si viene alguno, â€œfijaâ€ ese filtro
        if clus:
            clusters_effective = [clus]
        if ven:
            vendors_effective = [ven]
        if tech:
            technologies_effective = [tech]

    # -------- Paginado (usa histo/heatmap state) --------
    page = int((hm_page_state or {}).get("page", 1))
    page_sz = int((hm_page_state or {}).get("page_size", 50))
    offset = max(0, (page - 1) * page_sz)
    limit = max(1, page_sz)

    # -------- Clave de estado: cambia si cambia filtro/pÃ¡gina/selecciÃ³n/dominio --------
    base_state_key = (
        _hm_key(
            fecha,
            networks,
            technologies_effective,
            vendors_effective,
            clusters_effective,
            offset,
            limit,
            revision=trigger_revision,
        )
        + f"|dom={domain}"
    )
    state_key = f"{base_state_key}|selw={selected_wave}"

    # Evita recomputar si es exactamente lo mismo (y no fue click de wave)
    if _LAST_HI_KEY.get(domain) == state_key and ctx.triggered_id != "histo-selected-wave":
        _perf_log(
            f"histograma_{domain.lower()}",
            perf_start,
            [("cache_hit", time.perf_counter())],
            {"cached": True},
        )
        return None, None, None, True  # cache-hit: el callback que llama usa no_update

    # -------- Fechas HOY/AYER --------
    try:
        today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()

    yday_dt = today_dt - timedelta(days=1)
    today_str = today_dt.strftime("%Y-%m-%d")
    yday_str = yday_dt.strftime("%Y-%m-%d")

    # -------- Datos TS (cacheados) --------
    df_ts = _fetch_df_ts_cached(
        today_str, yday_str,
        networks,
        technologies_effective,
        vendors_effective,
        clusters_effective,
        revision=trigger_revision,
    )
    perf_marks.append(("df_ts", time.perf_counter()))

    # -------- Redes efectivas para el heat/histo --------
    if networks:
        nets_heat = networks
    else:
        nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) if (
            df_ts is not None and not df_ts.empty and "network" in df_ts.columns
        ) else []

    # -------- Meta para heat/histo (alarmados) --------
    # Nota: aquÃ­ usas fetch directo (si quieres, puedes cambiarlo por _fetch_alarm_meta_cached)
    df_meta_heat, alarm_keys_set = _fetch_alarm_meta_cached(
        today_str,
        vendors_effective,
        clusters_effective,
        nets_heat,
        technologies_effective,
    )
    perf_marks.append(("alarm_meta", time.perf_counter()))

    # -------- Payloads para overlay --------
    cached_payload = _cache_get(_HI_PAYLOAD_CACHE, base_state_key, _HI_PAYLOAD_TTL)
    if cached_payload is not None:
        pct_payload, unit_payload, page_info = cached_payload
    else:
        if df_meta_heat is not None and not df_meta_heat.empty and nets_heat:
            traffic_metric = "ps_traff_gb" if str(domain).upper() != "CS" else "cs_traff_erl"

            pct_payload, unit_payload, page_info = build_histo_payloads_fast(
                df_meta=df_meta_heat,
                df_ts=df_ts,
                UMBRAL_CFG=UM_MANAGER.config(),
                networks=nets_heat,
                valores_order=_valores_by_domain(domain),
                today=today_str, yday=yday_str,
                alarm_keys=alarm_keys_set,
                alarm_only=False,
                offset=offset, limit=limit,
                traffic_metric=traffic_metric,
            )
        else:
            pct_payload = unit_payload = None
            page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}
        _cache_set(_HI_PAYLOAD_CACHE, base_state_key, (pct_payload, unit_payload, page_info))
    perf_marks.append(("payloads", time.perf_counter()))

    # -------- Figuras overlay (sin selected_x) --------
    cached_figs = _cache_get(_HI_FIG_CACHE, state_key, _HI_FIG_TTL)
    if cached_figs is not None:
        fig_pct, fig_unit = cached_figs
    else:
        cfg = UM_MANAGER.config()
        fig_pct = build_overlay_waves_figure(
            pct_payload,
            UMBRAL_CFG=cfg,
            mode="severity",
            height=420,
            smooth_win=3,
            opacity=0.28,
            line_width=1.2,
            decimals=2,
            show_yaxis_ticks=True,
            selected_wave=selected_wave,
            show_traffic_bars=True,
            traffic_agg="mean",
            traffic_decimals=1
        ) if pct_payload else go.Figure()

        fig_unit = build_overlay_waves_figure(
            unit_payload,
            UMBRAL_CFG=cfg,
            mode="progress",
            height=420,
            smooth_win=3,
            opacity=0.25,
            line_width=1.2,
            decimals=0,
            show_yaxis_ticks=True,
            selected_wave=selected_wave,
            show_traffic_bars=False,
        ) if unit_payload else go.Figure()
        _cache_set(_HI_FIG_CACHE, state_key, (fig_pct, fig_unit))
    perf_marks.append(("figures", time.perf_counter()))

    # Marca el estado como â€œya renderizadoâ€
    _LAST_HI_KEY[domain] = state_key
    _perf_log(
        f"histograma_{domain.lower()}",
        perf_start,
        perf_marks,
        {"rows": int((page_info or {}).get("showing", 0)), "page_size": limit},
    )
    return fig_pct, fig_unit, page_info, False


def _render_api_main_heatmap(fecha, networks, technologies, vendors, clusters, page, page_sz, order_by):
    data = main_visuals_api_client.fetch_main_heatmap(
        fecha=fecha,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        page=page,
        page_size=page_sz,
        order_by=order_by,
        thresholds_snapshot=load_threshold_cfg(),
    )
    pct_payload = data.get("pct_payload")
    unit_payload = data.get("unit_payload")
    page_info = data.get("page_info") or {"total_rows": 0, "offset": 0, "limit": page_sz, "showing": 0}

    nrows = len((pct_payload or unit_payload or {}).get("y") or [])
    hm_height = _hm_height(nrows)
    fig_pct = build_heatmap_figure(pct_payload, height=hm_height, decimals=2) if pct_payload else go.Figure()
    fig_unit = build_heatmap_figure(unit_payload, height=hm_height, decimals=0) if unit_payload else go.Figure()
    table_component = (
        render_heatmap_summary_table(pct_payload, unit_payload, pct_decimals=2, unit_decimals=0)
        if (pct_payload or unit_payload)
        else dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")
    )
    return table_component, fig_pct, fig_unit, page_info


def _render_api_histogram(domain, fecha, networks, technologies, vendors, clusters, page, page_sz, selected_wave=None):
    data = main_visuals_api_client.fetch_histogram(
        fecha=fecha,
        domain=domain,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        page=page,
        page_size=page_sz,
        thresholds_snapshot=load_threshold_cfg(),
    )
    pct_payload = data.get("pct_payload")
    unit_payload = data.get("unit_payload")
    page_info = data.get("page_info") or {"total_rows": 0, "offset": 0, "limit": page_sz, "showing": 0}
    has_traffic = bool((pct_payload or {}).get("traffic_raw"))
    fig_pct = build_overlay_waves_figure(
        pct_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="severity",
        height=420,
        smooth_win=3,
        opacity=0.28,
        line_width=1.2,
        decimals=2,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=has_traffic,
        traffic_agg="mean",
        traffic_decimals=1,
    ) if pct_payload else go.Figure()

    fig_unit = build_overlay_waves_figure(
        unit_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="progress",
        height=420,
        smooth_win=3,
        opacity=0.25,
        line_width=1.2,
        decimals=0,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=False,
    ) if unit_payload else go.Figure()
    return fig_pct, fig_unit, page_info


def heatmap_callbacks(app):

    # -------------------------------------------------
    # 8) Render HEATMAP (tabla + figuras + paginado)
    # -------------------------------------------------
    @app.callback(
        Output("hm-table-container", "children", allow_duplicate=True),
        Output("hm-pct", "figure"),
        Output("hm-unit", "figure"),
        Output("hm-page-indicator", "children", allow_duplicate=True),
        Output("hm-total-rows-banner", "children", allow_duplicate=True),
        Output("heatmap-page-info", "data"),
        Output("hm-pct-dates", "children"),
        Output("hm-pct-hours", "children"),
        Output("hm-unit-dates", "children"),
        Output("hm-unit-hours", "children"),
        Input("heatmap-trigger", "data"),
        State("f-fecha", "date"),
        State("applied-filters-store", "data"),
        State("heatmap-page-state", "data"),
        State("hm-order-by", "value"),
        prevent_initial_call=True,
    )
    def refresh_heatmaps(_trigger, fecha, applied_filters, hm_page_state, hm_order_by):
        global _LAST_HEATMAP_KEY
        perf_start = time.perf_counter()
        perf_marks = []
        trigger_source = (_trigger or {}).get("source")
        trigger_revision = _trigger_revision(_trigger)

        # Normaliza filtros
        applied_filters = applied_filters or {}
        networks = _as_list(_applied_value(applied_filters, "network"))
        technologies = _as_list(_applied_value(applied_filters, "technology"))
        vendors = _as_list(_applied_value(applied_filters, "vendor"))
        clusters = _as_list(_applied_value(applied_filters, "cluster"))

        # -------- PaginaciÃ³n HEATMAP --------
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # Normaliza modo de orden
        hm_order_by_norm = (hm_order_by or "alarm_hours")
        hm_order_by_norm = str(hm_order_by_norm).strip().lower()

        if DATA_SOURCE == "api":
            dates_children, hours_children = _build_time_header_children_by_dates(fecha)
            try:
                table_component, fig_pct, fig_unit, page_info = _render_api_main_heatmap(
                    fecha,
                    networks,
                    technologies,
                    vendors,
                    clusters,
                    page,
                    page_sz,
                    hm_order_by_norm,
                )
            except Exception as exc:
                logger.warning("No se pudo cargar heatmap principal via heatmap_query: %s", exc)
                page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}
                return (
                    dbc.Alert(f"No se pudo cargar heatmap desde API: {exc}", color="warning", className="mb-0"),
                    go.Figure(),
                    go.Figure(),
                    "Pagina 1 de 1",
                    "Sin resultados.",
                    page_info,
                    dates_children,
                    hours_children,
                    dates_children,
                    hours_children,
                )

            total = int(page_info.get("total_rows", 0))
            showing = int(page_info.get("showing", 0))
            start_i = int(page_info.get("offset", 0)) + 1 if showing else 0
            end_i = start_i + showing - 1 if showing else 0
            total_pg = max(1, math.ceil(total / max(1, page_sz)))
            hm_indicator = f"Pagina {page} de {total_pg}"
            hm_banner = "Sin filas." if total == 0 else f"Mostrando {start_i}-{end_i} de {total} filas"
            _perf_log("refresh_heatmaps_api", perf_start, [("api_payload", time.perf_counter())], {"rows": showing})
            return (
                table_component,
                fig_pct,
                fig_unit,
                hm_indicator,
                hm_banner,
                page_info,
                dates_children,
                hours_children,
                dates_children,
                hours_children,
            )

        # State key incluye filtros + pÃ¡gina + orden
        state_key = _hm_key(
            fecha, networks, technologies, vendors, clusters, offset, limit, revision=trigger_revision
        ) + f"|ord={hm_order_by_norm}"

        # Si no cambiÃ³ nada, no re-renderiza
        if _LAST_HEATMAP_KEY == state_key:
            _perf_log(
                "refresh_heatmaps",
                perf_start,
                [("cache_hit", time.perf_counter())],
                {"cached": True},
            )
            dates_children, hours_children = _build_time_header_children_by_dates(fecha)
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                dates_children,
                hours_children,
                dates_children,
                hours_children,
            )

        # -------- Fechas HOY/AYER (sin hora) --------
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()

        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # -------- Datos TS (cacheados) --------
        try:
            df_ts = _fetch_df_ts_cached(
                today_str, yday_str, networks, technologies, vendors, clusters, revision=trigger_revision
            )
        except Exception as exc:
            logger.warning("No se pudo cargar heatmap principal via data access: %s", exc)
            df_ts = pd.DataFrame()
        perf_marks.append(("df_ts", time.perf_counter()))

        # -------- Redes efectivas --------
        if networks:
            nets_heat = networks
        else:
            nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) \
                if not df_ts.empty and "network" in df_ts.columns else []

        # -------- Meta alarmados (cacheada) --------
        df_meta_heat, alarm_keys_set = _fetch_alarm_meta_cached(
            today_str,
            vendors,
            clusters,
            nets_heat,
            technologies
        )
        perf_marks.append(("alarm_meta", time.perf_counter()))

        rank_key = _hm_key(
            fecha, networks, technologies, vendors, clusters, 0, 0, revision=trigger_revision
        ) + f"|ord={hm_order_by_norm}|rank"
        rank_cached = _cache_get(_HM_RANK_CACHE, rank_key, _HM_RANK_TTL)
        if rank_cached is not None:
            ranked_rows, rank_meta = rank_cached
        else:
            if df_meta_heat is not None and not df_meta_heat.empty and nets_heat:
                ranked_rows, rank_meta = build_heatmap_ranked_rows_fast(
                    df_meta=df_meta_heat,
                    df_ts=df_ts,
                    UMBRAL_CFG=UM_MANAGER.config(),
                    networks=nets_heat,
                    valores_order=("PS_RRC", "CS_RRC", "PS_S1", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB"),
                    today=today_str, yday=yday_str,
                    alarm_keys=alarm_keys_set,
                    alarm_only=False,
                    order_by=hm_order_by_norm
                )
            else:
                ranked_rows, rank_meta = None, None
            _cache_set(_HM_RANK_CACHE, rank_key, (ranked_rows, rank_meta))
        perf_marks.append(("ranking", time.perf_counter()))

        # -------- Payloads (cacheados por state_key) --------
        cached = _cache_get(_HM_PAYLOAD_CACHE, state_key, _HM_PAYLOAD_TTL)
        if cached is not None:
            pct_payload, unit_payload, page_info = cached
        else:
            if ranked_rows is not None and rank_meta is not None and not ranked_rows.empty:
                pct_payload, unit_payload, page_info = build_heatmap_payloads_from_ranked_rows(
                    ranked_rows,
                    df_ts=df_ts,
                    UMBRAL_CFG=UM_MANAGER.config(),
                    rank_meta=rank_meta,
                    offset=offset,
                    limit=limit,
                )
            else:
                pct_payload = unit_payload = None
                page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

            _cache_set(_HM_PAYLOAD_CACHE, state_key, (pct_payload, unit_payload, page_info))
        perf_marks.append(("payloads", time.perf_counter()))

        # -------- Altura del heatmap (para que cuadre con #filas) --------
        nrows = len((pct_payload or unit_payload or {}).get("y") or [])
        hm_height = _hm_height(nrows)

        # -------- Figuras --------
        fig_pct = build_heatmap_figure(pct_payload, height=hm_height, decimals=2) if pct_payload else go.Figure()
        fig_unit = build_heatmap_figure(unit_payload, height=hm_height, decimals=0) if unit_payload else go.Figure()
        perf_marks.append(("figures", time.perf_counter()))

        # -------- Tabla resumen --------
        if pct_payload or unit_payload:
            table_component = render_heatmap_summary_table(
                pct_payload, unit_payload, pct_decimals=2, unit_decimals=0
            )
        else:
            table_component = dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")
        perf_marks.append(("summary_table", time.perf_counter()))

        # -------- Indicadores de paginaciÃ³n --------
        total = int(page_info.get("total_rows", 0))
        showing = int(page_info.get("showing", 0))
        start_i = int(page_info.get("offset", 0)) + 1 if showing else 0
        end_i = start_i + showing - 1 if showing else 0
        total_pg = max(1, math.ceil(total / max(1, page_sz)))

        hm_indicator = f"PÃ¡gina {page} de {total_pg}"
        hm_banner = "Sin filas." if total == 0 else f"Mostrando {start_i}â€“{end_i} de {total} filas"
        dates_children, hours_children = _build_time_header_children_by_dates(fecha)

        # Marca renderizado
        _LAST_HEATMAP_KEY = state_key
        _perf_log(
            "refresh_heatmaps",
            perf_start,
            perf_marks,
            {"rows": showing, "page_size": limit, "order": hm_order_by_norm},
        )
        return (
            table_component,
            fig_pct,
            fig_unit,
            hm_indicator,
            hm_banner,
            page_info,
            dates_children,
            hours_children,
            dates_children,
            hours_children,
        )

    # -------------------------------------------------
    # Controlador: dispara el â€œtriggerâ€ cuando cambian filtros/pÃ¡gina/orden
    # (asÃ­ el callback pesado solo depende de heatmap-trigger)
    # -------------------------------------------------
    @app.callback(
        Output("heatmap-trigger", "data"),
        Input("data-ready-store", "data"),
        Input("f-fecha", "date"),
        Input("applied-filters-store", "data"),
        Input("heatmap-page-state", "data"),
        Input("hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap al cargar
    )
    def heatmap_trigger_controller(_ready, _fecha, _applied_filters, _page_state, _ord):
        # Un timestamp basta para â€œforzarâ€ la actualizaciÃ³n
        return {
            "ts": time.time(),
            "source": "data_ready" if ctx.triggered_id == "data-ready-store" else "ui",
            "slot": ((_ready or {}).get("slot") or {}),
            "updated_at": (_ready or {}).get("updated_at"),
        }

    # -------------------------------------------------
    # Reset de paginaciÃ³n del heatmap cuando cambian filtros/tamaÃ±o/orden
    # -------------------------------------------------
    @app.callback(
        Output("heatmap-page-state", "data"),
        Input("f-fecha", "date"),
        Input("applied-filters-store", "data"),
        Input("hm-page-size", "value"),
        Input("hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap
    )
    def hm_reset_page_on_filters(_fecha, _applied_filters, hm_page_size, _ord):
        return reset_page_state(hm_page_size, default_size=50)

    # -------------------------------------------------
    # Botones prev/next del heatmap
    # -------------------------------------------------
    @app.callback(
        Output("heatmap-page-state", "data", allow_duplicate=True),
        Input("hm-page-prev", "n_clicks"),
        Input("hm-page-next", "n_clicks"),
        State("heatmap-page-state", "data"),
        prevent_initial_call=True,
    )
    def hm_paginate(n_prev, n_next, state):
        return paginate_state(state, prev_id="hm-page-prev", next_id="hm-page-next", default_size=50)

    # -------------------------------------------------
    # 8) Histograma PS (figuras + page_info)
    # -------------------------------------------------
    @app.callback(
        Output("hi-pct-ps", "figure"),
        Output("hi-unit-ps", "figure"),
        Output("histo-page-info", "data"),
        Input("histo-trigger", "data"),
        Input("histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("applied-filters-store", "data"),
        State("histo-page-state", "data"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_histograma_ps(_trigger, sel_wave, fecha, applied_filters, hm_page_state, link_state):
        applied_filters = applied_filters or {}
        networks = _as_list(_applied_value(applied_filters, "network"))
        technologies = _as_list(_applied_value(applied_filters, "technology"))
        vendors = _as_list(_applied_value(applied_filters, "vendor"))
        clusters = _as_list(_applied_value(applied_filters, "cluster"))
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        if DATA_SOURCE == "api":
            try:
                fig_pct, fig_unit, page_info = _render_api_histogram(
                    "PS", fecha, networks, technologies, vendors, clusters, page, page_sz, selected_wave=sel_wave
                )
                return fig_pct, fig_unit, page_info
            except Exception as exc:
                logger.warning("No se pudo cargar histograma PS via heatmap_query: %s", exc)
                return go.Figure(), go.Figure(), {"total_rows": 0, "offset": 0, "limit": page_sz, "showing": 0}
        fig_pct, fig_unit, page_info, is_cache_hit = _build_histograma_for_domain(
            "PS", sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state, _trigger
        )
        # Si fue â€œcache hitâ€ no actualizamos nada
        if is_cache_hit:
            return no_update, no_update, no_update
        return fig_pct, fig_unit, page_info

    # -------------------------------------------------
    # Histograma CS (figuras)
    # -------------------------------------------------
    @app.callback(
        Output("hi-pct-cs", "figure"),
        Output("hi-unit-cs", "figure"),
        Input("histo-trigger", "data"),
        Input("histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("applied-filters-store", "data"),
        State("histo-page-state", "data"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_histograma_cs(_trigger, sel_wave, fecha, applied_filters, hm_page_state, link_state):
        applied_filters = applied_filters or {}
        networks = _as_list(_applied_value(applied_filters, "network"))
        technologies = _as_list(_applied_value(applied_filters, "technology"))
        vendors = _as_list(_applied_value(applied_filters, "vendor"))
        clusters = _as_list(_applied_value(applied_filters, "cluster"))
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        if DATA_SOURCE == "api":
            try:
                fig_pct, fig_unit, _page_info = _render_api_histogram(
                    "CS", fecha, networks, technologies, vendors, clusters, page, page_sz, selected_wave=sel_wave
                )
                return fig_pct, fig_unit
            except Exception as exc:
                logger.warning("No se pudo cargar histograma CS via heatmap_query: %s", exc)
                return go.Figure(), go.Figure()
        fig_pct, fig_unit, _page_info, is_cache_hit = _build_histograma_for_domain(
            "CS", sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state, _trigger
        )
        if is_cache_hit:
            return no_update, no_update
        return fig_pct, fig_unit

    # -------------------------------------------------
    # Trigger del histograma (para re-render cuando cambian filtros/pÃ¡gina/link)
    # -------------------------------------------------
    @app.callback(
        Output("histo-trigger", "data"),
        Input("data-ready-store", "data"),
        Input("f-fecha", "date"),
        Input("applied-filters-store", "data"),
        Input("hm-page-size", "value"),
        Input("topoff-link-state", "data"),
        prevent_initial_call=False,
    )
    def histo_trigger_controller(_ready, _fecha, _applied_filters, _page_size, _link_state):
        if not _ready:
            return no_update
        return {
            "ts": time.time(),
            "source": "data_ready" if ctx.triggered_id == "data-ready-store" else "ui",
            "slot": ((_ready or {}).get("slot") or {}),
            "updated_at": (_ready or {}).get("updated_at"),
        }

    # -------------------------------------------------
    # Reset de paginaciÃ³n de histograma cuando cambian filtros/tamaÃ±o/link
    # (OJO: aquÃ­ usas hm-page-size y botones hm-page-prev/next)
    # -------------------------------------------------
    @app.callback(
        Output("histo-page-state", "data"),
        Input("f-fecha", "date"),
        Input("applied-filters-store", "data"),
        Input("hm-page-size", "value"),
        Input("topoff-link-state", "data"),
        prevent_initial_call=False,
    )
    def hi_reset_page_on_filters(_fecha, _applied_filters, hm_page_size, _link_state):
        return reset_page_state(hm_page_size, default_size=50)

    # -------------------------------------------------
    # PaginaciÃ³n del histograma (usa botones del heatmap)
    # -------------------------------------------------
    @app.callback(
        Output("histo-page-state", "data", allow_duplicate=True),
        Input("hm-page-prev", "n_clicks"),
        Input("hm-page-next", "n_clicks"),
        State("histo-page-state", "data"),
        prevent_initial_call=True,
    )
    def hi_paginate(n_prev, n_next, state):
        return paginate_state(state, prev_id="hm-page-prev", next_id="hm-page-next", default_size=50)

    # -------------------------------------------------
    # Click: seleccionar una wave (PS %)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data"),
        Input("hi-pct-ps", "clickData"),
        State("hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_pct_ps(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]
        return {"series_key": series_key}

    # -------------------------------------------------
    # Click: seleccionar una wave (PS UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-unit-ps", "clickData"),
        State("hi-unit-ps", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_unit_ps(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]
        return {"series_key": series_key}

    # -------------------------------------------------
    # Click: seleccionar una wave (CS %)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-pct-cs", "clickData"),
        State("hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_pct_cs(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]
        return {"series_key": series_key}

    # -------------------------------------------------
    # Click: seleccionar una wave (CS UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-unit-cs", "clickData"),
        State("hi-unit-cs", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_unit_cs(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]
        return {"series_key": series_key}

    # -------------------------------------------------
    # Doble click / autoscale: limpiar selecciÃ³n de wave
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-pct-ps", "relayoutData"),
        Input("hi-unit-ps", "relayoutData"),
        Input("hi-pct-cs", "relayoutData"),
        Input("hi-unit-cs", "relayoutData"),
        prevent_initial_call=True,
    )
    def clear_wave_on_doubleclick(r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit):
        def is_autosize(r):
            return bool(r) and (
                r.get("autosize") is True
                or r.get("xaxis.autorange") is True
                or r.get("yaxis.autorange") is True
            )

        # Si alguno hizo autoscale -> limpiamos selecciÃ³n
        if any(is_autosize(r) for r in [r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit]):
            return {}
        return no_update

    # -------------------------------------------------
    # Sync de legend: PS (lo que ocultas en % tambiÃ©n se oculta en UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("hi-unit-ps", "figure", allow_duplicate=True),
        Input("hi-pct-ps", "restyleData"),
        State("hi-unit-ps", "figure"),
        State("hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_ps(restyle, unit_fig, pct_fig):
        if not restyle or not unit_fig or not pct_fig:
            return no_update
        try:
            update, idxs = restyle
        except Exception:
            return no_update
        if "visible" not in update:
            return no_update

        vis_update = update["visible"]
        if not isinstance(vis_update, (list, tuple)):
            vis_update = [vis_update] * len(idxs)

        new_unit_fig = unit_fig.copy()
        data_unit = new_unit_fig.get("data", [])
        data_pct = pct_fig.get("data", [])

        # Copia el "visible" de la traza en % hacia la traza equivalente en UNIT
        for v, idx in zip(vis_update, idxs):
            if idx is None or idx >= len(data_pct):
                continue
            trace_pct = data_pct[idx]
            cluster = trace_pct.get("legendgroup") or trace_pct.get("name")
            if not cluster:
                continue

            for t in data_unit:
                if t.get("legendgroup") == cluster or t.get("name") == cluster:
                    t["visible"] = v

        return new_unit_fig

    # -------------------------------------------------
    # Sync de legend: CS
    # -------------------------------------------------
    @app.callback(
        Output("hi-unit-cs", "figure", allow_duplicate=True),
        Input("hi-pct-cs", "restyleData"),
        State("hi-unit-cs", "figure"),
        State("hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_cs(restyle, unit_fig, pct_fig):
        if not restyle or not unit_fig or not pct_fig:
            return no_update
        try:
            update, idxs = restyle
        except Exception:
            return no_update
        if "visible" not in update:
            return no_update

        vis_update = update["visible"]
        if not isinstance(vis_update, (list, tuple)):
            vis_update = [vis_update] * len(idxs)

        new_unit_fig = unit_fig.copy()
        data_unit = new_unit_fig.get("data", [])
        data_pct = pct_fig.get("data", [])

        for v, idx in zip(vis_update, idxs):
            if idx is None or idx >= len(data_pct):
                continue
            trace_pct = data_pct[idx]
            cluster = trace_pct.get("legendgroup") or trace_pct.get("name")
            if not cluster:
                continue

            for t in data_unit:
                if t.get("legendgroup") == cluster or t.get("name") == cluster:
                    t["visible"] = v

        return new_unit_fig
