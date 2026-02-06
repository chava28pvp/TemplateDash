# Callback para exportar a Excel EXACTAMENTE la misma página que estás viendo en la tabla,
from io import BytesIO
from dash import Output, Input, State, callback, dcc
import pandas as pd
import numpy as np
import math

from xlsxwriter import Workbook  # pip install xlsxwriter

from src.dataAccess.data_access import (
    fetch_kpis_paginated_severity_global_sort,
    fetch_kpis_paginated_severity_sort,
    COLMAP,
)
from src.Utils.alarmados import load_threshold_cfg
from components.main.main_table import (
    pivot_by_network,
    expand_groups_for_networks,
    _resolve_sort_col,
    ROW_KEYS
)

def _as_list(x):
    """Normaliza un input a lista:
    - None -> None
    - list/tuple -> list
    - valor -> [valor]
    """
    if x is None: return None
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def xlsx_col_letter(idx: int) -> str:
    """Convierte índice de columna 0-based a letra Excel:
    0->A, 1->B, ..., 25->Z, 26->AA, etc.
    """
    s = ""; idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s

def _parse_col(col):
    """Separa columnas tipo 'NET__kpi' en (net, kpi).
    Si no trae prefijo, regresa (None, col).
    """
    return col.split("__", 1) if "__" in col else (None, col)

def _severity_cfg_for(kpi, net, cfg):
    """Obtiene (orientation, thresholds) de severidad para una KPI y network.
    Soporta:
    - cfg['severity'][kpi] directo
    - cfg['severity'][kpi]['default'] y cfg['severity'][kpi]['per_network'][NET]
    """
    sev = (cfg.get("severity") or {}).get(kpi)
    if not sev: return None, None

    # Caso: configuración por default + override por network
    if "default" in sev or "per_network" in sev:
        base = sev.get("default", {})
        per  = sev.get("per_network", {}) or {}
        use  = per.get(net, {}) if (net and net in per) else base

        # orientation: usa override si existe; si no, cae al base; si no, al sev raíz
        orientation = use.get("orientation", base.get("orientation", sev.get("orientation")))
        thresholds  = use.get("thresholds", base.get("thresholds"))
    else:
        # Caso simple: orientation/thresholds directo
        orientation = sev.get("orientation")
        thresholds  = sev.get("thresholds")

    return orientation, thresholds

def _progress_max_for(kpi, net, cfg):
    """Obtiene el máximo (max) para progress/data bars en Excel para una KPI.
    Prioridad:
    1) per_network[NET].max
    2) default.max
    3) max directo
    4) fallback 100
    """
    progress_cfg = (cfg.get("progress") or {})
    pk = progress_cfg.get(kpi, {})
    per = (pk.get("per_network") or {})

    if net and net in per and isinstance(per[net], dict) and "max" in per[net]:
        return per[net]["max"]
    if "default" in pk and "max" in pk["default"]:
        return pk["default"]["max"]
    if "max" in pk:
        return pk["max"]
    return 100

def _add_severity_rules(workbook, worksheet, col_j, n_rows, orientation, thr, colors):
    """Aplica formato condicional 4-bandas (excelente/bueno/regular/critico)
    a una columna específica (col_j) para las filas de datos.
    """
    if not orientation or not thr: return

    # Rango de datos: fila 2 a fila n_rows+1 (porque fila 1 es header)
    from_idx, to_idx = 2, n_rows + 1
    col_letter = xlsx_col_letter(col_j)
    rng = f"{col_letter}{from_idx}:{col_letter}{to_idx}"

    # Formatos (background) por severidad
    fmt_ex = workbook.add_format({"bg_color": (colors or {}).get("excelente", "#2ecc71")})
    fmt_bu = workbook.add_format({"bg_color": (colors or {}).get("bueno",     "#f1c40f")})
    fmt_re = workbook.add_format({"bg_color": (colors or {}).get("regular",   "#e67e22")})
    fmt_cr = workbook.add_format({"bg_color": (colors or {}).get("critico",   "#e74c3c")})

    # Helper: convertir a float si existe
    def v(x): return None if x is None else float(x)

    ex = v(thr.get("excelente")); bu = v(thr.get("bueno"))
    re = v(thr.get("regular"));   cr = v(thr.get("critico"))

    # Reglas según orientación (higher_is_better vs lower_is_better)
    if orientation == "higher_is_better":
        # Menor a regular -> crítico
        if re is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"<", "value":re,"format":fmt_cr})
        # Entre regular y bueno -> regular
        if re is not None and bu is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":re,"maximum":bu,"format":fmt_re})
        # Entre bueno y excelente -> bueno
        if bu is not None and ex is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":bu,"maximum":ex,"format":fmt_bu})
        # >= excelente -> excelente
        if ex is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":">=","value":ex,"format":fmt_ex})
    else:
        # Mayor a regular -> crítico (porque menor es mejor)
        if re is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":">", "value":re,"format":fmt_cr})
        # Entre bueno y regular -> regular
        if bu is not None and re is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":bu,"maximum":re,"format":fmt_re})
        # Entre excelente y bueno -> bueno
        if ex is not None and bu is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":ex,"maximum":bu,"format":fmt_bu})
        # <= excelente -> excelente
        if ex is not None:
            worksheet.conditional_format(rng, {"type":"cell","criteria":"<=","value":ex,"format":fmt_ex})

def export_callback(app):
    # Callback: cuando el usuario da click en "export-excel", genera un XLSX y lo descarga
    @callback(
        Output("download-excel", "data"),
        Input("export-excel", "n_clicks"),
        State("f-fecha", "date"),
        State("f-hora", "value"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("f-sort-mode", "value"),
        State("sort-state", "data"),
        State("page-state", "data"),
        prevent_initial_call=True,
    )
    def do_export(_, fecha, hora, networks, techs, vendors, clusters,
                  sort_mode, sort_state, page_state):

        # 1) Exporta LA MISMA PÁGINA que está viendo el usuario en la tabla
        networks = _as_list(networks); techs = _as_list(techs)
        vendors  = _as_list(vendors);  clusters = _as_list(clusters)
        page      = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # Preparar sort si aplica (para el query global)
        sort_by = None
        sort_net = None
        ascending = True

        # Nota: aquí solo usamos sort_state si NO estamos en "global" puro,
        #       tal como lo tenías originalmente.
        if sort_mode != "global" and sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))
            if "__" in col:
                sort_net, sort_by = col.split("__", 1)   # NET__kpi -> ("NET","kpi")
            else:
                sort_by = col

        # 1.A) Traer DF de la página, usando el mismo endpoint que la tabla
        if sort_mode == "alarmado":
            # Modo alarmado: ya viene ordenado por severidad/alarmas desde backend
            df_page, _ = fetch_kpis_paginated_severity_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=techs or None,
                page=page, page_size=page_size,
            )
            safe_sort_state = None  # en export respetamos orden “ya resuelto” por backend
        else:
            # Modo global: ordena por severidad y opcionalmente por columna (si se pidió)
            df_page, _ = fetch_kpis_paginated_severity_global_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=techs or None,
                page=page, page_size=page_size,
                sort_by_friendly=sort_by if (sort_by in COLMAP) else None,
                sort_net=sort_net,
                ascending=ascending,
            )
            safe_sort_state = None

        # Si no hay datos, descargamos un txt para feedback claro
        if df_page is None or df_page.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # 2) Pivot igual que la tabla (si viene en formato long con columna "network")
        is_long = "network" in df_page.columns
        if is_long:
            nets_for_pivot = networks or sorted(df_page["network"].dropna().unique().tolist())
            df_wide = pivot_by_network(df_page, networks=nets_for_pivot)
        else:
            df_wide = df_page.copy()

        if df_wide is None or df_wide.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # ---------- ORDER FIX: replica orden visual de la tabla ----------
        # 3) Construimos un orden ordinal basado en el orden real de df_page (antes del pivot)
        KEY_COLS = list(ROW_KEYS)  # ["fecha","hora","vendor","noc_cluster","technology"]
        tuples_in_order = list(dict.fromkeys(
            map(tuple, df_page[KEY_COLS].itertuples(index=False, name=None))
        ))
        order_map = {t: i for i, t in enumerate(tuples_in_order)}

        # Asegura que las keys estén como columnas (por si pivot dejó MultiIndex)
        if isinstance(df_wide.index, pd.MultiIndex) or df_wide.index.name is not None:
            df_wide = df_wide.reset_index()

        # Columna ordinal oculta para poder ordenar exactamente como la tabla
        df_wide["_ord"] = df_wide[KEY_COLS].apply(
            lambda r: order_map.get(tuple(r.values.tolist()), 10**9), axis=1
        )
        # ---------------------------------------------------------------

        # 4) Orden de columnas igual que la tabla (keys primero + métricas)
        _, metric_order, _ = expand_groups_for_networks(
            networks or (sorted(df_page["network"].dropna().unique()) if is_long else [])
        )
        visible_order = KEY_COLS + metric_order
        cols_final = [c for c in visible_order if c in df_wide.columns]

        # Garantiza que las keys queden al inicio aunque falte algo en metric_order
        for k in KEY_COLS:
            if k not in cols_final and k in df_wide.columns:
                cols_final.insert(0, k)

        # 5) Orden de filas:
        # - si hubiera un sort_state “seguro”, ordena por [col_resuelta, _ord]
        # - si no, respeta SOLO _ord (orden visual)
        if safe_sort_state:
            col_req = (safe_sort_state or {}).get("column")
            resolved = _resolve_sort_col(df_wide, metric_order, col_req)
            if resolved in df_wide.columns:
                asc = bool((safe_sort_state or {}).get("ascending", True))
                df_wide = df_wide.sort_values(
                    by=[resolved, "_ord"],
                    ascending=[asc, True],
                    na_position="last"
                )
            else:
                df_wide = df_wide.sort_values("_ord")
        else:
            df_wide = df_wide.sort_values("_ord")

        # DF final a exportar
        df_out = df_wide[cols_final].copy()

        # 6) NO tocar keys; normaliza SOLO KPIs a numérico (para que Excel las reconozca)
        ID_KEYS = {"fecha", "hora", "vendor", "noc_cluster", "technology", "network"}
        KPI_SET = set(COLMAP.keys()) - ID_KEYS
        for colname in df_out.columns:
            _, kpi = _parse_col(colname)
            if kpi in KPI_SET:
                df_out[colname] = pd.to_numeric(df_out[colname], errors="coerce")

        # 7) Sanitiza NaN/Inf (Excel no los quiere como números)
        df_out = df_out.replace([np.inf, -np.inf], pd.NA)
        df_out = df_out.where(pd.notna(df_out), None)

        # 8) Escribir XLSX con formatos (severity + progress)
        raw_cfg = load_threshold_cfg()

        # Por ahora fijo "main" (si luego quieres, lo haces dinámico por estado)
        profile_name = "main"
        profiles = raw_cfg.get("profiles") or {}
        profile_cfg = profiles.get(profile_name, {})

        # Config simplificada que esperan los helpers
        cfg = {
            "severity": profile_cfg.get("severity", {}),
            "progress": profile_cfg.get("progress", {}),
            "colors": raw_cfg.get("colors", {}),
        }
        colors = cfg.get("colors") or {}

        bio = BytesIO()
        workbook = Workbook(bio, {'in_memory': True})
        worksheet = workbook.add_worksheet("KPIs")

        # Encabezados (fila 1)
        header_fmt = workbook.add_format({"bold": True})
        for j, name in enumerate(df_out.columns):
            worksheet.write(0, j, name, header_fmt)

        # Filas (datos) con escritura segura por tipo
        for i, row in enumerate(df_out.itertuples(index=False, name=None), start=1):
            for j, val in enumerate(row):
                if val is None:
                    worksheet.write_blank(i, j, None)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    if isinstance(val, float) and not math.isfinite(val):
                        worksheet.write_blank(i, j, None)
                    else:
                        worksheet.write_number(i, j, float(val))
                else:
                    worksheet.write(i, j, str(val))

        n_rows = len(df_out)

        # Auto ancho de columnas (muestra 50 filas para estimar)
        for j, col in enumerate(df_out.columns):
            sample = [str(v) for v in df_out[col][:50] if v is not None]
            width = max([len(col)] + [len(s) for s in sample] + [10])
            worksheet.set_column(j, j, min(width, 40))

        # 9) Data bars SOLO para KPIs de progress
        for j, col in enumerate(df_out.columns):
            net, kpi = _parse_col(col)
            if kpi in KPI_SET and kpi in (cfg.get("progress") or {}):
                col_letter = xlsx_col_letter(j)
                rng = f"{col_letter}2:{col_letter}{n_rows+1}"
                maxv = _progress_max_for(kpi, net, cfg)
                worksheet.conditional_format(rng, {
                    "type": "data_bar",
                    "min_type": "num", "min_value": 0,
                    "max_type": "num", "max_value": maxv,
                })

        # 10) Umbrales 4-bandas SOLO KPIs (severidad)
        for j, col in enumerate(df_out.columns):
            net, kpi = _parse_col(col)
            if kpi not in KPI_SET:
                continue
            orientation, thr = _severity_cfg_for(kpi, net, cfg)
            _add_severity_rules(workbook, worksheet, j, n_rows, orientation, thr, colors)

        workbook.close()
        bio.seek(0)

        # Nombre de archivo con info de página y filtros
        fname = f"kpis_p{page}_sz{page_size}_{fecha or 'fecha'}_{(hora or 'Todas')[:5]}_fmt.xlsx"
        return dcc.send_bytes(bio.read(), filename=fname)
