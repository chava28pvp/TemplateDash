# src/callbacks/export_callback.py
from io import BytesIO
from dash import Output, Input, State, callback, dcc
import pandas as pd
import numpy as np
import math

from xlsxwriter import Workbook  # pip install xlsxwriter

from src.dataAccess.data_access import (
    fetch_kpis_paginated,
    fetch_kpis_paginated_severity_global_sort,
    fetch_kpis_paginated_severity_sort,
    COLMAP,
)
from src.Utils.alarmados import load_threshold_cfg
from components.Tables.main_table import (
    pivot_by_network,
    expand_groups_for_networks,
    _resolve_sort_col,
    ROW_KEYS
)

def _as_list(x):
    if x is None: return None
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def xlsx_col_letter(idx: int) -> str:
    s = ""; idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s

def _parse_col(col):
    return col.split("__", 1) if "__" in col else (None, col)

def _severity_cfg_for(kpi, net, cfg):
    sev = (cfg.get("severity") or {}).get(kpi)
    if not sev: return None, None
    if "default" in sev or "per_network" in sev:
        base = sev.get("default", {})
        per  = sev.get("per_network", {}) or {}
        use  = per.get(net, {}) if (net and net in per) else base
        orientation = use.get("orientation", base.get("orientation", sev.get("orientation")))
        thresholds  = use.get("thresholds", base.get("thresholds"))
    else:
        orientation = sev.get("orientation")
        thresholds  = sev.get("thresholds")
    return orientation, thresholds

def _progress_max_for(kpi, net, cfg):
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
    if not orientation or not thr: return
    from_idx, to_idx = 2, n_rows + 1
    col_letter = xlsx_col_letter(col_j)
    rng = f"{col_letter}{from_idx}:{col_letter}{to_idx}"
    fmt_ex = workbook.add_format({"bg_color": (colors or {}).get("excelente", "#2ecc71")})
    fmt_bu = workbook.add_format({"bg_color": (colors or {}).get("bueno",     "#f1c40f")})
    fmt_re = workbook.add_format({"bg_color": (colors or {}).get("regular",   "#e67e22")})
    fmt_cr = workbook.add_format({"bg_color": (colors or {}).get("critico",   "#e74c3c")})
    def v(x): return None if x is None else float(x)
    ex = v(thr.get("excelente")); bu = v(thr.get("bueno"))
    re = v(thr.get("regular"));   cr = v(thr.get("critico"))
    if orientation == "higher_is_better":
        if re is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"<", "value":re,"format":fmt_cr})
        if re is not None and bu is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":re,"maximum":bu,"format":fmt_re})
        if bu is not None and ex is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":bu,"maximum":ex,"format":fmt_bu})
        if ex is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":">=","value":ex,"format":fmt_ex})
    else:
        if re is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":">", "value":re,"format":fmt_cr})
        if bu is not None and re is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":bu,"maximum":re,"format":fmt_re})
        if ex is not None and bu is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"between","minimum":ex,"maximum":bu,"format":fmt_bu})
        if ex is not None: worksheet.conditional_format(rng, {"type":"cell","criteria":"<=","value":ex,"format":fmt_ex})

def export_callback(app):
    @callback(
        Output("download-excel", "data"),
        Input("export-excel", "n_clicks"),
        State("f-fecha", "date"),
        State("f-hora", "value"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("f-sort-mode", "value"),   # "global" | "alarmado"
        State("sort-state", "data"),     # {"column":..., "ascending":...}
        State("page-state", "data"),     # {"page":..., "page_size":...}
        prevent_initial_call=True,
    )
    def do_export(_, fecha, hora, networks, techs, vendors, clusters,
                  sort_mode, sort_state, page_state):

        # 1) Misma página que la tabla
        networks = _as_list(networks); techs = _as_list(techs)
        vendors  = _as_list(vendors);  clusters = _as_list(clusters)
        page      = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        sort_by = None; sort_net = None; ascending = True
        if sort_mode != "alarmado" and sort_state and sort_state.get("column"):
            col = sort_state["column"]; ascending = bool(sort_state.get("ascending", True))
            if "__" in col: sort_net, sort_by = col.split("__", 1)
            else:           sort_by = col

        if sort_mode == "alarmado":
            df_page, _ = fetch_kpis_paginated_severity_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=techs or None,
                page=page, page_size=page_size,
            )
            safe_sort_state = None
        else:
            if sort_by and sort_by in COLMAP:
                df_page, _ = fetch_kpis_paginated_severity_global_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=techs or None,
                    page=page, page_size=page_size,
                    sort_by_friendly=sort_by, sort_net=sort_net, ascending=ascending,
                )
            else:
                df_page, _ = fetch_kpis_paginated(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=techs or None,
                    page=page, page_size=page_size,
                )
            safe_sort_state = sort_state

        if df_page is None or df_page.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # 2) Pivot igual que la tabla
        is_long = "network" in df_page.columns
        if is_long:
            nets_for_pivot = networks or sorted(df_page["network"].dropna().unique().tolist())
            df_wide = pivot_by_network(df_page, networks=nets_for_pivot)
        else:
            df_wide = df_page.copy()

        if df_wide is None or df_wide.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # ---------- ORDER FIX: replica orden visual de la tabla ----------
        KEY_COLS = list(ROW_KEYS)  # ["fecha","hora","vendor","noc_cluster","technology"]
        # mapa de orden según la página tal como salió de la consulta
        tuples_in_order = list(dict.fromkeys(
            map(tuple, df_page[KEY_COLS].itertuples(index=False, name=None))
        ))
        order_map = {t: i for i, t in enumerate(tuples_in_order)}
        # asegura que keys existan como columnas
        if isinstance(df_wide.index, pd.MultiIndex) or df_wide.index.name is not None:
            df_wide = df_wide.reset_index()
        # columna ordinal oculta
        df_wide["_ord"] = df_wide[KEY_COLS].apply(
            lambda r: order_map.get(tuple(r.values.tolist()), 10**9), axis=1
        )
        # ---------------------------------------------------------------

        # Orden de columnas como la tabla
        _, metric_order, _ = expand_groups_for_networks(
            networks or (sorted(df_page["network"].dropna().unique()) if is_long else [])
        )
        visible_order = KEY_COLS + metric_order
        cols_final = [c for c in visible_order if c in df_wide.columns]
        for k in KEY_COLS:
            if k not in cols_final and k in df_wide.columns:
                cols_final.insert(0, k)

        # Orden de filas:
        # - si hay sort explícito (global), ordena por [resolved, _ord]
        # - si no, respeta solo _ord (orden visual original)
        if safe_sort_state:
            col_req = (safe_sort_state or {}).get("column")
            resolved = _resolve_sort_col(df_wide, metric_order, col_req)
            if resolved in df_wide.columns:
                asc = bool((safe_sort_state or {}).get("ascending", True))
                df_wide = df_wide.sort_values(by=[resolved, "_ord"],
                                              ascending=[asc, True],
                                              na_position="last")
            else:
                df_wide = df_wide.sort_values("_ord")
        else:
            df_wide = df_wide.sort_values("_ord")

        df_out = df_wide[cols_final].copy()

        # 3) NO tocar keys; normaliza SOLO KPIs
        ID_KEYS = {"fecha", "hora", "vendor", "noc_cluster", "technology", "network"}
        KPI_SET = set(COLMAP.keys()) - ID_KEYS
        for colname in df_out.columns:
            _, kpi = _parse_col(colname)
            if kpi in KPI_SET:
                df_out[colname] = pd.to_numeric(df_out[colname], errors="coerce")

        # Sanitiza NaN/Inf
        df_out = df_out.replace([np.inf, -np.inf], pd.NA)
        df_out = df_out.where(pd.notna(df_out), None)

        # 4) Escribir XLSX con formatos
        raw_cfg = load_threshold_cfg()

        profile_name = "main"  # o lo que venga de algún estado si luego quieres hacerlo dinámico
        profiles = raw_cfg.get("profiles") or {}
        profile_cfg = profiles.get(profile_name, {})

        # Rearmamos cfg tal como lo esperan los helpers
        cfg = {
            "severity": profile_cfg.get("severity", {}),
            "progress": profile_cfg.get("progress", {}),
            "colors": raw_cfg.get("colors", {}),
        }
        colors = cfg.get("colors") or {}

        bio = BytesIO()
        workbook = Workbook(bio, {'in_memory': True})
        worksheet = workbook.add_worksheet("KPIs")

        # Encabezados
        header_fmt = workbook.add_format({"bold": True})
        for j, name in enumerate(df_out.columns):
            worksheet.write(0, j, name, header_fmt)

        # Filas seguras
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

        # Auto ancho
        for j, col in enumerate(df_out.columns):
            sample = [str(v) for v in df_out[col][:50] if v is not None]
            width = max([len(col)] + [len(s) for s in sample] + [10])
            worksheet.set_column(j, j, min(width, 40))

        # Data bars SOLO KPIs de progreso
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

        # Umbrales 4-bandas SOLO KPIs
        for j, col in enumerate(df_out.columns):
            net, kpi = _parse_col(col)
            if kpi not in KPI_SET:
                continue
            orientation, thr = _severity_cfg_for(kpi, net, cfg)
            _add_severity_rules(workbook, worksheet, j, n_rows, orientation, thr, colors)

        workbook.close()
        bio.seek(0)

        fname = f"kpis_p{page}_sz{page_size}_{fecha or 'fecha'}_{(hora or 'Todas')[:5]}_fmt.xlsx"
        return dcc.send_bytes(bio.read(), filename=fname)
