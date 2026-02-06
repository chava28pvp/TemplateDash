from io import BytesIO
from dash import Output, Input, State, callback, dcc
import pandas as pd
import numpy as np
import math
from xlsxwriter import Workbook  # pip install xlsxwriter

# === DATA ACCESS TOPOFF ===
from src.dataAccess.data_acess_topoff import (
    fetch_topoff_paginated,
    fetch_topoff_paginated_global_sort,
    fetch_topoff_paginated_severity_global_sort,
)

# === UMBRALES / COLORES JSON ===
from src.Utils.alarmados import load_threshold_cfg


# ======================================================
# Helpers
# ======================================================
def _as_list(x):
    """Asegura que un filtro siempre sea lista (o None)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def xlsx_col_letter(idx: int) -> str:
    """Convierte índice 0-based a letra Excel: 0->A, 1->B, 26->AA, etc."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def _parse_col(col):
    """
    Si viene con prefijo tipo 'NET__kpi' lo separa en (NET, kpi).
    Si no, regresa (None, col).
    """
    return col.split("__", 1) if "__" in col else (None, col)


def _severity_cfg_for(kpi, net, cfg):
    """
    Extrae orientación y thresholds del JSON para un KPI.
    Soporta:
      - formato simple: cfg["severity"][kpi] = {"orientation":..., "thresholds":...}
      - formato con default/per_network
    """
    sev = (cfg.get("severity") or {}).get(kpi)
    if not sev:
        return None, None

    # Caso con default/per_network (más flexible)
    if "default" in sev or "per_network" in sev:
        base = sev.get("default", {})
        per = sev.get("per_network", {}) or {}
        use = per.get(net, {}) if (net and net in per) else base

        orientation = use.get("orientation", base.get("orientation", sev.get("orientation")))
        thresholds = use.get("thresholds", base.get("thresholds"))
    else:
        # Caso simple
        orientation = sev.get("orientation")
        thresholds = sev.get("thresholds")

    return orientation, thresholds


def _progress_max_for(kpi, net, cfg):
    """
    Obtiene el max para data bars (progress) del JSON.
    Prioridad:
      per_network -> default -> max directo -> 100
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
    """
    Aplica formato condicional de 4 bandas (excelente/bueno/regular/critico)
    a toda una columna de Excel.
    """
    if not orientation or not thr:
        return

    from_idx, to_idx = 2, n_rows + 1  # filas con datos (1 es header)
    col_letter = xlsx_col_letter(col_j)
    rng = f"{col_letter}{from_idx}:{col_letter}{to_idx}"

    # Colores por defecto si el JSON no trae
    fmt_ex = workbook.add_format({"bg_color": (colors or {}).get("excelente", "#2ecc71")})
    fmt_bu = workbook.add_format({"bg_color": (colors or {}).get("bueno", "#f1c40f")})
    fmt_re = workbook.add_format({"bg_color": (colors or {}).get("regular", "#e67e22")})
    fmt_cr = workbook.add_format({"bg_color": (colors or {}).get("critico", "#e74c3c")})

    def v(x):
        return None if x is None else float(x)

    ex = v(thr.get("excelente"))
    bu = v(thr.get("bueno"))
    re = v(thr.get("regular"))
    cr = v(thr.get("critico"))

    # higher_is_better: excelente = alto, crítico = bajo
    if orientation == "higher_is_better":
        if re is not None:
            worksheet.conditional_format(rng, {"type": "cell", "criteria": "<", "value": re, "format": fmt_cr})
        if re is not None and bu is not None:
            worksheet.conditional_format(
                rng, {"type": "cell", "criteria": "between", "minimum": re, "maximum": bu, "format": fmt_re}
            )
        if bu is not None and ex is not None:
            worksheet.conditional_format(
                rng, {"type": "cell", "criteria": "between", "minimum": bu, "maximum": ex, "format": fmt_bu}
            )
        if ex is not None:
            worksheet.conditional_format(rng, {"type": "cell", "criteria": ">=", "value": ex, "format": fmt_ex})

    # lower_is_better: excelente = bajo, crítico = alto
    else:
        if re is not None:
            worksheet.conditional_format(rng, {"type": "cell", "criteria": ">", "value": re, "format": fmt_cr})
        if bu is not None and re is not None:
            worksheet.conditional_format(
                rng, {"type": "cell", "criteria": "between", "minimum": bu, "maximum": re, "format": fmt_re}
            )
        if ex is not None and bu is not None:
            worksheet.conditional_format(
                rng, {"type": "cell", "criteria": "between", "minimum": ex, "maximum": bu, "format": fmt_bu}
            )
        if ex is not None:
            worksheet.conditional_format(rng, {"type": "cell", "criteria": "<=", "value": ex, "format": fmt_ex})


# ======================================================
# Export callback TopOff
# ======================================================
def export_topoff_callback(app):

    @callback(
        Output("topoff-download-excel", "data"),
        Input("topoff-export-excel", "n_clicks"),

        # filtros base
        State("f-fecha", "date"),
        State("f-technology", "value"),
        State("f-vendor", "value"),

        # filtros mini TopOff
        State("topoff-site-filter", "value"),
        State("topoff-rnc-filter", "value"),
        State("topoff-nodeb-filter", "value"),

        # estado de sort/paginado/mode (los mismos que tabla)
        State("topoff-sort-state", "data"),  # {"column": "...", "ascending": ...}
        State("topoff-page-state", "data"),  # {"page":..., "page_size":...}
        State("topoff-order-mode", "value"),  # "recent" | "alarmado"

        prevent_initial_call=True,
    )
    def do_export_topoff(_, fecha, techs, vendors, sites, rncs, nodebs, sort_state, page_state, order_mode):

        # ---------- Normaliza filtros a listas ----------
        techs = _as_list(techs)
        vendors = _as_list(vendors)
        sites = _as_list(sites)
        rncs = _as_list(rncs)
        nodebs = _as_list(nodebs)

        # ---------- Misma página que está viendo el usuario ----------
        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # ---------- Sort (si el usuario clickeó header) ----------
        sort_by = None
        ascending = True
        if sort_state and sort_state.get("column"):
            sort_by = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))

        # ---------- Modo de orden de TopOff ----------
        order_mode = (order_mode or "recent").lower()
        is_alarmado = (order_mode == "alarmado")

        # ---------- Parámetros comunes para queries ----------
        common_kwargs = dict(
            fecha=fecha,
            technologies=techs or None,
            vendors=vendors or None,
            sites=sites or None,
            rncs=rncs or None,
            nodebs=nodebs or None,
            page=page,
            page_size=page_size,
        )

        # ---------- MISMA lógica que la tabla ----------
        # - "alarmado": orden global por severidad (y opcional por header)
        # - "recent": orden por fecha/hora (y opcional por header si aplica)
        if is_alarmado:
            df_page, _total = fetch_topoff_paginated_severity_global_sort(
                **common_kwargs,
                sort_by_friendly=sort_by,
                ascending=ascending,
            )
        else:
            if sort_by:
                df_page, _total = fetch_topoff_paginated_global_sort(
                    **common_kwargs,
                    sort_by_friendly=sort_by,
                    ascending=ascending,
                )
            else:
                df_page, _total = fetch_topoff_paginated(**common_kwargs)

        # ---------- Sin datos ----------
        if df_page is None or df_page.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # ---------- Preparar DF de salida ----------
        df_out = df_page.copy()

        # Estas columnas son “identificadores” (no KPIs) => NO se convierten a número
        ID_KEYS = {
            "fecha", "hora", "technology", "vendor",
            "region", "province", "municipality",
            "site_att", "rnc", "nodeb", "valores",
        }

        # KPIs = todo lo demás
        KPI_COLS = [c for c in df_out.columns if c not in ID_KEYS]

        # Normaliza KPIs a numérico (lo demás queda texto)
        for colname in KPI_COLS:
            df_out[colname] = pd.to_numeric(df_out[colname], errors="coerce")

        # Limpia NaN/Inf para que Excel no truene
        df_out = df_out.replace([np.inf, -np.inf], pd.NA)
        df_out = df_out.where(pd.notna(df_out), None)

        # ---------- Cargar configuración (perfil: topoff) ----------
        raw_cfg = load_threshold_cfg()
        profiles = raw_cfg.get("profiles") or {}
        profile_cfg = profiles.get("topoff", {})

        cfg = {
            "severity": profile_cfg.get("severity", {}),
            "progress": profile_cfg.get("progress", {}),
            "colors": raw_cfg.get("colors", {}),
        }
        colors = cfg.get("colors") or {}

        # ---------- Crear XLSX en memoria ----------
        bio = BytesIO()
        workbook = Workbook(bio, {"in_memory": True})
        worksheet = workbook.add_worksheet("TopOff")

        # Encabezados
        header_fmt = workbook.add_format({"bold": True})
        for j, name in enumerate(df_out.columns):
            worksheet.write(0, j, name, header_fmt)

        # Datos (escribe números como número para que funcionen los formatos)
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

        # Auto ancho de columnas (muestra de 50)
        for j, col in enumerate(df_out.columns):
            sample = [str(v) for v in df_out[col][:50] if v is not None]
            width = max([len(col)] + [len(s) for s in sample] + [10])
            worksheet.set_column(j, j, min(width, 40))

        # ---------- Data bars para KPIs de progreso ----------
        for j, col in enumerate(df_out.columns):
            net, kpi = _parse_col(col)
            if kpi in (cfg.get("progress") or {}):
                col_letter = xlsx_col_letter(j)
                rng = f"{col_letter}2:{col_letter}{n_rows + 1}"
                maxv = _progress_max_for(kpi, net, cfg)
                worksheet.conditional_format(
                    rng,
                    {
                        "type": "data_bar",
                        "min_type": "num",
                        "min_value": 0,
                        "max_type": "num",
                        "max_value": maxv,
                    },
                )

        # ---------- Bandas de severidad (4 colores) ----------
        # Se aplican SOLO a KPIs (no a keys)
        for j, col in enumerate(df_out.columns):
            net, kpi = _parse_col(col)
            if kpi not in KPI_COLS:
                continue
            orientation, thr = _severity_cfg_for(kpi, net, cfg)
            _add_severity_rules(workbook, worksheet, j, n_rows, orientation, thr, colors)

        workbook.close()
        bio.seek(0)

        # ---------- Nombre del archivo ----------
        fname = f"topoff_{order_mode}_p{page}_sz{page_size}_{fecha or 'fecha'}_fmt.xlsx"

        # Devuelve el archivo a Dash para descarga
        return dcc.send_bytes(bio.read(), filename=fname)
