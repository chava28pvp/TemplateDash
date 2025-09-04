# components/Grids/main_grid.py
from dash import html
import dash_ag_grid as dag
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from components.Tables.main_table import (
    BASE_GROUPS, ROW_KEYS, prefixed_progress_cols, prefixed_severity_cols,
    pivot_by_network
)

# 🔧 Usa tus utils reales
from src.Utils.umbrales.utils_umbrales import cell_severity, progress_cfg

# Diccionario de nombres para mostrar en las columnas
DISPLAY_NAME_BASE = {
    "fecha": "Fecha", "hora": "Hora", "vendor": "Vendor",
    "technology": "Tech", "noc_cluster": "Cluster", "integrity": "Integrity",
    "ps_traff_delta": "DELTA", "ps_traff_gb": "GB",
    "ps_rrc_ia_percent": "%IA", "ps_rrc_fail": "FAIL",
    "ps_rab_ia_percent": "%IA", "ps_rab_fail": "FAIL",
    "ps_s1_ia_percent": "%IA", "ps_s1_fail": "FAIL",
    "ps_drop_dc_percent": "%DC", "ps_drop_abnrel": "ABNREL",
    "cs_traff_delta": "DELTA", "cs_traff_erl": "ERL",
    "cs_rrc_ia_percent": "%IA", "cs_rrc_fail": "FAIL",
    "cs_rab_ia_percent": "%IA", "cs_rab_fail": "FAIL",
    "cs_drop_dc_percent": "%DC", "cs_drop_abnrel": "ABNREL",
}


def _fmt_header(base: str) -> str:
    return DISPLAY_NAME_BASE.get(base, base).upper()


def _mk_left_key_cols() -> List[Dict[str, Any]]:
    defs = []
    for k in ROW_KEYS:
        defs.append({
            "headerName": _fmt_header(k),
            "field": k,
            "pinned": "left",
            "sortable": True,
            "filter": "agTextColumnFilter",
            "minWidth": 110 if k != "vendor" else 140,
            "suppressMenu": False,
            "cellClass": "td-key",
            "headerClass": "center-header",
            "valueFormatter": {
                "function": """
                    function(params){
                      const v = params.value;
                      if(v===null||v===undefined||v==='') return '';
                      if(typeof v==='number' && !isFinite(v)) return '';
                      const num = Number(v);
                      if(Number.isNaN(num)) return v;
                      if(Number.isInteger(num)) return num.toLocaleString();
                      return num.toLocaleString(undefined,{minimumFractionDigits:1, maximumFractionDigits:1});
                    }
                """
            },
        })
    return defs


def _mk_metric_col_def(field: str, base_name: str, *, is_progress: bool,
                       is_severity: bool, network: Optional[str]) -> Dict[str, Any]:
    """
    Crea la definición de una columna de métrica.

    field: 'ATT__ps_rrc_fail'
    base_name: 'ps_rrc_fail'
    """
    col = {
        "headerName": _fmt_header(base_name),
        "field": field,
        "type": "numericColumn",
        "sortable": True,
        "filter": "agNumberColumnFilter",
        "suppressMenu": False,
        "minWidth": 110,
        "comparator": {
            "function": """
                function(valueA, valueB) {
                    if ((valueA === null || valueA === undefined || valueA === '') &&
                        (valueB === null || valueB === undefined || valueB === '')) {
                        return 0;
                    }
                    if (valueA === null || valueA === undefined || valueA === '') {
                        return 1;
                    }
                    if (valueB === null || valueB === undefined || valueB === '') {
                        return -1;
                    }
                    return Number(valueA) - Number(valueB);
                }
            """
        },
        "valueFormatter": {
            "function": """
                function(params){
                  const v = params.value;
                  if(v===null||v===undefined||v==='') return '';
                  if(typeof v==='number' && !isFinite(v)) return '';
                  const num = Number(v);
                  if(Number.isNaN(num)) return v;
                  if(Number.isInteger(num)) return num.toLocaleString();
                  return num.toLocaleString(undefined,{minimumFractionDigits:1, maximumFractionDigits:1});
                }
            """
        },
    }

    # 🔵 Barra de progreso (usa tus min/max/decimals/label)
    if is_progress:
        cfg = progress_cfg(base_name, network=network)  # {'min','max','decimals','label',...}
        col["cellRenderer"] = {
            "function": f"""
                function(params){{
                  const v = Number(params.value);
                  if(!isFinite(v)) return '';
                  const min = {float(cfg.get("min", 0.0))};
                  const max = {float(cfg.get("max", 100.0))};
                  const widthPx = {int(cfg.get("width_px", 120))};
                  const decimals = {int(cfg.get("decimals", 1))};
                  const pct = Math.max(0, Math.min(100, ((v-min)/(max-min))*100));
                  const labelTpl = "{cfg.get("label", "{value:.1f}")}";
                  const label = labelTpl.replace('{{value:.1f}}', v.toFixed(decimals));
                  return `<div class="kb" style="--kb-width:${{widthPx}}px"><div class="kb__fill" style="width:${{pct}}%">${{label}}</div></div>`;
                }}
            """
        }
        # Tooltip opcional con el % normalizado:
        col["tooltipValueGetter"] = {
            "function": f"""
                function(params){{
                  const v = Number(params.value);
                  if(!isFinite(v)) return '';
                  const min = {float(cfg.get("min", 0.0))};
                  const max = {float(cfg.get("max", 100.0))};
                  const pct = Math.max(0, Math.min(100, ((v-min)/(max-min))*100));
                  return Math.round(pct) + '%';
                }}
            """
        }

    # 🟢 Umbrales de severidad (usa columna derivada __sev calculada en Python)
    if is_severity:
        sev_field = f"{field}__sev"  # p.ej., 'ATT__ps_rrc_ia_percent__sev'
        col["cellClassRules"] = {
            "sev-excelente": {"condition": f"(params.data && params.data['{sev_field}'] === 'excelente')"},
            "sev-bueno":     {"condition": f"(params.data && params.data['{sev_field}'] === 'bueno')"},
            "sev-regular":   {"condition": f"(params.data && params.data['{sev_field}'] === 'regular')"},
            "sev-critico":   {"condition": f"(params.data && params.data['{sev_field}'] === 'critico')"},
        }
        col["tooltipValueGetter"] = {"function": f"params => params.data?.['{sev_field}'] ?? ''"}

    return col


def _mk_network_groups(networks: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Crea definiciones de columnas con jerarquía:
    [Keys (pinned)] + [Network -> BaseGroup -> Columns]
    """
    left = _mk_left_key_cols()
    metric_order = []
    groups = []

    progress_set = prefixed_progress_cols(networks)
    severity_set = prefixed_severity_cols(networks)

    for net in networks:
        group_children = []
        for grp_title, base_cols in BASE_GROUPS:
            cols_in_group = []
            for base in base_cols:
                field = f"{net}__{base}"
                metric_order.append(field)
                cols_in_group.append(_mk_metric_col_def(
                    field, base,
                    is_progress=(field in progress_set),
                    is_severity=(field in severity_set),
                    network=net
                ))
            group_children.append({
                "headerName": grp_title,
                "marryChildren": True,
                "children": cols_in_group
            })
        groups.append({
            "headerName": net,
            "marryChildren": True,
            "children": group_children
        })

    column_defs = left + groups
    return column_defs, metric_order


def _enrich_severity_labels(df_wide: pd.DataFrame, networks: List[str]) -> pd.DataFrame:
    """
    Agrega columnas '__sev' por cada métrica de severidad usando tu util cell_severity
    con override por network. Trabaja sobre DF wide (columnas tipo 'NET__metric').
    """
    df = df_wide.copy()
    sev_fields = prefixed_severity_cols(networks)  # ej: {'ATT__ps_rrc_ia_percent', ...}
    for field in sev_fields:
        try:
            net, base = field.split("__", 1)
        except ValueError:
            # si no cumple el patrón, salta
            continue
        sev_col = f"{field}__sev"
        if field in df.columns:
            # vectorizado ligero
            df[sev_col] = df[field].apply(lambda v: cell_severity(base, v, network=net))
        else:
            # si la métrica no existe en DF, al menos crea la columna sev vacía
            df[sev_col] = ""
    return df


def build_grid(df_in: pd.DataFrame, networks: Optional[List[str]] = None) -> html.Div:
    """
    Construye la grid a partir de un DataFrame.

    Args:
        df_in: DataFrame con los datos (formato long o wide)
        networks: Lista de redes a incluir. Si es None, se derivan del DataFrame

    Returns:
        Componente AgGrid o mensaje de error
    """
    if df_in is None or df_in.empty:
        return html.Div("Sin datos para los filtros seleccionados.",
                        className="alert alert-warning my-3")

    is_long = "network" in df_in.columns

    # Derivar redes del DataFrame si no se proporcionan
    if not networks:
        if is_long:
            networks = sorted(df_in["network"].dropna().unique().tolist())
        else:
            nets = set()
            for c in df_in.columns:
                if "__" in c:
                    nets.add(c.split("__", 1)[0])
            networks = sorted(nets)

    # Convertir a formato wide si es necesario
    df_wide = pivot_by_network(df_in, networks=networks) if is_long else df_in.copy()
    if df_wide is None or df_wide.empty:
        return html.Div("Sin datos para las redes seleccionadas.",
                        className="alert alert-warning my-3")

    # ➕ Calcula etiquetas de severidad en Python (respeta overrides por network)
    df_wide = _enrich_severity_labels(df_wide, networks)

    column_defs, metric_order = _mk_network_groups(networks)
    row_data = df_wide.to_dict("records")

    grid = dag.AgGrid(
        id="kpi-grid",
        className="ag-theme-quartz kpi-grid",
        columnDefs=column_defs,
        rowData=row_data,
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
            "wrapText": False,
            "suppressHeaderMenuButton": False,
            "valueFormatter": {
                "function": """
                    function(params){
                      const v = params.value;
                      if(v===null||v===undefined||v==='') return '';
                      if(typeof v==='number' && !isFinite(v)) return '';
                      const num = Number(v);
                      if(Number.isNaN(num)) return v;
                      if(Number.isInteger(num)) return num.toLocaleString();
                      return num.toLocaleString(undefined,{minimumFractionDigits:1, maximumFractionDigits:1});
                    }
                """
            },
        },
        columnSize="responsiveSizeToFit",
        dashGridOptions={
            "ensureDomOrder": True,
            "suppressColumnMoveAnimation": True,
            "animateRows": False,
            "rowSelection": "single",
            "domLayout": "autoHeight",
            "pagination": True,
            "paginationPageSize": 200,
            "rowBuffer": 40,
        },
        dangerously_allow_code=True,
    )
    return grid
