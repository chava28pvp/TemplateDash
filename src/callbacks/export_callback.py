from dash import Output, Input, State, callback, dcc
import pandas as pd
from src.data_access import (
    fetch_kpis_paginated,
    fetch_kpis_paginated_global_sort,
    fetch_kpis_paginated_alarm_sort,
    COLMAP,
)
from components.Tables.main_table import (
    pivot_by_network,
    expand_groups_for_networks,
    _resolve_sort_col,
    ROW_KEYS,  # típicamente: ["fecha","hora","vendor","noc_cluster","technology"]
)

def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

# ---------- helper para fijar el orden de FILAS como en la tabla ----------
def _apply_table_row_order(df_page: pd.DataFrame,
                           df_wide: pd.DataFrame,
                           key_cols: list[str],
                           metric_order: list[str],
                           safe_sort_state: dict | None):
    """
    Replica el orden visual de la tabla:
    - Primero arma un orden estable (_ord) con base en df_page[key_cols].
    - Si hay sort explícito (global), ordena por [resolved, _ord].
    - Si no, solo por _ord.
    """
    # 1) asegurar keys como columnas (no índice)
    if isinstance(df_wide.index, pd.MultiIndex) or df_wide.index.name is not None:
        df_wide = df_wide.reset_index()

    # (opcional) normaliza posibles nombres de BD que lleguen capitalizados
    rename_fix = {
        "Date": "fecha", "Time": "hora", "Vendor": "vendor",
        "Noc_Cluster": "noc_cluster", "Technology": "technology",
    }
    cols_to_fix = {k: v for k, v in rename_fix.items() if k in df_wide.columns}
    if cols_to_fix:
        df_wide = df_wide.rename(columns=cols_to_fix)

    # 2) mapa de orden según la página (lo que el usuario ve)
    tuples_in_order = list(dict.fromkeys(
        map(tuple, df_page[key_cols].itertuples(index=False, name=None))
    ))
    order_map = {t: i for i, t in enumerate(tuples_in_order)}

    # 3) columna ordinal oculta (para mantener orden visual)
    df_wide["_ord"] = df_wide[key_cols].apply(
        lambda r: order_map.get(tuple(r.values.tolist()), 10**9), axis=1
    )

    # 4) aplicar sort explícito (si corresponde)
    if safe_sort_state and safe_sort_state.get("column"):
        col_req = safe_sort_state["column"]
        asc = bool(safe_sort_state.get("ascending", True))
        resolved = _resolve_sort_col(df_wide, metric_order, col_req)
        if resolved in df_wide.columns:
            df_wide = df_wide.sort_values(by=[resolved, "_ord"],
                                          ascending=[asc, True],
                                          na_position="last")
        else:
            # si no existe la col (p.ej. red distinta), cae al orden visual
            df_wide = df_wide.sort_values("_ord")
    else:
        df_wide = df_wide.sort_values("_ord")

    return df_wide.drop(columns=["_ord"])


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

        networks = _as_list(networks)
        techs    = _as_list(techs)
        vendors  = _as_list(vendors)
        clusters = _as_list(clusters)

        page      = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # --- sort explícito (solo en "global") ---
        sort_by = None
        sort_net = None
        ascending = True
        if sort_mode != "alarmado" and sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))
            if "__" in col:
                sort_net, base = col.split("__", 1)
                sort_by = base
            else:
                sort_by = col

        # --- misma fuente de datos que la tabla (SOLO la página actual) ---
        if sort_mode == "alarmado":
            df_page, _ = fetch_kpis_paginated_alarm_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=techs or None,
                page=page, page_size=page_size,
            )
            safe_sort_state = None
        else:
            if sort_by and sort_by in COLMAP:
                df_page, _ = fetch_kpis_paginated_global_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=techs or None,
                    page=page, page_size=page_size,
                    sort_by_friendly=sort_by,
                    sort_net=sort_net,
                    ascending=ascending,
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

        # --- pivot a formato wide igual que la tabla ---
        is_long = "network" in df_page.columns
        if is_long:
            nets_for_pivot = networks or sorted(df_page["network"].dropna().unique().tolist())
            df_wide = pivot_by_network(df_page, networks=nets_for_pivot)
        else:
            df_wide = df_page.copy()

        if df_wide is None or df_wide.empty:
            return dcc.send_string("Sin datos para exportar.", filename="vacio.txt")

        # --- orden de columnas igual que la tabla ---
        _, metric_order, _ = expand_groups_for_networks(
            networks or (sorted(df_page["network"].dropna().unique()) if is_long else [])
        )
        visible_order = ROW_KEYS + metric_order
        cols_final = [c for c in visible_order if c in df_wide.columns]

        # --- **ORDEN DE FILAS** idéntico a la tabla ---
        df_wide = _apply_table_row_order(
            df_page=df_page,
            df_wide=df_wide,
            key_cols=list(ROW_KEYS),
            metric_order=metric_order,
            safe_sort_state=safe_sort_state if sort_mode != "alarmado" else None,
        )

        df_out = df_wide[cols_final].copy()

        # --- descarga (sin estilos) ---
        filename = f"kpis_p{page}_sz{page_size}_{fecha or 'fecha'}_{(hora or 'Todas')[:5]}.xlsx"
        return dcc.send_data_frame(df_out.to_excel, filename, index=False)
