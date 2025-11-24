# callbacks/topoff_callbacks.py
import math
from dash import Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from src.dataAccess.data_acess_topoff import (
    fetch_topoff_paginated,
    fetch_topoff_paginated_global_sort,
    fetch_topoff_paginated_severity_global_sort,
    fetch_topoff_distinct_options,
)
from components.topoff.topoff import render_topoff_table

DEFAULT_PAGE_SIZE = 50
DEFAULT_SORT_STATE = {"column": None, "ascending": True}


def register_topoff_callbacks(app):

    # -----------------------------------------------------
    # Toggle filtros mini TopOff (Collapse)
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-filters-collapse", "is_open"),
        Input("topoff-filters-toggle", "n_clicks"),
        State("topoff-filters-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_topoff_filters(n, is_open):
        return (not is_open) if n else is_open

    # -----------------------------------------------------
    # Cargar opciones distintas para Site/RNC/NodeB
    # en base a filtros superiores (fecha/tech/vendor).
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-site-filter", "options"),
        Output("topoff-rnc-filter", "options"),
        Output("topoff-nodeb-filter", "options"),
        Input("f-fecha", "date"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        prevent_initial_call=False,
    )
    def load_topoff_options(fecha, technologies, vendors):
        sites, rncs, nodebs = fetch_topoff_distinct_options(
            fecha=fecha,
            technologies=technologies,
            vendors=vendors,
        )
        return (
            [{"label": s, "value": s} for s in sites],
            [{"label": r, "value": r} for r in rncs],
            [{"label": n, "value": n} for n in nodebs],
        )

    # -----------------------------------------------------
    # Paginación Prev/Next
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-page-state", "data"),
        Input("topoff-page-prev", "n_clicks"),
        Input("topoff-page-next", "n_clicks"),
        State("topoff-page-state", "data"),
        prevent_initial_call=True,
    )
    def paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": DEFAULT_PAGE_SIZE}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", DEFAULT_PAGE_SIZE))
        trig = ctx.triggered_id
        if trig == "topoff-page-prev":
            page = max(1, page - 1)
        elif trig == "topoff-page-next":
            page = page + 1
        return {"page": page, "page_size": ps}

    # -----------------------------------------------------
    # Sort header (click inmediato por timestamp)
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-sort-state", "data"),
        Output("topoff-sort-last-ts", "data"),
        Input({"type": "topoff-sort-btn", "col": ALL}, "n_clicks_timestamp"),
        State({"type": "topoff-sort-btn", "col": ALL}, "id"),
        State("topoff-sort-state", "data"),
        State("topoff-sort-last-ts", "data"),
        prevent_initial_call=True,
    )
    def on_click_sort(ts_list, ids_list, sort_state, last_ts):
        sort_state = sort_state or {"column": None, "ascending": True}
        last_ts = last_ts or 0

        safe_ts = [(t or 0) for t in (ts_list or [])]
        best_ts = max(safe_ts, default=0)

        if best_ts <= last_ts:
            raise PreventUpdate

        idx = safe_ts.index(best_ts)
        clicked = ids_list[idx]["col"]

        if sort_state.get("column") == clicked:
            sort_state["ascending"] = not sort_state.get("ascending", True)
        else:
            sort_state["column"] = clicked
            sort_state["ascending"] = True

        return sort_state, best_ts

    # -----------------------------------------------------
    # Reset a page 1 cuando cambie:
    # - page_size
    # - orden (recent/alarmado)
    # - filtros mini (site/rnc/nodeb)
    # - sort manual
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-page-state", "data", allow_duplicate=True),
        Input("topoff-page-size", "value"),
        Input("topoff-order-mode", "value"),
        Input("topoff-site-filter", "value"),
        Input("topoff-rnc-filter", "value"),
        Input("topoff-nodeb-filter", "value"),
        Input("topoff-sort-state", "data"),
        State("topoff-page-state", "data"),
        prevent_initial_call=True,
    )
    def reset_page_on_any_change(ps, _mode, _s, _r, _n, _sort, page_state):
        ps = max(1, int(ps or DEFAULT_PAGE_SIZE))
        current_page = int((page_state or {}).get("page", 1))
        current_ps = int((page_state or {}).get("page_size", DEFAULT_PAGE_SIZE))

        if current_page == 1 and current_ps == ps:
            raise PreventUpdate

        return {"page": 1, "page_size": ps}

    # -----------------------------------------------------
    # Render tabla + indicadores
    # -----------------------------------------------------
    @app.callback(
        Output("topoff-table-container", "children"),
        Output("topoff-page-indicator", "children"),
        Output("topoff-total-rows-banner", "children"),
        Input("topoff-page-state", "data"),
        Input("topoff-sort-state", "data"),
        Input("f-fecha", "date"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("topoff-site-filter", "value"),
        Input("topoff-rnc-filter", "value"),
        Input("topoff-nodeb-filter", "value"),
        Input("topoff-order-mode", "value"),
        prevent_initial_call=False,
    )
    def refresh_table(
        page_state, sort_state,
        fecha, technologies, vendors,
        sites, rncs, nodebs,
        sort_mode
    ):
        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", DEFAULT_PAGE_SIZE))

        # sort manual por header
        sort_by = None
        ascending = True
        if sort_state and sort_state.get("column"):
            sort_by = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))

        sort_mode = (sort_mode or "recent").lower()
        order_mode = "alarmado" if sort_mode == "alarmado" else "recent"

        common_kwargs = dict(
            fecha=fecha,
            technologies=technologies,
            vendors=vendors,
            sites=sites,
            rncs=rncs,
            nodebs=nodebs,
            page=page,
            page_size=page_size,
        )

        if order_mode == "alarmado":
            df, total = fetch_topoff_paginated_severity_global_sort(
                **common_kwargs,
                sort_by_friendly=sort_by,
                ascending=ascending,
            )
        else:
            if sort_by:
                df, total = fetch_topoff_paginated_global_sort(
                    **common_kwargs,
                    sort_by_friendly=sort_by,
                    ascending=ascending,
                )
            else:
                df, total = fetch_topoff_paginated(**common_kwargs)

        if df is None or df.empty:
            empty = dbc.Alert("Sin datos para mostrar.", color="warning")
            return empty, "Página 1 de 1", "Sin resultados."

        table = render_topoff_table(df, sort_state=sort_state)

        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)
        indicator = f"Página {page_corrected} de {total_pages}"

        banner = "Sin resultados." if (total or 0) == 0 else (
            f"Mostrando {(page_corrected - 1) * page_size + 1}–"
            f"{min(page_corrected * page_size, total)} de {total} registros"
        )

        return table, indicator, banner

