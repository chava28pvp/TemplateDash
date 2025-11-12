# callbacks/topoff_callbacks.py
import math
from dash import Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from src.dataAccess.data_acess_topoff import fetch_topoff_paginated
from components.Tables.topoff import render_topoff_table

DEFAULT_PAGE_SIZE = 50

def register_topoff_callbacks(app):
    # 1) Reset page si cambia page-size
    @app.callback(
        Output("topoff-page-state", "data"),
        Input("topoff-page-size", "value"),
        prevent_initial_call=True,
    )
    def reset_page_on_pagesize(ps):
        ps = max(1, int(ps or DEFAULT_PAGE_SIZE))
        return {"page": 1, "page_size": ps}

    # 2) Prev/Next
    @app.callback(
        Output("topoff-page-state", "data", allow_duplicate=True),
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

    # 3) Render tabla + indicadores
    @app.callback(
        Output("topoff-table-container", "children"),
        Output("topoff-page-indicator", "children"),
        Output("topoff-total-rows-banner", "children"),
        Input("topoff-page-state", "data"),
        prevent_initial_call=False,
    )
    def refresh_table(page_state):
        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", DEFAULT_PAGE_SIZE))

        # Sin filtros por ahora: solo pagina
        df, total = fetch_topoff_paginated(page=page, page_size=page_size)

        if df is None or df.empty:
            empty_alert = dbc.Alert("Sin datos para mostrar.", color="warning")
            return empty_alert, "Página 1 de 1", "Sin resultados."

        table = render_topoff_table(df)

        total_pages = max(1, math.ceil((total or 0) / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)

        indicator = f"Página {page_corrected} de {total_pages}"
        banner = "Sin resultados." if (total or 0) == 0 else \
            f"Mostrando {(page_corrected - 1) * page_size + 1}–{min(page_corrected * page_size, total)} de {total} registros"

        return table, indicator, banner
