# components/simple_tables.py
from dash import html
import dash_bootstrap_components as dbc
import math

def _fmt(v):
    if v is None: return "-"
    if isinstance(v, (int,)) and not isinstance(v, bool): return f"{v:,}"
    if isinstance(v, float):
        # 3 decimales, separador de miles
        if math.isnan(v): return "-"
        return f"{v:,.3f}"
    return str(v)

def render_simple_table(df, title, columns):
    """
    columns: lista de tuplas (label_visible, key_df)
    """
    if df is None or df.empty:
        return dbc.Card(dbc.CardBody([html.H4(title, className="mb-3"),
                                      dbc.Alert("Sin datos.", color="secondary")]),
                        className="shadow-sm")
    header = html.Thead(html.Tr([html.Th(lbl) for (lbl, _) in columns]))
    body_rows = []
    for _, row in df.iterrows():
        tds = []
        for _, key in columns:
            val = row.get(key, None)
            # alinea números a la derecha
            cls = "text-end" if isinstance(val, (int, float)) else ""
            tds.append(html.Td(_fmt(val), className=cls))
        body_rows.append(html.Tr(tds))
    body = html.Tbody(body_rows)
    table = dbc.Table([header, body], bordered=True, hover=True, striped=False,
                      size="sm", className="mini-table")
    return dbc.Card(dbc.CardBody([html.H4(title, className="mb-3"), table]), className="shadow-sm")
