from dash import html, dcc
import dash_bootstrap_components as dbc
from src.Utils.utils_time import default_date_str, default_hour_str

def build_filters(
    network_options=None,
    technology_options=None,
    vendor_options=None,
    cluster_options=None
):
    """
    Panel de filtros para: Network, Technology, Vendor, Noc_Cluster, Date, Time.
    *_options: listas de strings (puedes pasarlas desde callbacks o al inicializar).
    """
    network_options   = network_options   or ["ATT", "TEF", "NET"]  # <-- default útil
    technology_options= technology_options or []
    vendor_options    = vendor_options     or []
    cluster_options   = cluster_options    or []

    hours = [f"{h:02d}:00:00" for h in range(24)]

    return dbc.Card(
        dbc.CardBody([
            html.H4("Filtros", className="mb-3"),

            # Fila 1: Fecha / Hora / Network / Technology
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fecha"),
                    dcc.DatePickerSingle(
                        id="f-fecha",
                        date=default_date_str(),
                        display_format="YYYY-MM-DD",
                        clearable=False,
                        persistence=True, persistence_type="session"
                    )
                ], md=3),

                dbc.Col([
                    dbc.Label("Hora"),
                    dcc.Dropdown(
                        id="f-hora",
                        options=[{"label": "Todas", "value": "Todas"}]
                                + [{"label": h[:5], "value": h} for h in hours],
                        value=default_hour_str(),   # o "Todas" si quieres todo por defecto
                        clearable=False,
                        persistence=True, persistence_type="session"
                    )
                ], md=3),

                # >>> NUEVO/CLAVE: Network multi-selección (ID consistente con tus callbacks)
                dbc.Col([
                    dbc.Label("Network"),
                    dcc.Dropdown(
                        id="f-network",
                        options=[{"label": n, "value": n} for n in network_options],
                        value=network_options,          # << todas por defecto
                        multi=True,
                        clearable=False,
                        placeholder="Selecciona network",
                        persistence=True, persistence_type="session"
                    )
                ], md=3),

                dbc.Col([
                    dbc.Label("Technology"),
                    dcc.Dropdown(
                        id="f-technology",
                        options=[{"label": t, "value": t} for t in technology_options],
                        value=technology_options,       # << todas por defecto si vienen
                        multi=True,
                        placeholder="Selecciona technology",
                        persistence=True, persistence_type="session"
                    )
                ], md=3),
            ], className="g-3"),

            # Fila 2: Vendor / Cluster
            dbc.Row([
                dbc.Col([
                    dbc.Label("Vendor"),
                    dcc.Dropdown(
                        id="f-vendor",
                        options=[{"label": v, "value": v} for v in vendor_options],
                        value=vendor_options,            # << todas por defecto si vienen
                        multi=True,
                        placeholder="Selecciona vendor",
                        persistence=True, persistence_type="session"
                    )
                ], md=6),

                dbc.Col([
                    dbc.Label("Cluster"),
                    dcc.Dropdown(
                        id="f-cluster",
                        options=[{"label": c, "value": c} for c in cluster_options],
                        value=cluster_options,           # << todas por defecto si vienen
                        multi=True,
                        placeholder="Selecciona cluster",
                        persistence=True, persistence_type="session"
                    )
                ], md=6),
            ], className="g-3"),

            # (Opcional) Intervalo de refresco local
            # dcc.Interval(id="refresh-interval"),
            # Fila 3: Modo de orden
            dbc.Row([
                dbc.Col([
                    dbc.Label("Orden"),
                    dcc.RadioItems(
                        id="f-sort-mode",
                        options=[
                            {"label": "Alarmado", "value": "alarmado"},
                            {"label": "Global", "value": "global"},
                        ],
                        value="alarmado",  # por defecto usa el orden de alarmas
                        inline=True,
                        persistence=True, persistence_type="session"
                    ),
                ], md=12),
            ], className="g-3"),

        ]),
        className="shadow-sm"
    )
