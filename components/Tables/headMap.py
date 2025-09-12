# === components/Tables/heatmap_valores.py ===
from dash import html, dcc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Configuración de KPIs por "valores"
# -----------------------------
VALORES_MAP = {
    "PS_RCC":  ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RCC":  ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB":  ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB":  ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
}

ROW_KEYS = ["technology", "vendor", "valores", "noc_cluster"]


# -----------------------------
# Helpers
# -----------------------------

def _normalize_ts_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Normaliza df para que matchee las llaves y valores del heatmap:
      - fecha -> 'YYYY-MM-DD' (str)
      - hora  -> 'HH:MM:SS' (str)
      - keys  -> str
      - value_col -> numérico (coerce)
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    if "fecha" in out.columns:
        out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "hora" in out.columns:
        # mantiene 'HH:MM:SS'; si viene número, lo formatea
        def _fmt_h(x):
            if x is None:
                return None
            if hasattr(x, "hour"):
                return f"{int(x.hour):02d}:00:00"
            s = str(x)
            if ":" in s:
                # asume 'HH:MM[:SS]'
                try:
                    h = int(s.split(":")[0])
                    return f"{h:02d}:00:00"
                except Exception:
                    return None
            try:
                h = int(float(s))
                return f"{h:02d}:00:00"
            except Exception:
                return None
        out["hora"] = out["hora"].map(_fmt_h)

    for c in ("technology", "vendor", "noc_cluster", "valores", "network"):
        if c in out.columns:
            out[c] = out[c].astype(str)

    if value_col in out.columns:
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    return out


def _safe_hour_to_idx(v):
    if v is None:
        return None
    # pandas Timestamp / datetime
    if hasattr(v, "hour"):
        h = int(v.hour)
        return h if 0 <= h <= 23 else None
    s = str(v)
    try:
        # 'HH:MM:SS' o 'H' (numérico)
        if ":" in s:
            h = int(s.split(":")[0])
        else:
            h = int(float(s))
        return h if 0 <= h <= 23 else None
    except Exception:
        return None



def _infer_networks(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "network" not in df.columns:
        return []
    return sorted(df["network"].dropna().unique().tolist())


def _row_label(r: pd.Series) -> str:
    """Etiqueta compacta para el eje Y (fila)."""
    vend = "" if pd.isna(r["vendor"]) else str(r["vendor"])
    return f"{r['technology']} | {vend[:6]} | {r['valores']} | {r['noc_cluster']}"


def _build_matrix(df_ts: pd.DataFrame, rows_df: pd.DataFrame, metric: str, network: str, today_str: str, yday_str: str):
    """
    Construye la matriz Z (n_rows x 49) para un 'network' dado:
      columnas: A00..A23 | gap(NaN) | H00..H23  => 24 + 1 + 24 = 49
      filas: rows_df (cada fila es technology/vendor/valores/cluster)
    df_ts debe tener columnas: ['fecha','hora','technology','vendor','noc_cluster','network', metric]
    """
    n_rows = len(rows_df)
    xlabels = [f"A{h:02d}" for h in range(24)] + [""] + [f"H{h:02d}" for h in range(24)]
    ylabels = rows_df.apply(_row_label, axis=1).tolist()

    if n_rows == 0:
        # Evita heatmap vacío (causa error en Plotly)
        return np.full((0, 49), np.nan), ylabels, xlabels

    if df_ts is None or df_ts.empty or metric not in df_ts.columns:
        Z = np.full((n_rows, 49), np.nan)
        return Z, ylabels, xlabels

    # Filtra por network si existe
    base = df_ts[df_ts["network"] == network] if "network" in df_ts.columns else df_ts

    # Pre-indexa por claves para acceso rápido: (fecha, tech, vend, clus, valores) -> array de 24
    cols_key = ["fecha", "technology", "vendor", "noc_cluster", "valores"]
    idx = {}
    for (fecha, tech, vend, clus, val), g in base.groupby(cols_key, sort=False):
        arr = [np.nan] * 24
        if metric in g.columns:
            for _, row in g.iterrows():
                hh = _safe_hour_to_idx(row.get("hora"))
                if hh is None:
                    continue
                valnum = row.get(metric, np.nan)
                try:
                    arr[hh] = float(valnum) if valnum is not None else np.nan
                except Exception:
                    arr[hh] = np.nan
        idx[(fecha, tech, vend, clus, val)] = arr

    # Construye Z concatenando ayer + gap + hoy por cada fila
    Z_list = []
    for _, rr in rows_df.iterrows():
        key_today = (today_str, rr["technology"], rr["vendor"], rr["noc_cluster"], rr["valores"])
        key_yday  = (yday_str,  rr["technology"], rr["vendor"], rr["noc_cluster"], rr["valores"])
        arr_today = idx.get(key_today, [np.nan] * 24)
        arr_yday  = idx.get(key_yday,  [np.nan] * 24)
        row_vals  = arr_yday + [np.nan] + arr_today
        Z_list.append(row_vals)

    Z = np.array(Z_list, dtype=float)
    return Z, ylabels, xlabels


def _fmt_val(v, is_percent: bool):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}%" if is_percent else f"{v:.0f}"


def _hover_text(Z, xlabels, ylabels, is_percent: bool):
    text = []
    for i, row in enumerate(Z):
        r = []
        for j, v in enumerate(row):
            r.append(f"Fila: {ylabels[i]}<br>Col: {xlabels[j]}<br>Valor: {_fmt_val(v, is_percent)}")
        text.append(r)
    return text


def _heat_trace(Z, xlabels, ylabels, zmin, zmax, colorscale, cbar_title, use_gl=True, is_percent=False):
    """
    Crea la traza de heatmap:
      - use_gl=True  -> Heatmapgl + hover con text/hoverinfo (rápido)
      - use_gl=False -> Heatmap   + hovertemplate (personalizable)
    """
    if use_gl:
        # HeatmapGL no acepta hovertemplate → usamos text + hoverinfo
        return go.Heatmapgl(
            z=Z, x=xlabels, y=ylabels,
            zmin=zmin, zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=cbar_title),
            text=_hover_text(Z, xlabels, ylabels, is_percent=is_percent),
            hoverinfo="text",
        )
    else:
        return go.Heatmap(
            z=Z, x=xlabels, y=ylabels,
            zmin=zmin, zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=cbar_title),
            hovertemplate="Fila: %{y}<br>Col: %{x}<br>Valor: %{z}<extra></extra>",
        )


# -----------------------------
# Render principal
# -----------------------------
def render_valores_heatmaps(
    df_meta: pd.DataFrame,         # snapshot paginado (filas visibles)
    df_ts: pd.DataFrame,           # timeseries HOY+AYER (24h completas, sin filtrar por 'hora')
    *,
    networks=None,
    title="KPIs por 'valores' — Heatmap 24h (Ayer | Hoy)",
    valores_order=("PS_RCC","CS_RCC","PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    show_percent=True,
    show_unit=False,
    pct_range=(80, 100),           # rango fijo para %
    unit_quantile_range=(0.1, 0.95), # auto-escala robusta para UNIT
    use_gl=True                    # True=Heatmapgl, False=Heatmap
):
    """
    Devuelve un Div con un dcc.Graph que contiene subplots:
      - Por cada network, hasta dos columnas: [%] y/o [UNIT]
      - Filas = (technology, vendor, valores, cluster) del df_meta paginado
      - Columnas = A00..A23 (ayer) | gap | H00..H23 (hoy)
    """
    # Validaciones mínimas
    if df_meta is None or df_meta.empty:
        return dcc.Markdown("**Sin datos para los filtros seleccionados.**")
    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return dcc.Markdown("**Sin redes para construir el heatmap.**")
    if not (show_percent or show_unit):
        return dcc.Markdown("**Selecciona al menos un tipo de métrica (%, UNIT).**")

    # Prepara filas (expande meta por 'valores')
    base_cols = ["technology", "vendor", "noc_cluster"]
    meta_base = df_meta.drop_duplicates(subset=base_cols)[base_cols].reset_index(drop=True)

    rows = []
    for _, r in meta_base.iterrows():
        for v in valores_order:
            rows.append({
                "technology": r["technology"],
                "vendor": r["vendor"],
                "noc_cluster": r["noc_cluster"],
                "valores": v
            })
    rows_df = pd.DataFrame(rows, columns=["technology","vendor","noc_cluster","valores"])

    # 🚧 Guard: si no hay filas visibles, evita traza vacía
    if rows_df.empty:
        return dcc.Markdown("**No hay filas para mostrar en el heatmap con los filtros actuales.**")

    # Fechas hoy/ayer (intenta inferir de df_ts)
    try:
        today_str = pd.to_datetime(df_ts["fecha"]).max().date().strftime("%Y-%m-%d")
    except Exception:
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
    yday_str = (datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Define subplots
    cols_per_network = int(show_percent) + int(show_unit)
    total_cols = max(1, cols_per_network) * len(networks)
    subplot_titles = []
    for net in networks:
        if show_percent: subplot_titles.append(f"{net} – %")
        if show_unit:    subplot_titles.append(f"{net} – UNIT")

    fig = make_subplots(
        rows=1, cols=total_cols,
        shared_yaxes=True,
        horizontal_spacing=0.06 if total_cols > 1 else 0.02,
        subplot_titles=subplot_titles
    )

    current_col = 1
    # --- % ---
    if show_percent:
        zmin_pct, zmax_pct = pct_range

        for net in networks:
            # Unifica métricas % bajo 'value' según 'valores'
            frames = []
            if df_ts is not None and not df_ts.empty:
                for v in valores_order:
                    metric_pct, _ = VALORES_MAP.get(v, (None, None))
                    if metric_pct and (metric_pct in df_ts.columns):
                        cols = [c for c in ["fecha","hora","technology","vendor","noc_cluster","network", metric_pct] if c in df_ts.columns]
                        tmp = df_ts[cols].copy()
                        tmp["valores"] = v
                        tmp.rename(columns={metric_pct: "value"}, inplace=True)
                        frames.append(tmp)
            df_ts_pct = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            df_ts_pct = _normalize_ts_frame(df_ts_pct, value_col="value")

            Z, ylabels, xlabels = _build_matrix(
                df_ts=df_ts_pct.rename(columns={"value": "__value__"}),
                rows_df=rows_df,
                metric="__value__",
                network=net,
                today_str=today_str,
                yday_str=yday_str,
            )

            # 🚧 Guard: si no hay valores numéricos, evita GL vacío
            if not np.isfinite(Z).any():
                return dcc.Markdown("**No hay datos numéricos para el heatmap de % con los filtros actuales.**")

            heat = _heat_trace(
                Z=Z, xlabels=xlabels, ylabels=ylabels,
                zmin=zmin_pct, zmax=zmax_pct,
                colorscale="Viridis", cbar_title="%",
                use_gl=use_gl, is_percent=True
            )
            fig.add_trace(heat, row=1, col=current_col)
            current_col += 1

    # --- UNIT ---
    if show_unit:
        vals_unit = []
        if df_ts is not None and not df_ts.empty:
            for v in valores_order:
                _, metric_unit = VALORES_MAP.get(v, (None, None))
                if metric_unit and (metric_unit in df_ts.columns):
                    vals_unit.extend(pd.to_numeric(df_ts[metric_unit], errors="coerce").dropna().tolist())

        if len(vals_unit) > 0:
            zmin_u = float(np.quantile(vals_unit, unit_quantile_range[0]))
            zmax_u = float(np.quantile(vals_unit, unit_quantile_range[1]))
            if zmin_u == zmax_u:
                zmax_u = zmin_u + 1.0
        else:
            zmin_u, zmax_u = 0.0, 1.0

        for net in networks:
            frames = []
            if df_ts is not None and not df_ts.empty:
                for v in valores_order:
                    _, metric_unit = VALORES_MAP.get(v, (None, None))
                    if metric_unit and (metric_unit in df_ts.columns):
                        cols = [c for c in ["fecha","hora","technology","vendor","noc_cluster","network", metric_unit] if c in df_ts.columns]
                        tmp = df_ts[cols].copy()
                        tmp["valores"] = v
                        tmp.rename(columns={metric_unit: "value"}, inplace=True)
                        frames.append(tmp)
            df_ts_unit = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            df_ts_unit = _normalize_ts_frame(df_ts_unit, value_col="value")

            Z, ylabels, xlabels = _build_matrix(
                df_ts=df_ts_unit.rename(columns={"value": "__value__"}),
                rows_df=rows_df,
                metric="__value__",
                network=net,
                today_str=today_str,
                yday_str=yday_str,
            )

            # 🚧 Guard: si no hay valores numéricos, evita GL vacío
            if not np.isfinite(Z).any():
                return dcc.Markdown("**No hay datos numéricos para el heatmap UNIT con los filtros actuales.**")

            heat = _heat_trace(
                Z=Z, xlabels=xlabels, ylabels=ylabels,
                zmin=zmin_u, zmax=zmax_u,
                colorscale="Inferno", cbar_title="UNIT",
                use_gl=use_gl, is_percent=False
            )
            fig.add_trace(heat, row=1, col=current_col)
            current_col += 1

    # Layout
    fig.update_layout(
        height=max(400, int(24 * len(rows_df))),  # ~24px por fila; ajusta a gusto
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#202225", paper_bgcolor="#202225",
        font=dict(color="#eaeaea", size=11),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return html.Div([
        html.H4(title, className="grid-title"),
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": False, "staticPlot": True},
            style={"height": "100%"}
        ),
    ])
