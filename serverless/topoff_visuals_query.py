from mat import *
from datetime import datetime, timedelta


TABLE = "Resources_dashboardTopoff"
VIEW = "topoff_visuals"
MAX_SCAN_ROWS = 100000
MAX_PAGE_SIZE = 2000
DEFAULT_PAGE_SIZE = 50


COLMAP = {
    "fecha": "Date",
    "hora": "Time",
    "technology": "Technology",
    "vendor": "Vendor",
    "region": "Region",
    "province": "Province",
    "municipality": "Municipality",
    "cluster": "Noc_Cluster",
    "site_att": "Site_att",
    "rnc": "RNC",
    "nodeb": "NodeB",
    "ps_traff_gb": "PS_TRAFF_GB",
    "ps_rrc_ia_percent": "PS_RRC__IA",
    "ps_rrc_fail": "PS_RRC_FAIL",
    "ps_rab_ia_percent": "PS_RAB__IA",
    "ps_rab_fail": "PS_RAB_FAIL",
    "ps_s1_ia_percent": "PS_S1__IA",
    "ps_s1_fail": "PS_S1_FAIL",
    "ps_drop_dc_percent": "PS_DROP__DC",
    "ps_drop_abnrel": "PS_DROP_ABNREL",
    "cs_traff_erl": "CS_TRAFF_ERL",
    "cs_rrc_ia_percent": "CS_RRC__IA",
    "cs_rrc_fail": "CS_RRC_FAIL",
    "cs_rab_ia_percent": "CS_RAB__IA",
    "cs_rab_fail": "CS_RAB_FAIL",
    "cs_drop_dc_percent": "CS_DROP__DC",
    "cs_drop_abnrel": "CS_DROP_ABNREL",
    "unav": "Unav",
    "rtx_tnl_tx_percent": "G_RTX4G_TNL__Tx",
    "tnl_abn": "TNL_ABN",
    "tnl_fail": "TNL_FAIL",
}
REVERSE_COLMAP = {v: k for k, v in COLMAP.items()}

BASE_FIELDS = [COLMAP[k] for k in [
    "fecha", "hora", "technology", "vendor", "region", "province", "municipality",
    "cluster", "site_att", "rnc", "nodeb",
    "ps_traff_gb", "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail", "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_traff_erl", "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail", "cs_drop_dc_percent", "cs_drop_abnrel",
    "unav", "rtx_tnl_tx_percent", "tnl_abn", "tnl_fail",
]]

META_COLS = ["technology", "vendor", "region", "province", "municipality", "cluster", "site_att", "rnc", "nodeb"]
VALORES_MAP = {
    "PS_RRC": ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RRC": ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB": ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB": ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
    "PS_S1": ("ps_s1_ia_percent", "ps_s1_fail"),
    "RTX_TNL": ("rtx_tnl_tx_percent", "tnl_abn"),
}
HEATMAP_ORDER = ("PS_RRC", "CS_RRC", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB")
PS_ORDER = ("PS_RRC", "PS_DROP", "PS_RAB")
CS_ORDER = ("CS_RRC", "CS_DROP", "CS_RAB")
SEVERITY_KPIS = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
    "rtx_tnl_tx_percent",
]
PROGRESS_DEFAULTS = {
    "ps_rrc_fail": (0.0, 9000.0),
    "ps_rab_fail": (0.0, 9000.0),
    "ps_s1_fail": (0.0, 9000.0),
    "ps_drop_abnrel": (0.0, 9000.0),
    "cs_rrc_fail": (0.0, 9000.0),
    "cs_rab_fail": (0.0, 9000.0),
    "cs_drop_abnrel": (0.0, 9000.0),
    "tnl_abn": (0.0, 9000.0),
}


def serverless_function_handler(params, context):
    try:
        params = params or {}
        operation = str(params.get("operation") or "contract").strip().lower()
        matclient = MATClient()

        if operation in ("contract", "topoff_visuals_contract"):
            return ok_response("contract", data=contract())
        if operation in ("heatmap", "main_heatmap"):
            return handle_heatmap(matclient, params)
        if operation in ("histogram", "histograma", "histo"):
            return handle_histogram(matclient, params)

        return error_response("Operacion no soportada: %s" % operation)
    except Exception as exc:
        return error_response(str(exc))


def contract():
    return {
        "version": "topoff-visuals-v1",
        "resource": TABLE,
        "operations": {
            "heatmap": {"params": ["fecha", "filters", "pagination", "order_by", "thresholds_snapshot"]},
            "histogram": {"params": ["fecha", "domain=PS|CS", "filters", "pagination", "thresholds_snapshot"]},
        },
    }


def handle_heatmap(matclient, params):
    today, yday = resolve_dates(params)
    page, page_size, offset = parse_pagination(params)
    rows = fetch_48h_rows(matclient, params, today, yday)
    meta_rows, alarm_keys = build_alarm_meta(rows, params)
    pct_payload, unit_payload, page_info = build_payloads(
        rows=rows,
        meta_rows=meta_rows,
        valores_order=HEATMAP_ORDER,
        params=params,
        today=today,
        yday=yday,
        offset=offset,
        limit=page_size,
        mode="heatmap",
        order_by=str(params.get("order_by") or "alarm_bins_pct"),
        alarm_keys=alarm_keys,
    )
    return ok_response(
        "heatmap",
        data={"pct_payload": pct_payload, "unit_payload": unit_payload, "page_info": page_info},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"rows_scanned": len(rows), "meta_rows": len(meta_rows)},
    )


def handle_histogram(matclient, params):
    today, yday = resolve_dates(params)
    page, page_size, offset = parse_pagination(params)
    domain = str(params.get("domain") or ((params.get("options") or {}).get("domain")) or "PS").upper()
    rows = fetch_48h_rows(matclient, params, today, yday)
    meta_rows, alarm_keys = build_alarm_meta(rows, params)
    valores_order = CS_ORDER if domain == "CS" else PS_ORDER
    pct_payload, unit_payload, page_info = build_payloads(
        rows=rows,
        meta_rows=meta_rows,
        valores_order=valores_order,
        params=params,
        today=today,
        yday=yday,
        offset=offset,
        limit=page_size,
        mode="histogram",
        order_by="unit",
        alarm_keys=alarm_keys,
        domain=domain,
    )
    return ok_response(
        "histogram",
        data={"pct_payload": pct_payload, "unit_payload": unit_payload, "page_info": page_info},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"rows_scanned": len(rows), "meta_rows": len(meta_rows), "domain": domain},
    )


def fetch_48h_rows(matclient, params, today, yday):
    where = build_where(params, today, yday)
    max_rows = option_int(params, "max_rows", MAX_SCAN_ROWS)
    return normalize_rows(fetch_all_rows(
        matclient,
        where=where,
        fields=BASE_FIELDS,
        order_by=[{"Date": "asc"}, {"Time": "asc"}],
        max_rows=max_rows,
    ))


def fetch_all_rows(matclient, *, where, fields, order_by, max_rows):
    max_rows = max(1, min(int(max_rows or MAX_SCAN_ROWS), MAX_SCAN_ROWS))
    rows = []
    offset = 0
    limit = min(MAX_PAGE_SIZE, max_rows)
    while len(rows) < max_rows:
        field_block = "\n".join(fields)
        query = """
        query topoffVisualRows($where: Resources_dashboardTopoff_bool_exp, $limit: Int!, $offset: Int!, $order_by: [Resources_dashboardTopoff_order_by!]) {
          Resources_dashboardTopoff(where: $where, limit: $limit, offset: $offset, order_by: $order_by) {
            %s
          }
        }
        """ % field_block
        result = matclient.graphQL.execute(
            operation=query,
            variables={"where": where, "limit": limit, "offset": offset, "order_by": order_by},
        )
        if result.get("errors"):
            raise Exception("Error GraphQL consultando dashboardTopoff: %s" % result.get("errors"))
        chunk = (result.get("data") or {}).get(TABLE) or []
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
    return rows[:max_rows]


def build_where(params, today, yday):
    where = {"_and": [{"Date": {"_in": [yday, today]}}]}
    add_in_filter(where, "Technology", filter_value(params, "technologies", "technology"))
    add_in_filter(where, "Vendor", filter_value(params, "vendors", "vendor"))
    add_in_filter(where, "Noc_Cluster", filter_value(params, "clusters", "cluster"))
    add_in_filter(where, "Site_att", filter_value(params, "sites", "site_att"))
    add_in_filter(where, "RNC", filter_value(params, "rncs", "rnc"))
    add_in_filter(where, "NodeB", filter_value(params, "nodebs", "nodeb"))
    return where


def build_alarm_meta(rows, params):
    grouped = {}
    alarm_keys = set()
    for row in rows or []:
        key = meta_key(row)
        acc = grouped.setdefault(key, {"row": meta_item(row), "max_flags": {}, "flag_hits": 0})
        row_flags = 0
        for metric in SEVERITY_KPIS:
            flag = 1 if metric_severity_level(metric, row.get(metric), params) >= 4 else 0
            acc["max_flags"][metric] = max(acc["max_flags"].get(metric, 0), flag)
            row_flags += flag
        acc["flag_hits"] += row_flags
        if row_flags > 0:
            alarm_keys.add(key)

    ranked = []
    for key, acc in grouped.items():
        item = dict(acc["row"])
        item["_alarm_score"] = sum(acc["max_flags"].values())
        item["_flag_hits"] = acc["flag_hits"]
        ranked.append(item)

    ranked = sorted(
        ranked,
        key=lambda r: (-int(r.get("_alarm_score") or 0), -int(r.get("_flag_hits") or 0), str(r.get("vendor") or ""), str(r.get("technology") or ""), str(r.get("nodeb") or "")),
    )
    return ranked, alarm_keys


def build_payloads(rows, meta_rows, valores_order, params, today, yday, offset, limit, mode, order_by, alarm_keys, domain=None):
    if not meta_rows:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0, "height": 300}

    row_index = build_row_index(rows, today, yday)
    rows_all = []
    for meta in meta_rows:
        for valores in valores_order:
            item = dict(meta)
            item["valores"] = valores
            enrich_visual_item(item, row_index, params, today, yday)
            rows_all.append(item)

    if mode == "histogram":
        rows_all = sorted(rows_all, key=lambda r: (-(r.get("max_unit") if r.get("max_unit") is not None else -1), str(r.get("nodeb") or ""), str(r.get("valores") or "")))
    else:
        if order_by in ("alarm_bins_unit", "alarm_hours_unit", "unit"):
            rows_all = sorted(rows_all, key=lambda r: (-(r.get("last_unit_offset") if r.get("last_unit_offset") is not None else -1), -(r.get("max_unit") if r.get("max_unit") is not None else -1), str(r.get("nodeb") or "")))
        else:
            rows_all = sorted(rows_all, key=lambda r: (-(r.get("last_alarm_offset") if r.get("last_alarm_offset") is not None else -1), -int(r.get("alarm_bins") or 0), -(r.get("max_pct_score") if r.get("max_pct_score") is not None else -1), str(r.get("nodeb") or "")))
            rows_all = [r for r in rows_all if int(r.get("has_any_pct_sample") or 0) == 1]

    total_rows = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    page_rows = rows_all[start:end]
    x_dt = build_x_dt_15m(yday) + build_x_dt_15m(today)

    z_pct, z_unit, z_pct_raw, z_unit_raw = [], [], [], []
    y_labels, row_detail, row_last_ts, row_max_pct, row_max_unit = [], [], [], [], []
    traffic_rows_raw, traffic_by_key = [], {}
    all_unit_scores = []

    traffic_metric = None
    if mode == "histogram":
        traffic_metric = "cs_traff_erl" if domain == "CS" else "ps_traff_gb"

    for item in page_rows:
        key = meta_key(item)
        valores = item.get("valores")
        pct_col, unit_col = VALORES_MAP.get(valores, (None, None))

        y = "%s/%s/%s/%s/%s/%s/%s/%s/%s/%s" % (
            item.get("technology"), item.get("vendor"), item.get("region"), item.get("province"),
            item.get("municipality"), item.get("site_att"), item.get("rnc"), item.get("nodeb"),
            item.get("cluster"), valores,
        )
        detail = "%s/%s/%s/%s/%s/%s/%s/%s/%s/%s" % (
            item.get("technology"), item.get("vendor"), item.get("region"), item.get("province"),
            item.get("municipality"), item.get("cluster"), item.get("site_att"), item.get("rnc"),
            item.get("nodeb"), valores,
        )
        y_labels.append(str(item.get("nodeb") or "") if mode == "histogram" else y)
        row_detail.append(detail)

        raw_pct = row192(row_index, key, pct_col)
        raw_unit = row192(row_index, key, unit_col)
        z_pct_raw.append(raw_pct)
        z_unit_raw.append(raw_unit)

        if mode == "histogram":
            z_pct.append([None if v is None else metric_severity_level(pct_col, v, params) for v in raw_pct])
        else:
            z_pct.append([None if v is None else severity_bucket(pct_col, v, params) for v in raw_pct])

        unit_norm = [normalize_progress(unit_col, v, params) if v is not None else None for v in raw_unit]
        z_unit.append(unit_norm)
        all_unit_scores.extend([v for v in unit_norm if v is not None])

        traffic_raw = row192(row_index, key, traffic_metric) if traffic_metric else [None] * 192
        traffic_rows_raw.append(traffic_raw)
        traffic_by_key[detail] = traffic_raw

        last_idx = last_valid_index(raw_unit) if last_valid_index(raw_unit) is not None else last_valid_index(raw_pct)
        row_last_ts.append(str(x_dt[last_idx]).replace("T", " ")[:16] if last_idx is not None else "")
        row_max_pct.append(max([v for v in raw_pct if isinstance(v, (int, float))], default=None))
        row_max_unit.append(max([v for v in raw_unit if isinstance(v, (int, float))], default=None))

    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "severity",
        "zmin": -0.5 if mode == "histogram" else 0.0,
        "zmax": 3.5 if mode == "histogram" else 3.0,
        "title": "% IA / % DC (TopOff)",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
        "traffic_rows_raw": traffic_rows_raw,
        "traffic_by_key": traffic_by_key,
    }
    unit_payload = {
        "z": z_unit,
        "z_raw": z_unit_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",
        "zmin": min(all_unit_scores) if all_unit_scores else 0.0,
        "zmax": max(all_unit_scores) if all_unit_scores else 1.0,
        "title": "Unidades",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
        "traffic_rows_raw": traffic_rows_raw,
        "traffic_by_key": traffic_by_key,
    }
    page_info = {
        "total_rows": int(total_rows),
        "offset": int(start),
        "limit": int(limit),
        "showing": len(page_rows),
        "height": max(300, int(len(page_rows) * 26 + 170)),
    }
    return pct_payload, unit_payload, page_info


def build_row_index(rows, today, yday):
    out = {}
    for row in rows or []:
        off = offset192(row.get("fecha"), row.get("hora"), today, yday)
        if off is None:
            continue
        out.setdefault(meta_key(row), {})[off] = row
    return out


def enrich_visual_item(item, row_index, params, today, yday):
    key = meta_key(item)
    pct_col, unit_col = VALORES_MAP.get(item.get("valores"), (None, None))
    raw_pct = row192(row_index, key, pct_col)
    raw_unit = row192(row_index, key, unit_col)
    alarm_offsets = [i for i, v in enumerate(raw_pct) if metric_severity_level(pct_col, v, params) >= 4]
    unit_offsets = [i for i, v in enumerate(raw_unit) if v is not None]
    pct_scores = [severity_bucket(pct_col, v, params) for v in raw_pct if v is not None]
    item["alarm_bins"] = len(alarm_offsets)
    item["last_alarm_offset"] = max(alarm_offsets) if alarm_offsets else None
    item["last_unit_offset"] = max(unit_offsets) if unit_offsets else None
    item["max_pct_score"] = max(pct_scores) if pct_scores else None
    item["max_unit"] = max([v for v in raw_unit if isinstance(v, (int, float))], default=None)
    item["has_any_pct_sample"] = 1 if any(v is not None for v in raw_pct) else 0


def row192(row_index, key, metric):
    if not metric:
        return [None] * 192
    rows_by_offset = row_index.get(key) or {}
    return [rows_by_offset.get(off, {}).get(metric) for off in range(192)]


def normalize_rows(rows):
    out = []
    for row in rows or []:
        item = {}
        for real, friendly in REVERSE_COLMAP.items():
            if real in row:
                value = row.get(real)
                if friendly == "cluster":
                    value = clean_text(value) or clean_text(row.get("Site_att"))
                elif friendly == "hora":
                    value = normalize_time(value)
                elif friendly in META_COLS:
                    value = clean_text(value)
                else:
                    value = to_float(value) if friendly not in ("fecha",) else value
                item[friendly] = value
        out.append(item)
    return out


def meta_key(row):
    return tuple(clean_text(row.get(col)) or "" for col in META_COLS)


def meta_item(row):
    return {col: clean_text(row.get(col)) or "" for col in META_COLS}


def offset192(fecha, hora, today, yday):
    if str(fecha) not in (str(today), str(yday)):
        return None
    q = q15(hora)
    if q is None:
        return None
    return q + (96 if str(fecha) == str(today) else 0)


def q15(hora):
    try:
        parts = str(hora).split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh * 4 + (mm // 15)
    except Exception:
        return None
    return None


def build_x_dt_15m(day):
    return ["%sT%02d:%02d:00" % (day, h, m) for h in range(24) for m in (0, 15, 30, 45)]


def metric_severity_level(metric, raw_value, params):
    value = to_float(raw_value)
    if metric is None or value is None:
        return 0
    cfg = metric_threshold_config(metric, params)
    thr = cfg.get("thresholds") or {}
    orientation = cfg.get("orientation") or "lower_is_better"
    exc = safe_float(thr.get("excelente"), 0.0)
    bue = safe_float(thr.get("bueno"), exc)
    reg = safe_float(thr.get("regular"), bue)
    cri = safe_float(thr.get("critico"), reg)
    if orientation == "higher_is_better":
        if value <= cri:
            return 4
        if value <= reg:
            return 3
        if value <= bue:
            return 2
        if value <= exc:
            return 1
        return 0
    if value >= cri:
        return 4
    if value >= reg:
        return 3
    if value >= bue:
        return 2
    if value >= exc:
        return 1
    return 0


def severity_bucket(metric, raw_value, params):
    level = metric_severity_level(metric, raw_value, params)
    if level >= 4:
        return 3
    if level == 3:
        return 2
    if level == 2:
        return 1
    return 0


def metric_threshold_config(metric, params):
    snapshot = threshold_snapshot(params)
    profiles = snapshot.get("profiles") or {}
    profile = profiles.get("topoff") or profiles.get("main") or {}
    sev = profile.get("severity") or {}
    cfg = sev.get(metric) or {}
    return (cfg.get("default") or cfg) or {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    }


def normalize_progress(metric, raw_value, params):
    value = to_float(raw_value)
    if metric is None or value is None:
        return None
    snapshot = threshold_snapshot(params)
    profiles = snapshot.get("profiles") or {}
    profile = profiles.get("topoff") or profiles.get("main") or {}
    progress = profile.get("progress") or {}
    cfg = progress.get(metric) or {}
    vmin, vmax = PROGRESS_DEFAULTS.get(metric, (0.0, 9000.0))
    vmin = safe_float(cfg.get("min"), vmin)
    vmax = safe_float(cfg.get("max"), vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def parse_pagination(params):
    pagination = params.get("pagination") or {}
    page = safe_int(pagination.get("page"), safe_int(params.get("page"), 1))
    page_size = safe_int(pagination.get("page_size"), safe_int(params.get("page_size"), DEFAULT_PAGE_SIZE))
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    return page, page_size, (page - 1) * page_size


def resolve_dates(params):
    fecha = filter_value(params, "fecha")
    if fecha:
        today = str(fecha)
    else:
        today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        yday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    return today, yday


def option_int(params, name, default):
    options = params.get("options") or {}
    return safe_int(options.get(name), default)


def threshold_snapshot(params):
    options = params.get("options") or {}
    return params.get("thresholds_snapshot") or options.get("thresholds_snapshot") or options.get("thresholds") or {}


def filter_value(params, *names):
    filters = params.get("filters") or {}
    for name in names:
        if params.get(name) not in (None, ""):
            return params.get(name)
        if filters.get(name) not in (None, ""):
            return filters.get(name)
    return None


def add_in_filter(where, field, values):
    values = as_list(values)
    if values:
        where.setdefault("_and", []).append({field: {"_in": values}})


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    return [value] if value != "" else []


def last_valid_index(values):
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] is not None:
            return idx
    return None


def normalize_time(value):
    if value is None:
        return None
    s = str(value).strip()
    if len(s) == 5:
        return s + ":00"
    return s[:8]


def clean_text(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def ok_response(operation, data=None, pagination=None, meta=None):
    return {
        "ok": True,
        "success": True,
        "view": VIEW,
        "operation": operation,
        "data": data or {},
        "pagination": pagination or {},
        "meta": meta or {},
    }


def error_response(message):
    return {"ok": False, "success": False, "view": VIEW, "error": {"message": str(message)}}
