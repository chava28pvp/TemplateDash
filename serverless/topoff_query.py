from mat import *
from datetime import datetime, timedelta


TABLE = "Resources_dashboardTopoff"
VIEW = "topoff"
MAX_PAGE_SIZE = 2000
DEFAULT_PAGE_SIZE = 50
MAX_SCAN_ROWS = 100000


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
    "archivo_fuente": "Archivo_Fuente",
    "fecha_ejecucion": "Fecha_Ejecucion",
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
    "archivo_fuente", "fecha_ejecucion",
]]

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

NUMERIC_COLS = set(SEVERITY_KPIS + [
    "ps_traff_gb", "ps_rrc_fail", "ps_rab_fail", "ps_s1_fail", "ps_drop_abnrel",
    "cs_traff_erl", "cs_rrc_fail", "cs_rab_fail", "cs_drop_abnrel",
    "unav", "tnl_abn", "tnl_fail",
])


def serverless_function_handler(params, context):
    try:
        params = params or {}
        operation = str(params.get("operation") or "contract").strip().lower()
        matclient = MATClient()

        if operation in ("contract", "topoff_contract"):
            return ok_response("contract", data=contract())
        if operation in ("page", "table_page"):
            return handle_page(matclient, params)
        if operation in ("distinct_options", "catalogs"):
            return handle_distinct_options(matclient, params)
        if operation in ("latest_slot", "latest"):
            return handle_latest_slot(matclient)

        return error_response("Operacion no soportada: %s" % operation)
    except Exception as exc:
        return error_response(str(exc))


def contract():
    return {
        "version": "topoff-query-v1",
        "resource": TABLE,
        "operations": {
            "page": {"params": ["fecha", "hora", "filters", "pagination", "mode", "sort", "thresholds_snapshot"]},
            "distinct_options": {"params": ["fecha", "filters"]},
            "latest_slot": {"params": []},
        },
    }


def handle_page(matclient, params):
    page, page_size, offset = parse_pagination(params)
    mode = str(params.get("mode") or "recent").strip().lower()
    sort = params.get("sort") or {}
    sort_col = sort.get("column")
    ascending = bool(sort.get("ascending", True))
    where = build_where(params, include_hora=True)

    rows = fetch_all_rows(
        matclient,
        where=where,
        fields=BASE_FIELDS,
        order_by=[{"Date": "desc"}, {"Time": "desc"}],
        max_rows=option_int(params, "max_rows", MAX_SCAN_ROWS),
    )
    if mode == "alarmado":
        rows = sorted(
            rows,
            key=lambda r: severity_sort_key(r, params, sort_col, ascending),
        )
    elif mode == "sitio":
        rows = sorted(rows, key=lambda r: (str(r.get("Site_att") or ""), str(r.get("Date") or ""), str(r.get("Time") or "")))
    elif sort_col and sort_col in COLMAP:
        rows = sorted(
            rows,
            key=lambda r: generic_sort_key(r, sort_col),
            reverse=not ascending,
        )
    else:
        rows = sorted(rows, key=lambda r: (str(r.get("Date") or ""), str(r.get("Time") or "")), reverse=True)

    total = len(rows)
    page_rows = rows[offset:offset + page_size]

    return ok_response(
        "page",
        data={"rows": normalize_rows(page_rows), "total": int(total)},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"mode": mode},
    )


def handle_distinct_options(matclient, params):
    where = build_where(params, include_hora=False, include_topoff_filters=False)
    rows = fetch_all_rows(
        matclient,
        where=where,
        fields=["Site_att", "RNC", "NodeB"],
        order_by=[{"Site_att": "asc"}, {"RNC": "asc"}, {"NodeB": "asc"}],
        max_rows=option_int(params, "catalog_max_rows", 50000),
    )
    sites = sorted(set(clean_text(r.get("Site_att")) for r in rows if clean_text(r.get("Site_att"))))
    rncs = sorted(set(clean_text(r.get("RNC")) for r in rows if clean_text(r.get("RNC"))))
    nodebs = sorted(set(clean_text(r.get("NodeB")) for r in rows if clean_text(r.get("NodeB"))))
    return ok_response("distinct_options", data={"sites": sites, "rncs": rncs, "nodebs": nodebs})


def handle_latest_slot(matclient):
    query = """
    query latestTopoff {
      Resources_dashboardTopoff(
        where: {Date: {_is_null: false}, Time: {_is_null: false}},
        order_by: [{Date: desc}, {Time: desc}],
        limit: 1
      ) {
        Date
        Time
      }
    }
    """
    result = matclient.graphQL.execute(operation=query, variables={})
    if result.get("errors"):
        raise Exception("Error GraphQL latest_slot: %s" % result.get("errors"))
    rows = (result.get("data") or {}).get(TABLE) or []
    slot = None
    if rows:
        slot = {"fecha": rows[0].get("Date"), "hora": normalize_time(rows[0].get("Time"))}
    return ok_response("latest_slot", data={"slot": slot})


def fetch_all_rows(matclient, *, where, fields, order_by, max_rows):
    max_rows = max(1, min(int(max_rows or MAX_SCAN_ROWS), MAX_SCAN_ROWS))
    rows = []
    offset = 0
    limit = min(MAX_PAGE_SIZE, max_rows)
    while len(rows) < max_rows:
        chunk = execute_rows_query(
            matclient,
            where=where,
            fields=fields,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
    return rows[:max_rows]


def execute_rows_query(matclient, *, where, fields, order_by, limit, offset):
    field_block = "\n".join(fields)
    query = """
    query topoffRows($where: Resources_dashboardTopoff_bool_exp, $limit: Int!, $offset: Int!, $order_by: [Resources_dashboardTopoff_order_by!]) {
      Resources_dashboardTopoff(where: $where, limit: $limit, offset: $offset, order_by: $order_by) {
        %s
      }
    }
    """ % field_block
    result = matclient.graphQL.execute(
        operation=query,
        variables={"where": where, "limit": int(limit), "offset": int(offset), "order_by": order_by},
    )
    if result.get("errors"):
        raise Exception("Error GraphQL consultando dashboardTopoff: %s" % result.get("errors"))
    return (result.get("data") or {}).get(TABLE) or []


def fetch_count(matclient, where):
    query = """
    query topoffCount($where: Resources_dashboardTopoff_bool_exp) {
      Resources_dashboardTopoff_aggregate(where: $where) {
        aggregate { count }
      }
    }
    """
    result = matclient.graphQL.execute(operation=query, variables={"where": where})
    if result.get("errors"):
        raise Exception("Error GraphQL contando dashboardTopoff: %s" % result.get("errors"))
    return (((result.get("data") or {}).get("%s_aggregate" % TABLE) or {}).get("aggregate") or {}).get("count") or 0


def build_where(params, include_hora=True, include_topoff_filters=True):
    where = {"_and": []}
    fecha = filter_value(params, "fecha")
    fechas = fecha_with_prev(fecha)
    if fechas:
        where["_and"].append({"Date": {"_in": fechas}})

    if include_hora:
        h_range = hora_range(filter_value(params, "hora"))
        if h_range:
            start, end = h_range
            where["_and"].append({"Time": {"_gte": start}})
            where["_and"].append({"Time": {"_lt": end}})

    add_in_filter(where, "Technology", filter_value(params, "technologies", "technology"))
    add_in_filter(where, "Vendor", filter_value(params, "vendors", "vendor"))
    add_in_filter(where, "Noc_Cluster", filter_value(params, "clusters", "cluster"))

    if include_topoff_filters:
        add_in_filter(where, "Site_att", filter_value(params, "sites", "site_att"))
        add_in_filter(where, "RNC", filter_value(params, "rncs", "rnc"))
        add_in_filter(where, "NodeB", filter_value(params, "nodebs", "nodeb"))

    return where if where["_and"] else {}


def build_order_by(mode, sort_col, ascending):
    if mode == "sitio":
        return [{"Site_att": "asc"}, {"Date": "desc"}, {"Time": "desc"}]
    if sort_col and sort_col in COLMAP:
        direction = "asc" if ascending else "desc"
        return [{COLMAP[sort_col]: direction}, {"Date": "desc"}, {"Time": "desc"}]
    return [{"Date": "desc"}, {"Time": "desc"}]


def severity_sort_key(row, params, sort_col, ascending):
    score = row_severity_score(row, params)
    secondary = value_for_sort(row, sort_col)
    if sort_col and sort_col in NUMERIC_COLS and secondary is not None:
        secondary_key = float(secondary) if ascending else -float(secondary)
    elif sort_col:
        secondary_key = str(secondary or "")
        if not ascending:
            secondary_key = "".join(chr(255 - ord(ch)) for ch in secondary_key)
    else:
        secondary_key = ""
    return (-score, secondary_key, str(row.get("Date") or ""), str(row.get("Time") or ""))


def generic_sort_key(row, sort_col):
    value = value_for_sort(row, sort_col)
    if sort_col in NUMERIC_COLS:
        return (value is None, float(value or 0.0))
    return (value is None, str(value or ""))


def row_severity_score(row, params):
    return sum(metric_severity_level(metric, row.get(COLMAP[metric]), params) for metric in SEVERITY_KPIS)


def metric_severity_level(metric, raw_value, params):
    value = to_float(raw_value)
    if value is None:
        value = 0.0
    cfg = metric_threshold_config(metric, params)
    thresholds = cfg.get("thresholds") or {}
    orientation = cfg.get("orientation") or "lower_is_better"
    exc = safe_float(thresholds.get("excelente"), 0.0)
    bue = safe_float(thresholds.get("bueno"), exc)
    reg = safe_float(thresholds.get("regular"), bue)
    cri = safe_float(thresholds.get("critico"), reg)
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
                item[friendly] = value
        out.append(item)
    return out


def parse_pagination(params):
    pagination = params.get("pagination") or {}
    page = safe_int(pagination.get("page"), safe_int(params.get("page"), 1))
    page_size = safe_int(pagination.get("page_size"), safe_int(params.get("page_size"), DEFAULT_PAGE_SIZE))
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    return page, page_size, (page - 1) * page_size


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


def fecha_with_prev(fecha):
    if not fecha:
        return []
    try:
        d = datetime.strptime(str(fecha), "%Y-%m-%d")
        return [(d - timedelta(days=1)).strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")]
    except Exception:
        return [str(fecha)]


def hora_range(hora):
    if not hora:
        return None
    s = str(hora).strip()
    if not s or s.lower() == "todas":
        return None
    fmt = "%H:%M:%S" if len(s) > 5 else "%H:%M"
    try:
        dt = datetime.strptime(s, fmt)
    except ValueError:
        return None
    start = (dt - timedelta(hours=1)).replace(minute=0, second=0)
    end = (dt + timedelta(hours=1)).replace(minute=0, second=0)
    return start.strftime("%H:%M"), end.strftime("%H:%M")


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


def value_for_sort(row, friendly):
    if not friendly or friendly not in COLMAP:
        return None
    value = row.get(COLMAP[friendly])
    if friendly in NUMERIC_COLS:
        return to_float(value)
    return value


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
