from mat import *
from datetime import datetime, timedelta


TABLE = "Resources_dashboardMaster"
VIEW = "main"

API_CONTRACT_VERSION = "main-table-v1"
MAX_PAGE_SIZE = 1000
MAX_SCAN_ROWS = 200000
DEFAULT_PAGE_SIZE = 50
ALLOW_IN_MEMORY_SORT_FALLBACK = False
FORCE_IN_MEMORY_SORT_OPTION = "force_in_memory_sort"

SERVER_SORT_FIELDS = {
    "severity_score": "Severity_Score",
    "crit_count": "Crit_Count",
    "complete_flag": "Complete_Flag",
    "integrity_health_pct": "Integrity_Health_Pct",
}

COMPUTED_FIELDS = [
    SERVER_SORT_FIELDS["severity_score"],
    SERVER_SORT_FIELDS["crit_count"],
    SERVER_SORT_FIELDS["complete_flag"],
    SERVER_SORT_FIELDS["integrity_health_pct"],
]


COLMAP = {
    "fecha": "Date",
    "hora": "Time",
    "network": "Network",
    "technology": "Technology",
    "vendor": "Vendor",
    "noc_cluster": "Noc_Cluster",
    "integrity": "INTEGRITY",
    "integrity_deg_pct": "Integrity_Health_Pct",
    "ps_traff_delta": "PS_TRAFF_DELTA",
    "ps_traff_gb": "PS_TRAFF_GB",
    "ps_rrc_ia_percent": "PS_RRC__IA",
    "ps_rrc_fail": "PS_RRC_FAIL",
    "ps_rab_ia_percent": "PS_RAB__IA",
    "ps_rab_fail": "PS_RAB_FAIL",
    "ps_s1_ia_percent": "PS_S1__IA",
    "ps_s1_fail": "PS_S1_FAIL",
    "ps_drop_dc_percent": "PS_DROP__DC",
    "ps_drop_abnrel": "PS_DROP_ABNREL",
    "cs_traff_delta": "CS_TRAFF_DELTA",
    "cs_traff_erl": "CS_TRAFF_ERL",
    "cs_rrc_ia_percent": "CS_RRC__IA",
    "cs_rrc_fail": "CS_RRC_FAIL",
    "cs_rab_ia_percent": "CS_RAB__IA",
    "cs_rab_fail": "CS_RAB_FAIL",
    "cs_drop_dc_percent": "CS_DROP__DC",
    "cs_drop_abnrel": "CS_DROP_ABNREL",
    "archivo_fuente": "Archivo_Fuente",
    "fecha_ejecucion": "Fecha_Ejecucion",
}

REVERSE_COLMAP = {v: k for k, v in COLMAP.items()}

BASE_COLUMNS = [
    "fecha", "hora", "vendor", "noc_cluster", "network", "technology",
    "integrity", "integrity_deg_pct",
    "ps_traff_delta", "ps_traff_gb",
    "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail",
    "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_traff_delta", "cs_traff_erl",
    "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail",
    "cs_drop_dc_percent", "cs_drop_abnrel",
]

KEY_COLUMNS = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
BASE_FIELDS = [COLMAP[c] for c in BASE_COLUMNS]

SEVERITY_METRICS = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
]

PROGRESS_METRICS = [
    "ps_rrc_fail",
    "ps_rab_fail",
    "ps_s1_fail",
    "ps_drop_abnrel",
    "cs_rrc_fail",
    "cs_rab_fail",
    "cs_drop_abnrel",
]

DEFAULT_THRESHOLDS = {
    "ps_rrc_ia_percent": {
        "orientation": "higher_is_better",
        "thresholds": {"excelente": 99.0, "bueno": 98.5, "regular": 98.0, "critico": 97.5},
    },
    "ps_rab_ia_percent": {
        "orientation": "higher_is_better",
        "thresholds": {"excelente": 99.0, "bueno": 98.5, "regular": 98.0, "critico": 97.5},
    },
    "ps_s1_ia_percent": {
        "orientation": "higher_is_better",
        "thresholds": {"excelente": 99.0, "bueno": 98.5, "regular": 98.0, "critico": 97.5},
    },
    "ps_drop_dc_percent": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.0, "bueno": 2.0, "regular": 2.5, "critico": 3.0},
    },
    "cs_rrc_ia_percent": {
        "orientation": "higher_is_better",
        "thresholds": {"excelente": 99.0, "bueno": 98.5, "regular": 98.0, "critico": 97.5},
    },
    "cs_rab_ia_percent": {
        "orientation": "higher_is_better",
        "thresholds": {"excelente": 99.0, "bueno": 98.5, "regular": 98.0, "critico": 97.5},
    },
    "cs_drop_dc_percent": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.0, "bueno": 2.0, "regular": 2.5, "critico": 3.0},
    },
}


def serverless_function_handler(params, context):
    try:
        params = params or {}
        requested_view = str(params.get("view") or VIEW).strip().lower()
        if requested_view not in ("", VIEW):
            return error_response("Vista no soportada para esta API: %s" % requested_view)

        matclient = MATClient()
        operation = str(params.get("operation") or "page").strip().lower()

        if operation in ("page", "kpis", "table_page", "main_table_page"):
            return handle_page(matclient, params)
        if operation == "latest_slot":
            return handle_latest_slot(matclient)
        if operation in ("distinct_catalogs", "catalogs"):
            return handle_distinct_catalogs(matclient, params)
        if operation == "integrity_baseline_week":
            return handle_integrity_baseline_week(matclient, params)
        if operation == "progress_max_by_network":
            return handle_progress_max_by_network(matclient, params)
        if operation == "alarm_state":
            return handle_alarm_state(matclient, params)
        if operation == "alarm_meta":
            return handle_alarm_meta(matclient, params)
        if operation == "by_keys":
            return handle_by_keys(matclient, params)
        if operation in ("context", "table_context", "main_table_context"):
            return handle_context(matclient, params)
        if operation in ("contract", "table_contract", "main_table_contract"):
            return handle_table_contract()
        if operation == "computed_preview":
            return handle_computed_preview(matclient, params)
        if operation == "computed_columns_check":
            return handle_computed_columns_check(matclient, params)

        return error_response("Operacion no soportada: %s" % operation)
    except Exception as exc:
        return error_response(str(exc))


def handle_page(matclient, params):
    page, page_size, offset = parse_pagination(params)
    sort = params.get("sort") or {}
    mode = str(params.get("mode") or sort.get("mode") or "alarmado").strip().lower()
    columns = parse_columns(params)
    fields = columns_to_fields(unique(columns + SEVERITY_METRICS + ["integrity"]))
    where = build_where(params)

    if mode in ("alarmado", "global", "severity") or is_integrity_pct_sort(sort):
        options = params.get("options") or {}
        use_in_memory_sort = bool(options.get(FORCE_IN_MEMORY_SORT_OPTION, ALLOW_IN_MEMORY_SORT_FALLBACK))

        if not use_in_memory_sort:
            data = execute_rows_query(
                matclient,
                where=build_backend_page_where(where, mode),
                limit=page_size,
                offset=offset,
                order_by=build_backend_page_order_by(params, mode),
                include_total=True,
                fields=columns_to_fields(columns),
            )
            rows = normalize_rows(data["rows"], na_as_empty=get_na_as_empty(params))
            return ok_response(
                "page",
                data={"rows": rows, "total": data["total"]},
                rows=rows,
                total=data["total"],
                pagination={"page": page, "page_size": page_size, "offset": offset},
                meta={"mode": mode, "sorted_by": "backend_computed_fields"},
            )

        # Fallback only for diagnostics. It violates real backend pagination because
        # it scans filtered rows before slicing the requested page.
        raw_rows = fetch_all_rows(
            matclient,
            where=where,
            order_by=[{"Date": "desc"}, {"Time": "desc"}],
            fields=fields,
            max_rows=option_int(params.get("options") or {}, "max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
        )
        rows = normalize_rows(raw_rows, na_as_empty=get_na_as_empty(params))
        scored = score_and_sort_rows(rows, params, mode)
        page_rows = trim_columns([item["row"] for item in scored[offset:offset + page_size]], columns)
        return ok_response(
            "page",
            data={"rows": page_rows, "total": len(scored)},
            rows=page_rows,
            total=len(scored),
            pagination={"page": page, "page_size": page_size, "offset": offset},
            meta={"mode": mode, "sorted_by": "server_computed", "diagnostic": "in_memory_sort"},
        )

    include_total = bool((params.get("options") or {}).get("include_total", True))
    data = execute_rows_query(
        matclient,
        where=where,
        limit=page_size,
        offset=offset,
        order_by=build_order_by(params),
        include_total=include_total,
        fields=columns_to_fields(columns),
    )
    rows = normalize_rows(data["rows"], na_as_empty=get_na_as_empty(params))
    return ok_response(
        "page",
        data={"rows": rows, "total": data["total"]},
        rows=rows,
        total=data["total"] if include_total else None,
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"mode": mode, "sorted_by": build_order_by(params)},
    )


def build_backend_page_where(base_where, mode):
    where = dict(base_where or {})
    if mode == "alarmado":
        where[SERVER_SORT_FIELDS["crit_count"]] = {"_gt": 0}
    return where


def build_backend_page_order_by(params, mode):
    sort = params.get("sort") or {}
    sort_column = strip_network_prefix(sort.get("column"))
    ascending = bool(sort.get("ascending", True))
    direction = "asc" if ascending else "desc"

    if is_integrity_pct_sort(sort):
        return [
            {SERVER_SORT_FIELDS["integrity_health_pct"]: direction},
            {"Date": "desc"},
            {"Time": "desc"},
        ]

    real_column = COLMAP.get(sort_column)
    if real_column:
        return [
            {real_column: direction},
            {SERVER_SORT_FIELDS["severity_score"]: "desc"},
            {"Date": "desc"},
            {"Time": "desc"},
        ]

    if mode == "alarmado":
        return [
            {SERVER_SORT_FIELDS["crit_count"]: "desc"},
            {SERVER_SORT_FIELDS["complete_flag"]: "asc"},
            {"INTEGRITY": "desc"},
            {SERVER_SORT_FIELDS["severity_score"]: "desc"},
            {"Date": "desc"},
            {"Time": "desc"},
            {"Noc_Cluster": "asc"},
        ]

    return [
        {SERVER_SORT_FIELDS["severity_score"]: "desc"},
        {"Date": "desc"},
        {"Time": "desc"},
    ]


def handle_latest_slot(matclient):
    data = execute_rows_query(
        matclient,
        where={},
        limit=1,
        offset=0,
        order_by=[{"Date": "desc"}, {"Time": "desc"}],
        include_total=False,
        fields=["Date", "Time"],
    )
    rows = normalize_rows(data["rows"])
    slot = None
    if rows:
        slot = {"fecha": rows[0].get("fecha"), "hora": normalize_hour_out(rows[0].get("hora"))}
    return ok_response("latest_slot", data=slot)


def handle_distinct_catalogs(matclient, params):
    base_where = build_where(params, include_filter_keys=False)
    dep_where = build_where(params, include_filter_keys=True)

    base_rows = fetch_all_rows(
        matclient,
        where=base_where,
        order_by=[{"Date": "desc"}, {"Time": "desc"}],
        fields=["Date", "Time", "Network", "Technology"],
        max_rows=option_int(params.get("options") or {}, "catalog_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    dep_rows = fetch_all_rows(
        matclient,
        where=dep_where,
        order_by=[{"Vendor": "asc"}, {"Noc_Cluster": "asc"}],
        fields=["Vendor", "Noc_Cluster"],
        max_rows=option_int(params.get("options") or {}, "catalog_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )

    base = normalize_rows(base_rows)
    dep = normalize_rows(dep_rows)
    data = {
        "fechas": sorted_unique(base, "fecha", reverse=True),
        "horas": sorted_unique(base, "hora"),
        "networks": order_networks(sorted_unique(base, "network")),
        "technologies": sorted_unique(base, "technology"),
        "vendors": sorted_unique(dep, "vendor"),
        "clusters": sorted_unique(dep, "noc_cluster"),
    }
    return ok_response("distinct_catalogs", data=data)


def handle_integrity_baseline_week(matclient, params):
    fecha = filter_value(params, "fecha")
    if not fecha:
        return error_response("integrity_baseline_week requiere fecha")

    selected_dt = datetime.strptime(str(fecha), "%Y-%m-%d")
    current_monday = selected_dt - timedelta(days=selected_dt.weekday())
    prev_monday = current_monday - timedelta(days=7)
    prev_sunday = current_monday - timedelta(days=1)
    days = [(prev_monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    p = clone_params(params)
    p["fecha"] = None
    p["hora"] = None
    where = build_where(p, ignore_fecha=True, ignore_hora=True)
    where["Date"] = {"_gte": prev_monday.strftime("%Y-%m-%d"), "_lte": prev_sunday.strftime("%Y-%m-%d")}

    rows = fetch_all_rows(
        matclient,
        where=where,
        fields=["Network", "Vendor", "Noc_Cluster", "Technology", "INTEGRITY"],
        max_rows=option_int(params.get("options") or {}, "baseline_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    rows = normalize_rows(rows)
    data = compute_integrity_baseline(rows)
    return ok_response("integrity_baseline_week", data=data, rows=data, total=len(data), meta={"days": days})


def handle_progress_max_by_network(matclient, params):
    rows = fetch_all_rows(
        matclient,
        where=build_where(params),
        fields=["Network"] + [COLMAP[m] for m in PROGRESS_METRICS],
        max_rows=option_int(params.get("options") or {}, "progress_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    data = compute_progress_max(normalize_rows(rows))
    return ok_response("progress_max_by_network", data=data)


def handle_alarm_state(matclient, params):
    fecha = filter_value(params, "fecha")
    if not fecha:
        return error_response("alarm_state requiere fecha")

    p = clone_params(params)
    p["hora"] = None
    rows = fetch_all_rows(
        matclient,
        where=build_where(p),
        order_by=[
            {"Network": "asc"},
            {"Vendor": "asc"},
            {"Noc_Cluster": "asc"},
            {"Technology": "asc"},
            {"Date": "asc"},
            {"Time": "asc"},
        ],
        fields=["Date", "Time", "Network", "Vendor", "Noc_Cluster", "Technology"] + [COLMAP[m] for m in SEVERITY_METRICS],
        max_rows=option_int(params.get("options") or {}, "alarm_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    data = compute_alarm_state(normalize_rows(rows), params)
    return ok_response("alarm_state", data=data, rows=data, total=len(data))


def handle_alarm_meta(matclient, params):
    fecha = filter_value(params, "fecha")
    if not fecha:
        return error_response("alarm_meta requiere fecha")

    base_dt = datetime.strptime(str(fecha), "%Y-%m-%d")
    days = [(base_dt - timedelta(days=1)).strftime("%Y-%m-%d"), str(fecha)]
    p = clone_params(params)
    p["fecha"] = None
    p["hora"] = None
    where = build_where(p, ignore_fecha=True, ignore_hora=True)
    where["Date"] = {"_in": days}

    rows = fetch_all_rows(
        matclient,
        where=where,
        fields=["Technology", "Vendor", "Noc_Cluster", "Network"] + [COLMAP[m] for m in SEVERITY_METRICS],
        max_rows=option_int(params.get("options") or {}, "alarm_meta_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    rows = normalize_rows(rows)

    grouped = {}
    alarm_keys = set()
    for row in rows:
        if row_crit_count(row, params) <= 0:
            continue
        trio = (row.get("technology"), row.get("vendor"), row.get("noc_cluster"))
        key = (row.get("technology"), row.get("vendor"), row.get("noc_cluster"), row.get("network"))
        grouped[trio] = grouped.get(trio, 0) + 1
        alarm_keys.add(key)

    ranked = sorted(
        grouped.items(),
        key=lambda item: (-item[1], str(item[0][1]), str(item[0][0]), str(item[0][2])),
    )
    meta_rows = [
        {"technology": key[0], "vendor": key[1], "noc_cluster": key[2], "flag_hits": hits}
        for key, hits in ranked
    ]
    response = ok_response("alarm_meta", data=meta_rows, rows=meta_rows, total=len(meta_rows), meta={"days": days})
    response["alarm_keys"] = [
        {"technology": k[0], "vendor": k[1], "noc_cluster": k[2], "network": k[3]}
        for k in sorted(alarm_keys)
    ]
    return response


def handle_by_keys(matclient, params):
    row_keys = params.get("row_keys") or params.get("keys") or []
    if not row_keys:
        return ok_response("by_keys", data={"rows": [], "total": 0}, rows=[], total=0)

    where = build_where(params)
    key_or = []
    for raw_key in row_keys[:MAX_PAGE_SIZE]:
        key = normalize_row_key(raw_key)
        if not key:
            continue
        key_or.append({
            "_and": [
                {"Date": {"_eq": key["fecha"]}},
                {"Time": {"_eq": normalize_hour_in(key["hora"])}},
                {"Vendor": {"_eq": key["vendor"]}},
                {"Noc_Cluster": {"_eq": key["noc_cluster"]}},
                {"Technology": {"_eq": key["technology"]}},
            ]
        })

    if not key_or:
        return ok_response("by_keys", data={"rows": [], "total": 0}, rows=[], total=0)

    where = {"_and": [where, {"_or": key_or}]} if where else {"_or": key_or}
    columns = parse_columns(params)
    raw_rows = fetch_all_rows(
        matclient,
        where=where,
        fields=columns_to_fields(columns),
        max_rows=MAX_PAGE_SIZE,
    )
    rows = normalize_rows(raw_rows, na_as_empty=get_na_as_empty(params))
    order_map = {}
    for idx, raw_key in enumerate(row_keys):
        key = normalize_row_key(raw_key)
        if key:
            order_map[(key["fecha"], normalize_hour_out(key["hora"]), key["vendor"], key["noc_cluster"], key["technology"])] = idx
    rows.sort(key=lambda r: order_map.get(tuple(r.get(c) for c in KEY_COLUMNS), 10 ** 9))
    return ok_response("by_keys", data={"rows": rows, "total": len(rows)}, rows=rows, total=len(rows))


def handle_context(matclient, params):
    include = params.get("include") or {}
    data = {}
    if include.get("progress_max", True):
        data["progress_max"] = handle_progress_max_by_network(matclient, params).get("data")
    if include.get("alarm_state", True) and params.get("fecha"):
        data["alarm_state"] = handle_alarm_state(matclient, params).get("data")
    if include.get("integrity_baseline", True) and params.get("fecha"):
        data["integrity_baseline"] = handle_integrity_baseline_week(matclient, params).get("data")
    return ok_response("context", data=data)


def handle_table_contract():
    """
    Machine-readable contract for the Dash main table adapter.

    The dashboard dataAccess layer should call table_page for rows and
    table_context for the auxiliary payloads used by the renderer.
    """
    data = {
        "version": API_CONTRACT_VERSION,
        "operations": {
            "table_page": {
                "returns": {"rows": "list[dict]", "total": "int", "pagination": "dict"},
                "columns": list(BASE_COLUMNS),
                "filters": ["fecha", "hora", "networks", "technologies", "vendors", "clusters"],
                "sort": ["mode", "column", "ascending", "sort_net"],
            },
            "table_context": {
                "returns": {
                    "progress_max": "dict[str,float]",
                    "alarm_state": "list[dict]",
                    "integrity_baseline": "list[dict]",
                },
                "filters": ["fecha", "hora", "networks", "technologies", "vendors", "clusters"],
            },
        },
    }
    return ok_response("table_contract", data=data, meta={"contract_version": API_CONTRACT_VERSION})


def handle_computed_preview(matclient, params):
    """
    Diagnostic helper for the ingestion DAG:
    returns rows plus the four values that must be persisted in Resources_dashboardMaster.

    This does not update the table. Use it to validate that the DAG-side calculation
    matches the dashboard's current behavior before writing the computed columns.
    """
    page, page_size, offset = parse_pagination(params)
    options = params.get("options") or {}
    include_total = bool(options.get("include_total", False))
    include_integrity_baseline = bool(options.get("include_integrity_baseline", False))
    where = build_where(params)
    fields = columns_to_fields(unique(BASE_COLUMNS + SEVERITY_METRICS + ["integrity"]))
    data = execute_rows_query(
        matclient,
        where=where,
        limit=page_size,
        offset=offset,
        order_by=[
            {"Date": "desc"},
            {"Time": "desc"},
            {"Vendor": "asc"},
            {"Noc_Cluster": "asc"},
            {"Network": "asc"},
            {"Technology": "asc"},
        ],
        include_total=include_total,
        fields=fields,
    )
    rows = normalize_rows(data["rows"])
    baseline_map = {}
    if include_integrity_baseline:
        baseline_map = fetch_integrity_baseline_map_for_rows(matclient, rows, params)

    out = []
    for row in rows:
        computed = compute_dashboard_master_fields(row, params, baseline_map)
        item = dict(row)
        item.update(computed)
        item.update(compute_integrity_health_debug_fields(row, baseline_map))
        out.append(item)

    total = data["total"] if include_total else len(out)
    return ok_response(
        "computed_preview",
        data={"rows": out, "total": total},
        rows=out,
        total=total,
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={
            "include_total": include_total,
            "include_integrity_baseline": include_integrity_baseline,
            "total_is_page_count": not include_total,
        },
    )


def handle_computed_columns_check(matclient, params):
    """
    Fast check for persisted computed columns.
    It reads the stored values directly and skips diagnostic recalculation.
    """
    page, page_size, offset = parse_pagination(params)
    fields = unique([
        "Date",
        "Time",
        "Vendor",
        "Noc_Cluster",
        "Network",
        "Technology",
        "INTEGRITY",
    ] + COMPUTED_FIELDS)
    data = execute_rows_query(
        matclient,
        where=build_where(params),
        limit=page_size,
        offset=offset,
        order_by=[
            {"Date": "desc"},
            {"Time": "desc"},
            {"Vendor": "asc"},
            {"Noc_Cluster": "asc"},
            {"Network": "asc"},
            {"Technology": "asc"},
        ],
        include_total=False,
        fields=fields,
    )
    rows = normalize_rows(data["rows"], na_as_empty=get_na_as_empty(params))
    return ok_response(
        "computed_columns_check",
        data={"rows": rows, "total": len(rows)},
        rows=rows,
        total=len(rows),
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"total_is_page_count": True},
    )


def fetch_integrity_baseline_map_for_rows(matclient, rows, params):
    if not rows:
        return {}

    fechas = [row.get("fecha") for row in rows if row.get("fecha")]
    if not fechas:
        return {}

    # Use the newest date in the preview page. The ingestion DAG should compute
    # this per loaded date, using the previous Monday-Sunday week.
    fecha = max(fechas)
    return fetch_integrity_baseline_map_for_keys(matclient, fecha, rows, params)


def fetch_integrity_baseline_map_for_keys(matclient, fecha, key_rows, params):
    """
    Computes previous-week INTEGRITY averages only for the row keys being processed.

    This is intentionally narrower than handle_integrity_baseline_week(), which is used
    by the app context and can scan a large filtered week. The ingestion/preview path
    only needs baselines for the current batch/page.
    """
    try:
        selected_dt = datetime.strptime(str(fecha), "%Y-%m-%d")
    except Exception:
        return {}

    current_monday = selected_dt - timedelta(days=selected_dt.weekday())
    prev_monday = current_monday - timedelta(days=7)
    prev_sunday = current_monday - timedelta(days=1)

    key_or = []
    seen = set()
    for row in key_rows or []:
        key = (
            row.get("network"),
            row.get("vendor"),
            row.get("noc_cluster"),
            row.get("technology"),
        )
        if any(v in (None, "") for v in key) or key in seen:
            continue
        seen.add(key)
        key_or.append({
            "_and": [
                {"Network": {"_eq": key[0]}},
                {"Vendor": {"_eq": key[1]}},
                {"Noc_Cluster": {"_eq": key[2]}},
                {"Technology": {"_eq": key[3]}},
            ]
        })

    if not key_or:
        return {}

    where = {
        "_and": [
            {"Date": {"_gte": prev_monday.strftime("%Y-%m-%d"), "_lte": prev_sunday.strftime("%Y-%m-%d")}},
            {"_or": key_or},
        ]
    }
    rows = fetch_all_rows(
        matclient,
        where=where,
        fields=["Network", "Vendor", "Noc_Cluster", "Technology", "INTEGRITY"],
        max_rows=option_int(params.get("options") or {}, "preview_baseline_max_rows", MAX_SCAN_ROWS, MAX_SCAN_ROWS),
    )
    rows = normalize_rows(rows)
    baseline_rows = compute_integrity_baseline(rows)

    baseline_map = {}
    for item in baseline_rows:
        key = (
            item.get("network"),
            item.get("vendor"),
            item.get("noc_cluster"),
            item.get("technology"),
        )
        baseline_map[key] = item.get("integrity_week_avg")
    return baseline_map


def compute_dashboard_master_fields(row, params=None, integrity_baseline_map=None):
    """
    Pure calculation to reuse in the load DAG.

    Returns DB column names:
      - Severity_Score: sum of severity levels across severity KPIs.
      - Crit_Count: count of KPIs in critical level.
      - Complete_Flag: 0 when INTEGRITY >= 80, otherwise 1.
      - Integrity_Health_Pct: INTEGRITY / previous-week baseline * 100, capped 0..100.
    """
    params = params or {}
    integrity_baseline_map = integrity_baseline_map or {}

    severity_score = row_severity_score(row, params)
    crit_count = row_crit_count(row, params)

    integrity = to_float(row.get("integrity"))
    complete_flag = 0 if integrity is not None and integrity >= 80 else 1

    baseline_key = (
        row.get("network"),
        row.get("vendor"),
        row.get("noc_cluster"),
        row.get("technology"),
    )
    baseline = to_float(integrity_baseline_map.get(baseline_key))
    integrity_health_pct = None
    if integrity is not None and baseline is not None and baseline > 0:
        integrity_health_pct = max(0.0, min(100.0, (integrity / baseline) * 100.0))

    return {
        SERVER_SORT_FIELDS["severity_score"]: int(severity_score),
        SERVER_SORT_FIELDS["crit_count"]: int(crit_count),
        SERVER_SORT_FIELDS["complete_flag"]: int(complete_flag),
        SERVER_SORT_FIELDS["integrity_health_pct"]: integrity_health_pct,
    }


def compute_integrity_health_debug_fields(row, integrity_baseline_map=None):
    """
    Preview-only diagnostic fields. Do not persist these columns.
    """
    integrity_baseline_map = integrity_baseline_map or {}
    baseline_key = (
        row.get("network"),
        row.get("vendor"),
        row.get("noc_cluster"),
        row.get("technology"),
    )
    integrity = to_float(row.get("integrity"))
    baseline = to_float(integrity_baseline_map.get(baseline_key))
    raw_pct = None
    if integrity is not None and baseline is not None and baseline > 0:
        raw_pct = (integrity / baseline) * 100.0
    return {
        "Integrity_Baseline_Debug": baseline,
        "Integrity_Health_Raw_Pct_Debug": raw_pct,
    }


def parse_pagination(params):
    pagination = params.get("pagination") or {}
    page = safe_int(pagination.get("page"), safe_int(params.get("page"), 1))
    page_size = safe_int(pagination.get("page_size"), safe_int(params.get("page_size"), DEFAULT_PAGE_SIZE))
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    offset = (page - 1) * page_size
    return page, page_size, offset


def parse_columns(params):
    options = params.get("options") or {}
    raw = params.get("columns") or options.get("columns") or BASE_COLUMNS
    cols = []
    for col in as_list(raw):
        col = strip_network_prefix(str(col))
        if col in COLMAP and col not in cols:
            cols.append(col)
    return cols or list(BASE_COLUMNS)


def columns_to_fields(columns):
    return [COLMAP[c] for c in columns if c in COLMAP]


def build_where(params, include_filter_keys=True, ignore_fecha=False, ignore_hora=False):
    where = {}
    filters = params.get("filters") or {}
    fecha = filter_value(params, "fecha")
    hora = filter_value(params, "hora")

    if fecha and not ignore_fecha:
        where["Date"] = {"_eq": fecha}
    if hora and not ignore_hora and str(hora).strip().lower() != "todas":
        where["Time"] = {"_eq": normalize_hour_in(hora)}

    if include_filter_keys:
        add_in_filter(where, "Network", filter_value(params, "networks", "network"))
        add_in_filter(where, "Technology", filter_value(params, "technologies", "technology"))
        add_in_filter(where, "Vendor", filter_value(params, "vendors", "vendor"))
        add_in_filter(where, "Noc_Cluster", filter_value(params, "clusters", "cluster"))

    return where


def build_order_by(params):
    sort = params.get("sort") or {}
    column = strip_network_prefix(sort.get("column"))
    direction = "asc" if bool(sort.get("ascending", True)) else "desc"
    real_column = COLMAP.get(column)
    if real_column:
        return [{real_column: direction}, {"Date": "desc"}, {"Time": "desc"}]
    return [{"Date": "desc"}, {"Time": "desc"}]


def execute_rows_query(matclient, where, limit, offset, order_by, include_total=False, fields=None):
    fields = fields or BASE_FIELDS
    aggregate_block = """
      Resources_dashboardMaster_aggregate(where: $where) {
        aggregate { count }
      }
    """ if include_total else ""

    query = """
    query getDashboardMaster(
      $where: Resources_dashboardMaster_bool_exp,
      $limit: Int!,
      $offset: Int!,
      $order_by: [Resources_dashboardMaster_order_by!]
    ) {
      Resources_dashboardMaster(
        where: $where,
        limit: $limit,
        offset: $offset,
        order_by: $order_by
      ) {
        %s
      }
      %s
    }
    """ % ("\n        ".join(fields), aggregate_block)

    result = matclient.graphQL.execute(
        operation=query,
        variables={"where": where, "limit": limit, "offset": offset, "order_by": order_by},
    )
    if result.get("errors"):
        raise Exception("Error GraphQL consultando %s: %s" % (TABLE, result.get("errors")))

    data = result.get("data") or {}
    rows = data.get(TABLE) or []
    total = 0
    if include_total:
        total = (((data.get("%s_aggregate" % TABLE) or {}).get("aggregate") or {}).get("count") or 0)
    return {"rows": rows, "total": int(total)}


def fetch_all_rows(matclient, where, order_by=None, fields=None, max_rows=MAX_SCAN_ROWS):
    rows = []
    max_rows = max(1, min(int(max_rows or MAX_SCAN_ROWS), MAX_SCAN_ROWS))
    page_size = min(MAX_PAGE_SIZE, max_rows)
    offset = 0
    order_by = order_by or [{"Date": "asc"}, {"Time": "asc"}]

    while len(rows) < max_rows:
        data = execute_rows_query(
            matclient,
            where=where,
            limit=page_size,
            offset=offset,
            order_by=order_by,
            include_total=False,
            fields=fields,
        )
        chunk = data["rows"]
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return rows[:max_rows]


def normalize_rows(rows, na_as_empty=False):
    out = []
    for row in rows or []:
        item = {}
        for key, value in row.items():
            new_key = REVERSE_COLMAP.get(key, key)
            if new_key == "hora":
                value = normalize_hour_out(value)
            if na_as_empty and (value is None or value == "NA"):
                value = ""
            item[new_key] = value
        out.append(item)
    return out


def score_and_sort_rows(rows, params, mode):
    sort = params.get("sort") or {}
    sort_column = strip_network_prefix(sort.get("column"))
    sort_net = sort.get("network") or sort.get("sort_net")
    ascending = bool(sort.get("ascending", True))

    scored = []
    for row in rows:
        severity_score = row_severity_score(row, params)
        crit_count = row_crit_count(row, params)

        if mode == "alarmado" and crit_count <= 0:
            continue

        metric_value = None
        if sort_column and sort_column in row:
            if sort_net and row.get("network") != sort_net:
                metric_value = None
            else:
                metric_value = to_float(row.get(sort_column))

        integrity = to_float(row.get("integrity"))
        complete_flag = 0 if integrity is not None and integrity >= 80 else 1

        scored.append({
            "row": row,
            "severity_score": severity_score,
            "crit_count": crit_count,
            "metric_value": metric_value,
            "complete_flag": complete_flag,
            "integrity": integrity,
        })

    def base_key(item):
        row = item["row"]
        return (
            str(row.get("fecha") or ""),
            str(row.get("hora") or ""),
            str(row.get("vendor") or ""),
            str(row.get("noc_cluster") or ""),
            str(row.get("technology") or ""),
        )

    if sort_column:
        scored.sort(
            key=lambda item: (
                item["metric_value"] is None,
                item["metric_value"] if ascending else -(item["metric_value"] or 0),
                -item["severity_score"],
                tuple(reversed(base_key(item))),
            )
        )
    elif mode == "alarmado":
        scored.sort(
            key=lambda item: (
                item["complete_flag"],
                -(item["integrity"] if item["integrity"] is not None else -1),
                -item["severity_score"],
                str(item["row"].get("noc_cluster") or ""),
            )
        )
    else:
        scored.sort(
            key=lambda item: (
                -item["severity_score"],
                str(item["row"].get("fecha") or ""),
                str(item["row"].get("hora") or ""),
            )
        )

    return scored


def row_severity_score(row, params):
    score = 0
    for metric in SEVERITY_METRICS:
        score += metric_severity_level(metric, row.get(metric), row.get("network"), params)
    return score


def row_crit_count(row, params):
    count = 0
    for metric in SEVERITY_METRICS:
        level = metric_severity_level(metric, row.get(metric), row.get("network"), params)
        if level >= 4:
            count += 1
    return count


def metric_severity_level(metric, raw_value, network, params):
    value = to_float(raw_value)
    if value is None:
        return 0

    cfg = metric_threshold_config(metric, network, params)
    thresholds = cfg.get("thresholds") or {}
    orientation = cfg.get("orientation", "lower_is_better")

    exc = to_float(thresholds.get("excelente"))
    bue = to_float(thresholds.get("bueno"))
    reg = to_float(thresholds.get("regular"))
    cri = to_float(thresholds.get("critico") or cfg.get("critical"))
    if cri is None:
        return 0

    if orientation == "higher_is_better":
        if value <= cri:
            return 4
        if reg is not None and value <= reg:
            return 3
        if bue is not None and value <= bue:
            return 2
        if exc is not None and value <= exc:
            return 1
        return 0

    if value >= cri:
        return 4
    if reg is not None and value >= reg:
        return 3
    if bue is not None and value >= bue:
        return 2
    if exc is not None and value >= exc:
        return 1
    return 0


def metric_threshold_config(metric, network, params):
    options = params.get("options") or {}
    snapshot = (
        params.get("thresholds_snapshot")
        or options.get("thresholds_snapshot")
        or options.get("thresholds")
        or {}
    )

    severity = None
    if snapshot.get("profiles"):
        severity = (((snapshot.get("profiles") or {}).get("main") or {}).get("severity") or {})
    elif snapshot.get("severity"):
        severity = snapshot.get("severity") or {}
    else:
        severity = snapshot

    cfg = severity.get(metric) or DEFAULT_THRESHOLDS.get(metric) or {}
    if cfg.get("per_network") and network in cfg.get("per_network"):
        net_cfg = cfg.get("per_network").get(network) or {}
        base = cfg.get("default") or cfg
        merged = dict(base)
        merged.update(net_cfg)
        return merged
    return cfg.get("default") or cfg


def compute_progress_max(rows):
    out = {}
    for row in rows:
        net = row.get("network")
        if not net:
            continue
        for metric in PROGRESS_METRICS:
            value = to_float(row.get(metric))
            if value is None:
                continue
            key = "%s__%s" % (net, metric)
            out[key] = value if key not in out else max(out[key], value)
    return out


def compute_alarm_state(rows, params):
    out = []
    for row in rows:
        out.append({
            "fecha": row.get("fecha"),
            "hora": row.get("hora"),
            "network": row.get("network"),
            "vendor": row.get("vendor"),
            "noc_cluster": row.get("noc_cluster"),
            "technology": row.get("technology"),
            "has_alarm": 1 if row_crit_count(row, params) > 0 else 0,
        })
    return out


def compute_integrity_baseline(rows):
    sums = {}
    counts = {}
    for row in rows:
        value = to_float(row.get("integrity"))
        if value is None:
            continue
        key = (row.get("network"), row.get("vendor"), row.get("noc_cluster"), row.get("technology"))
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1

    out = []
    for key, total in sums.items():
        count = counts.get(key) or 1
        out.append({
            "network": key[0],
            "vendor": key[1],
            "noc_cluster": key[2],
            "technology": key[3],
            "integrity_week_avg": total / count,
        })
    return out


def ok_response(operation, data=None, rows=None, total=None, pagination=None, meta=None):
    response = {
        "success": True,
        "ok": True,
        "view": VIEW,
        "operation": operation,
        "data": data if data is not None else (rows or []),
    }
    if rows is not None:
        response["rows"] = rows
    if total is not None:
        response["total"] = int(total)
    if pagination:
        response["pagination"] = pagination
        if total is not None:
            page_size = max(1, int(pagination.get("page_size") or DEFAULT_PAGE_SIZE))
            response["total_pages"] = int((int(total) + page_size - 1) / page_size)
    if meta:
        response["meta"] = meta
    return response


def error_response(message):
    return {
        "success": False,
        "ok": False,
        "view": VIEW,
        "error": {"message": str(message)},
        "rows": [],
        "data": [],
    }


def trim_columns(rows, columns):
    allowed = set(columns)
    return [{k: v for k, v in row.items() if k in allowed} for row in rows]


def normalize_row_key(raw_key):
    if isinstance(raw_key, dict):
        key = {c: raw_key.get(c) for c in KEY_COLUMNS}
    elif isinstance(raw_key, (list, tuple)) and len(raw_key) >= 5:
        key = dict(zip(KEY_COLUMNS, raw_key[:5]))
    else:
        return None
    if any(key.get(c) in (None, "") for c in KEY_COLUMNS):
        return None
    return key


def is_integrity_pct_sort(sort):
    col = str((sort or {}).get("column") or "")
    return col == "integrity_deg_pct" or col.endswith("__integrity_deg_pct")


def strip_network_prefix(column):
    if column is None:
        return None
    column = str(column)
    if "__" in column:
        return column.split("__", 1)[1]
    return column


def get_na_as_empty(params):
    return bool((params.get("options") or {}).get("na_as_empty", False))


def filter_value(params, *names):
    params = params or {}
    filters = params.get("filters") or {}
    for name in names:
        if name in params and params.get(name) not in (None, ""):
            return params.get(name)
        if name in filters and filters.get(name) not in (None, ""):
            return filters.get(name)
    return None


def add_in_filter(where, column, values):
    values = as_list(values)
    if values:
        where[column] = {"_in": values}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return [value]


def sorted_unique(rows, key, reverse=False):
    return sorted([str(v).strip() for v in {row.get(key) for row in rows} if v not in (None, "")], reverse=reverse)


def order_networks(values):
    preferred = ["NET", "ATT", "TEF"]
    first = [v for v in preferred if v in values]
    rest = sorted([v for v in values if v not in preferred])
    return first + rest


def unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def clone_params(params):
    out = dict(params or {})
    out["filters"] = dict((params or {}).get("filters") or {})
    return out


def safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def option_int(options, key, default, maximum):
    value = safe_int((options or {}).get(key), default)
    return max(1, min(value, maximum))


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_hour_in(value):
    text = str(value).strip()
    return text[:5] if len(text) >= 5 else text


def normalize_hour_out(value):
    if value is None:
        return None
    text = str(value).strip()
    if len(text) == 5:
        return "%s:00" % text
    if len(text) > 8:
        return text[:8]
    return text
