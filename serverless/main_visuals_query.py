from mat import *
from datetime import datetime, timedelta


TABLE = "Resources_dashboardMaster"
VIEW = "main_visuals"
MAX_SCAN_ROWS = 200000
MAX_PAGE_SIZE = 2000
DEFAULT_PAGE_SIZE = 50


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
}
REVERSE_COLMAP = {v: k for k, v in COLMAP.items()}

VALORES_MAP = {
    "PS_RRC": ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "PS_S1": ("ps_s1_ia_percent", "ps_s1_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "PS_RAB": ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RRC": ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
    "CS_RAB": ("cs_rab_ia_percent", "cs_rab_fail"),
}

SEVERITY_METRICS = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
]

DEFAULT_THRESHOLDS = {
    metric: {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    }
    for metric in SEVERITY_METRICS
}

DEFAULT_PROGRESS = {
    "ps_rrc_fail": {"min": 0.0, "max": 9000.0},
    "ps_rab_fail": {"min": 0.0, "max": 9000.0},
    "ps_s1_fail": {"min": 0.0, "max": 9000.0},
    "ps_drop_abnrel": {"min": 0.0, "max": 9000.0},
    "cs_rrc_fail": {"min": 0.0, "max": 9000.0},
    "cs_rab_fail": {"min": 0.0, "max": 9000.0},
    "cs_drop_abnrel": {"min": 0.0, "max": 9000.0},
}

BASE_FIELDS = [COLMAP[k] for k in [
    "fecha", "hora", "network", "technology", "vendor", "noc_cluster",
    "integrity", "integrity_deg_pct",
    "ps_traff_gb", "cs_traff_erl",
    "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail",
    "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail",
    "cs_drop_dc_percent", "cs_drop_abnrel",
]]


def serverless_function_handler(params, context):
    try:
        params = params or {}
        operation = str(params.get("operation") or "contract").strip().lower()
        matclient = MATClient()

        if operation in ("contract", "main_visuals_contract"):
            return ok_response("contract", data=contract())
        if operation in ("main_heatmap", "heatmap"):
            return handle_main_heatmap(matclient, params)
        if operation in ("histogram", "histograma", "histo"):
            return handle_histogram(matclient, params)
        if operation in ("integrity_heatmap", "heatmap_integrity"):
            return handle_integrity_heatmap(matclient, params)

        return error_response("Operacion no soportada: %s" % operation)
    except Exception as exc:
        return error_response(str(exc))


def contract():
    return {
        "version": "main-visuals-v1",
        "operations": {
            "main_heatmap": {
                "params": ["fecha", "filters", "pagination", "order_by", "thresholds_snapshot"],
                "returns": ["pct_payload", "unit_payload", "page_info"],
            },
            "histogram": {
                "params": ["fecha", "domain=PS|CS", "filters", "pagination", "thresholds_snapshot"],
                "returns": ["pct_payload", "unit_payload", "page_info"],
            },
            "integrity_heatmap": {
                "params": ["fecha", "filters", "pagination"],
                "returns": ["pct_payload", "unit_payload", "page_info"],
            },
        },
    }


def handle_main_heatmap(matclient, params):
    today, yday = resolve_dates(params)
    page, page_size, offset = parse_pagination(params)
    thresholds = threshold_snapshot(params)
    order_by = str(params.get("order_by") or ((params.get("options") or {}).get("order_by")) or "alarm_hours")
    rows = fetch_48h_rows(matclient, params, today, yday)
    networks = requested_networks(params, rows)
    df_meta, alarm_keys = build_alarm_meta(rows, thresholds)
    pct_payload, unit_payload, page_info = build_main_heatmap_payloads(
        rows=rows,
        df_meta=df_meta,
        alarm_keys=alarm_keys,
        networks=networks,
        thresholds=thresholds,
        today=today,
        yday=yday,
        offset=offset,
        limit=page_size,
        order_by=order_by,
    )
    return ok_response(
        "main_heatmap",
        data={"pct_payload": pct_payload, "unit_payload": unit_payload, "page_info": page_info},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"rows_scanned": len(rows), "networks": networks, "order_by": order_by},
    )


def handle_histogram(matclient, params):
    today, yday = resolve_dates(params)
    page, page_size, offset = parse_pagination(params)
    thresholds = threshold_snapshot(params)
    domain = str(params.get("domain") or ((params.get("options") or {}).get("domain")) or "PS").upper()
    if domain == "CS":
        valores_order = ("CS_RRC", "CS_DROP", "CS_RAB")
        traffic_metric = "cs_traff_erl"
    else:
        valores_order = ("PS_RRC", "PS_S1", "PS_DROP", "PS_RAB")
        traffic_metric = "ps_traff_gb"

    rows = fetch_48h_rows(matclient, params, today, yday)
    networks = requested_networks(params, rows)
    df_meta, alarm_keys = build_alarm_meta(rows, thresholds)
    pct_payload, unit_payload, page_info = build_histogram_payloads(
        rows=rows,
        df_meta=df_meta,
        networks=networks,
        thresholds=thresholds,
        today=today,
        yday=yday,
        offset=offset,
        limit=page_size,
        valores_order=valores_order,
        traffic_metric=traffic_metric,
        alarm_keys=alarm_keys,
    )
    return ok_response(
        "histogram",
        data={"pct_payload": pct_payload, "unit_payload": unit_payload, "page_info": page_info},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"rows_scanned": len(rows), "domain": domain, "networks": networks},
    )


def handle_integrity_heatmap(matclient, params):
    today, yday = resolve_dates(params)
    page, page_size, offset = parse_pagination(params)
    rows = fetch_48h_rows(matclient, params, today, yday)
    networks = requested_networks(params, rows)
    pct_payload, unit_payload, page_info = build_integrity_payloads(
        rows=rows,
        networks=networks,
        today=today,
        yday=yday,
        offset=offset,
        limit=page_size,
    )
    return ok_response(
        "integrity_heatmap",
        data={"pct_payload": pct_payload, "unit_payload": unit_payload, "page_info": page_info},
        pagination={"page": page, "page_size": page_size, "offset": offset},
        meta={"rows_scanned": len(rows), "networks": networks},
    )


def fetch_48h_rows(matclient, params, today, yday):
    where = build_where(params, today, yday)
    max_rows = safe_int(((params.get("options") or {}).get("max_rows")), MAX_SCAN_ROWS)
    max_rows = max(1, min(max_rows, MAX_SCAN_ROWS))
    rows = []
    offset = 0
    page_size = min(MAX_PAGE_SIZE, max_rows)
    while len(rows) < max_rows:
        chunk = execute_rows_query(
            matclient,
            where=where,
            limit=page_size,
            offset=offset,
            order_by=[{"Date": "asc"}, {"Time": "asc"}, {"Network": "asc"}],
            fields=BASE_FIELDS,
        )
        if not chunk:
            break
        rows.extend(normalize_rows(chunk))
        if len(chunk) < page_size:
            break
        offset += page_size
    return rows[:max_rows]


def execute_rows_query(matclient, where, limit, offset, order_by, fields):
    query = """
    query getDashboardMasterVisuals(
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
    }
    """ % ("\n        ".join(fields))
    result = matclient.graphQL.execute(
        operation=query,
        variables={"where": where, "limit": limit, "offset": offset, "order_by": order_by},
    )
    if result.get("errors"):
        raise Exception("Error GraphQL consultando %s: %s" % (TABLE, result.get("errors")))
    return ((result.get("data") or {}).get(TABLE) or [])


def build_where(params, today, yday):
    where = {"Date": {"_in": [yday, today]}}
    add_in_filter(where, "Network", filter_value(params, "networks", "network"))
    add_in_filter(where, "Technology", filter_value(params, "technologies", "technology"))
    add_in_filter(where, "Vendor", filter_value(params, "vendors", "vendor"))
    add_in_filter(where, "Noc_Cluster", filter_value(params, "clusters", "cluster"))
    return where


def normalize_rows(rows):
    out = []
    for row in rows or []:
        item = {}
        for key, value in row.items():
            new_key = REVERSE_COLMAP.get(key, key)
            if new_key == "hora":
                value = normalize_hour(value)
            item[new_key] = value
        out.append(item)
    return out


def build_alarm_meta(rows, thresholds):
    trio_hits = {}
    trio_metric_flags = {}
    alarm_keys = set()
    for row in rows:
        key4 = key4_from_row(row)
        flags = {}
        for metric in SEVERITY_METRICS:
            flags[metric] = 1 if severity_level(row.get(metric), metric, row.get("network"), thresholds) >= 4 else 0
        crit = sum(flags.values())
        if crit >= 1:
            alarm_keys.add(key4)
        trio = (row.get("technology"), row.get("vendor"), row.get("noc_cluster"))
        trio_hits[trio] = trio_hits.get(trio, 0) + crit
        metric_flags = trio_metric_flags.setdefault(trio, {})
        for metric, flag in flags.items():
            metric_flags[metric] = max(metric_flags.get(metric, 0), flag)
    trios = sorted(
        list(trio_hits.keys()),
        key=lambda t: (
            -sum((trio_metric_flags.get(t) or {}).values()),
            -trio_hits.get(t, 0),
            str(t[1]),
            str(t[0]),
            str(t[2]),
        ),
    )
    return [{"technology": t[0], "vendor": t[1], "noc_cluster": t[2]} for t in trios], alarm_keys


def build_main_heatmap_payloads(rows, df_meta, alarm_keys, networks, thresholds, today, yday, offset, limit, order_by):
    valores_order = ("PS_RRC", "CS_RRC", "PS_S1", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB")
    row_index = build_row_index(rows, today, yday)
    rows_all = []
    for meta in df_meta:
        for net in networks:
            for valores in valores_order:
                key4 = (meta.get("technology"), meta.get("vendor"), meta.get("noc_cluster"), net)
                if order_by in ("alarm", "alarm_only") and key4 not in alarm_keys:
                    continue
                item = {
                    "technology": meta.get("technology"),
                    "vendor": meta.get("vendor"),
                    "noc_cluster": meta.get("noc_cluster"),
                    "network": net,
                    "valores": valores,
                }
                enrich_rank(item, row_index, thresholds, today, yday)
                if order_by in ("alarm_hours", "hours", "alarm_hours_pct") and item["alarm_hours"] <= 0:
                    continue
                rows_all.append(item)

    rows_all = sort_visual_rows(rows_all, order_by)
    return payloads_from_rows(rows_all, row_index, thresholds, today, yday, offset, limit)


def build_histogram_payloads(rows, df_meta, networks, thresholds, today, yday, offset, limit, valores_order, traffic_metric, alarm_keys):
    row_index = build_row_index(rows, today, yday)
    rows_all = []
    for meta in df_meta:
        for net in networks:
            for valores in valores_order:
                item = {
                    "technology": meta.get("technology"),
                    "vendor": meta.get("vendor"),
                    "noc_cluster": meta.get("noc_cluster"),
                    "network": net,
                    "valores": valores,
                }
                enrich_rank(item, row_index, thresholds, today, yday)
                rows_all.append(item)
    rows_all = sorted(rows_all, key=lambda r: (-(r.get("max_unit") if r.get("max_unit") is not None else -1), str(r.get("noc_cluster"))))
    return payloads_from_rows(rows_all, row_index, thresholds, today, yday, offset, limit, y_as_cluster=True, traffic_metric=traffic_metric)


def build_integrity_payloads(rows, networks, today, yday, offset, limit):
    row_index = build_row_index(rows, today, yday)
    combos = set()
    for row in rows:
        if row.get("network") in networks:
            combos.add(key4_from_row(row))
    rows_all = []
    for tech, vend, clus, net in combos:
        vals = row48(row_index, (tech, vend, clus, net), "integrity_deg_pct")
        units = row48(row_index, (tech, vend, clus, net), "integrity")
        finite_vals = [v for v in vals if is_number(v)]
        last_bad = -1
        for i, v in enumerate(vals):
            if is_number(v) and float(v) < 80.0:
                last_bad = i
        rows_all.append({
            "technology": tech,
            "vendor": vend,
            "noc_cluster": clus,
            "network": net,
            "valores": "INTEGRITY",
            "last_bad": last_bad,
            "min_pct": min(finite_vals) if finite_vals else None,
            "max_unit": max([u for u in units if is_number(u)] or [-1]),
        })
    rows_all = sorted(
        rows_all,
        key=lambda r: (-(r["last_bad"]), r["min_pct"] if r["min_pct"] is not None else 10**9, -r["max_unit"], str(r["noc_cluster"])),
    )
    return integrity_payloads_from_rows(rows_all, row_index, today, yday, offset, limit)


def payloads_from_rows(rows_all, row_index, thresholds, today, yday, offset, limit, y_as_cluster=False, traffic_metric=None):
    total = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all[start:end]
    x_dt = x_axis(today, yday)
    z_pct, z_unit, z_pct_raw, z_unit_raw = [], [], [], []
    y_labels, row_detail, row_last_ts, row_max_pct, row_max_unit = [], [], [], [], []
    traffic_rows_raw, traffic_by_key = [], {}
    unit_scores = []

    for r in rows_page:
        key4 = (r["technology"], r["vendor"], r["noc_cluster"], r["network"])
        valores = r["valores"]
        pm, um = VALORES_MAP.get(valores, (None, None))
        detail = "%s/%s/%s/%s/%s" % (r["technology"], r["vendor"], r["noc_cluster"], r["network"], valores)
        y_labels.append(str(r["noc_cluster"]) if y_as_cluster else detail)
        row_detail.append(detail)

        raw_pct = row48(row_index, key4, pm)
        raw_unit = row48(row_index, key4, um)
        z_pct_raw.append(raw_pct)
        z_unit_raw.append(raw_unit)
        z_pct.append([pct_color(v, pm, r["network"], thresholds) for v in raw_pct])
        unit_row = [progress_norm(v, um, r["network"], thresholds) for v in raw_unit]
        z_unit.append(unit_row)
        unit_scores.extend([v for v in unit_row if v is not None])

        valid = [i for i, v in enumerate(raw_unit) if is_number(v)] or [i for i, v in enumerate(raw_pct) if is_number(v)]
        row_last_ts.append(str(x_dt[valid[-1]]).replace("T", " ")[:16] if valid else "")
        row_max_pct.append(max([v for v in raw_pct if is_number(v)] or [None]))
        row_max_unit.append(max([v for v in raw_unit if is_number(v)] or [None]))

        if traffic_metric:
            trow = row48(row_index, key4, traffic_metric)
            traffic_rows_raw.append(trow)
            traffic_by_key[detail] = trow

    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "severity",
        "zmin": 0.0,
        "zmax": 1.0,
        "title": "% IA / % DC",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }
    unit_payload = {
        "z": z_unit,
        "z_raw": z_unit_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",
        "zmin": min(unit_scores) if unit_scores else 0.0,
        "zmax": max(unit_scores) if unit_scores else 1.0,
        "title": "Unidades",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }
    if traffic_metric:
        pct_payload["traffic_metric"] = traffic_metric
        pct_payload["traffic_raw"] = traffic_rows_raw
        pct_payload["traffic_by_key"] = traffic_by_key
        pct_payload["zmin"] = -0.5
        pct_payload["zmax"] = 3.5
    return pct_payload, unit_payload, {"total_rows": total, "offset": start, "limit": int(limit), "showing": len(rows_page)}


def integrity_payloads_from_rows(rows_all, row_index, today, yday, offset, limit):
    total = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all[start:end]
    x_dt = x_axis(today, yday)
    z_pct, z_unit, z_pct_raw, z_unit_raw, y_labels, row_detail = [], [], [], [], [], []
    for r in rows_page:
        key4 = (r["technology"], r["vendor"], r["noc_cluster"], r["network"])
        detail = "%s/%s/%s/%s/INTEGRITY" % key4
        y_labels.append(detail)
        row_detail.append(detail)
        raw_pct = row48(row_index, key4, "integrity_deg_pct")
        raw_unit = row48(row_index, key4, "integrity")
        z_pct_raw.append(raw_pct)
        z_unit_raw.append(raw_unit)
        z_pct.append([integrity_color(v) for v in raw_pct])
        z_unit.append([progress_simple(v) for v in raw_unit])
    base = {
        "x_dt": x_dt,
        "y": y_labels,
        "row_detail": row_detail,
        "zmin": 0.0,
        "zmax": 1.0,
    }
    pct_payload = dict(base, z=z_pct, z_raw=z_pct_raw, color_mode="severity", title="% Integridad")
    unit_payload = dict(base, z=z_unit, z_raw=z_unit_raw, color_mode="progress", title="Integridad")
    return pct_payload, unit_payload, {"total_rows": total, "offset": start, "limit": int(limit), "showing": len(rows_page)}


def enrich_rank(item, row_index, thresholds, today, yday):
    key4 = (item["technology"], item["vendor"], item["noc_cluster"], item["network"])
    pm, um = VALORES_MAP.get(item["valores"], (None, None))
    raw_pct = row48(row_index, key4, pm)
    raw_unit = row48(row_index, key4, um)
    scores = [pct_score(v, pm, item["network"], thresholds) for v in raw_pct]
    levels = [severity_level(v, pm, item["network"], thresholds) for v in raw_pct]
    item["alarm_hours"] = sum(1 for lv in levels if lv >= 1)
    item["crit_hours"] = sum(1 for lv in levels if lv >= 4)
    item["max_score"] = max([s for s in scores if s is not None] or [-1])
    item["max_unit"] = max([v for v in raw_unit if is_number(v)] or [-1])
    item["stair"] = [scores[i] if levels[i] >= 1 else None for i in range(48)]


def sort_visual_rows(rows_all, order_by):
    if order_by in ("unit", "alarm_hours_unit", "alarm_hours_fail"):
        return sorted(rows_all, key=lambda r: (-safe_float(r.get("max_unit"), -1), str(r.get("noc_cluster"))))
    return sorted(
        rows_all,
        key=lambda r: (
            -int(r.get("crit_hours") or 0),
            -int(r.get("alarm_hours") or 0),
            -safe_float(r.get("max_score"), -1),
            tuple(-(int((r.get("stair") or [None] * 48)[i] * 1000000)) if (r.get("stair") or [None] * 48)[i] is not None else 1 for i in range(47, -1, -1)),
            str(r.get("vendor")),
            str(r.get("technology")),
            str(r.get("noc_cluster")),
        ),
    )


def build_row_index(rows, today, yday):
    out = {}
    for row in rows:
        key4 = key4_from_row(row)
        off = offset48(row, today, yday)
        if off is None:
            continue
        out.setdefault(key4, {})[off] = row
    return out


def row48(row_index, key4, metric):
    if not metric:
        return [None] * 48
    by_off = row_index.get(key4) or {}
    return [by_off.get(off, {}).get(metric) for off in range(48)]


def row_crit_count(row, thresholds):
    total = 0
    net = row.get("network")
    for metric in SEVERITY_METRICS:
        if severity_level(row.get(metric), metric, net, thresholds) >= 4:
            total += 1
    return total


def severity_level(value, metric, network, thresholds):
    value = to_float(value)
    if value is None:
        return -1
    cfg = severity_cfg(metric, network, thresholds)
    thr = cfg.get("thresholds") or {}
    orientation = cfg.get("orientation", "lower_is_better")
    exc = to_float(thr.get("excelente"))
    bue = to_float(thr.get("bueno"))
    reg = to_float(thr.get("regular"))
    cri = to_float(thr.get("critico"))
    if cri is None:
        return -1
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


def pct_score(value, metric, network, thresholds):
    value = to_float(value)
    if value is None:
        return None
    cfg = severity_cfg(metric, network, thresholds)
    thr = cfg.get("thresholds") or {}
    orient = cfg.get("orientation", "lower_is_better")
    exc = to_float(thr.get("excelente")) or 0.0
    cri = to_float(thr.get("critico"))
    if cri is None or cri == exc:
        return None
    if orient == "higher_is_better":
        score = 1.0 - ((cri - value) / (cri - exc))
    else:
        score = (value - exc) / (cri - exc)
    return max(0.0, min(score, 2.0))


def pct_color(value, metric, network, thresholds):
    score = pct_score(value, metric, network, thresholds)
    if score is None:
        return None
    level = severity_level(value, metric, network, thresholds)
    if level >= 4:
        return 1.0
    return min(score, 0.999)


def progress_norm(value, metric, network, thresholds):
    value = to_float(value)
    if value is None:
        return None
    cfg = progress_cfg(metric, network, thresholds)
    mn = to_float(cfg.get("min")) or 0.0
    mx = to_float(cfg.get("max")) or 1.0
    if mx <= mn:
        mx = mn + 1.0
    return max(0.0, min(1.0, (value - mn) / (mx - mn)))


def progress_simple(value):
    value = to_float(value)
    if value is None:
        return None
    return max(0.0, min(1.0, value / 100.0))


def integrity_color(value):
    value = to_float(value)
    if value is None:
        return None
    return max(0.0, min(1.0, value / 100.0))


def severity_cfg(metric, network, thresholds):
    severity = extract_profile(thresholds).get("severity") or {}
    cfg = severity.get(metric) or DEFAULT_THRESHOLDS.get(metric) or {}
    per_net = cfg.get("per_network") or {}
    if network in per_net:
        merged = dict(cfg.get("default") or cfg)
        merged.update(per_net.get(network) or {})
        cfg = merged
    else:
        cfg = cfg.get("default") or cfg
    thr = dict(cfg.get("thresholds") or {})
    for key in ("excelente", "bueno", "regular", "critico"):
        if key not in thr:
            thr[key] = thr.get("regular", 0.0)
    cfg = dict(cfg)
    cfg["thresholds"] = thr
    cfg.setdefault("orientation", "lower_is_better")
    return cfg


def progress_cfg(metric, network, thresholds):
    progress = extract_profile(thresholds).get("progress") or {}
    cfg = progress.get(metric) or DEFAULT_PROGRESS.get(metric) or {}
    per_net = cfg.get("per_network") or {}
    if network in per_net:
        merged = dict(cfg.get("default") or cfg)
        merged.update(per_net.get(network) or {})
        return merged
    return cfg.get("default") or cfg


def extract_profile(thresholds):
    thresholds = thresholds or {}
    if thresholds.get("profiles"):
        return (thresholds.get("profiles") or {}).get("main") or {}
    return thresholds


def threshold_snapshot(params):
    options = params.get("options") or {}
    return params.get("thresholds_snapshot") or options.get("thresholds_snapshot") or options.get("thresholds") or {}


def resolve_dates(params):
    fecha = filter_value(params, "fecha")
    try:
        today_dt = datetime.strptime(str(fecha), "%Y-%m-%d") if fecha else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()
    today = today_dt.strftime("%Y-%m-%d")
    yday = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return today, yday


def parse_pagination(params):
    pagination = params.get("pagination") or {}
    page = safe_int(pagination.get("page"), safe_int(params.get("page"), 1))
    page_size = safe_int(pagination.get("page_size"), safe_int(params.get("page_size"), DEFAULT_PAGE_SIZE))
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    offset = (page - 1) * page_size
    return page, page_size, offset


def requested_networks(params, rows):
    values = as_list(filter_value(params, "networks", "network"))
    if not values:
        values = sorted_unique(rows, "network")
    preferred = ["NET", "ATT", "TEF"]
    return [n for n in preferred if n in values] + [n for n in values if n not in preferred]


def key4_from_row(row):
    return (row.get("technology"), row.get("vendor"), row.get("noc_cluster"), row.get("network"))


def offset48(row, today, yday):
    fecha = str(row.get("fecha") or "")
    hora = normalize_hour(row.get("hora"))
    try:
        h = int(str(hora)[:2])
    except Exception:
        return None
    if h < 0 or h > 23:
        return None
    if fecha == today:
        return h + 24
    if fecha == yday:
        return h
    return None


def x_axis(today, yday):
    return ["%sT%02d:00:00" % (yday, h) for h in range(24)] + ["%sT%02d:00:00" % (today, h) for h in range(24)]


def filter_value(params, *names):
    filters = params.get("filters") or {}
    for name in names:
        if params.get(name) not in (None, ""):
            return params.get(name)
        if filters.get(name) not in (None, ""):
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
        if not value or value.lower() == "todas":
            return []
        return [value]
    return [value]


def sorted_unique(rows, key):
    return sorted([str(v).strip() for v in {row.get(key) for row in rows} if v not in (None, "")])


def normalize_hour(value):
    if value is None:
        return None
    text = str(value)
    if len(text) == 5:
        return text + ":00"
    return text[:8]


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def safe_float(value, default=0.0):
    out = to_float(value)
    return default if out is None else out


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def is_number(value):
    return to_float(value) is not None


def ok_response(operation, data=None, pagination=None, meta=None):
    response = {
        "success": True,
        "ok": True,
        "view": VIEW,
        "operation": operation,
        "data": data if data is not None else {},
    }
    if pagination:
        response["pagination"] = pagination
    if meta:
        response["meta"] = meta
    return response


def error_response(message):
    return {
        "success": False,
        "ok": False,
        "view": VIEW,
        "error": {"message": str(message)},
        "data": {},
    }
