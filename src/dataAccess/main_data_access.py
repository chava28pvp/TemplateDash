from src.config import DATA_SOURCE

if DATA_SOURCE == "api":
    from src.dataAccess.main_api_access import (
        COLMAP,
        fetch_integrity_baseline_week,
        fetch_kpis,
        fetch_kpis_by_keys,
        fetch_kpis_paginated_severity_global_sort,
        fetch_kpis_paginated_severity_sort,
        fetch_alarm_meta_for_heatmap,
        fetch_latest_available_slot,
        fetch_main_alarm_state,
        fetch_main_distinct_catalogs,
        fetch_progress_max_by_network,
    )
else:
    from src.dataAccess.data_access import (
        COLMAP,
        fetch_integrity_baseline_week,
        fetch_kpis,
        fetch_kpis_by_keys,
        fetch_kpis_paginated_severity_global_sort,
        fetch_kpis_paginated_severity_sort,
        fetch_alarm_meta_for_heatmap,
        fetch_latest_available_slot,
        fetch_main_alarm_state,
        fetch_main_distinct_catalogs,
        fetch_progress_max_by_network,
    )
