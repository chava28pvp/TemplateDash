from dash import ctx


def reset_page_state(page_size, default_size=50):
    ps = max(1, int(page_size or default_size))
    return {"page": 1, "page_size": ps}


def paginate_state(state, prev_id, next_id, default_size=50):
    state = state or {"page": 1, "page_size": default_size}
    page = int(state.get("page", 1))
    ps = int(state.get("page_size", default_size))

    trig = ctx.triggered_id
    if trig == prev_id:
        page = max(1, page - 1)
    elif trig == next_id:
        page = page + 1

    return {"page": page, "page_size": ps}


def toggle_bool(n_clicks, current_value):
    if not n_clicks:
        return current_value
    return not current_value
