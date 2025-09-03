# extensions.py
from flask_caching import Cache

# Solo declaras la instancia (sin init_app aquí)
cache = Cache(config={
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 60,
})
