import os


os.environ.setdefault("ENV", "api")

from app import run_local_server


if __name__ == "__main__":
    run_local_server(open_browser=True)
