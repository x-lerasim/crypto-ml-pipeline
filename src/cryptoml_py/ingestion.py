import time
import requests


def fetch_history(api_cfg):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(api_cfg["history-days"]) * 24 * 3600 * 1000
    return fetch_range(api_cfg, start_ms, now_ms)


def fetch_range(api_cfg, start_ms, end_ms):
    url = (
        f'{api_cfg["base-url"]}/assets/{api_cfg["asset"]}/history'
        f'?interval={api_cfg["interval"]}&start={start_ms}&end={end_ms}'
    )
    response = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=float(api_cfg["timeout-ms"]) / 1000.0,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    out = []
    for row in data:
        try:
            out.append(
                {
                    "priceUsd": float(row.get("priceUsd")),
                    "time": int(row.get("time")),
                    "date": str(row.get("date")),
                }
            )
        except (TypeError, ValueError):
            continue
    return out
