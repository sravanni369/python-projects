"""Fetch hourly air-quality history from the Open-Meteo Air Quality API.

Free, no API key. Source: https://air-quality-api.open-meteo.com/v1/air-quality
"""

import json
import urllib.error
import urllib.request

API = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Cities carrying a heavy PM2.5 burden, in India and the USA.
CITIES = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Kolkata": (22.57, 88.36),
    "Los Angeles": (34.05, -118.24),
    "New York": (40.71, -74.01),
    "Fresno": (36.74, -119.79),
}

VARIABLES = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]


def build_url(lat, lon, past_days):
    """Return the request URL for one city."""
    return (
        f"{API}?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(VARIABLES)}"
        f"&past_days={past_days}&forecast_days=0"
    )


def fetch_city(lat, lon, past_days=92, timeout=30):
    """Fetch one city's hourly series.

    Returns (times, columns) where columns maps variable name to a list of
    floats. Hours where any variable is missing are dropped, so every column
    has the same length and no None values.
    """
    url = build_url(lat, lon, past_days)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)

    hourly = payload.get("hourly", {})
    if not hourly:
        return [], {name: [] for name in VARIABLES}

    times = hourly.get("time", [])
    raw = {name: hourly.get(name, []) for name in VARIABLES}

    keep = [
        i
        for i in range(len(times))
        if all(raw[name][i] is not None for name in VARIABLES)
    ]
    columns = {name: [float(raw[name][i]) for i in keep] for name in VARIABLES}
    return [times[i] for i in keep], columns


def fetch_all(cities=None, past_days=92):
    """Fetch every city. Cities that error out are skipped, not faked."""
    cities = cities or CITIES
    out = {}
    for name, (lat, lon) in cities.items():
        try:
            times, columns = fetch_city(lat, lon, past_days)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"  {name:<12} FAILED: {exc}")
            continue
        if not times:
            print(f"  {name:<12} no usable rows, skipped")
            continue
        print(f"  {name:<12} {len(times):>5} hourly rows  {times[0]} to {times[-1]}")
        out[name] = (times, columns)
    return out
