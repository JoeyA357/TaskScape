
from typing import Dict, Any, List, Tuple
import os
import requests
from math import radians, sin, cos, sqrt, atan2



def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance between two lat/lon points in kilometers."""
    R = 6371.0  

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _get_google_key() -> str:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. Add it to your .env file."
        )
    return api_key


def _geocode_location(user_location: str) -> Tuple[float, float]:
    """
    Turn a text location into (lat, lon) using Google Geocoding.
    Falls back to Nominatim only if Google fails.
    """
    if not user_location.strip():
        raise ValueError("Empty location string.")

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"address": user_location, "key": api_key}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                loc = results[0]["geometry"]["location"]
                return float(loc["lat"]), float(loc["lng"])
        except Exception:
            pass

    def _nominatim_query(q: str):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": q,
            "format": "json",
            "limit": 1,
        }
        headers = {
            "User-Agent": "TaskScapeAI/1.0 (student project)",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    results = _nominatim_query(user_location)
    if results:
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        return lat, lon

    results = _nominatim_query(user_location + ", Lebanon")
    if results:
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        return lat, lon

    raise ValueError(f"No coordinates found for location: {user_location}")


def _infer_place_profiles_for_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decide which place profiles to search for based on the task.

    Each profile is a dict:
      {"type": <google_place_type_or_None>, "keyword": <optional keyword or None>}
    """
    ttype = (task.get("task_type") or "").lower()
    prefer_outside = bool(task.get("prefer_outside", False))


    if ttype in ["study", "project"]:
        if prefer_outside:
            return [
                {"type": "library", "keyword": None},
                {"type": "cafe", "keyword": None},
                {"type": "point_of_interest", "keyword": "coworking space"},
            ]
        else:
            return [
                {"type": "library", "keyword": None},
                {"type": "university", "keyword": "library"},
            ]
    elif ttype in ["meeting"]:
        return [
            {"type": "cafe", "keyword": None},
            {"type": "restaurant", "keyword": "coffee"},
        ]
    else:
        return [
            {"type": "cafe", "keyword": None},
            {"type": "library", "keyword": None},
        ]


def _google_places_nearby(
    lat: float,
    lon: float,
    radius_m: int,
    profiles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Call Google Places Nearby Search for each profile and merge results.
    """
    api_key = _get_google_key()
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    seen_place_ids = set()
    all_results: List[Dict[str, Any]] = []

    for prof in profiles:
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "key": api_key,
        }
        if prof.get("type"):
            params["type"] = prof["type"]
        if prof.get("keyword"):
            params["keyword"] = prof["keyword"]

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            all_results.append(
                {
                    "error": f"Google Places error for profile {prof}: {e}"
                }
            )
            continue

        status = data.get("status", "")
        if status not in ["OK", "ZERO_RESULTS"]:
            all_results.append(
                {
                    "error": f"Google Places status={status} for profile {prof}"
                }
            )
            continue

        for result in data.get("results", []):
            place_id = result.get("place_id")
            if not place_id or place_id in seen_place_ids:
                continue
            seen_place_ids.add(place_id)
            all_results.append(result)

    return [r for r in all_results if "error" not in r]


def _google_elements_to_places(
    elements: List[Dict[str, Any]],
    user_lat: float,
    user_lon: float,
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert Google Places results into the place dict format used by the UI.
    """
    results: List[Dict[str, Any]] = []
    ttype = (task.get("task_type") or "").lower()

    for el in elements:
        name = el.get("name")
        if not name:
            continue

        geometry = el.get("geometry", {})
        loc = geometry.get("location", {})
        plat = loc.get("lat")
        plon = loc.get("lng")
        if plat is None or plon is None:
            continue

        plat_f = float(plat)
        plon_f = float(plon)

        dist_km = _haversine_km(user_lat, user_lon, plat_f, plon_f)

        types = el.get("types", []) or []
        if "library" in types:
            best_for = "deep focus / study"
        elif "cafe" in types or "restaurant" in types:
            best_for = "casual study / meetings"
        elif "university" in types:
            best_for = "study / group work"
        else:
            best_for = "general work / study"

        notes = f"Suggested for {ttype or 'your task'}."

        results.append(
            {
                "name": name,
                "distance_km": round(dist_km, 2),
                "best_for": best_for,
                "notes": notes,
                "lat": plat_f,
                "lon": plon_f,
            }
        )

    results.sort(key=lambda p: p["distance_km"])
    return results[:8]



def suggest_places_for_task_with_coords(
    task: Dict[str, Any],
    lat: float,
    lon: float,
    radius_km: float,
) -> List[Dict[str, Any]]:
    """
    Suggest nearby places when we already know the user's coordinates.
    Uses Google Places Nearby Search.
    """
    profiles = _infer_place_profiles_for_task(task)
    radius_m = int(radius_km * 1000)

    try:
        elements = _google_places_nearby(lat, lon, radius_m, profiles)
    except Exception as e:
        return [
            {
                "name": "Place search error",
                "distance_km": 0.0,
                "best_for": "N/A",
                "notes": f"Error while calling Google Places: {e}",
            }
        ]

    if not elements:
        return []

    return _google_elements_to_places(elements, lat, lon, task)


def suggest_places_for_task(
    task: Dict[str, Any],
    user_location: str,
    radius_km: float,
) -> List[Dict[str, Any]]:
    """
    Suggest nearby places using:
      - Google Geocoding for the string location
      - Google Places Nearby Search for the actual places
    """
    if not user_location.strip():
        return []

    try:
        lat, lon = _geocode_location(user_location)
    except Exception as e:
        return [
            {
                "name": "Location not found",
                "distance_km": 0.0,
                "best_for": "N/A",
                "notes": f"Could not geocode '{user_location}': {e}",
            }
        ]

    return suggest_places_for_task_with_coords(task, lat, lon, radius_km)