# location_module.py

from typing import Dict, Any, List

def suggest_places_for_task(
    task: Dict[str, Any],
    user_location: str,
    radius_km: float,
) -> List[Dict[str, Any]]:
    """
    Inputs:
      task: single task dict (title, type, duration_min, prefer_outside, etc.)
      user_location: text location (e.g., "LAU Beirut")
      radius_km: search radius in kilometers

    Output: list of places like:
    [
      {
        "name": "Quiet Beans Café",
        "distance_km": 0.8,
        "best_for": "deep focus / study",
        "notes": "Usually quiet before 5pm, good Wi-Fi.",
      },
      ...
    ]
    """
    # TODO: implement API call & filtering
    ...
