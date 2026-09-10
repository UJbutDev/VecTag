from trajectory.db import get_plate_reference, get_detections_by_matched_plate_id
from datetime import datetime
import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_trajectory(plate_number: str):
    ref = get_plate_reference(plate_number)

    if not ref:
        return {"plate": plate_number, "path": [], "vehicle_info": None}

    rows = get_detections_by_matched_plate_id(ref["id"])

    path = []
    for row in rows:
        cam = row.get("cameras")
        if not cam:
            continue
        path.append({
            "camera_id": row["camera_id"],
            "camera_name": cam["camera_name"],
            "latitude": cam["latitude"],
            "longitude": cam["longitude"],
            "time": row["detected_at"],
            "confidence": row.get("confidence")
        })

    total_distance_km = 0.0
    for i in range(1, len(path)):
        total_distance_km += haversine_km(
            path[i - 1]["latitude"], path[i - 1]["longitude"],
            path[i]["latitude"], path[i]["longitude"]
        )

    avg_speed_kmph = None
    if len(path) >= 2:
        try:
            t_start = datetime.fromisoformat(path[0]["time"])
            t_end = datetime.fromisoformat(path[-1]["time"])
            duration_hours = (t_end - t_start).total_seconds() / 3600
            if duration_hours > 0:
                avg_speed_kmph = total_distance_km / duration_hours
        except (ValueError, TypeError):
            avg_speed_kmph = None

    vehicle_info = {
        "owner_name": ref.get("owner_name", "Unknown"),
        "vehicle_type": ref.get("vehicle_type", "Unknown"),
        "status": ref.get("status", "unknown"),
        "first_seen": path[0]["time"] if path else None,
        "last_seen": path[-1]["time"] if path else None,
        "start_location": path[0]["camera_name"] if path else None,
        "end_location": path[-1]["camera_name"] if path else None,
        "total_distance_km": round(total_distance_km, 2),
        "avg_speed_kmph": round(avg_speed_kmph, 1) if avg_speed_kmph is not None else None
    }

    return {"plate": plate_number, "path": path, "vehicle_info": vehicle_info}