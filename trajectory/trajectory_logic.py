from trajectory.db import get_detections_for_plate


def build_trajectory(plate_number: str):
    rows = get_detections_for_plate(plate_number)

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
            "time": row["detected_at"]
        })

    return {"plate": plate_number, "path": path}