from trajectory.db import get_all_detections_with_status, get_all_cameras
from collections import Counter


def get_dashboard_stats():
    rows = get_all_detections_with_status()

    total_scans = len(rows)
    total_matches = sum(1 for r in rows if r.get("match_status") == "matched")
    total_alerts = sum(
        1 for r in rows
        if r.get("plates_reference") and r["plates_reference"].get("status") in ("stolen", "blacklisted")
    )

    camera_counts = Counter()
    for r in rows:
        cam_id = r.get("camera_id")
        if cam_id:
            camera_counts[cam_id] += 1

    cameras = {c["id"]: c["camera_name"] for c in get_all_cameras()}
    scans_by_camera = [
        {"camera_name": cameras.get(cam_id, "Unknown"), "count": count}
        for cam_id, count in camera_counts.items()
    ]

    return {
        "total_scans": total_scans,
        "total_matches": total_matches,
        "total_alerts": total_alerts,
        "scans_by_camera": scans_by_camera
    }