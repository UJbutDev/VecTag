import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_all_cameras():
    response = supabase.table("cameras").select("*").execute()
    return response.data


def insert_camera(camera_name: str, latitude: float, longitude: float):
    response = supabase.table("cameras").insert({
        "camera_name": camera_name,
        "latitude": latitude,
        "longitude": longitude
    }).execute()
    return response.data


def get_plate_reference(plate_number: str):
    response = (
        supabase.table("plates_reference")
        .select("*")
        .eq("plate_number", plate_number)
        .execute()
    )
    data = response.data
    return data[0] if data else None


def get_detections_by_matched_plate_id(matched_plate_id: int):
    # Only pulls rows the fuzzy-matcher already confirmed belong to this
    # vehicle (match_status = 'matched'), regardless of how noisy the raw
    # OCR text was on any individual sighting.
    response = (
        supabase.table("plate_detections")
        .select("*, cameras(camera_name, latitude, longitude)")
        .eq("matched_plate_id", matched_plate_id)
        .eq("match_status", "matched")
        .order("detected_at")
        .execute()
    )
    return response.data


def get_all_detections_with_status():
    response = (
        supabase.table("plate_detections")
        .select("*, plates_reference(status)")
        .execute()
    )
    return response.data