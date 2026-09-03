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


def get_detections_for_plate(plate_number: str):
    response = (
        supabase.table("plate_detections")
        .select("*, cameras(camera_name, latitude, longitude)")
        .eq("cleaned_text", plate_number)
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