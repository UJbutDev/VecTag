import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_all_reference_plates():
    """Read-only fetch of plates_reference for fuzzy matching."""
    response = supabase.table("plates_reference").select("*").execute()
    return response.data


def camera_exists(camera_id: int) -> bool:
    """Read-only check against cameras table."""
    response = supabase.table("cameras").select("id").eq("id", camera_id).execute()
    return len(response.data) > 0


def insert_detection(row: dict):
    """Write-only insert into plate_detections."""
    response = supabase.table("plate_detections").insert(row).execute()
    return response.data[0] if response.data else None