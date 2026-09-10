from fastapi import APIRouter
from pydantic import BaseModel
from trajectory.camera_manager import list_cameras, add_camera
from trajectory.trajectory_logic import build_trajectory
from trajectory.analytics import get_dashboard_stats

router = APIRouter()


class CameraIn(BaseModel):
    camera_name: str
    latitude: float
    longitude: float


@router.get("/cameras")
def get_cameras():
    return list_cameras()


@router.post("/cameras")
def create_camera(camera: CameraIn):
    return add_camera(camera.camera_name, camera.latitude, camera.longitude)


@router.get("/trajectory/{plate_number}")
def get_trajectory(plate_number: str):
    return build_trajectory(plate_number)


@router.get("/dashboard/stats")
def dashboard_stats():
    return get_dashboard_stats()