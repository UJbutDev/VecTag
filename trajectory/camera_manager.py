from trajectory.db import get_all_cameras, insert_camera


def list_cameras():
    return get_all_cameras()


def add_camera(camera_name: str, latitude: float, longitude: float):
    return insert_camera(camera_name, latitude, longitude)