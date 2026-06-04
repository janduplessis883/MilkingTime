from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt


@dataclass(frozen=True)
class GeoFence:
    center_lat: float
    center_lon: float
    radius_m: float
    grace_minutes: int


def distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    earth_radius_m = 6_371_000
    d_lat = radians(lat_b - lat_a)
    d_lon = radians(lon_b - lon_a)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(d_lon / 2) ** 2
    )
    return 2 * earth_radius_m * asin(sqrt(a))


def is_inside_geofence(lat: float, lon: float, fence: GeoFence) -> tuple[bool, float]:
    distance = distance_m(lat, lon, fence.center_lat, fence.center_lon)
    return distance <= fence.radius_m, distance

