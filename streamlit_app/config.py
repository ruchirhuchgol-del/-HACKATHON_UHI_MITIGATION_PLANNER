
from dataclasses import dataclass
from typing import Dict, Any

# Color palette for all components
COLOR_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "danger": "#d62728",
    "high": "#d62728",
    "medium": "#ff7f0e",
    "low": "#2ca02c",
}

# City configurations
@dataclass
class CityConfig:
    name: str
    center_lat: float
    center_lon: float
    zoom_level: int
    tile_layer: str = "OpenStreetMap"

CITY_CONFIGS = {
    "pune": CityConfig("Pune", 18.5204, 73.8567, 11),
    "nashik": CityConfig("Nashik", 19.9975, 73.7898, 11),
    "default": CityConfig("Default", 18.5204, 73.8567, 11),
}

# Thresholds for vulnerability and temperature
VULNERABILITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}

TEMPERATURE_THRESHOLDS = {
    "high": 40.0,
    "medium": 35.0,
}

# Priority thresholds
PRIORITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}