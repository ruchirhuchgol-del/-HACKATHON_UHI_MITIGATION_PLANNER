
from __future__ import annotations
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import config and utils
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import numpy as np
from folium.plugins import HeatMap

# Import configuration with error handling
try:
    from config import (
        COLOR_PALETTE, CITY_CONFIGS, 
        VULNERABILITY_THRESHOLDS, TEMPERATURE_THRESHOLDS
    )
except ImportError as e:
    # Define fallback values if import fails
    logging.warning(f"Failed to import config: {e}. Using fallback values.")
    COLOR_PALETTE = {
        "high": "#d32f2f",      # Red for high values
        "medium": "#f57c00",    # Orange for medium values
        "low": "#388e3c",       # Green for low values
        "primary": "#1976d2",   # Blue for primary elements
        "danger": "#d32f2f",    # Red for danger elements
        "warning": "#f57c00",   # Orange for warning elements
        "success": "#388e3c"    # Green for success elements
    }
    
    CITY_CONFIGS = {
        "pune": {"center": [18.5204, 73.8567], "zoom": 11},
        "nashik": {"center": [19.9975, 73.7898], "zoom": 11},
        "default": {"center": [18.5204, 73.8567], "zoom": 11}
    }
    
    VULNERABILITY_THRESHOLDS = {"high": 0.7, "medium": 0.4}
    TEMPERATURE_THRESHOLDS = {"high": 40.0, "medium": 35.0}

# Import utils with error handling
try:
    from utils import safe_html
except ImportError as e:
    logging.warning(f"Failed to import utils: {e}. Using fallback function.")
    def safe_html(text: Any, default: str = "") -> str:
        """Fallback function to convert text to HTML-safe string."""
        if text is None:
            return default
        import html
        return html.escape(str(text))

class MapType(Enum):
    """Enumeration for supported map types."""
    VULNERABILITY = "vulnerability"
    TEMPERATURE = "temperature"
    HYBRID = "hybrid"

class LegendType(Enum):
    """Enumeration for legend types."""
    VULNERABILITY = "vulnerability"
    TEMPERATURE = "temperature"
    CUSTOM = "custom"

@dataclass
class MapConfiguration:
    """Configuration class for map settings."""
    center_lat: float
    center_lon: float
    zoom_level: int
    width: str = "100%"
    height: str = "100%"
    tile_layer: str = "OpenStreetMap"

@dataclass
class MarkerStyle:
    """Style configuration for map markers."""
    color: str
    size: float
    opacity: float = 0.7
    fill_opacity: float = 0.7

class MapRendererError(Exception):
    """Custom exception for MapRenderer errors."""
    pass

class DataValidationError(MapRendererError):
    """Exception raised for data validation errors."""
    pass

class ConfigurationError(MapRendererError):
    """Exception raised for configuration errors."""
    pass

class MapRenderer:
    """
    Professional map renderer for heat vulnerability index visualization.
    """
    
    # Class-level constants
    DEFAULT_CONFIGS = {
        "pune": MapConfiguration(18.5204, 73.8567, 11),
        "nashik": MapConfiguration(19.9975, 73.7898, 11),
        "default": MapConfiguration(18.5204, 73.8567, 11)
    }
    
    VULNERABILITY_THRESHOLDS = VULNERABILITY_THRESHOLDS
    TEMPERATURE_THRESHOLDS = TEMPERATURE_THRESHOLDS
    
    COLOR_SCHEMES = {
        "vulnerability": {"high": COLOR_PALETTE["high"], "medium": COLOR_PALETTE["medium"], "low": COLOR_PALETTE["low"]},
        "temperature": {"high": COLOR_PALETTE["danger"], "medium": COLOR_PALETTE["warning"], "low": COLOR_PALETTE["success"]},
        "default": COLOR_PALETTE["primary"]
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize MapRenderer with optional configuration."""
        self.logger = self._setup_logger()
        self.config = self._load_configuration(config_path)
        self.logger.info("MapRenderer initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with proper configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_configuration(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if config_path and config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                self.logger.info("Using default configuration")
                return {}
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _get_city_config(self, city_name: str) -> MapConfiguration:
        """Get configuration for specified city."""
        city_key = city_name.lower().strip()
        return self.DEFAULT_CONFIGS.get(city_key, self.DEFAULT_CONFIGS["default"])
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate input data for required columns."""
        if data is None:
            raise DataValidationError("Data cannot be None")
        
        if data.empty:
            raise DataValidationError("Data cannot be empty")
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for valid coordinates
        if 'lat' in data.columns and 'lon' in data.columns:
            invalid_coords = data[
                (data['lat'].isna()) | (data['lon'].isna()) |
                (data['lat'] < -90) | (data['lat'] > 90) |
                (data['lon'] < -180) | (data['lon'] > 180)
            ]
            if not invalid_coords.empty:
                self.logger.warning(f"Found {len(invalid_coords)} rows with invalid coordinates")
    
    def create_base_map(
        self,
        city_name: str = "pune",
        custom_config: Optional[MapConfiguration] = None
    ) -> folium.Map:
        """Create a base map with proper configuration."""
        try:
            config = custom_config or self._get_city_config(city_name)
            
            map_obj = folium.Map(
                location=[config.center_lat, config.center_lon],
                zoom_start=config.zoom_level,
                tiles=config.tile_layer,
                width=config.width,
                height=config.height,
                prefer_canvas=True  # Better performance for large datasets
            )
            
            self.logger.debug(f"Base map created for {city_name}")
            return map_obj
            
        except Exception as e:
            raise MapRendererError(f"Failed to create base map: {e}") from e
    
    def _determine_marker_style(
        self,
        value: float,
        style_type: str,
        thresholds: Dict[str, float],
        color_scheme: Dict[str, str]
    ) -> MarkerStyle:
        """Determine marker style based on value and thresholds."""
        if pd.isna(value):
            return MarkerStyle(color=self.COLOR_SCHEMES["default"], size=8)
        
        if value >= thresholds["high"]:
            category = "high"
            size = 12
        elif value >= thresholds["medium"]:
            category = "medium"
            size = 10
        else:
            category = "low"
            size = 8
        
        return MarkerStyle(
            color=color_scheme[category],
            size=size,
            opacity=0.8,
            fill_opacity=0.6
        )
    
    def add_heat_layer(
        self,
        map_obj: folium.Map,
        heat_data: pd.DataFrame,
        value_column: str = "temperature",
        name: str = "Heat Map"
    ) -> folium.Map:
        """Add heat layer to the map with proper error handling."""
        try:
            if heat_data is None or heat_data.empty:
                self.logger.warning("No heat data provided")
                return map_obj
            
            self._validate_data(heat_data, ['lat', 'lon', value_column])
            
            # Prepare heat data
            heat_data_clean = heat_data.dropna(subset=['lat', 'lon', value_column])
            heat_points = [
                [row['lat'], row['lon'], row[value_column]]
                for _, row in heat_data_clean.iterrows()
            ]
            
            if heat_points:
                HeatMap(
                    heat_points,
                    name=name,
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=25,
                    blur=15
                ).add_to(map_obj)
                
                self.logger.info(f"Added heat layer with {len(heat_points)} points")
            else:
                self.logger.warning("No valid heat points to display")
            
            return map_obj
            
        except Exception as e:
            self.logger.error(f"Failed to add heat layer: {e}")
            raise MapRendererError(f"Failed to add heat layer: {e}") from e
    
    def add_marker_layer(
        self,
        map_obj: folium.Map,
        marker_data: pd.DataFrame,
        style_column: Optional[str] = None,
        map_type: MapType = MapType.VULNERABILITY
    ) -> folium.Map:
        """Add marker layer with intelligent styling."""
        try:
            if marker_data is None or marker_data.empty:
                self.logger.warning("No marker data provided")
                return map_obj
            
            self._validate_data(marker_data, ['lat', 'lon'])
            
            # Determine thresholds and color scheme based on map type
            if map_type == MapType.VULNERABILITY:
                thresholds = self.VULNERABILITY_THRESHOLDS
                color_scheme = self.COLOR_SCHEMES["vulnerability"]
            elif map_type == MapType.TEMPERATURE:
                thresholds = self.TEMPERATURE_THRESHOLDS
                color_scheme = self.COLOR_SCHEMES["temperature"]
            else:
                thresholds = {"high": float('inf'), "medium": float('inf')}
                color_scheme = self.COLOR_SCHEMES["default"]
            
            markers_added = 0
            
            for idx, row in marker_data.iterrows():
                try:
                    # Get coordinates
                    lat, lon = row['lat'], row['lon']
                    
                    # Determine style
                    if style_column and style_column in row:
                        style = self._determine_marker_style(
                            row[style_column], map_type.value, thresholds, color_scheme
                        )
                    else:
                        style = MarkerStyle(color=self.COLOR_SCHEMES["default"], size=8)
                    
                    # Create popup content
                    popup_content = self._create_popup_content(row, map_type)
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=style.size,
                        popup=folium.Popup(popup_content, max_width=350),
                        color=style.color,
                        weight=2,
                        fill=True,
                        fillColor=style.color,
                        fillOpacity=style.fill_opacity,
                        opacity=style.opacity
                    ).add_to(map_obj)
                    
                    markers_added += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add marker for row {idx}: {e}")
                    continue
            
            self.logger.info(f"Added {markers_added} markers to map")
            return map_obj
            
        except Exception as e:
            self.logger.error(f"Failed to add marker layer: {e}")
            raise MapRendererError(f"Failed to add marker layer: {e}") from e
    
    def _create_popup_content(self, row: pd.Series, map_type: MapType) -> str:
        """Create comprehensive popup content for markers."""
        # Base content
        neighborhood = row.get('neighborhood', 'Unknown')
        content = f"<div style='font-family: Arial, sans-serif;'>"
        content += f"<h4 style='margin: 0 0 10px 0; color: #2c3e50;'>"
        content += f"{safe_html(neighborhood).replace('_', ' ').title()}</h4>"
        
        # Add relevant metrics based on map type
        if map_type in [MapType.TEMPERATURE, MapType.HYBRID]:
            if 'lst' in row and not pd.isna(row['lst']):
                content += f"<p><b>🌡️ Temperature:</b> {row['lst']:.1f}°C</p>"
        
        if map_type in [MapType.VULNERABILITY, MapType.HYBRID]:
            if 'hvi' in row and not pd.isna(row['hvi']):
                hvi_level = self._get_vulnerability_level(row['hvi'])
                content += f"<p><b>🏠 HVI:</b> {row['hvi']:.3f} ({hvi_level})</p>"
            
            if 'sensitivity' in row and not pd.isna(row['sensitivity']):
                content += f"<p><b>📊 Sensitivity:</b> {row['sensitivity']:.3f}</p>"
            
            if 'adaptive_capacity' in row and not pd.isna(row['adaptive_capacity']):
                content += f"<p><b>🔧 Adaptive Capacity:</b> {row['adaptive_capacity']:.3f}</p>"
        
        # Additional information
        if 'population' in row and not pd.isna(row['population']):
            content += f"<p><b>👥 Population:</b> {int(row['population']):,}</p>"
        
        content += "</div>"
        return content
    
    def _get_vulnerability_level(self, hvi_value: float) -> str:
        """Get vulnerability level description."""
        if hvi_value >= self.VULNERABILITY_THRESHOLDS["high"]:
            return "High Risk"
        elif hvi_value >= self.VULNERABILITY_THRESHOLDS["medium"]:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def add_legend(self, map_obj: folium.Map, legend_type: LegendType) -> folium.Map:
        """Add professional legend to the map."""
        try:
            legend_html = self._generate_legend_html(legend_type)
            if legend_html:
                map_obj.get_root().html.add_child(folium.Element(legend_html))
                self.logger.debug(f"Added {legend_type.value} legend")
            
            return map_obj
            
        except Exception as e:
            self.logger.warning(f"Failed to add legend: {e}")
            return map_obj
    
    def _generate_legend_html(self, legend_type: LegendType) -> str:
        """Generate HTML for legend based on type."""
        base_style = """
        position: fixed; bottom: 50px; left: 50px; width: 180px;
        background-color: rgba(255, 255, 255, 0.95); 
        border: 2px solid #333; border-radius: 8px;
        z-index: 9999; font-size: 13px; padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        """
        
        if legend_type == LegendType.VULNERABILITY:
            return f"""
            <div style="{base_style}">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🏠 Vulnerability Index</h4>
                <div><span style="color: {self.COLOR_SCHEMES['vulnerability']['high']}; font-size: 16px;">●</span> High Risk (≥ 0.7)</div>
                <div><span style="color: {self.COLOR_SCHEMES['vulnerability']['medium']}; font-size: 16px;">●</span> Medium Risk (0.4-0.7)</div>
                <div><span style="color: {self.COLOR_SCHEMES['vulnerability']['low']}; font-size: 16px;">●</span> Low Risk (< 0.4)</div>
            </div>
            """
        elif legend_type == LegendType.TEMPERATURE:
            return f"""
            <div style="{base_style}">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🌡️ Temperature (LST)</h4>
                <div><span style="color: {self.COLOR_SCHEMES['temperature']['high']}; font-size: 16px;">●</span> High (≥ 40°C)</div>
                <div><span style="color: {self.COLOR_SCHEMES['temperature']['medium']}; font-size: 16px;">●</span> Medium (35-40°C)</div>
                <div><span style="color: {self.COLOR_SCHEMES['temperature']['low']}; font-size: 16px;">●</span> Low (< 35°C)</div>
            </div>
            """
        return ""
    
    def render_map(
        self,
        data: pd.DataFrame,
        city_name: str,
        map_type: MapType = MapType.VULNERABILITY,
        include_heat_layer: bool = False,
        render_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Comprehensive map rendering method."""
        try:
            if render_params is None:
                render_params = {"width": 700, "height": 500}
            
            # Create base map
            map_obj = self.create_base_map(city_name)
            
            # Add marker layer
            style_column = None
            if map_type == MapType.VULNERABILITY:
                style_column = 'hvi'
                legend_type = LegendType.VULNERABILITY
            elif map_type == MapType.TEMPERATURE:
                style_column = 'lst'
                legend_type = LegendType.TEMPERATURE
            else:
                legend_type = LegendType.VULNERABILITY  # Default
            
            map_obj = self.add_marker_layer(map_obj, data, style_column, map_type)
            
            # Add heat layer if requested
            if include_heat_layer and style_column:
                map_obj = self.add_heat_layer(map_obj, data, style_column)
            
            # Add legend
            map_obj = self.add_legend(map_obj, legend_type)
            
            # Render map
            folium_static(
                map_obj,
                width=render_params.get("width", 700),
                height=render_params.get("height", 500)
            )
            
            self.logger.info(f"Successfully rendered {map_type.value} map for {city_name}")
            
        except Exception as e:
            self.logger.error(f"Map rendering failed: {e}")
            raise MapRendererError(f"Map rendering failed: {e}") from e
    
    def export_map(
        self,
        data: pd.DataFrame,
        city_name: str,
        output_path: Path,
        map_type: MapType = MapType.VULNERABILITY
    ) -> None:
        """Export map to HTML file."""
        try:
            # Create map
            map_obj = self.create_base_map(city_name)
            
            style_column = 'hvi' if map_type == MapType.VULNERABILITY else 'lst'
            legend_type = LegendType.VULNERABILITY if map_type == MapType.VULNERABILITY else LegendType.TEMPERATURE
            
            map_obj = self.add_marker_layer(map_obj, data, style_column, map_type)
            map_obj = self.add_legend(map_obj, legend_type)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            map_obj.save(str(output_path))
            
            self.logger.info(f"Map exported to {output_path}")
            
        except Exception as e:
            raise MapRendererError(f"Map export failed: {e}") from e

# Convenience functions for backward compatibility
def create_vulnerability_map(data: pd.DataFrame, city_name: str) -> None:
    """Create and render vulnerability map."""
    renderer = MapRenderer()
    renderer.render_map(data, city_name, MapType.VULNERABILITY)

def create_temperature_map(data: pd.DataFrame, city_name: str) -> None:
    """Create and render temperature map."""
    renderer = MapRenderer()
    renderer.render_map(data, city_name, MapType.TEMPERATURE)