import earthaccess
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import re
import requests

class MODISProcessor:
    def __init__(self, target_crs="EPSG:4326"):
        self.auth = earthaccess.login()
        self.target_crs = target_crs
        
    # NEW METHOD: Extract tile identifier from URL
    def extract_tile(self, url):
        """Extract tile (e.g., h25v07) from MODIS URL"""
        match = re.search(r'\.(h\d{2}v\d{2})\.', url)
        if match:
            return match.group(1)
        return None
    
    def download_hdf(self, url, output_dir):
        """Download HDF file from NASA Earthdata with validation"""
        try:
            session = self.auth.get_session()
            response = session.get(url, stream=True)
            response.raise_for_status()  # Check for download errors
            
            filename = url.split('/')[-1]
            filepath = Path(output_dir) / filename
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return filepath
        except Exception as e:
            raise Exception(f"Download failed for {url}: {str(e)}")
    
    def process_hdf(self, hdf_path, city_bbox):
        """Process HDF file to extract LST data with proper dimension handling"""
        try:
            # Open HDF file
            ds = xr.open_dataset(hdf_path, engine='netcdf4')
            
            # Extract LST and convert to Celsius
            lst = ds['LST_Day_1km'] * 0.02 - 273.15
            
            # DEBUG: Print dimension names to understand structure
            print(f"Original dimensions: {lst.dims}")
            print(f"Original coords: {lst.coords}")
            
            # FIX: Handle MODIS dimension names
            # MODIS often uses non-standard dimension names
            dim_mapping = {}
            
            # Check for common MODIS dimension patterns
            if 'YDim' in lst.dims and 'XDim' in lst.dims:
                dim_mapping = {'YDim': 'y', 'XDim': 'x'}
            elif 'lat' in lst.dims and 'lon' in lst.dims:
                dim_mapping = {'lat': 'y', 'lon': 'x'}
            elif 'Latitude' in lst.dims and 'Longitude' in lst.dims:
                dim_mapping = {'Latitude': 'y', 'Longitude': 'x'}
            elif 'dim_0' in lst.dims and 'dim_1' in lst.dims:
                dim_mapping = {'dim_0': 'y', 'dim_1': 'x'}
            
            # Rename dimensions if mapping found
            if dim_mapping:
                lst = lst.rename(dim_mapping)
                print(f"Renamed dimensions to: {lst.dims}")
            
            # Set spatial dimensions explicitly
            if 'y' in lst.dims and 'x' in lst.dims:
                lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
            else:
                # If standard dimensions not found, try to infer
                print("Warning: Standard dimensions not found, attempting to infer...")
                
                # Look for coordinate variables
                coord_vars = [var for var in lst.coords if var in lst.dims]
                if len(coord_vars) >= 2:
                    # Assume first is y, second is x
                    lst = lst.rename({coord_vars[0]: 'y', coord_vars[1]: 'x'})
                    lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
                else:
                    raise Exception("Could not identify spatial dimensions")
            
            # Set CRS (MODIS Sinusoidal projection)
            lst.rio.write_crs("EPSG:32643", inplace=True)
            
            # Create city boundary GeoDataFrame
            bbox = box(*city_bbox)
            gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326")
            
            # Clip to city boundary
            city_lst = lst.rio.clip(gdf.geometry)
            
            return city_lst
        except Exception as e:
            print(f"Detailed error info:")
            print(f"  Dataset variables: {list(ds.data_vars.keys())}")
            print(f"  Dataset dimensions: {ds.dims}")
            print(f"  Dataset coords: {list(ds.coords.keys())}")
            if 'LST_Day_1km' in ds:
                lst_var = ds['LST_Day_1km']
                print(f"  LST dimensions: {lst_var.dims}")
                print(f"  LST coords: {list(lst_var.coords.keys())}")
            raise Exception(f"Processing failed for {hdf_path}: {str(e)}")
    
    def extract_date(self, filename):
        """Extract date from MODIS filename using regex"""
        # Example filename: MOD11A1.A2020150.h25v06.061.2020152032535.hdf
        match = re.search(r'\.A(\d{7})\.', filename)
        if match:
            year = match.group(1)[:4]
            doy = match.group(1)[4:]
            return f"{year}-{doy}"  # Format: YYYY-DOY
        return "unknown_date"
    
    # UPDATED METHOD: Now accepts tiles parameter and uses tile-based filtering
    def batch_process(self, links_file, city_name, city_bbox, output_dir, tiles=None):
        """Process multiple HDF files for a city"""
        # Read links
        try:
            with open(links_file, 'r') as f:
                links = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            raise Exception(f"Failed to read links file: {str(e)}")
        
        # NEW: Debug all unique tiles found in the links file
        all_tiles = set()
        for link in links:
            tile = self.extract_tile(link)
            if tile:
                all_tiles.add(tile)
        print(f"Found {len(all_tiles)} unique tiles: {sorted(all_tiles)}")
        
        # UPDATED: Filter by tiles if provided
        if tiles:
            print(f"Filtering by tiles: {tiles}")
            city_links = []
            for link in links:
                tile = self.extract_tile(link)
                if tile in tiles:
                    city_links.append(link)
        else:
            # Fallback to original method (city name in URL)
            city_links = [link for link in links if city_name.lower() in link.lower()]
        
        if not city_links:
            raise Exception(f"No links found for city: {city_name}. Available tiles: {sorted(all_tiles)}")
        
        # Create output directories
        raw_dir = Path(output_dir) / "raw"
        processed_dir = Path(output_dir) / "processed"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Process each file
        processed_data = []
        for link in tqdm(city_links, desc=f"Processing {city_name}"):
            try:
                # Download
                hdf_path = self.download_hdf(link, raw_dir)
                
                # Process
                lst = self.process_hdf(hdf_path, city_bbox)
                
                # Save processed data
                filename = Path(link).name
                date = self.extract_date(filename)
                tile = self.extract_tile(link)
                output_path = processed_dir / f"{city_name}_lst_{date}_{tile}.tif"
                lst.rio.to_raster(str(output_path))
                
                # Store metadata
                processed_data.append({
                    'file': str(output_path),
                    'date': date,
                    'tile': tile,
                    'min_temp': float(lst.min().values),
                    'max_temp': float(lst.max().values),
                    'mean_temp': float(lst.mean().values)
                })
                
            except Exception as e:
                print(f"Error processing {link}: {str(e)}")
                continue
        
        if not processed_data:
            raise Exception("No files were successfully processed")
            
        return pd.DataFrame(processed_data)