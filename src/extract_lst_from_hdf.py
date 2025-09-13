import h5py
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
import glob
from tqdm import tqdm
import re
import os
import json
import struct

def check_pyhdf():
    """Check if pyhdf is available and import it"""
    try:
        from pyhdf.SD import SD, SDC
        return SD, SDC
    except ImportError:
        return None, None

def detect_hdf_format(file_path):
    """Detect if a file is HDF4 or HDF5 format"""
    # Read the first 8 bytes to check the signature
    try:
        with open(file_path, 'rb') as f:
            signature = f.read(8)
        
        # HDF5 signature: \211HDF\r\n\032\n
        if signature == b'\211HDF\r\n\032\n':
            return "HDF5"
        
        # HDF4 signature: \016\003\023\001
        elif signature[:4] == b'\016\003\023\001':
            return "HDF4"
        
        # If neither signature matches, try to determine by attempting to open
        else:
            try:
                # Try opening as HDF5
                with h5py.File(file_path, 'r') as f:
                    return "HDF5"
            except (OSError, IOError):
                # If HDF5 fails, try HDF4
                SD, SDC = check_pyhdf()
                if SD:
                    try:
                        with SD(str(file_path), SDC.READ) as hdf:
                            return "HDF4"
                    except Exception:
                        pass
                return "UNKNOWN"
    except Exception as e:
        print(f"Error detecting HDF format: {e}")
        return "UNKNOWN"

def extract_lst_from_hdf5(hdf_path, output_dir):
    """Extract LST data from HDF5 file and save as GeoTIFF"""
    try:
        with h5py.File(hdf_path, 'r') as f:
            print(f"Processing HDF5: {hdf_path.name}")
            
            # Look for LST dataset in common locations
            lst_dataset = None
            lst_path = None
            
            # Common MODIS HDF structure patterns
            possible_paths = [
                'LST_Day_1km',
                'MODIS_Grid_Daily_1km_LST/LST_Day_1km',
                'Data/LST_Day_1km',
                'HDFEOS/GRIDS/MOD_Grid_Daily_1km_LST/Data_Fields/LST_Day_1km',
                'Grid/LST_Day_1km'
            ]
            
            # Try each possible path
            for path in possible_paths:
                try:
                    if path in f:
                        lst_dataset = f[path]
                        lst_path = path
                        print(f"  Found LST at: {path}")
                        break
                except:
                    continue
            
            # If not found in common paths, search recursively
            if lst_dataset is None:
                def find_lst(name, obj):
                    nonlocal lst_dataset, lst_path
                    if isinstance(obj, h5py.Dataset) and 'LST' in name and '1km' in name:
                        lst_dataset = obj
                        lst_path = name
                        return obj
                    return None
                
                f.visititems(find_lst)
            
            if lst_dataset is None:
                print(f"  LST dataset not found in HDF5 structure")
                return None
            
            # Get the data and metadata
            lst_data = lst_dataset[:]
            
            # Get attributes for conversion
            scale_factor = lst_dataset.attrs.get('scale_factor', 0.02)
            add_offset = lst_dataset.attrs.get('add_offset', 0)
            fill_value = lst_dataset.attrs.get('_FillValue', None)
            
            # Print metadata for debugging
            print(f"  Data shape: {lst_data.shape}")
            print(f"  Data type: {lst_data.dtype}")
            print(f"  Scale factor: {scale_factor}")
            print(f"  Add offset: {add_offset}")
            if fill_value is not None:
                print(f"  Fill value: {fill_value}")
            
            # Convert from Kelvin to Celsius
            lst_data = lst_data * scale_factor + add_offset - 273.15
            
            # Handle fill values
            if fill_value is not None:
                fill_value_celsius = fill_value * scale_factor + add_offset - 273.15
                lst_data = np.where(lst_data == fill_value_celsius, np.nan, lst_data)
            
            # Create xarray DataArray
            if len(lst_data.shape) == 2:
                # Create coordinate arrays
                y_coords = np.arange(lst_data.shape[0])
                x_coords = np.arange(lst_data.shape[1])
                
                lst = xr.DataArray(
                    lst_data,
                    dims=['y', 'x'],
                    coords={
                        'y': y_coords,
                        'x': x_coords
                    },
                    name='LST'
                )
                
                # Set spatial dimensions
                lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
                
                # Set CRS (MODIS Sinusoidal projection)
                lst.rio.write_crs("EPSG:32643", inplace=True)
                
                return lst
            else:
                print(f"  Unexpected LST shape: {lst_data.shape}")
                return None
                
    except Exception as e:
        print(f"  Error processing HDF5 {hdf_path.name}: {e}")
        return None

def extract_lst_from_hdf4(hdf_path, output_dir):
    """Extract LST data from HDF4 file and save as GeoTIFF"""
    # Check for pyhdf availability
    SD, SDC = check_pyhdf()
    if not SD:
        print(f"  Cannot process HDF4 file {hdf_path.name}: pyhdf not available")
        print("  Please ensure pyhdf is properly installed")
        print("  Try: pip install --force-reinstall pyhdf")
        return None
    
    try:
        # Open HDF4 file
        hdf_file = SD(str(hdf_path), SDC.READ)
        print(f"Processing HDF4: {hdf_path.name}")
        
        # Get dataset information
        datasets_dict = hdf_file.datasets()
        
        # Look for LST dataset
        lst_dataset = None
        lst_name = None
        
        # Common MODIS LST dataset names
        possible_names = [
            'LST_Day_1km',
            'LST_Night_1km'
        ]
        
        # Try each possible name
        for name in possible_names:
            if name in datasets_dict:
                lst_name = name
                lst_dataset = hdf_file.select(name)
                print(f"  Found LST at: {name}")
                break
        
        # If not found in common names, search for any dataset with 'LST' in name
        if lst_dataset is None:
            for name in datasets_dict.keys():
                if 'LST' in name:
                    lst_name = name
                    lst_dataset = hdf_file.select(name)
                    print(f"  Found LST at: {name}")
                    break
        
        if lst_dataset is None:
            print(f"  LST dataset not found in {hdf_path.name}")
            hdf_file.end()
            return None
        
        # Get the data
        lst_data = lst_dataset.get()
        
        # Get attributes
        attrs = lst_dataset.attributes()
        scale_factor = attrs.get('scale_factor', 0.02)
        add_offset = attrs.get('add_offset', 0)
        fill_value = attrs.get('_FillValue', None)
        
        # Print metadata for debugging
        print(f"  Data shape: {lst_data.shape}")
        print(f"  Data type: {lst_data.dtype}")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Add offset: {add_offset}")
        if fill_value is not None:
            print(f"  Fill value: {fill_value}")
        
        # Convert from Kelvin to Celsius
        lst_data = lst_data * scale_factor + add_offset - 273.15
        
        # Handle fill values
        if fill_value is not None:
            fill_value_celsius = fill_value * scale_factor + add_offset - 273.15
            lst_data = np.where(lst_data == fill_value_celsius, np.nan, lst_data)
        
        # Close the HDF4 file
        hdf_file.end()
        
        # Create xarray DataArray
        if len(lst_data.shape) == 2:
            # Create coordinate arrays
            y_coords = np.arange(lst_data.shape[0])
            x_coords = np.arange(lst_data.shape[1])
            
            lst = xr.DataArray(
                lst_data,
                dims=['y', 'x'],
                coords={
                    'y': y_coords,
                    'x': x_coords
                },
                name='LST'
            )
            
            # Set spatial dimensions
            lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
            
            # Set CRS (MODIS Sinusoidal projection)
            lst.rio.write_crs("EPSG:32643", inplace=True)
            
            return lst
        else:
            print(f"  Unexpected LST shape: {lst_data.shape}")
            return None
            
    except Exception as e:
        print(f"  Error processing HDF4 {hdf_path.name}: {e}")
        return None

def extract_lst_from_hdf(hdf_path, output_dir):
    """Extract LST data from HDF file (either HDF4 or HDF5)"""
    # First detect the file format
    file_format = detect_hdf_format(hdf_path)
    print(f"Detected format: {file_format} for {hdf_path.name}")
    
    if file_format == "HDF5":
        return extract_lst_from_hdf5(hdf_path, output_dir)
    elif file_format == "HDF4":
        return extract_lst_from_hdf4(hdf_path, output_dir)
    else:
        print(f"  Unknown HDF format for {hdf_path.name}")
        return None

def extract_lst_from_xml(xml_path, output_dir):
    """Extract LST data from CRM.XML file"""
    try:
        print(f"Processing XML: {xml_path.name}")
        
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Initialize variables to store LST data
        lst_data = None
        scale_factor = 0.02
        add_offset = 0
        fill_value = None
        
        # Look for LST data in XML structure
        # This is a generic approach - actual structure may vary
        for elem in root.iter():
            # Check for LST-related tags
            if 'LST' in elem.tag or 'Temperature' in elem.tag:
                # Try to get value as text
                if elem.text and elem.text.strip():
                    try:
                        # Try to parse as JSON if it's a complex structure
                        try:
                            data = json.loads(elem.text)
                            if isinstance(data, dict) and 'values' in data:
                                lst_data = np.array(data['values'])
                                if 'scale_factor' in data:
                                    scale_factor = data['scale_factor']
                                if 'add_offset' in data:
                                    add_offset = data['add_offset']
                                if 'fill_value' in data:
                                    fill_value = data['fill_value']
                                break
                        except json.JSONDecodeError:
                            # Not JSON, try to parse as simple value
                            value = float(elem.text)
                            # Create a 1x1 array for single values
                            lst_data = np.array([[value]])
                            break
                    except ValueError:
                        pass
            
            # Check for attributes that might contain LST data
            for attr_name, attr_value in elem.attrib.items():
                if 'LST' in attr_name or 'Temperature' in attr_name:
                    try:
                        value = float(attr_value)
                        lst_data = np.array([[value]])
                        break
                    except ValueError:
                        pass
        
        if lst_data is None:
            print(f"  No LST data found in {xml_path.name}")
            return None
        
        # Convert from Kelvin to Celsius if needed
        # Check if values are in Kelvin (typically > 200)
        if np.nanmean(lst_data) > 200:
            lst_data = lst_data - 273.15
        
        # Create xarray DataArray
        if len(lst_data.shape) == 2:
            # Create coordinate arrays
            y_coords = np.arange(lst_data.shape[0])
            x_coords = np.arange(lst_data.shape[1])
            
            lst = xr.DataArray(
                lst_data,
                dims=['y', 'x'],
                coords={
                    'y': y_coords,
                    'x': x_coords
                },
                name='LST'
            )
            
            # Set spatial dimensions
            lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
            
            # Set CRS (MODIS Sinusoidal projection)
            lst.rio.write_crs("EPSG:32643", inplace=True)
            
            return lst
        else:
            print(f"  Unexpected LST shape: {lst_data.shape}")
            return None
            
    except Exception as e:
        print(f"  Error processing {xml_path.name}: {e}")
        return None

def extract_lst_from_jpg(jpg_path, output_dir):
    """Extract LST data from JPG file (thermal image)"""
    try:
        print(f"Processing JPG: {jpg_path.name}")
        
        # Open the image
        img = Image.open(jpg_path)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # If it's a color image, convert to grayscale
        if len(img_array.shape) == 3:
            # Convert to grayscale using luminosity method
            img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Normalize to 0-1 range
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Convert to temperature using a simple linear approximation
        # This is a basic approach - actual calibration would require more information
        # Assuming the image represents temperatures in a range (e.g., 20-45°C)
        min_temp = 20.0
        max_temp = 45.0
        lst_data = min_temp + img_array * (max_temp - min_temp)
        
        # Create xarray DataArray
        y_coords = np.arange(lst_data.shape[0])
        x_coords = np.arange(lst_data.shape[1])
        
        lst = xr.DataArray(
            lst_data,
            dims=['y', 'x'],
            coords={
                'y': y_coords,
                'x': x_coords
            },
            name='LST'
        )
        
        # Set spatial dimensions
        lst = lst.rio.set_spatial_dims(x_dim='x', y_dim='y')
        
        # Set CRS (MODIS Sinusoidal projection)
        lst.rio.write_crs("EPSG:32643", inplace=True)
        
        return lst
        
    except Exception as e:
        print(f"  Error processing {jpg_path.name}: {e}")
        return None

def extract_date_from_filename(filename):
    """Extract date from MODIS filename"""
    # Format: MOD11A2.A2025217.h25v06.061.2025227044159.hdf
    match = re.search(r'\.A(\d{7})\.', filename)
    if match:
        year = match.group(1)[:4]
        doy = match.group(1)[4:]
        return f"{year}-{doy}"
    return "unknown"

def process_city_files(city_name):
    """Process all files (HDF, XML, JPG) for a specific city"""
    # Define paths
    raw_dir = Path(f"F:/Urban/UHI_MIGITAGION_PLANNER-/data/processed/{city_name}/raw")
    processed_dir = Path(f"F:/Urban/UHI_MIGITAGION_PLANNER-/data/processed/{city_name}/processed")
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files
    all_files = list(raw_dir.glob("*"))
    print(f"\nFound {len(all_files)} files for {city_name}")
    
    if not all_files:
        print(f"No files found in {raw_dir}")
        return None
    
    # Process each file
    results = []
    success_count = 0
    
    for file_path in tqdm(all_files, desc=f"Processing {city_name} files"):
        try:
            # Extract date from filename
            date = extract_date_from_filename(file_path.name)
            
            # Process based on file extension
            if file_path.suffix.lower() == '.hdf':
                lst = extract_lst_from_hdf(file_path, processed_dir)
                file_type = 'HDF'
            elif file_path.suffix.lower() == '.xml':
                lst = extract_lst_from_xml(file_path, processed_dir)
                file_type = 'XML'
            elif file_path.suffix.lower() == '.jpg':
                lst = extract_lst_from_jpg(file_path, processed_dir)
                file_type = 'JPG'
            else:
                print(f"  Skipping unsupported file: {file_path.name}")
                continue
            
            if lst is not None:
                # Create output filename
                output_filename = f"{city_name}_lst_{date}_{file_type.lower()}.tif"
                output_path = processed_dir / output_filename
                
                # Save as GeoTIFF
                lst.rio.to_raster(str(output_path))
                
                # Calculate statistics
                valid_data = lst.values[~np.isnan(lst.values)]
                if len(valid_data) > 0:
                    result = {
                        'file': str(output_path),
                        'original_file': str(file_path),
                        'date': date,
                        'file_type': file_type,
                        'min_temp': float(np.min(valid_data)),
                        'max_temp': float(np.max(valid_data)),
                        'mean_temp': float(np.mean(valid_data)),
                        'std_temp': float(np.std(valid_data)),
                        'valid_pixels': int(len(valid_data)),
                        'total_pixels': int(lst.values.size),
                        'coverage': float(len(valid_data)) / lst.values.size
                    }
                    results.append(result)
                    success_count += 1
                    print(f"  Saved: {output_filename}")
                else:
                    print(f"  No valid LST data in {file_path.name}")
                
        except Exception as e:
            print(f"  Failed to process {file_path.name}: {e}")
    
    # Save metadata
    if results:
        df = pd.DataFrame(results)
        metadata_path = processed_dir / f"{city_name}_lst_metadata.csv"
        df.to_csv(metadata_path, index=False)
        print(f"\nSaved metadata to: {metadata_path}")
        print(f"Successfully processed {success_count}/{len(all_files)} files for {city_name}")
        
        # Print summary statistics
        print(f"\nSummary for {city_name}:")
        print(f"  Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
        print(f"  Mean temperature: {df['mean_temp'].mean():.1f}°C ± {df['mean_temp'].std():.1f}°C")
        print(f"  Average data coverage: {df['coverage'].mean()*100:.1f}%")
        
        # Print file type breakdown
        file_type_counts = df['file_type'].value_counts()
        print(f"  File types processed:")
        for file_type, count in file_type_counts.items():
            print(f"    {file_type}: {count} files")
        
        return df
    else:
        print(f"No files were processed successfully for {city_name}")
        return None

def create_combined_metadata():
    """Create combined metadata file for both cities"""
    combined_results = []
    
    for city in ['pune', 'nashik']:
        metadata_path = Path(f"F:/Urban/UHI_MIGITAGION_PLANNER-/data/processed/{city}/processed/{city}_lst_metadata.csv")
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            df['city'] = city
            combined_results.append(df)
    
    if combined_results:
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_path = Path("F:/Urban/UHI_MIGITAGION_PLANNER-/data/processed/combined_lst_metadata.csv")
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined metadata saved to: {combined_path}")
        
        # Print combined summary
        print(f"\nCombined Summary:")
        print(f"  Total files processed: {len(combined_df)}")
        print(f"  Temperature range: {combined_df['min_temp'].min():.1f}°C to {combined_df['max_temp'].max():.1f}°C")
        print(f"  Mean temperature: {combined_df['mean_temp'].mean():.1f}°C ± {combined_df['mean_temp'].std():.1f}°C")
        
        # Print city and file type breakdown
        city_counts = combined_df['city'].value_counts()
        file_type_counts = combined_df['file_type'].value_counts()
        
        print(f"  Files by city:")
        for city, count in city_counts.items():
            print(f"    {city}: {count} files")
        
        print(f"  Files by type:")
        for file_type, count in file_type_counts.items():
            print(f"    {file_type}: {count} files")
        
        return combined_df
    else:
        print("No metadata files found to combine")
        return None

def main():
    """Main processing function"""
    print("Starting LST extraction from MODIS files (HDF, XML, JPG)...")
    
    # Check for pyhdf availability
    SD, SDC = check_pyhdf()
    if not SD:
        print("\nWarning: pyhdf library not available.")
        print("HDF4 files will not be processed unless you install pyhdf:")
        print("pip install --force-reinstall pyhdf")
    
    # Process each city
    all_results = []
    for city in ['pune', 'nashik']:
        print(f"\n{'='*50}")
        print(f"Processing {city.upper()}")
        print(f"{'='*50}")
        
        result = process_city_files(city)
        if result is not None:
            all_results.append(result)
    
    # Create combined metadata
    combined_metadata = create_combined_metadata()
    
    # Final summary
    print(f"\n{'='*50}")
    print("PROCESSING COMPLETE")
    print(f"{'='*50}")
    
    if all_results:
        total_files = sum(len(df) for df in all_results)
        print(f"Total files processed: {total_files}")
        
        # Overall temperature statistics
        all_temps = []
        for df in all_results:
            all_temps.extend(df['mean_temp'].tolist())
        
        print(f"Overall temperature range: {min(all_temps):.1f}°C to {max(all_temps):.1f}°C")
        print(f"Overall mean temperature: {np.mean(all_temps):.1f}°C ± {np.std(all_temps):.1f}°C")
        
        print(f"\nOutput directories:")
        for city in ['pune', 'nashik']:
            processed_dir = Path(f"F:/Urban/UHI_MIGITAGION_PLANNER-/data/processed/{city}/processed")
            tif_files = list(processed_dir.glob("*.tif"))
            print(f"  {city}: {len(tif_files)} GeoTIFF files")
            print(f"    Directory: {processed_dir}")
    
    print("\nLST extraction complete!")

if __name__ == "__main__":
    main()