import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label, center_of_mass
from shapely.geometry import Point
import geopandas as gpd
import json
from datetime import datetime
import traceback

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_processed_data(city_name, combined_metadata_path, base_dir):
    """Load processed LST data for a city using combined metadata"""
    # Load combined metadata
    combined_metadata = pd.read_csv(combined_metadata_path)
    
    # Filter for the specific city
    city_metadata = combined_metadata[combined_metadata['city'] == city_name].copy()
    
    if city_metadata.empty:
        raise FileNotFoundError(f"No metadata found for city: {city_name}")
    
    # Load LST data
    lst_data = []
    for _, row in city_metadata.iterrows():
        try:
            # Construct the full path to the GeoTIFF file
            file_path = base_dir / f"data/processed/{city_name}/processed" / Path(row['file']).name
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            lst = xr.open_dataarray(file_path)
            lst_data.append({
                'date': row['date'],
                'lst': lst,
                'min_temp': row['min_temp'],
                'max_temp': row['max_temp'],
                'mean_temp': row['mean_temp'],
                'file_type': row.get('file_type', 'unknown'),
                'coverage': row.get('coverage', 1.0)
            })
        except Exception as e:
            print(f"Error loading {row['file']}: {str(e)}")
            continue
    
    return city_metadata, lst_data

def analyze_combined_metadata(combined_metadata_path):
    """Perform detailed analysis on the combined metadata CSV"""
    # Load combined metadata
    df = pd.read_csv(combined_metadata_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%j')
    
    # Initialize results dictionary
    results = {
        'summary': {},
        'by_city': {},
        'by_file_type': {},
        'temporal_analysis': {},
        'temperature_stats': {}
    }
    
    # Overall summary
    results['summary'] = {
        'total_files': len(df),
        'cities': df['city'].unique().tolist(),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        },
        'temp_range': {
            'min': float(df['min_temp'].min()),
            'max': float(df['max_temp'].max()),
            'mean': float(df['mean_temp'].mean())
        }
    }
    
    # Analysis by city
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        results['by_city'][city] = {
            'file_count': len(city_data),
            'temp_range': {
                'min': float(city_data['min_temp'].min()),
                'max': float(city_data['max_temp'].max()),
                'mean': float(city_data['mean_temp'].mean())
            },
            'avg_coverage': float(city_data['coverage'].mean()),
            'file_types': city_data['file_type'].value_counts().to_dict()
        }
    
    # Analysis by file type
    for file_type in df['file_type'].unique():
        type_data = df[df['file_type'] == file_type]
        results['by_file_type'][file_type] = {
            'file_count': len(type_data),
            'temp_range': {
                'min': float(type_data['min_temp'].min()),
                'max': float(type_data['max_temp'].max()),
                'mean': float(type_data['mean_temp'].mean())
            },
            'cities': type_data['city'].unique().tolist()
        }
    
    # Temporal analysis
    df_sorted = df.sort_values('date')
    results['temporal_analysis'] = {
        'overall_trends': {
            'min_temp': float(np.polyfit(range(len(df_sorted)), df_sorted['min_temp'], 1)[0]),
            'max_temp': float(np.polyfit(range(len(df_sorted)), df_sorted['max_temp'], 1)[0]),
            'mean_temp': float(np.polyfit(range(len(df_sorted)), df_sorted['mean_temp'], 1)[0])
        }
    }
    
    # Temperature statistics
    results['temperature_stats'] = {
        'correlation_matrix': df[['min_temp', 'max_temp', 'mean_temp', 'coverage']].corr().to_dict(),
        'percentiles': {
            'min_temp': {f'p{p}': float(df['min_temp'].quantile(p/100)) for p in [10, 25, 50, 75, 90]},
            'max_temp': {f'p{p}': float(df['max_temp'].quantile(p/100)) for p in [10, 25, 50, 75, 90]},
            'mean_temp': {f'p{p}': float(df['mean_temp'].quantile(p/100)) for p in [10, 25, 50, 75, 90]}
        }
    }
    
    return results

def generate_combined_plots(combined_metadata_path, output_dir):
    """Generate plots based on combined metadata"""
    # Load combined metadata
    df = pd.read_csv(combined_metadata_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%j')
    
    # Create output directory
    plot_dir = output_dir / "plots/combined"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Temperature trends by city
    plt.figure(figsize=(14, 8))
    for city in df['city'].unique():
        city_data = df[df['city'] == city].sort_values('date')
        plt.plot(city_data['date'], city_data['mean_temp'], 'o-', label=city.title())
    
    plt.title('Mean Temperature Trends by City')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'temp_trends_by_city.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Temperature distribution by city
    plt.figure(figsize=(12, 6))
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        plt.hist(city_data['mean_temp'], alpha=0.6, label=city.title(), bins=20)
    
    plt.title('Mean Temperature Distribution by City')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / 'temp_distribution_by_city.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Temperature correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['min_temp', 'max_temp', 'mean_temp', 'coverage']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Temperature and Coverage Correlation Matrix')
    plt.savefig(plot_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Temperature range by file type
    plt.figure(figsize=(12, 6))
    file_types = df['file_type'].unique()
    x = np.arange(len(file_types))
    width = 0.25
    
    min_temps = [df[df['file_type'] == ft]['min_temp'].mean() for ft in file_types]
    mean_temps = [df[df['file_type'] == ft]['mean_temp'].mean() for ft in file_types]
    max_temps = [df[df['file_type'] == ft]['max_temp'].mean() for ft in file_types]
    
    plt.bar(x - width, min_temps, width, label='Min Temp')
    plt.bar(x, mean_temps, width, label='Mean Temp')
    plt.bar(x + width, max_temps, width, label='Max Temp')
    
    plt.title('Temperature Statistics by File Type')
    plt.xlabel('File Type')
    plt.ylabel('Temperature (°C)')
    plt.xticks(x, file_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / 'temp_by_filetype.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Data coverage over time
    plt.figure(figsize=(12, 6))
    for city in df['city'].unique():
        city_data = df[df['city'] == city].sort_values('date')
        plt.plot(city_data['date'], city_data['coverage'], 'o-', label=city.title())
    
    plt.title('Data Coverage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Coverage (fraction)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'coverage_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plots saved to: {plot_dir}")

def detect_hotspots(lst_array, threshold_percentile=90):
    """Detect hotspots in LST data"""
    try:
        # Calculate threshold
        valid_data = lst_array[~np.isnan(lst_array)]
        if len(valid_data) == 0:
            return [], None
            
        threshold = np.percentile(valid_data, threshold_percentile)
        
        # Create binary mask
        hotspot_mask = lst_array > threshold
        
        # Label connected regions
        labeled_array, num_features = label(hotspot_mask)
        
        # Calculate properties for each hotspot
        hotspots = []
        for i in range(1, num_features + 1):
            hotspot_pixels = lst_array[labeled_array == i]
            centroid = center_of_mass(labeled_array == i)
            
            hotspots.append({
                'id': i,
                'size': np.sum(labeled_array == i),
                'mean_temp': float(np.mean(hotspot_pixels)),
                'max_temp': float(np.max(hotspot_pixels)),
                'centroid': centroid
            })
        
        return hotspots, float(threshold)
    except Exception as e:
        print(f"Error in hotspot detection: {str(e)}")
        return [], None

def analyze_temporal_trends(metadata_df):
    """Analyze temperature trends over time"""
    # Convert date to datetime
    metadata_df['date'] = pd.to_datetime(metadata_df['date'], format='%Y-%j')
    metadata_df = metadata_df.sort_values('date')
    
    # Calculate trends
    trends = {
        'min_temp_trend': float(np.polyfit(range(len(metadata_df)), metadata_df['min_temp'], 1)[0]),
        'max_temp_trend': float(np.polyfit(range(len(metadata_df)), metadata_df['max_temp'], 1)[0]),
        'mean_temp_trend': float(np.polyfit(range(len(metadata_df)), metadata_df['mean_temp'], 1)[0])
    }
    
    return metadata_df, trends

def generate_summary_plots(city_name, metadata_df, lst_data, output_dir):
    """Generate summary plots for the city"""
    try:
        # Create output directory
        plot_dir = output_dir / f"plots/{city_name}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Temperature trends over time
        plt.figure(figsize=(12, 6))
        plt.plot(metadata_df['date'], metadata_df['min_temp'], 'b-', label='Min Temp')
        plt.plot(metadata_df['date'], metadata_df['max_temp'], 'r-', label='Max Temp')
        plt.plot(metadata_df['date'], metadata_df['mean_temp'], 'g-', label='Mean Temp')
        plt.title(f'{city_name.title()} Temperature Trends')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'temperature_trends.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Temperature distribution - using sampling to avoid memory issues
        plt.figure(figsize=(10, 6))
        
        # Sample size per file - adjust based on available memory
        sample_size = 10000
        all_temps = []
        
        for data in lst_data:
            try:
                # Get valid temperature values
                lst_array = data['lst'].values
                valid_temps = lst_array[~np.isnan(lst_array)]
                
                if len(valid_temps) > 0:
                    # If we have more data than sample_size, take a random sample
                    if len(valid_temps) > sample_size:
                        # Use random sampling without replacement
                        sampled_temps = np.random.choice(valid_temps, size=sample_size, replace=False)
                    else:
                        sampled_temps = valid_temps
                    
                    all_temps.extend(sampled_temps)
            except Exception as e:
                print(f"Error sampling temperatures from file: {str(e)}")
                continue
        
        if all_temps:  # Only create histogram if we have data
            plt.hist(all_temps, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{city_name.title()} Temperature Distribution (Sampled)')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(plot_dir / 'temperature_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print("No valid temperature data found for distribution plot")
        
        # 3. Hotspot analysis for most recent data
        if lst_data:
            recent_data = lst_data[-1]['lst']
            hotspots, threshold = detect_hotspots(recent_data.values)
            
            if threshold is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(recent_data.values, cmap='hot', vmin=np.nanmin(recent_data.values), vmax=np.nanmax(recent_data.values))
                plt.colorbar(label='Temperature (°C)')
                plt.title(f'{city_name.title()} Hotspots (Threshold: {threshold:.1f}°C)')
                
                # Mark hotspot centroids
                for hotspot in hotspots[:5]:  # Top 5 hotspots
                    plt.plot(hotspot['centroid'][1], hotspot['centroid'][0], 'wo', markersize=10)
                
                plt.savefig(plot_dir / 'hotspots.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"Plots saved to: {plot_dir}")
        return True
    except Exception as e:
        print(f"Error generating plots for {city_name}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    # Define paths
    base_dir = Path("F:/Urban/UHI_MIGITAGION_PLANNER-")
    combined_metadata_path = base_dir / "data/processed/combined_lst_metadata.csv"
    cities = ['pune', 'nashik']
    
    # First, perform detailed analysis on the combined metadata
    print("\nPerforming detailed analysis on combined metadata...")
    try:
        combined_analysis = analyze_combined_metadata(combined_metadata_path)
        
        # Save combined analysis results
        combined_results_path = base_dir / "data/processed/combined_analysis_results.json"
        with open(combined_results_path, 'w') as f:
            json.dump(combined_analysis, f, indent=2)
        print(f"Combined analysis results saved to: {combined_results_path}")
        
        # Generate combined plots
        generate_combined_plots(combined_metadata_path, base_dir)
        
        # Print summary of combined analysis
        print("\nCombined Metadata Analysis Summary:")
        print(f"Total files processed: {combined_analysis['summary']['total_files']}")
        print(f"Cities analyzed: {', '.join(combined_analysis['summary']['cities'])}")
        print(f"Date range: {combined_analysis['summary']['date_range']['start']} to {combined_analysis['summary']['date_range']['end']}")
        print(f"Overall temperature range: {combined_analysis['summary']['temp_range']['min']:.1f}°C to {combined_analysis['summary']['temp_range']['max']:.1f}°C")
        print(f"Overall mean temperature: {combined_analysis['summary']['temp_range']['mean']:.1f}°C")
        
        print("\nTemperature trends (°C/day):")
        print(f"  Min: {combined_analysis['temporal_analysis']['overall_trends']['min_temp']:.4f}")
        print(f"  Max: {combined_analysis['temporal_analysis']['overall_trends']['max_temp']:.4f}")
        print(f"  Mean: {combined_analysis['temporal_analysis']['overall_trends']['mean_temp']:.4f}")
    except Exception as e:
        print(f"Error in combined metadata analysis: {str(e)}")
        traceback.print_exc()
    
    # Process each city
    for city_name in cities:
        print(f"\n{'='*50}")
        print(f"Analyzing {city_name.title()} LST data...")
        print(f"{'='*50}")
        
        try:
            # Load processed data
            metadata_df, lst_data = load_processed_data(city_name, combined_metadata_path, base_dir)
            print(f"Loaded {len(lst_data)} LST files for {city_name}")
            
            if not lst_data:
                print(f"No valid LST data found for {city_name}")
                continue
            
            # Analyze temporal trends
            metadata_df, trends = analyze_temporal_trends(metadata_df)
            print(f"Temperature trends (°C/day):")
            print(f"  Min: {trends['min_temp_trend']:.4f}")
            print(f"  Max: {trends['max_temp_trend']:.4f}")
            print(f"  Mean: {trends['mean_temp_trend']:.4f}")
            
            # Initialize hotspots and threshold
            hotspots = []
            threshold = None
            
            # Detect hotspots in most recent data
            if lst_data:
                recent_lst = lst_data[-1]['lst']
                hotspots, threshold = detect_hotspots(recent_lst.values)
                
                if threshold is not None:
                    print(f"Detected {len(hotspots)} hotspots (threshold: {threshold:.1f}°C)")
                    
                    # Top 5 hotspots
                    top_hotspots = sorted(hotspots, key=lambda x: x['max_temp'], reverse=True)[:5]
                    print("Top 5 hotspots:")
                    for i, hotspot in enumerate(top_hotspots):
                        print(f"  {i+1}. Size: {hotspot['size']} pixels, Max Temp: {hotspot['max_temp']:.1f}°C")
                else:
                    print("No hotspots detected - threshold could not be calculated")
            
            # Generate summary plots
            plot_success = generate_summary_plots(city_name, metadata_df, lst_data, base_dir)
            
            # Convert trends to Python floats
            trends_python = {k: float(v) for k, v in trends.items()}
            
            # Save analysis results
            results = {
                'city': city_name,
                'num_files': len(lst_data),
                'date_range': f"{metadata_df['date'].min().strftime('%Y-%m-%d')} to {metadata_df['date'].max().strftime('%Y-%m-%d')}",
                'temp_range': f"{metadata_df['min_temp'].min():.1f}°C to {metadata_df['max_temp'].max():.1f}°C",
                'mean_temp': float(metadata_df['mean_temp'].mean()),
                'trends': trends_python,
                'num_hotspots': len(hotspots),
                'hotspot_threshold': threshold,
                'file_types': metadata_df['file_type'].value_counts().to_dict(),
                'avg_coverage': float(metadata_df['coverage'].mean()),
                'plots_generated': plot_success
            }
            
            # Save results
            results_path = base_dir / f"data/processed/{city_name}_analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Analysis results saved to: {results_path}")
            
        except Exception as e:
            print(f"Failed to analyze {city_name}: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()