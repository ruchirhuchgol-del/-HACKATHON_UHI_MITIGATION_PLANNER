import sys
import os
from pathlib import Path
import pandas as pd
# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from modis_processor import MODISProcessor

def main():
    # Initialize processor
    processor = MODISProcessor()
    
    # Define city information (bbox and tiles)
    city_info = {
        'pune': {
            'bbox': (73.7, 18.4, 74.1, 18.6),
            'tiles': ['h25v06']  # Tile that covers Pune
        },
        'nashik': {
            'bbox': (73.7, 19.9, 74.1, 20.1),
            'tiles': ['h25v07']  # Tile that covers Nashik
        }
    }
    
    # Define file paths using pathlib
    base_dir = Path(__file__).parent.parent  # Get project root directory
    links_file = base_dir / "data/raw/Modis_punenashik"
    output_dir = base_dir / "data/processed"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each city
    for city_name, info in city_info.items():
        print(f"\nProcessing {city_name.title()} data...")
        
        try:
            # Create city-specific output directory
            city_output_dir = output_dir / city_name
            city_output_dir.mkdir(exist_ok=True)
            
            # DEBUG: Print the parameters being passed
            print(f"Calling batch_process with:")
            print(f"  links_file: {links_file}")
            print(f"  city_name: {city_name}")
            print(f"  city_bbox: {info['bbox']}")
            print(f"  output_dir: {city_output_dir}")
            print(f"  tiles: {info['tiles']}")
            
            # Process MODIS data - FIXED: Now passing tiles parameter
            result_df = processor.batch_process(
                links_file=links_file,
                city_name=city_name,
                city_bbox=info['bbox'],
                output_dir=city_output_dir,
                tiles=info['tiles']  # Changed from city_tiles to tiles
            )
            
            # Validate result
            if result_df.empty:
                print(f"Warning: No data processed for {city_name}")
                continue
                
            # Save metadata
            metadata_path = city_output_dir / f"{city_name}_metadata.csv"
            result_df.to_csv(metadata_path, index=False)
            
            print(f"Successfully processed {len(result_df)} files for {city_name}")
            print(f"Metadata saved to: {metadata_path}")
            
            # Print summary statistics
            print(f"\nSummary Statistics for {city_name}:")
            print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
            print(f"Temperature range: {result_df['min_temp'].min():.1f}°C to {result_df['max_temp'].max():.1f}°C")
            print(f"Average temperature: {result_df['mean_temp'].mean():.1f}°C")
            
        except FileNotFoundError as e:
            print(f"File not found error for {city_name}: {str(e)}")
        except PermissionError as e:
            print(f"Permission error for {city_name}: {str(e)}")
        except Exception as e:
            print(f"Failed to process {city_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()