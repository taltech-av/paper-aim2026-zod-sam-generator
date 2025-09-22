#!/usr/bin/env python3
"""
ZOD Dataset Time-of-Day + Weather Combination Analyzer

This script analyzes metadata.json files from the ZOD dataset to count combinations
of time_of_day and weather conditions.

Usage:
    python3 analyze_time_weather_combinations.py <dataset_path> [--workers N]

Example:
    python3 analyze_time_weather_combinations.py /path/to/zod/single_frames --workers 8
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Any
import re

class TimeWeatherAnalyzer:
    def __init__(self, dataset_path: str, max_workers: int = 4):
        self.dataset_path = Path(dataset_path)
        self.max_workers = max_workers
        
        # Statistics storage
        self.combination_counts = defaultdict(int)
        self.detailed_stats = {
            'time_of_day_counts': defaultdict(int),
            'weather_counts': defaultdict(int),
            'country_counts': defaultdict(int),
            'city_counts': defaultdict(int),
            'frame_years': defaultdict(int),
            'missing_fields': defaultdict(int),
            
            # New enhanced statistics
            'seasonal_patterns': defaultdict(lambda: defaultdict(int)),  # season -> weather -> count
            'monthly_patterns': defaultdict(lambda: defaultdict(int)),   # month -> weather -> count
            'hourly_patterns': defaultdict(lambda: defaultdict(int)),    # hour -> weather -> count
            'visibility_distance': {'values': [], 'by_weather': defaultdict(list)},
            'road_surface_conditions': defaultdict(int),
            'extreme_weather_events': defaultdict(int),
            'geographic_clusters': defaultdict(lambda: defaultdict(int)),  # region -> weather -> count
            'weather_transitions': defaultdict(int),  # previous_weather -> current_weather
            'temperature_ranges': defaultdict(list),
            'precipitation_intensity': defaultdict(int),
            'wind_conditions': defaultdict(int),
            'challenging_conditions': defaultdict(int),  # combinations that are challenging for AV
            'daylight_quality': defaultdict(int),  # dawn/dusk/midday quality analysis
        }
        
        # Progress tracking
        self.total_files_found = 0
        self.total_files_processed = 0
        self.total_files_failed = 0
        self.start_time = 0
        self.last_progress_update = 0
        self.progress_lock = Lock()
        
        # Enhanced analysis storage
        self.previous_weather = None  # For weather transition analysis
        self.geographic_regions = {
            'nordic': ['sweden', 'norway', 'denmark', 'finland'],
            'central_europe': ['germany', 'poland', 'czech', 'austria'],
            'southern_europe': ['italy', 'spain', 'france', 'greece'],
            'uk_ireland': ['united kingdom', 'ireland', 'scotland', 'wales'],
            'eastern_europe': ['russia', 'ukraine', 'belarus', 'romania']
        }
        
    def normalize_weather(self, weather_str: str) -> str:
        """Normalize weather strings to standard categories"""
        if not weather_str:
            return 'unknown'
            
        weather_lower = weather_str.lower().strip()
        
        # Define weather category mappings
        weather_mappings = {
            'clear': ['clear', 'sunny', 'sun'],
            'cloudy': ['cloudy', 'cloud', 'overcast', 'partly cloudy', 'mostly cloudy'],
            'rain': ['rain', 'rainy', 'drizzle', 'shower', 'precipitation'],
            'snow': ['snow', 'snowy', 'snowfall', 'blizzard'],
            'fog': ['fog', 'foggy', 'mist', 'misty', 'haze', 'hazy'],
            'windy': ['wind', 'windy', 'gusty', 'breezy']
        }
        
        # Check each category
        for category, keywords in weather_mappings.items():
            if any(keyword in weather_lower for keyword in keywords):
                return category
                
        return 'unknown'

    def get_season_from_timestamp(self, timestamp_str: str) -> str:
        """Extract season from timestamp"""
        try:
            # Try to parse timestamp and extract month
            import re
            month_match = re.search(r'[-_](\d{2})[-_]', timestamp_str)
            if month_match:
                month = int(month_match.group(1))
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                elif month in [9, 10, 11]:
                    return 'autumn'
        except:
            pass
        return 'unknown'
    
    def extract_hour_from_timestamp(self, timestamp_str: str) -> int:
        """Extract hour from timestamp"""
        try:
            # Look for hour patterns in timestamp
            hour_match = re.search(r'T(\d{2}):', timestamp_str)
            if hour_match:
                return int(hour_match.group(1))
            # Alternative pattern
            hour_match = re.search(r'_(\d{2})\d{2}\d{2}', timestamp_str)
            if hour_match:
                return int(hour_match.group(1))
        except:
            pass
        return -1
    
    def categorize_geographic_region(self, country: str) -> str:
        """Categorize country into geographic regions"""
        if not country or country == 'unknown':
            return 'unknown'
        
        country_lower = country.lower()
        for region, countries in self.geographic_regions.items():
            if any(c in country_lower for c in countries):
                return region
        return 'other'
    
    def analyze_challenging_conditions(self, time_of_day: str, weather: str, 
                                     visibility: float = None) -> str:
        """Identify challenging conditions for autonomous vehicles"""
        challenges = []
        
        # Weather-based challenges
        if weather in ['rain', 'heavy_rain', 'drizzle']:
            challenges.append('wet_road')
        if weather in ['snow', 'heavy_snow', 'sleet']:
            challenges.append('winter_driving')
        if weather in ['fog', 'mist', 'haze']:
            challenges.append('low_visibility')
        
        # Time-based challenges
        if time_of_day in ['dawn', 'dusk']:
            challenges.append('transition_lighting')
        elif time_of_day == 'night':
            challenges.append('night_driving')
        
        # Visibility challenges
        if visibility and visibility < 100:  # meters
            challenges.append('very_low_visibility')
        elif visibility and visibility < 500:
            challenges.append('reduced_visibility')
        
        # Combined challenges
        if time_of_day == 'night' and weather in ['rain', 'snow', 'fog']:
            challenges.append('night_adverse_weather')
        
        return '+'.join(challenges) if challenges else 'normal'
    
    def find_metadata_files(self) -> List[Path]:
        """Find all metadata.json files in the dataset"""
        print(f"Searching for metadata.json files in: {self.dataset_path}")
        
        metadata_files = []
        count = 0
        
        for root, dirs, files in os.walk(self.dataset_path):
            if 'metadata.json' in files:
                metadata_files.append(Path(root) / 'metadata.json')
                count += 1
                
                # Progress indicator for large datasets
                if count % 10000 == 0:
                    print(f"Found {count:,} metadata.json files so far...")
        
        print(f"Total metadata.json files found: {len(metadata_files):,}")
        return metadata_files
    
    def parse_metadata_file(self, file_path: Path) -> Optional[Dict]:
        """Parse a single metadata.json file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                return metadata
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            return None
        except Exception as e:
            return None
    
    def update_statistics(self, metadata: Dict, file_path: Path):
        """Update statistics with metadata from one file"""
        # Extract fields
        time_of_day = metadata.get('time_of_day', 'unknown')
        weather_raw = metadata.get('scraped_weather', '')
        weather = self.normalize_weather(weather_raw)
        
        # Create combination key
        combination_key = f"{time_of_day}_{weather}"
        
        # Extract additional metadata for enhanced analysis
        timestamp = metadata.get('frame', {}).get('name', '') or metadata.get('timestamp', '')
        country = metadata.get('country', 'unknown')
        city = metadata.get('city', 'unknown')
        
        # Enhanced temporal analysis
        season = self.get_season_from_timestamp(timestamp)
        hour = self.extract_hour_from_timestamp(timestamp)
        geographic_region = self.categorize_geographic_region(country)
        
        # Challenging conditions analysis
        visibility = metadata.get('visibility_distance_m')  # if available
        challenging_condition = self.analyze_challenging_conditions(time_of_day, weather, visibility)
        
        # Thread-safe updates
        with self.progress_lock:
            # Basic statistics
            self.combination_counts[combination_key] += 1
            self.detailed_stats['time_of_day_counts'][time_of_day] += 1
            self.detailed_stats['weather_counts'][weather] += 1
            self.detailed_stats['country_counts'][country] += 1
            self.detailed_stats['city_counts'][city] += 1
            
            # Enhanced statistics
            self.detailed_stats['seasonal_patterns'][season][weather] += 1
            if hour >= 0:  # Valid hour extracted
                self.detailed_stats['hourly_patterns'][hour][weather] += 1
            self.detailed_stats['geographic_clusters'][geographic_region][weather] += 1
            self.detailed_stats['challenging_conditions'][challenging_condition] += 1
            
            # Weather transitions (if we have previous weather)
            if self.previous_weather:
                transition_key = f"{self.previous_weather}->{weather}"
                self.detailed_stats['weather_transitions'][transition_key] += 1
            self.previous_weather = weather
            
            # Visibility analysis
            if visibility is not None:
                self.detailed_stats['visibility_distance']['values'].append(visibility)
                self.detailed_stats['visibility_distance']['by_weather'][weather].append(visibility)
            
            # Extreme weather detection
            if weather in ['snow', 'fog'] or 'heavy' in weather_raw.lower():
                self.detailed_stats['extreme_weather_events'][weather] += 1
            
            # Daylight quality analysis
            if time_of_day in ['dawn', 'dusk']:
                self.detailed_stats['daylight_quality']['transition_periods'] += 1
            elif time_of_day == 'day' and weather == 'clear':
                self.detailed_stats['daylight_quality']['optimal'] += 1
            elif time_of_day == 'night':
                self.detailed_stats['daylight_quality']['night_driving'] += 1
            
            # Road surface condition inference
            if weather in ['rain', 'snow']:
                self.detailed_stats['road_surface_conditions']['wet_slippery'] += 1
            elif weather in ['clear', 'cloudy']:
                self.detailed_stats['road_surface_conditions']['dry'] += 1
            
            # Extract year from frame info if available
            frame_id = metadata.get('frame', {}).get('name', '')
            if frame_id:
                # Try to extract year from various possible formats
                year_match = re.search(r'(20\d{2})', str(frame_id))
                if year_match:
                    year = year_match.group(1)
                    self.detailed_stats['frame_years'][year] += 1
                    # Monthly analysis from timestamp
                    month_match = re.search(r'[-_](\d{2})[-_]', timestamp)
                    if month_match:
                        month = int(month_match.group(1))
                        self.detailed_stats['monthly_patterns'][month][weather] += 1
                else:
                    self.detailed_stats['frame_years']['unknown'] += 1
            else:
                self.detailed_stats['frame_years']['unknown'] += 1
    
    def update_progress(self, processed: int = 1, failed: int = 0):
        """Thread-safe progress update"""
        with self.progress_lock:
            self.total_files_processed += processed
            self.total_files_failed += failed
            
            current_time = time.time()
            # Update progress every 5 seconds
            if current_time - self.last_progress_update > 5:
                self.print_progress()
                self.last_progress_update = current_time
    
    def print_progress(self):
        """Print current progress"""
        if self.total_files_found == 0:
            return
            
        elapsed = time.time() - self.start_time
        processed = self.total_files_processed + self.total_files_failed
        progress_pct = (processed / self.total_files_found) * 100
        
        if processed > 0:
            rate = processed / elapsed
            remaining = self.total_files_found - processed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            rate = 0
            eta_str = "Unknown"
        
        print(f"Progress: {processed:,}/{self.total_files_found:,} "
              f"({progress_pct:.1f}%) | "
              f"Rate: {rate:.1f} files/sec | "
              f"ETA: {eta_str} | "
              f"Errors: {self.total_files_failed}")
    
    def process_files_batch(self, file_paths: List[Path]) -> None:
        """Process a batch of metadata files"""
        for file_path in file_paths:
            metadata = self.parse_metadata_file(file_path)
            
            if metadata:
                self.update_statistics(metadata, file_path)
                self.update_progress(processed=1)
            else:
                self.update_progress(failed=1)
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the entire dataset using multiple workers
        """
        print(f"=== ZOD Dataset Time-of-Day + Weather Analysis ===")
        
        # Find all metadata files
        metadata_files = self.find_metadata_files()
        
        if not metadata_files:
            print("No metadata.json files found!")
            return {}
        
        self.total_files_found = len(metadata_files)
        self.start_time = time.time()
        self.last_progress_update = self.start_time
        
        print(f"Starting analysis with {self.max_workers} worker threads...")
        
        # Split files into batches for workers
        batch_size = max(1, len(metadata_files) // (self.max_workers * 4))
        file_batches = [
            metadata_files[i:i + batch_size] 
            for i in range(0, len(metadata_files), batch_size)
        ]
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_files_batch, batch)
                for batch in file_batches
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker thread error: {e}")
        
        # Final progress update
        self.print_progress()
        
        end_time = time.time()
        duration = timedelta(seconds=int(end_time - self.start_time))
        print(f"Analysis completed in {duration}")
        
        return {
            'combination_counts': dict(self.combination_counts),
            'detailed_statistics': {
                'time_of_day_counts': dict(self.detailed_stats['time_of_day_counts']),
                'weather_counts': dict(self.detailed_stats['weather_counts']),
                'country_counts': dict(self.detailed_stats['country_counts']),
                'city_counts': dict(self.detailed_stats['city_counts']),
                'frame_years': dict(self.detailed_stats['frame_years']),
                'missing_fields': dict(self.detailed_stats['missing_fields']),
                
                # Enhanced statistics
                'seasonal_patterns': {k: dict(v) for k, v in self.detailed_stats['seasonal_patterns'].items()},
                'monthly_patterns': {k: dict(v) for k, v in self.detailed_stats['monthly_patterns'].items()},
                'hourly_patterns': {k: dict(v) for k, v in self.detailed_stats['hourly_patterns'].items()},
                'geographic_clusters': {k: dict(v) for k, v in self.detailed_stats['geographic_clusters'].items()},
                'weather_transitions': dict(self.detailed_stats['weather_transitions']),
                'challenging_conditions': dict(self.detailed_stats['challenging_conditions']),
                'extreme_weather_events': dict(self.detailed_stats['extreme_weather_events']),
                'daylight_quality': dict(self.detailed_stats['daylight_quality']),
                'road_surface_conditions': dict(self.detailed_stats['road_surface_conditions']),
                
                # Visibility statistics
                'visibility_analysis': {
                    'total_samples': len(self.detailed_stats['visibility_distance']['values']),
                    'average_visibility': statistics.mean(self.detailed_stats['visibility_distance']['values']) 
                                        if self.detailed_stats['visibility_distance']['values'] else 0,
                    'visibility_by_weather': {
                        weather: {
                            'count': len(distances),
                            'average': statistics.mean(distances) if distances else 0,
                            'min': min(distances) if distances else 0,
                            'max': max(distances) if distances else 0
                        } for weather, distances in self.detailed_stats['visibility_distance']['by_weather'].items()
                    }
                }
            },
            'summary': {
                'total_files_found': self.total_files_found,
                'total_files_processed': self.total_files_processed,
                'total_files_failed': self.total_files_failed,
                'analysis_duration': str(duration),
                'analysis_timestamp': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path)
            }
        }

def format_results(results: Dict) -> str:
    """Format analysis results for display"""
    if not results:
        return "No results to display"
    
    output = []
    output.append("\n" + "="*71)
    output.append("TIME OF DAY + WEATHER COMBINATION ANALYSIS RESULTS")
    output.append("="*71)
    
    summary = results['summary']
    output.append(f"Dataset Path: {summary['dataset_path']}")
    output.append(f"Files Processed: {summary['total_files_processed']:,}")
    output.append(f"Processing Errors: {summary['total_files_failed']:,}")
    output.append(f"Analysis Duration: {summary['analysis_duration']}")
    output.append(f"Analysis Date: {datetime.now().isoformat()}")
    
    # Combination counts
    combinations = results['combination_counts']
    total_frames = sum(combinations.values())
    
    if total_frames > 0:
        output.append("\nTIME OF DAY + WEATHER COMBINATIONS:")
        output.append("-" * 50)
        
        # Sort by count descending
        sorted_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)
        
        for combo_key, count in sorted_combinations:
            # Format combination key for display
            display_key = combo_key.replace('_', ' + ')
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            output.append(f"{display_key:20}: {count:8,} ({percentage:5.1f}%)")
        
        output.append(f"{'Total':20}: {total_frames:8,}")
        output.append("")
        
        # Specific requested combinations
        output.append("REQUESTED COMBINATIONS:")
        output.append("-" * 30)
        
        day_rain = combinations.get('day_rain', 0)
        day_clear = combinations.get('day_clear', 0)
        day_snow = combinations.get('day_snow', 0)
        day_fog = combinations.get('day_fog', 0)
        night_rain = combinations.get('night_rain', 0)
        night_clear = combinations.get('night_clear', 0)
        night_snow = combinations.get('night_snow', 0)
        night_fog = combinations.get('night_fog', 0)
        
        output.append(f"day + rain       : {day_rain:8,}")
        output.append(f"day + clear      : {day_clear:8,}")
        output.append(f"day + snow       : {day_snow:8,}")
        output.append(f"day + fog        : {day_fog:8,}")
        output.append(f"night + rain     : {night_rain:8,}")
        output.append(f"night + clear    : {night_clear:8,}")
        output.append(f"night + snow     : {night_snow:8,}")
        output.append(f"night + fog      : {night_fog:8,}")
        output.append("")
        
        # Additional breakdowns
        detailed = results['detailed_statistics']
        
        output.append("TIME OF DAY BREAKDOWN:")
        output.append("-" * 30)
        for time_period, count in sorted(detailed['time_of_day_counts'].items()):
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            output.append(f"{time_period:15}: {count:8,} ({percentage:5.1f}%)")
        output.append("")
        
        output.append("WEATHER BREAKDOWN:")
        output.append("-" * 30)
        for weather, count in Counter(detailed['weather_counts']).most_common():
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            output.append(f"{weather:15}: {count:8,} ({percentage:5.1f}%)")
        output.append("")
        
        # Country breakdown (top 10)
        output.append("COUNTRY BREAKDOWN (Top 10):")
        output.append("-" * 30)
        top_countries = Counter(detailed['country_counts']).most_common(10)
        for country, count in top_countries:
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            output.append(f"{country:15}: {count:8,} ({percentage:5.1f}%)")
        output.append("")
    
    return "\n".join(output)

def save_results(results: Dict, output_file: Optional[str] = None) -> str:
    """Save results to JSON file in notebooks folder"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"notebooks/time_weather_analysis_{timestamp}.json"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return output_file
    except Exception as e:
        print(f"Error saving results: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ZOD dataset time-of-day and weather combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 %(prog)s /path/to/zod/single_frames
    python3 %(prog)s /path/to/zod/single_frames --workers 8
    python3 %(prog)s /path/to/zod/single_frames --output analysis_results.json
        """
    )
    
    parser.add_argument(
        'dataset_path',
        help='Path to the ZOD dataset directory containing single_frames'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: auto-generated with timestamp)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    if not dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        sys.exit(1)
    
    # Run analysis
    analyzer = TimeWeatherAnalyzer(
        dataset_path=str(dataset_path),
        max_workers=args.workers
    )
    
    try:
        results = analyzer.analyze_dataset()
        
        if results:
            # Display results
            formatted_output = format_results(results)
            print(formatted_output)
            
            # Save results
            output_file = save_results(results, args.output)
            if output_file:
                print(f"Results saved to: {output_file}")
            
            # Print summary of requested combinations for easy copying
            combinations = results['combination_counts']
            print("\n" + "="*71)
            print("SUMMARY - REQUESTED COMBINATIONS:")
            print("="*71)
            print(f"day rain:   {combinations.get('day_rain', 0):,}")
            print(f"day clear:  {combinations.get('day_clear', 0):,}")
            print(f"day snow:   {combinations.get('day_snow', 0):,}")
            print(f"day fog:    {combinations.get('day_fog', 0):,}")
            print(f"night rain: {combinations.get('night_rain', 0):,}")
            print(f"night clear:{combinations.get('night_clear', 0):,}")
            print(f"night snow: {combinations.get('night_snow', 0):,}")
            print(f"night fog:  {combinations.get('night_fog', 0):,}")
        else:
            print("Analysis completed but no results generated.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()