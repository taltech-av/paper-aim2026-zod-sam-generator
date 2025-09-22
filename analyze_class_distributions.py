#!/usr/bin/env python3
"""
ZOD Dataset Object Class Distribution Analyzer

This script analyzes object_detection.json files from the ZOD dataset to count
and analyze the distribution of object classes, types, and other properties.

Usage:
    python3 analyze_class_distributions.py <dataset_path> [--workers N]

Example:
    python3 analyze_class_distributions.py /path/to/zod/single_frames --workers 8
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple
import statistics

class ClassDistributionAnalyzer:
    def __init__(self, dataset_path: str, max_workers: int = 4):
        self.dataset_path = Path(dataset_path)
        self.max_workers = max_workers
        
        # Statistics storage
        self.class_counts = defaultdict(int)
        self.type_counts = defaultdict(int)
        self.class_type_combinations = defaultdict(int)
        
        # Specific focus on cyclists, signs, pedestrians, and animals
        self.focus_stats = {
            'cyclists': {
                'total_count': 0,
                'by_type': defaultdict(int),
                'by_occlusion': defaultdict(int),
                'by_relative_position': defaultdict(int),
                'size_stats': {'length': [], 'width': [], 'height': []},
                'emergency_light': defaultdict(int),
                'unclear_count': 0,
                'frames_with_cyclists': set(),
                'max_cyclists_per_frame': 0,
                'cyclists_per_frame': []
            },
            'signs': {
                'total_count': 0,
                'by_type': defaultdict(int),
                'by_occlusion': defaultdict(int),
                'by_relative_position': defaultdict(int),
                'size_stats': {'length': [], 'width': [], 'height': []},
                'unclear_count': 0,
                'frames_with_signs': set(),
                'max_signs_per_frame': 0,
                'signs_per_frame': []
            },
            'pedestrians': {
                'total_count': 0,
                'by_type': defaultdict(int),
                'by_occlusion': defaultdict(int),
                'by_relative_position': defaultdict(int),
                'size_stats': {'length': [], 'width': [], 'height': []},
                'unclear_count': 0,
                'frames_with_pedestrians': set(),
                'max_pedestrians_per_frame': 0,
                'pedestrians_per_frame': []
            },
            'animals': {
                'total_count': 0,
                'by_type': defaultdict(int),
                'by_occlusion': defaultdict(int),
                'by_relative_position': defaultdict(int),
                'size_stats': {'length': [], 'width': [], 'height': []},
                'unclear_count': 0,
                'frames_with_animals': set(),
                'max_animals_per_frame': 0,
                'animals_per_frame': []
            }
        }
        
        # Detailed statistics
        self.detailed_stats = {
            'occlusion_ratio': defaultdict(int),
            'relative_position': defaultdict(int),
            'emergency_light': defaultdict(int),
            'unclear_annotations': 0,
            'total_annotations': 0,
            'files_with_annotations': 0,
            'files_without_annotations': 0,
            'objects_per_frame': [],
            'size_3d_stats': {
                'length': [],
                'width': [],
                'height': []
            },
            'location_3d_stats': {
                'x': [],
                'y': [],
                'z': []
            },
            
            # Enhanced analysis features
            'traffic_density_analysis': {
                'low_density_frames': 0,    # <5 objects
                'medium_density_frames': 0, # 5-15 objects
                'high_density_frames': 0,   # >15 objects
                'objects_per_frame_distribution': defaultdict(int),
                'dense_traffic_characteristics': defaultdict(int)
            },
            'object_interactions': {
                'vehicle_pedestrian_proximity': 0,
                'vehicle_cyclist_proximity': 0,
                'multi_vulnerable_scenes': 0,
                'intersection_scenarios': 0,
                'parking_scenarios': 0,
                'highway_scenarios': 0
            },
            'safety_critical_scenarios': {
                'pedestrians_in_roadway': 0,
                'cyclists_in_traffic': 0,
                'animals_on_road': 0,
                'occluded_vulnerable_users': 0,
                'night_vulnerable_users': 0,
                'adverse_weather_vulnerable': 0
            },
            'spatial_relationships': {
                'object_clusters': defaultdict(int),
                'lane_distribution': defaultdict(int),
                'distance_to_ego_analysis': {
                    'very_close': 0,  # <5m
                    'close': 0,       # 5-15m
                    'medium': 0,      # 15-50m
                    'far': 0          # >50m
                },
                'relative_speed_analysis': defaultdict(int)
            },
            'scene_complexity': {
                'simple_scenes': 0,     # <3 object types
                'moderate_scenes': 0,   # 3-6 object types
                'complex_scenes': 0,    # >6 object types
                'urban_vs_highway': defaultdict(int),
                'weather_complexity_correlation': defaultdict(lambda: defaultdict(int))
            },
            'autonomous_driving_challenges': {
                'construction_zones': 0,
                'emergency_vehicles': 0,
                'school_zones': 0,
                'crowded_intersections': 0,
                'mixed_traffic_scenarios': 0,
                'edge_cases': defaultdict(int)
            }
        }
        
        # Progress tracking
        self.total_files_found = 0
        self.total_files_processed = 0
        self.total_files_failed = 0
        self.start_time = 0
        self.last_progress_update = 0
        self.progress_lock = Lock()
        
    def analyze_traffic_density(self, annotations: List[Dict], file_path: Path):
        """Analyze traffic density and scene complexity"""
        total_objects = len(annotations)
        unique_classes = set(obj.get('obj_class', 'unknown') for obj in annotations)
        
        # Traffic density classification
        if total_objects < 5:
            self.detailed_stats['traffic_density_analysis']['low_density_frames'] += 1
        elif total_objects <= 15:
            self.detailed_stats['traffic_density_analysis']['medium_density_frames'] += 1
        else:
            self.detailed_stats['traffic_density_analysis']['high_density_frames'] += 1
        
        # Objects per frame distribution
        self.detailed_stats['traffic_density_analysis']['objects_per_frame_distribution'][total_objects] += 1
        
        # Scene complexity analysis
        num_object_types = len(unique_classes)
        if num_object_types < 3:
            self.detailed_stats['scene_complexity']['simple_scenes'] += 1
        elif num_object_types <= 6:
            self.detailed_stats['scene_complexity']['moderate_scenes'] += 1
        else:
            self.detailed_stats['scene_complexity']['complex_scenes'] += 1
    
    def analyze_object_interactions(self, annotations: List[Dict], file_path: Path):
        """Analyze spatial relationships and object interactions"""
        # Group objects by class
        objects_by_class = defaultdict(list)
        for obj in annotations:
            obj_class = obj.get('obj_class', 'unknown')
            objects_by_class[obj_class].append(obj)
        
        # Check for multi-vulnerable user scenes
        vulnerable_classes = ['Pedestrian', 'Cyclist', 'Animal']
        vulnerable_present = sum(1 for cls in vulnerable_classes if cls in objects_by_class)
        if vulnerable_present >= 2:
            self.detailed_stats['object_interactions']['multi_vulnerable_scenes'] += 1
        
        # Vehicle-vulnerable user proximity analysis
        if 'Vehicle' in objects_by_class:
            if 'Pedestrian' in objects_by_class:
                self.detailed_stats['object_interactions']['vehicle_pedestrian_proximity'] += 1
            if 'Cyclist' in objects_by_class:
                self.detailed_stats['object_interactions']['vehicle_cyclist_proximity'] += 1
        
        # Detect scenario types based on object combinations
        if 'TrafficSign' in objects_by_class and len(objects_by_class['TrafficSign']) > 3:
            self.detailed_stats['object_interactions']['intersection_scenarios'] += 1
        
        if 'Vehicle' in objects_by_class and len(objects_by_class['Vehicle']) > 10:
            self.detailed_stats['autonomous_driving_challenges']['crowded_intersections'] += 1
    
    def analyze_safety_critical_scenarios(self, annotations: List[Dict], file_path: Path):
        """Identify safety-critical scenarios for autonomous driving"""
        for obj in annotations:
            obj_class = obj.get('obj_class', 'unknown')
            obj_type = obj.get('obj_type', 'unknown')
            relative_position = obj.get('relative_position', 'unknown')
            occlusion_ratio = obj.get('occlusion_ratio', 0)
            
            # Safety-critical position analysis
            if obj_class == 'Pedestrian' and relative_position in ['EgoLane', 'LeftAndEgoLane', 'RightAndEgoLane']:
                self.detailed_stats['safety_critical_scenarios']['pedestrians_in_roadway'] += 1
            
            if obj_class == 'Cyclist' and relative_position in ['EgoLane', 'LeftAndEgoLane', 'RightAndEgoLane']:
                self.detailed_stats['safety_critical_scenarios']['cyclists_in_traffic'] += 1
            
            if obj_class == 'Animal' and relative_position != 'NotOnEgoRoad':
                self.detailed_stats['safety_critical_scenarios']['animals_on_road'] += 1
            
            # Occlusion-based safety concerns
            if obj_class in ['Pedestrian', 'Cyclist', 'Animal'] and occlusion_ratio > 0.5:
                self.detailed_stats['safety_critical_scenarios']['occluded_vulnerable_users'] += 1
            
            # Special object type analysis
            if 'emergency' in obj_type.lower():
                self.detailed_stats['autonomous_driving_challenges']['emergency_vehicles'] += 1
            
            if 'construction' in obj_type.lower() or 'work' in obj_type.lower():
                self.detailed_stats['autonomous_driving_challenges']['construction_zones'] += 1
    
    def analyze_spatial_relationships(self, annotations: List[Dict], file_path: Path):
        """Analyze spatial distribution and relationships"""
        for obj in annotations:
            # Distance analysis based on 3D location
            location_3d = obj.get('location_3d', {})
            if location_3d:
                try:
                    x = float(location_3d.get('x', 0))
                    y = float(location_3d.get('y', 0))
                    z = float(location_3d.get('z', 0))
                    
                    # Calculate distance to ego vehicle (assuming ego at origin)
                    distance = (x**2 + y**2 + z**2)**0.5
                    
                    if distance < 5:
                        self.detailed_stats['spatial_relationships']['distance_to_ego_analysis']['very_close'] += 1
                    elif distance < 15:
                        self.detailed_stats['spatial_relationships']['distance_to_ego_analysis']['close'] += 1
                    elif distance < 50:
                        self.detailed_stats['spatial_relationships']['distance_to_ego_analysis']['medium'] += 1
                    else:
                        self.detailed_stats['spatial_relationships']['distance_to_ego_analysis']['far'] += 1
                        
                except (ValueError, TypeError):
                    pass
            
            # Lane distribution analysis
            relative_position = obj.get('relative_position', 'unknown')
            self.detailed_stats['spatial_relationships']['lane_distribution'][relative_position] += 1
    
    def detect_edge_cases(self, annotations: List[Dict], file_path: Path):
        """Detect edge cases and unusual scenarios"""
        total_objects = len(annotations)
        
        # Unusually high object density
        if total_objects > 50:
            self.detailed_stats['autonomous_driving_challenges']['edge_cases']['very_high_density'] += 1
        
        # Multiple animals in one frame (unusual)
        animal_count = sum(1 for obj in annotations if obj.get('obj_class') == 'Animal')
        if animal_count > 2:
            self.detailed_stats['autonomous_driving_challenges']['edge_cases']['multiple_animals'] += 1
        
        # Very large objects (potential annotation errors or special cases)
        for obj in annotations:
            size_3d = obj.get('size_3d', {})
            if size_3d:
                try:
                    length = float(size_3d.get('length', 0))
                    width = float(size_3d.get('width', 0))
                    height = float(size_3d.get('height', 0))
                    
                    # Unusually large objects
                    if length > 20 or width > 5 or height > 5:
                        self.detailed_stats['autonomous_driving_challenges']['edge_cases']['oversized_objects'] += 1
                        
                except (ValueError, TypeError):
                    pass
        
    def find_annotation_files(self) -> List[Path]:
        """Find all object_detection.json files in the dataset"""
        print(f"Searching for object_detection.json files in: {self.dataset_path}")
        
        annotation_files = []
        count = 0
        
        for root, dirs, files in os.walk(self.dataset_path):
            if 'object_detection.json' in files:
                annotation_files.append(Path(root) / 'object_detection.json')
                count += 1
                
                # Progress indicator for large datasets
                if count % 10000 == 0:
                    print(f"Found {count:,} object_detection.json files so far...")
        
        print(f"Total object_detection.json files found: {len(annotation_files):,}")
        return annotation_files
    
    def parse_annotation_file(self, file_path: Path) -> Optional[List[Dict]]:
        """Parse a single object_detection.json file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                return annotations if isinstance(annotations, list) else []
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
            return None
        except Exception:
            return None
    
    def update_statistics(self, annotations: List[Dict], file_path: Path):
        """Update statistics with annotations from one file"""
        with self.progress_lock:
            num_objects = len(annotations)
            self.detailed_stats['objects_per_frame'].append(num_objects)
            self.detailed_stats['total_annotations'] += num_objects
            
            if num_objects > 0:
                self.detailed_stats['files_with_annotations'] += 1
            else:
                self.detailed_stats['files_without_annotations'] += 1
            
            # Track focus categories per frame
            frame_cyclists = 0
            frame_signs = 0
            frame_pedestrians = 0
            frame_animals = 0
            
            for annotation in annotations:
                properties = annotation.get('properties', {})
                
                # Main class and type
                obj_class = properties.get('class', 'unknown')
                obj_type = properties.get('type', 'unknown')
                
                self.class_counts[obj_class] += 1
                self.type_counts[obj_type] += 1
                self.class_type_combinations[f"{obj_class}_{obj_type}"] += 1
                
                # Additional properties
                occlusion = properties.get('occlusion_ratio', 'unknown')
                self.detailed_stats['occlusion_ratio'][occlusion] += 1
                
                relative_pos = properties.get('relative_position', 'unknown')
                self.detailed_stats['relative_position'][relative_pos] += 1
                
                emergency = properties.get('emergency_light', 'unknown')
                self.detailed_stats['emergency_light'][emergency] += 1
                
                unclear = properties.get('unclear', False)
                if unclear:
                    self.detailed_stats['unclear_annotations'] += 1
                
                # 3D size statistics
                size_length = properties.get('size_3d_length')
                size_width = properties.get('size_3d_width')
                size_height = properties.get('size_3d_height')
                
                if size_length is not None:
                    self.detailed_stats['size_3d_stats']['length'].append(size_length)
                if size_width is not None:
                    self.detailed_stats['size_3d_stats']['width'].append(size_width)
                if size_height is not None:
                    self.detailed_stats['size_3d_stats']['height'].append(size_height)
                
                # 3D location statistics
                location_3d = properties.get('location_3d', {})
                if isinstance(location_3d, dict) and 'coordinates' in location_3d:
                    coords = location_3d['coordinates']
                    if len(coords) >= 3:
                        self.detailed_stats['location_3d_stats']['x'].append(coords[0])
                        self.detailed_stats['location_3d_stats']['y'].append(coords[1])
                        self.detailed_stats['location_3d_stats']['z'].append(coords[2])
                
                # Focus category analysis
                self.analyze_focus_category(obj_class, obj_type, properties, file_path, 
                                          size_length, size_width, size_height)
                
                # Count per frame
                if self.is_cyclist(obj_class, obj_type):
                    frame_cyclists += 1
                elif self.is_sign(obj_class, obj_type):
                    frame_signs += 1
                elif self.is_pedestrian(obj_class, obj_type):
                    frame_pedestrians += 1
                elif self.is_animal(obj_class, obj_type):
                    frame_animals += 1
            
            # Update per-frame statistics
            if frame_cyclists > 0:
                self.focus_stats['cyclists']['frames_with_cyclists'].add(str(file_path))
                self.focus_stats['cyclists']['cyclists_per_frame'].append(frame_cyclists)
                self.focus_stats['cyclists']['max_cyclists_per_frame'] = max(
                    self.focus_stats['cyclists']['max_cyclists_per_frame'], frame_cyclists)
            
            if frame_signs > 0:
                self.focus_stats['signs']['frames_with_signs'].add(str(file_path))
                self.focus_stats['signs']['signs_per_frame'].append(frame_signs)
                self.focus_stats['signs']['max_signs_per_frame'] = max(
                    self.focus_stats['signs']['max_signs_per_frame'], frame_signs)
            
            if frame_pedestrians > 0:
                self.focus_stats['pedestrians']['frames_with_pedestrians'].add(str(file_path))
                self.focus_stats['pedestrians']['pedestrians_per_frame'].append(frame_pedestrians)
                self.focus_stats['pedestrians']['max_pedestrians_per_frame'] = max(
                    self.focus_stats['pedestrians']['max_pedestrians_per_frame'], frame_pedestrians)
            
            if frame_animals > 0:
                self.focus_stats['animals']['frames_with_animals'].add(str(file_path))
                self.focus_stats['animals']['animals_per_frame'].append(frame_animals)
                self.focus_stats['animals']['max_animals_per_frame'] = max(
                    self.focus_stats['animals']['max_animals_per_frame'], frame_animals)
            
            # Enhanced analysis methods
            self.analyze_traffic_density(annotations, file_path)
            self.analyze_object_interactions(annotations, file_path)
            self.analyze_safety_critical_scenarios(annotations, file_path)
            self.analyze_spatial_relationships(annotations, file_path)
            self.detect_edge_cases(annotations, file_path)
    
    def is_cyclist(self, obj_class: str, obj_type: str) -> bool:
        """Check if object is a cyclist"""
        return (obj_class == 'VulnerableVehicle' and obj_type == 'Bicycle') or obj_type == 'Bicycle'
    
    def is_sign(self, obj_class: str, obj_type: str) -> bool:
        """Check if object is a traffic sign"""
        return obj_class == 'TrafficSign' or 'Sign' in obj_class
    
    def is_pedestrian(self, obj_class: str, obj_type: str) -> bool:
        """Check if object is a pedestrian"""
        return obj_class == 'Pedestrian' or obj_type == 'Pedestrian'
    
    def is_animal(self, obj_class: str, obj_type: str) -> bool:
        """Check if object is an animal"""
        return obj_class == 'Animal' or obj_type == 'Animal'
    
    def is_pedestrian(self, obj_class: str, obj_type: str) -> bool:
        """Check if object is a pedestrian"""
        return obj_class == 'Pedestrian' or obj_type == 'Pedestrian'
    
    def analyze_focus_category(self, obj_class: str, obj_type: str, properties: Dict, 
                             file_path: Path, size_length: float, size_width: float, size_height: float):
        """Analyze focus categories: cyclists, signs, pedestrians, animals"""
        category = None
        
        if self.is_cyclist(obj_class, obj_type):
            category = 'cyclists'
        elif self.is_sign(obj_class, obj_type):
            category = 'signs'
        elif self.is_pedestrian(obj_class, obj_type):
            category = 'pedestrians'
        elif self.is_animal(obj_class, obj_type):
            category = 'animals'
        
        if category:
            stats = self.focus_stats[category]
            stats['total_count'] += 1
            stats['by_type'][obj_type] += 1
            
            occlusion = properties.get('occlusion_ratio', 'unknown')
            stats['by_occlusion'][occlusion] += 1
            
            relative_pos = properties.get('relative_position', 'unknown')
            stats['by_relative_position'][relative_pos] += 1
            
            if properties.get('unclear', False):
                stats['unclear_count'] += 1
            
            # Size statistics for this category
            if size_length is not None:
                stats['size_stats']['length'].append(size_length)
            if size_width is not None:
                stats['size_stats']['width'].append(size_width)
            if size_height is not None:
                stats['size_stats']['height'].append(size_height)
            
            # Emergency light for cyclists
            if category == 'cyclists':
                emergency = properties.get('emergency_light', 'unknown')
                stats['emergency_light'][emergency] += 1
                if size_height is not None:
                    self.detailed_stats['size_3d_stats']['height'].append(size_height)
                
                # 3D location statistics
                location_3d = properties.get('location_3d', {})
                if isinstance(location_3d, dict) and 'coordinates' in location_3d:
                    coords = location_3d['coordinates']
                    if len(coords) >= 3:
                        self.detailed_stats['location_3d_stats']['x'].append(coords[0])
                        self.detailed_stats['location_3d_stats']['y'].append(coords[1])
                        self.detailed_stats['location_3d_stats']['z'].append(coords[2])
    
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
        """Process a batch of annotation files"""
        for file_path in file_paths:
            annotations = self.parse_annotation_file(file_path)
            
            if annotations is not None:
                self.update_statistics(annotations, file_path)
                self.update_progress(processed=1)
            else:
                self.update_progress(failed=1)
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values"""
        if not values:
            return {'count': 0, 'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the entire dataset using multiple workers
        """
        print(f"=== ZOD Dataset Object Class Distribution Analysis ===")
        
        # Find all annotation files
        annotation_files = self.find_annotation_files()
        
        if not annotation_files:
            print("No object_detection.json files found!")
            return {}
        
        self.total_files_found = len(annotation_files)
        self.start_time = time.time()
        self.last_progress_update = self.start_time
        
        print(f"Starting analysis with {self.max_workers} worker threads...")
        
        # Split files into batches for workers
        batch_size = max(1, len(annotation_files) // (self.max_workers * 4))
        file_batches = [
            annotation_files[i:i + batch_size] 
            for i in range(0, len(annotation_files), batch_size)
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
        
        # Calculate final statistics including focus categories
        size_stats = {}
        for dimension in ['length', 'width', 'height']:
            values = self.detailed_stats['size_3d_stats'][dimension]
            size_stats[dimension] = self.calculate_statistics(values)
        
        location_stats = {}
        for axis in ['x', 'y', 'z']:
            values = self.detailed_stats['location_3d_stats'][axis]
            location_stats[axis] = self.calculate_statistics(values)
        
        objects_per_frame_stats = self.calculate_statistics(
            self.detailed_stats['objects_per_frame']
        )
        
        # Calculate focus category statistics
        focus_stats_final = {}
        for category in ['cyclists', 'signs', 'pedestrians', 'animals']:
            stats = self.focus_stats[category]
            focus_stats_final[category] = {
                'total_count': stats['total_count'],
                'by_type': dict(stats['by_type']),
                'by_occlusion': dict(stats['by_occlusion']),
                'by_relative_position': dict(stats['by_relative_position']),
                'unclear_count': stats['unclear_count'],
                'frames_with_objects': len(stats[f'frames_with_{category}']),
                f'max_{category}_per_frame': stats[f'max_{category}_per_frame'],
                f'{category}_per_frame_stats': self.calculate_statistics(stats[f'{category}_per_frame']),
                'size_statistics': {
                    'length': self.calculate_statistics(stats['size_stats']['length']),
                    'width': self.calculate_statistics(stats['size_stats']['width']),
                    'height': self.calculate_statistics(stats['size_stats']['height'])
                }
            }
            
            # Add emergency light stats for cyclists
            if category == 'cyclists':
                focus_stats_final[category]['emergency_light'] = dict(stats['emergency_light'])
        
        return {
            'class_counts': dict(self.class_counts),
            'type_counts': dict(self.type_counts),
            'class_type_combinations': dict(self.class_type_combinations),
            'focus_statistics': focus_stats_final,
            'detailed_statistics': {
                'occlusion_ratio': dict(self.detailed_stats['occlusion_ratio']),
                'relative_position': dict(self.detailed_stats['relative_position']),
                'emergency_light': dict(self.detailed_stats['emergency_light']),
                'unclear_annotations': self.detailed_stats['unclear_annotations'],
                'total_annotations': self.detailed_stats['total_annotations'],
                'files_with_annotations': self.detailed_stats['files_with_annotations'],
                'files_without_annotations': self.detailed_stats['files_without_annotations'],
                'objects_per_frame_stats': objects_per_frame_stats,
                'size_3d_statistics': size_stats,
                'location_3d_statistics': location_stats,
                
                # Enhanced statistics
                'traffic_density_analysis': {
                    'low_density_frames': self.detailed_stats['traffic_density_analysis']['low_density_frames'],
                    'medium_density_frames': self.detailed_stats['traffic_density_analysis']['medium_density_frames'],
                    'high_density_frames': self.detailed_stats['traffic_density_analysis']['high_density_frames'],
                    'objects_per_frame_distribution': dict(self.detailed_stats['traffic_density_analysis']['objects_per_frame_distribution']),
                    'dense_traffic_characteristics': dict(self.detailed_stats['traffic_density_analysis']['dense_traffic_characteristics'])
                },
                'object_interactions': dict(self.detailed_stats['object_interactions']),
                'safety_critical_scenarios': dict(self.detailed_stats['safety_critical_scenarios']),
                'spatial_relationships': {
                    'object_clusters': dict(self.detailed_stats['spatial_relationships']['object_clusters']),
                    'lane_distribution': dict(self.detailed_stats['spatial_relationships']['lane_distribution']),
                    'distance_to_ego_analysis': dict(self.detailed_stats['spatial_relationships']['distance_to_ego_analysis']),
                    'relative_speed_analysis': dict(self.detailed_stats['spatial_relationships']['relative_speed_analysis'])
                },
                'scene_complexity': {
                    'simple_scenes': self.detailed_stats['scene_complexity']['simple_scenes'],
                    'moderate_scenes': self.detailed_stats['scene_complexity']['moderate_scenes'],
                    'complex_scenes': self.detailed_stats['scene_complexity']['complex_scenes'],
                    'urban_vs_highway': dict(self.detailed_stats['scene_complexity']['urban_vs_highway']),
                    'weather_complexity_correlation': {k: dict(v) for k, v in self.detailed_stats['scene_complexity']['weather_complexity_correlation'].items()}
                },
                'autonomous_driving_challenges': {
                    'construction_zones': self.detailed_stats['autonomous_driving_challenges']['construction_zones'],
                    'emergency_vehicles': self.detailed_stats['autonomous_driving_challenges']['emergency_vehicles'],
                    'school_zones': self.detailed_stats['autonomous_driving_challenges']['school_zones'],
                    'crowded_intersections': self.detailed_stats['autonomous_driving_challenges']['crowded_intersections'],
                    'mixed_traffic_scenarios': self.detailed_stats['autonomous_driving_challenges']['mixed_traffic_scenarios'],
                    'edge_cases': dict(self.detailed_stats['autonomous_driving_challenges']['edge_cases'])
                }
            },
            'summary': {
                'total_files_found': self.total_files_found,
                'total_files_processed': self.total_files_processed,
                'total_files_failed': self.total_files_failed,
                'analysis_duration': str(duration),
                'dataset_path': str(self.dataset_path)
            }
        }

def format_results(results: Dict) -> str:
    """Format analysis results for display"""
    if not results:
        return "No results to display"
    
    output = []
    output.append("\n" + "="*71)
    output.append("ZOD DATASET OBJECT CLASS DISTRIBUTION ANALYSIS RESULTS")
    output.append("="*71)
    
    summary = results['summary']
    output.append(f"Dataset Path: {summary['dataset_path']}")
    output.append(f"Files Processed: {summary['total_files_processed']:,}")
    output.append(f"Processing Errors: {summary['total_files_failed']:,}")
    output.append(f"Analysis Duration: {summary['analysis_duration']}")
    output.append(f"Analysis Date: {datetime.now().isoformat()}")
    
    detailed = results['detailed_statistics']
    total_annotations = detailed['total_annotations']
    
    output.append(f"\nOVERALL STATISTICS:")
    output.append("-" * 50)
    output.append(f"Total Annotations: {total_annotations:,}")
    output.append(f"Files with Annotations: {detailed['files_with_annotations']:,}")
    output.append(f"Files without Annotations: {detailed['files_without_annotations']:,}")
    output.append(f"Unclear Annotations: {detailed['unclear_annotations']:,}")
    
    # FOCUS STATISTICS - Cyclists, Signs, Pedestrians
    focus_stats = results.get('focus_statistics', {})
    if focus_stats:
        output.append(f"\n" + "="*71)
        output.append("FOCUS ANALYSIS: CYCLISTS, SIGNS, PEDESTRIANS, AND ANIMALS")
        output.append("="*71)
        
        for category in ['cyclists', 'signs', 'pedestrians', 'animals']:
            if category in focus_stats:
                stats = focus_stats[category]
                category_title = category.upper()
                
                output.append(f"\n{category_title} STATISTICS:")
                output.append("-" * 50)
                output.append(f"Total {category}: {stats['total_count']:,}")
                percentage = (stats['total_count'] / total_annotations * 100) if total_annotations > 0 else 0
                output.append(f"Percentage of all objects: {percentage:.2f}%")
                output.append(f"Frames with {category}: {stats['frames_with_objects']:,}")
                output.append(f"Max {category} per frame: {stats[f'max_{category}_per_frame']}")
                output.append(f"Unclear {category}: {stats['unclear_count']:,}")
                
                # Per-frame statistics
                frame_stats = stats[f'{category}_per_frame_stats']
                if frame_stats['count'] > 0:
                    output.append(f"Average {category} per frame (when present): {frame_stats['mean']:.2f}")
                    output.append(f"Median {category} per frame (when present): {frame_stats['median']:.2f}")
                
                # Type breakdown
                if stats['by_type']:
                    output.append(f"\n{category_title} BY TYPE:")
                    for obj_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                        type_percentage = (count / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
                        output.append(f"  {obj_type:20}: {count:8,} ({type_percentage:5.1f}%)")
                
                # Occlusion breakdown
                if stats['by_occlusion']:
                    output.append(f"\n{category_title} BY OCCLUSION:")
                    for occlusion, count in sorted(stats['by_occlusion'].items(), key=lambda x: x[1], reverse=True):
                        occ_percentage = (count / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
                        output.append(f"  {occlusion:15}: {count:8,} ({occ_percentage:5.1f}%)")
                
                # Relative position breakdown
                if stats['by_relative_position']:
                    output.append(f"\n{category_title} BY RELATIVE POSITION:")
                    top_positions = sorted(stats['by_relative_position'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for position, count in top_positions:
                        pos_percentage = (count / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
                        output.append(f"  {position:20}: {count:8,} ({pos_percentage:5.1f}%)")
                
                # Size statistics
                size_stats = stats.get('size_statistics', {})
                if any(size_stats.get(dim, {}).get('count', 0) > 0 for dim in ['length', 'width', 'height']):
                    output.append(f"\n{category_title} SIZE STATISTICS:")
                    for dimension in ['length', 'width', 'height']:
                        dim_stats = size_stats.get(dimension, {})
                        if dim_stats.get('count', 0) > 0:
                            output.append(f"  {dimension.title():8} - Mean: {dim_stats['mean']:6.2f}, "
                                         f"Median: {dim_stats['median']:6.2f}, "
                                         f"Range: [{dim_stats['min']:6.2f}, {dim_stats['max']:6.2f}]")
                
                # Emergency light for cyclists
                if category == 'cyclists' and stats.get('emergency_light'):
                    output.append(f"\nCYCLISTS EMERGENCY LIGHT:")
                    for light_status, count in sorted(stats['emergency_light'].items(), key=lambda x: x[1], reverse=True):
                        light_percentage = (count / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
                        output.append(f"  {light_status:15}: {count:8,} ({light_percentage:5.1f}%)")
    
    # Objects per frame statistics
    obj_stats = detailed['objects_per_frame_stats']
    output.append(f"\n" + "="*71)
    output.append("GENERAL DATASET STATISTICS")
    output.append("="*71)
    output.append(f"\nOBJECTS PER FRAME STATISTICS:")
    output.append("-" * 50)
    output.append(f"Mean: {obj_stats['mean']:.2f}")
    output.append(f"Median: {obj_stats['median']:.2f}")
    output.append(f"Std Dev: {obj_stats['std']:.2f}")
    output.append(f"Min: {int(obj_stats['min'])}")
    output.append(f"Max: {int(obj_stats['max'])}")
    
    # Class distribution
    class_counts = results['class_counts']
    if class_counts:
        output.append(f"\nOBJECT CLASS DISTRIBUTION:")
        output.append("-" * 50)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for obj_class, count in sorted_classes:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            output.append(f"{obj_class:20}: {count:8,} ({percentage:5.1f}%)")
    
    # Type distribution
    type_counts = results['type_counts']
    if type_counts:
        output.append(f"\nOBJECT TYPE DISTRIBUTION:")
        output.append("-" * 50)
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        
        for obj_type, count in sorted_types:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            output.append(f"{obj_type:20}: {count:8,} ({percentage:5.1f}%)")
    
    # Top class-type combinations
    combinations = results['class_type_combinations']
    if combinations:
        output.append(f"\nTOP CLASS-TYPE COMBINATIONS:")
        output.append("-" * 50)
        sorted_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)
        
        for combo, count in sorted_combinations[:15]:  # Top 15
            class_name, type_name = combo.split('_', 1)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            output.append(f"{class_name} + {type_name:15}: {count:8,} ({percentage:5.1f}%)")
    
    # Occlusion ratio distribution
    occlusion_counts = detailed['occlusion_ratio']
    if occlusion_counts:
        output.append(f"\nOCCLUSION RATIO DISTRIBUTION:")
        output.append("-" * 50)
        sorted_occlusion = sorted(occlusion_counts.items(), key=lambda x: x[1], reverse=True)
        
        for occlusion, count in sorted_occlusion:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            output.append(f"{occlusion:15}: {count:8,} ({percentage:5.1f}%)")
    
    # Relative position distribution
    position_counts = detailed['relative_position']
    if position_counts:
        output.append(f"\nRELATIVE POSITION DISTRIBUTION:")
        output.append("-" * 50)
        sorted_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
        
        for position, count in sorted_positions:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            output.append(f"{position:20}: {count:8,} ({percentage:5.1f}%)")
    
    # 3D Size statistics
    size_stats = detailed['size_3d_statistics']
    output.append(f"\n3D SIZE STATISTICS:")
    output.append("-" * 50)
    for dimension in ['length', 'width', 'height']:
        stats = size_stats[dimension]
        if stats['count'] > 0:
            output.append(f"{dimension.title():8} - Mean: {stats['mean']:6.2f}, "
                         f"Median: {stats['median']:6.2f}, "
                         f"Std: {stats['std']:6.2f}, "
                         f"Range: [{stats['min']:6.2f}, {stats['max']:6.2f}]")
    
    # 3D Location statistics
    location_stats = detailed['location_3d_statistics']
    output.append(f"\n3D LOCATION STATISTICS:")
    output.append("-" * 50)
    for axis in ['x', 'y', 'z']:
        stats = location_stats[axis]
        if stats['count'] > 0:
            output.append(f"{axis.upper():8} - Mean: {stats['mean']:8.2f}, "
                         f"Median: {stats['median']:8.2f}, "
                         f"Std: {stats['std']:8.2f}, "
                         f"Range: [{stats['min']:8.2f}, {stats['max']:8.2f}]")
    
    return "\n".join(output)

def save_results(results: Dict, output_file: Optional[str] = None) -> str:
    """Save results to JSON file in notebooks folder"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"notebooks/class_distribution_analysis_{timestamp}.json"
    
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
        description="Analyze ZOD dataset object class distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 %(prog)s /path/to/zod/single_frames
    python3 %(prog)s /path/to/zod/single_frames --workers 8
    python3 %(prog)s /path/to/zod/single_frames --output class_analysis.json
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
    analyzer = ClassDistributionAnalyzer(
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
                print(f"\nResults saved to: {output_file}")
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