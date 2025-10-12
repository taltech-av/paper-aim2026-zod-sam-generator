#!/usr/bin/env python3
"""
Pre-filter ZOD frames BEFORE SAM processing
Analyzes frame quality to avoid wasting 2-3 days on bad frames

Filters OUT frames with:
- Too many tiny boxes (poor annotation quality)
- Too many overlapping boxes (SAM confusion)
- Excessive huge boxes (annotation errors)

Prioritizes frames with:
- Pedestrians, cyclists, signs
- Clean object annotations
- Reasonable object sizes
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
from zod import ZodFrames
from zod.constants import AnnotationProject, Anonymization
import numpy as np


class FrameQualityAnalyzer:
    def __init__(self, zod_root: str, output_file: str, strict_mode: bool = False):
        self.zod_root = Path(zod_root)
        self.output_file = Path(output_file)
        self.strict_mode = strict_mode
        
        # Quality thresholds (stricter in strict mode)
        if strict_mode:
            # STRICT MODE: For highest quality training data only
            self.min_quality_score = 2000  # Minimum acceptable score
            self.max_tiny_boxes = 30  # Too many tiny boxes = bad quality
            self.max_overlaps = 30  # Too many overlaps = confusion
            self.min_priority_classes = 2  # Must have at least 2 types of priority objects
            self.max_huge_boxes = 2  # Maximum huge boxes allowed
        else:
            # NORMAL MODE: Original thresholds
            self.min_quality_score = -100  # Very permissive
            self.max_tiny_boxes = 50  # Too many tiny boxes = bad quality
            self.max_overlaps = 50  # Too many overlaps = confusion
            self.min_priority_classes = 1  # At least 1 priority object
            self.max_huge_boxes = 5  # More permissive
        
        # Class-specific minimum size thresholds (matches process_zod_dataset.py)
        self.min_size_by_class = {
            'Vehicle': 30,
            'VulnerableVehicle': 15,  # Cyclists - SMALL
            'Pedestrian': 15,  # SMALL for distant people
            'TrafficSign': 15,  # SMALL signs are important
            'TrafficSignal': 15,  # SMALL lights are important
            'TrafficLight': 15,  # SMALL lights are important
            'TrafficGuide': 15,  # SMALL guides are important
            'LaneMarking': 20,
            'PoleObject': 30,
        }
        
        # Max box size threshold (percentage of image)
        self.max_box_percentage = 0.30  # 30% of image
        
        # Priority classes (for quality scoring)
        self.priority_classes = {
            'Pedestrian': 100,
            'VulnerableVehicle': 80,  # Cyclists
            'TrafficSign': 50,
            'TrafficSignal': 40,
            'TrafficLight': 40,
            'TrafficGuide': 30,
        }
        
    def check_overlap(self, box_a, box_b):
        """Check if two boxes overlap significantly"""
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b
        
        # Calculate intersection
        x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
        y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
        
        if x_overlap > 0 and y_overlap > 0:
            overlap_area = x_overlap * y_overlap
            area_a = (x2_a - x1_a) * (y2_a - y1_a)
            area_b = (x2_b - x1_b) * (y2_b - y1_b)
            
            # Significant if >10% of either box
            if overlap_area > 0.1 * min(area_a, area_b):
                return True
        return False
    
    def analyze_frame(self, frame, frame_id: str):
        """Analyze single frame quality and return metrics"""
        
        metrics = {
            'frame_id': frame_id,
            'quality_score': 0,
            'has_road_data': False,
            'has_priority_objects': False,
            'total_objects': 0,
            'priority_objects': {},
            'tiny_boxes': 0,
            'huge_boxes': 0,
            'overlaps': 0,
            'issues': [],
            'should_process': True,
        }
        
        try:
            # 1. Check road/lane annotations
            lane_anno = frame.get_annotation(AnnotationProject.LANE_MARKINGS)
            if lane_anno and hasattr(lane_anno, 'ego_road'):
                ego_road = lane_anno.ego_road
                if ego_road and hasattr(ego_road, 'polygon') and ego_road.polygon is not None:
                    if len(ego_road.polygon) > 0:
                        metrics['has_road_data'] = True
                        metrics['quality_score'] += 50  # Big bonus for road data
            
            if not metrics['has_road_data']:
                metrics['issues'].append('no_road_data')
                metrics['quality_score'] -= 100  # Big penalty
            
            # 2. Analyze object annotations
            obj_anno = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            if not obj_anno:
                metrics['issues'].append('no_objects')
                metrics['should_process'] = False
                return metrics
            
            metrics['total_objects'] = len(obj_anno)
            
            # Get image size
            camera_frame = frame.info.get_key_camera_frame(Anonymization.BLUR)
            img_width = 3848  # ZOD standard size
            img_height = 2168
            img_area = img_width * img_height
            
            # Collect all boxes for overlap checking
            boxes_with_class = []
            
            for obj in obj_anno:
                if not hasattr(obj, 'box2d') or obj.box2d is None:
                    continue
                
                box = obj.box2d
                if box.xmin >= box.xmax or box.ymin >= box.ymax:
                    continue
                
                width = box.xmax - box.xmin
                height = box.ymax - box.ymin
                area = width * height
                
                # Check if priority class
                if obj.name in self.priority_classes:
                    metrics['has_priority_objects'] = True
                    metrics['priority_objects'][obj.name] = metrics['priority_objects'].get(obj.name, 0) + 1
                    metrics['quality_score'] += self.priority_classes[obj.name]
                
                # Class-specific size threshold
                min_size = self.min_size_by_class.get(obj.name, 30)
                
                # Check for tiny boxes (below class-specific threshold)
                if width < min_size or height < min_size:
                    metrics['tiny_boxes'] += 1
                else:
                    # Only count non-tiny boxes for overlaps
                    boxes_with_class.append((box.xmin, box.ymin, box.xmax, box.ymax, obj.name))
                
                # Check for huge boxes
                if area > img_area * self.max_box_percentage:
                    metrics['huge_boxes'] += 1
                    metrics['quality_score'] -= 20
            
            # 3. Check for overlaps (only among non-tiny boxes)
            overlap_count = 0
            for i in range(len(boxes_with_class)):
                for j in range(i+1, len(boxes_with_class)):
                    if self.check_overlap(boxes_with_class[i][:4], boxes_with_class[j][:4]):
                        overlap_count += 1
            
            metrics['overlaps'] = overlap_count
            
            # 4. Apply quality penalties
            if metrics['tiny_boxes'] > self.max_tiny_boxes:
                metrics['issues'].append(f'too_many_tiny_boxes_{metrics["tiny_boxes"]}')
                metrics['quality_score'] -= 50
            
            if metrics['overlaps'] > self.max_overlaps:
                metrics['issues'].append(f'too_many_overlaps_{metrics["overlaps"]}')
                metrics['quality_score'] -= 50
            
            if metrics['huge_boxes'] > 0:
                metrics['issues'].append(f'huge_boxes_{metrics["huge_boxes"]}')
            
            # Count number of different priority class types
            num_priority_types = len(metrics['priority_objects'])
            
            # 5. Determine if should process (STRICT MODE checks)
            # Skip if:
            # - No road data AND no priority objects
            # - Quality score too low
            # - Not enough priority class diversity (strict mode)
            # - Too many huge boxes (strict mode)
            
            if not metrics['has_road_data'] and not metrics['has_priority_objects']:
                metrics['should_process'] = False
                metrics['issues'].append('no_road_and_no_priority')
            
            if metrics['quality_score'] < self.min_quality_score:
                metrics['should_process'] = False
                metrics['issues'].append(f'quality_too_low_score_{metrics["quality_score"]}')
            
            if self.strict_mode:
                # Additional strict mode filters
                if num_priority_types < self.min_priority_classes:
                    metrics['should_process'] = False
                    metrics['issues'].append(f'insufficient_diversity_{num_priority_types}_types')
                
                if metrics['huge_boxes'] > self.max_huge_boxes:
                    metrics['should_process'] = False
                    metrics['issues'].append(f'too_many_huge_boxes_{metrics["huge_boxes"]}')
            
        except Exception as e:
            metrics['issues'].append(f'error_{str(e)}')
            metrics['should_process'] = False
        
        return metrics
    
    def analyze_dataset(self, max_frames: int = None):
        """Analyze all frames in dataset"""
        
        print("="*60)
        print("PRE-FILTERING ZOD FRAMES FOR SAM PROCESSING")
        print("="*60)
        
        print("\nLoading ZOD dataset...")
        zod = ZodFrames(dataset_root=self.zod_root, version="full")
        print(f"✓ Loaded {len(zod)} frames")
        
        # Get all frame IDs
        all_frame_ids = sorted(zod.get_all_ids())
        
        if max_frames:
            frame_ids = all_frame_ids[:max_frames]
            print(f"Analyzing first {max_frames} frames...")
        else:
            frame_ids = all_frame_ids
            print(f"Analyzing all {len(frame_ids)} frames...")
        
        results = []
        stats = {
            'total': 0,
            'should_process': 0,
            'should_skip': 0,
            'has_road_data': 0,
            'no_road_data': 0,
            'has_priority': 0,
            'issues': defaultdict(int),
            'priority_counts': defaultdict(int),
        }
        
        for frame_id in tqdm(frame_ids, desc="Analyzing frames"):
            try:
                frame = zod[frame_id]
                metrics = self.analyze_frame(frame, frame_id)
                results.append(metrics)
                
                # Update stats
                stats['total'] += 1
                if metrics['should_process']:
                    stats['should_process'] += 1
                else:
                    stats['should_skip'] += 1
                
                if metrics['has_road_data']:
                    stats['has_road_data'] += 1
                else:
                    stats['no_road_data'] += 1
                
                if metrics['has_priority_objects']:
                    stats['has_priority'] += 1
                
                for issue in metrics['issues']:
                    stats['issues'][issue] += 1
                
                for obj_class, count in metrics['priority_objects'].items():
                    stats['priority_counts'][obj_class] += count
                
            except Exception as e:
                print(f"\nError analyzing {frame_id}: {e}")
                stats['should_skip'] += 1
        
        # Sort by quality score
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Save results
        output_data = {
            'stats': dict(stats),
            'stats_issues': dict(stats['issues']),
            'stats_priority_counts': dict(stats['priority_counts']),
            'frames': results,
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal frames analyzed: {stats['total']}")
        print(f"Should process: {stats['should_process']} ({stats['should_process']/stats['total']*100:.1f}%)")
        print(f"Should skip: {stats['should_skip']} ({stats['should_skip']/stats['total']*100:.1f}%)")
        
        print(f"\nRoad data:")
        print(f"  With road data: {stats['has_road_data']} ({stats['has_road_data']/stats['total']*100:.1f}%)")
        print(f"  Without road data: {stats['no_road_data']} ({stats['no_road_data']/stats['total']*100:.1f}%)")
        
        print(f"\nPriority objects:")
        print(f"  Frames with priority objects: {stats['has_priority']} ({stats['has_priority']/stats['total']*100:.1f}%)")
        for obj_class, count in sorted(stats['priority_counts'].items(), key=lambda x: -x[1]):
            print(f"    {obj_class}: {count} instances")
        
        print(f"\nCommon issues:")
        for issue, count in sorted(stats['issues'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {issue}: {count} frames")
        
        print(f"\nTop 10 quality frames:")
        for i, frame in enumerate(results[:10], 1):
            print(f"  {i}. {frame['frame_id']}: score={frame['quality_score']}, "
                  f"priority={frame['priority_objects']}, issues={frame['issues'][:2]}")
        
        print(f"\nBottom 10 quality frames:")
        for i, frame in enumerate(results[-10:], 1):
            print(f"  {i}. {frame['frame_id']}: score={frame['quality_score']}, "
                  f"priority={frame['priority_objects']}, issues={frame['issues'][:2]}")
        
        print(f"\n✅ Results saved to: {self.output_file}")
        
        # Save filtered frame lists
        good_frames_file = self.output_file.parent / 'frames_to_process.txt'
        skip_frames_file = self.output_file.parent / 'frames_to_skip.txt'
        
        with open(good_frames_file, 'w') as f:
            for frame in results:
                if frame['should_process']:
                    f.write(f"{frame['frame_id']}\n")
        
        with open(skip_frames_file, 'w') as f:
            for frame in results:
                if not frame['should_process']:
                    f.write(f"{frame['frame_id']}\n")
        
        print(f"✅ Frame lists saved:")
        print(f"   Good frames: {good_frames_file}")
        print(f"   Skip frames: {skip_frames_file}")
        
        return results, stats


def main():
    parser = argparse.ArgumentParser(description="Pre-filter ZOD frames before SAM processing")
    parser.add_argument('--zod-root', type=str, default='/media/tom/ml/zod-data',
                       help='Path to ZOD dataset')
    parser.add_argument('--output', type=str, 
                       default='/media/tom/ml/projects/clft-zod/frame_quality_analysis.json',
                       help='Output JSON file')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Analyze only first N frames (for testing)')
    parser.add_argument('--strict', action='store_true',
                       help='Enable strict mode for highest quality data only')
    
    args = parser.parse_args()
    
    if args.strict:
        print("⚠️  STRICT MODE ENABLED - Filtering for highest quality frames only")
        print("   Thresholds:")
        print("   - Min quality score: 2000")
        print("   - Min priority class types: 2")
        print("   - Max tiny boxes: 30 (was 50)")
        print("   - Max overlaps: 30 (was 50)")
        print("   - Max huge boxes: 2 (was 5)")
        print()
    
    analyzer = FrameQualityAnalyzer(args.zod_root, args.output, strict_mode=args.strict)
    results, stats = analyzer.analyze_dataset(max_frames=args.max_frames)
    
    print(f"\n{'='*60}")
    print(f"TIME SAVED: Skipping {stats['should_skip']} bad frames")
    print(f"Will process {stats['should_process']} good frames")
    if stats['total'] > 0:
        time_saved_pct = stats['should_skip'] / stats['total'] * 100
        print(f"Estimated time savings: {time_saved_pct:.1f}% of 2-3 days = {time_saved_pct * 2 / 100:.1f}-{time_saved_pct * 3 / 100:.1f} days")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
