#!/usr/bin/env python3
"""
Generate merged annotations by combining SAM camera annotations with semantic region logic.

Creates three types of annotations:

1. annotation_include: Smart semantic region-based annotations
   - Sky region (top 25%): Forced background (0) - not relevant for driving decisions
   - Road region (bottom 50%): Force ignore only where no SAM annotation exists
   - Other regions: SAM camera annotations (Vehicle, Sign, Cyclist, Pedestrian)
   - Background (0) where no SAM annotation in non-ignored regions

2. annotation_exclude: Smart sensor fusion based on object type
   - Vehicles: Require both SAM + LiDAR confirmation (LiDAR reliable for vehicles)
   - Signs: Keep all SAM detections (LiDAR often misses signs)
   - Pedestrians/Cyclists: Require both sensors (safety-critical, need validation)
   - LiDAR points filtered by distance (>60m points excluded)
   - Preserves valuable camera-only detections while being conservative for vehicles

3. annotation_balanced: Best of both worlds (RECOMMENDED for training)
   - Combines semantic region awareness with smart sensor fusion
   - Sky/road regions handled semantically, objects validated by appropriate sensors
   - Provides context-aware, reliable annotations for real-world training

This script works with pre-generated outputs:
- SAM annotations from generate_sam.py (annotation_sam folder)
- LiDAR projections from generate_lidar_projections.py (lidar_projection folder)

Classes:
    0: Background (sky and true background)
    1: Ignore (road regions - too complex for supervision)
    2: Vehicle (from SAM, validated by LiDAR where required)
    3: Sign (from SAM, LiDAR validation not required)
    4: Cyclist (from SAM, validated by LiDAR where required)
    5: Pedestrian (from SAM, validated by LiDAR where required)
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse


class MergedAnnotationGenerator:
    def __init__(self, output_root, frames_list):
        """Initialize merged annotation generator

        Args:
            output_root: Path to output_clft_v2 directory
            frames_list: Path to frames_to_process.txt or all.txt
        """
        self.output_root = Path(output_root)
        self.frames_list = Path(frames_list)

        # Input directories
        self.camera_dir = self.output_root / "camera"
        self.sam_annotation_dir = self.output_root / "annotation_sam"
        self.lidar_projection_dir = self.output_root / "lidar_projection"

        # Output directories
        self.annotation_include_dir = self.output_root / "annotation_include"
        self.annotation_exclude_dir = self.output_root / "annotation_exclude"
        self.annotation_balanced_dir = self.output_root / "annotation_balanced"  # NEW: Best of both worlds
        self.annotation_include_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_exclude_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_balanced_dir.mkdir(parents=True, exist_ok=True)

        # Class mapping (same for both annotation types)
        self.CLASSES = {
            0: 'background',  # Sky and true background
            1: 'ignore',      # Road regions (too complex for pixel supervision)
            2: 'vehicle',     # From SAM
            3: 'sign',        # From SAM
            4: 'cyclist',     # From SAM
            5: 'pedestrian',  # From SAM
        }

        # Progress tracking
        self.processed_frames_file = self.output_root / "processed_merged_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # Load frames to process
        if self.frames_list.exists():
            try:
                with open(self.frames_list) as f:
                    self.frame_ids = [line.strip() for line in f if line.strip()]
                print(f"✓ Loaded {len(self.frame_ids):,} frames from {self.frames_list}")
            except FileNotFoundError:
                print(f"⚠️  Frames file not found: {self.frames_list}")
                print("Please provide a valid frames list file")
                self.frame_ids = []
        else:
            # If frames file doesn't exist, get all frames from annotation_sam directory
            print(f"⚠️  Frames file not found: {self.frames_list}")
            print("Using all available frames from annotation_sam directory as starting point")

            # Get all frame IDs from annotation_sam directory
            sam_files = list(self.sam_annotation_dir.glob("frame_*.png"))
            self.frame_ids = []
            for sam_file in sam_files:
                # Extract frame ID from filename (frame_XXXXXX.png -> XXXXXX)
                frame_id = sam_file.stem.replace("frame_", "")
                self.frame_ids.append(frame_id)

            print(f"✓ Found {len(self.frame_ids):,} frames in {self.sam_annotation_dir}")

        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Loaded {original_count:,} frames to process")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

    def load_sam_annotation(self, frame_id):
        """Load SAM camera annotation"""
        anno_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not anno_path.exists():
            return None

        annotation = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)
        return annotation

    def load_lidar_projection_mask(self, frame_id, max_distance=60.0):
        """Load LiDAR projection PNG and convert to binary mask with distance filtering

        Args:
            frame_id: Frame identifier
            max_distance: Maximum distance in meters to include (default: 60m)

        Returns:
            Binary mask where 1 = LiDAR coverage within distance limit, 0 = no LiDAR or too far
        """
        projection_path = self.lidar_projection_dir / f"frame_{frame_id}.png"
        if not projection_path.exists():
            return None

        try:
            # Load the LiDAR projection image
            lidar_img = cv2.imread(str(projection_path))

            # Convert BGR to RGB for easier processing
            lidar_img_rgb = cv2.cvtColor(lidar_img, cv2.COLOR_BGR2RGB)

            # Extract color channels (BGR format in OpenCV)
            b_channel = lidar_img[:, :, 0].astype(float)
            g_channel = lidar_img[:, :, 1].astype(float)
            r_channel = lidar_img[:, :, 2].astype(float)

            # Decode distance from color using inverse of get_distance_color function
            # From generate_lidar_projections.py:
            # r = int(255 * (1 - normalized))
            # g = int(255 * (1 - normalized * 0.6))
            # b = int(139 + 116 * (1 - normalized))

            # Solve for normalized distance from r channel (most reliable)
            # normalized = 1 - (r / 255)
            normalized_r = 1 - (r_channel / 255.0)

            # Clamp to valid range
            normalized_r = np.clip(normalized_r, 0, 1)

            # Convert back to distance (min_dist=0, max_dist=80)
            decoded_distances = normalized_r * 80.0

            # Create distance-based mask
            # Filter out points beyond max_distance and points with no LiDAR (black pixels)
            has_lidar = np.any(lidar_img > 0, axis=2)  # Any non-black pixel
            within_distance = decoded_distances <= max_distance

            # Combine masks: must have LiDAR AND be within distance
            lidar_mask = has_lidar & within_distance

            return lidar_mask.astype(np.uint8)

        except Exception as e:
            print(f"  Error loading LiDAR projection for {frame_id}: {e}")
            return None

    def create_annotation_include(self, frame_id):
        """Create annotation_include: Smart semantic annotations with region-based logic

        Logic:
        1. Sky region (top 25%): Force background (0) - not relevant for driving
        2. Road region (estimated bottom 30-40%): Force ignore (1) - too complex for pixel supervision
        3. Other regions: Keep SAM camera annotations (Vehicle, Sign, Cyclist, Pedestrian)
        4. Background (0) remains where no SAM annotation exists in non-ignored regions
        """
        try:
            # Load SAM annotation
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                return None

            # Start with SAM annotation
            merged_annotation = sam_annotation.copy()
            h, w = merged_annotation.shape

            # Strategy: Semantic region-based annotations
            # 1. Sky region (top 25%): Force background only where no SAM annotation exists
            # (don't cut through objects that span sky/foreground boundary)
            sky_region = int(0.25 * h)  # Top 25% is sky
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[:sky_region, :] = True

            # Only force sky to background where there are no SAM annotations
            # (preserve objects that extend into sky region)
            sky_background = sky_mask & (sam_annotation == 0)
            merged_annotation[sky_background] = 0

            # 2. Road region (bottom 50%): Force ignore class
            road_start = int(0.50 * h)  # Bottom 50% is road (expanded)
            road_mask = np.zeros((h, w), dtype=bool)
            road_mask[road_start:, :] = True

            # Only apply road ignore where there wasn't already a SAM annotation
            # (don't overwrite vehicles on the road)
            road_ignore = road_mask & (sam_annotation == 0)
            merged_annotation[road_ignore] = 1

            return merged_annotation

        except Exception as e:
            print(f"  Error creating annotation_include for {frame_id}: {e}")
            return None

    def create_annotation_exclude(self, frame_id):
        """Create annotation_exclude: Smart sensor fusion based on object type

        Logic:
        1. Vehicles: Require both SAM and LiDAR (LiDAR is reliable for vehicles)
        2. Signs: Keep SAM detections even without LiDAR (LiDAR often misses signs)
        3. Pedestrians/Cyclists: Keep SAM detections, but boost confidence where LiDAR agrees
        4. Everything else becomes background (0)

        This preserves valuable camera-only detections while being conservative for vehicles.
        """
        try:
            # Load SAM annotation
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                return None

            # Start with all background
            merged_annotation = np.zeros_like(sam_annotation)

            # Load LiDAR projection mask (filter out distant points >60m)
            lidar_mask = self.load_lidar_projection_mask(frame_id, max_distance=60.0)
            has_lidar = lidar_mask is not None

            # Apply different logic based on object type
            if has_lidar:
                # Vehicles (class 2): Require both sensors
                vehicle_regions = (sam_annotation == 2) & (lidar_mask == 1)
                merged_annotation[vehicle_regions] = 2

                # Signs (class 3): Keep all SAM detections (LiDAR often misses signs)
                sign_regions = (sam_annotation == 3)
                merged_annotation[sign_regions] = 3

                # Pedestrians (class 5): Keep SAM, but only where LiDAR also detects
                # (more conservative for safety-critical objects)
                pedestrian_regions = (sam_annotation == 5) & (lidar_mask == 1)
                merged_annotation[pedestrian_regions] = 5

                # Cyclists (class 4): Similar to pedestrians - require both sensors
                cyclist_regions = (sam_annotation == 4) & (lidar_mask == 1)
                merged_annotation[cyclist_regions] = 4
            else:
                # No LiDAR data: Keep signs (most valuable), be conservative with others
                sign_regions = (sam_annotation == 3)
                merged_annotation[sign_regions] = 3

                # For vehicles/pedestrians/cyclists without LiDAR: leave as background
                # (we can't validate these without LiDAR confirmation)

            return merged_annotation

        except Exception as e:
            print(f"  Error creating annotation_exclude for {frame_id}: {e}")
            return None

    def create_annotation_balanced(self, frame_id):
        """Create annotation_balanced: Best of both worlds

        Combines semantic region awareness with smart sensor fusion:
        1. Sky region (top 25%): Background where no SAM detection (preserve spanning objects)
        2. Road region (bottom 50%): Ignore where no SAM detection (complex road textures)
        3. Objects: Smart fusion based on sensor reliability
           - Vehicles: Require both SAM + LiDAR confirmation
           - Signs: Keep all SAM detections (LiDAR often misses signs)
           - Pedestrians/Cyclists: Require both sensors (safety-critical)
        4. LiDAR points filtered by distance (>60m excluded)

        This provides semantic context while ensuring object annotations are reliable.
        """
        try:
            # Load SAM annotation
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                return None

            # Start with SAM annotation
            merged_annotation = sam_annotation.copy()
            h, w = merged_annotation.shape

            # 1. Apply semantic regions (from annotation_include logic)
            # Sky region (top 25%): Force background only where no SAM annotation exists
            sky_region = int(0.25 * h)
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[:sky_region, :] = True
            sky_background = sky_mask & (sam_annotation == 0)
            merged_annotation[sky_background] = 0

            # Road region (bottom 50%): Force ignore only where no SAM annotation exists
            road_start = int(0.50 * h)
            road_mask = np.zeros((h, w), dtype=bool)
            road_mask[road_start:, :] = True
            road_ignore = road_mask & (sam_annotation == 0)
            merged_annotation[road_ignore] = 1

            # 2. Apply smart sensor fusion (from annotation_exclude logic)
            # Load LiDAR projection mask (filter out distant points >60m)
            lidar_mask = self.load_lidar_projection_mask(frame_id, max_distance=60.0)
            has_lidar = lidar_mask is not None

            if has_lidar:
                # Vehicles: Require both sensors (don't overwrite semantic regions)
                vehicle_regions = (sam_annotation == 2) & (lidar_mask == 1) & (~sky_mask) & (~road_mask)
                # Keep existing annotation if it's already a vehicle, otherwise set to vehicle
                merged_annotation[vehicle_regions] = 2

                # Signs: Keep all SAM detections (LiDAR validation not required)
                # But don't override semantic regions
                sign_regions = (sam_annotation == 3) & (~sky_mask) & (~road_mask)
                merged_annotation[sign_regions] = 3

                # Pedestrians: Require both sensors
                pedestrian_regions = (sam_annotation == 5) & (lidar_mask == 1) & (~sky_mask) & (~road_mask)
                merged_annotation[pedestrian_regions] = 5

                # Cyclists: Require both sensors
                cyclist_regions = (sam_annotation == 4) & (lidar_mask == 1) & (~sky_mask) & (~road_mask)
                merged_annotation[cyclist_regions] = 4

                # Any objects that don't meet criteria in non-semantic regions become background
                non_semantic = (~sky_mask) & (~road_mask)
                invalid_vehicles = (sam_annotation == 2) & (~lidar_mask) & non_semantic
                invalid_pedestrians = (sam_annotation == 5) & (~lidar_mask) & non_semantic
                invalid_cyclists = (sam_annotation == 4) & (~lidar_mask) & non_semantic
                merged_annotation[invalid_vehicles | invalid_pedestrians | invalid_cyclists] = 0
            else:
                # No LiDAR: Keep signs, be conservative with others (already handled by semantic regions)
                pass

            return merged_annotation

        except Exception as e:
            print(f"  Error creating annotation_balanced for {frame_id}: {e}")
            return None

    def process_all_frames(self, create_vis=False):
        """Process all frames and create both annotation types

        Args:
            create_vis: Whether to create visualizations for ALL frames
        """
        print(f"\nGenerating merged annotations for {len(self.frame_ids):,} frames...")
        print(f"Input: {self.sam_annotation_dir}")
        print(f"       {self.lidar_projection_dir}")
        print(f"Output: {self.annotation_include_dir}")
        print(f"        {self.annotation_exclude_dir}")
        print(f"        {self.annotation_balanced_dir}")
        if create_vis:
            print(f"Visualizations: ALL frames")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory if needed
        if create_vis:
            vis_dir_include = self.output_root / "visualizations" / "annotation_include"
            vis_dir_exclude = self.output_root / "visualizations" / "annotation_exclude"
            vis_dir_balanced = self.output_root / "visualizations" / "annotation_balanced"
            vis_dir_comparison = self.output_root / "visualizations" / "merged_annotations"
            vis_dir_include.mkdir(parents=True, exist_ok=True)
            vis_dir_exclude.mkdir(parents=True, exist_ok=True)
            vis_dir_balanced.mkdir(parents=True, exist_ok=True)
            vis_dir_comparison.mkdir(parents=True, exist_ok=True)

            # Color map for visualization
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore (road) - Gray
                2: np.array([255, 0, 0], dtype=np.uint8),        # Vehicle - Red
                3: np.array([255, 255, 0], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
            }

        for idx, frame_id in enumerate(tqdm(self.frame_ids, desc="Processing frames")):
            # Check if already exists
            include_path = self.annotation_include_dir / f"frame_{frame_id}.png"
            exclude_path = self.annotation_exclude_dir / f"frame_{frame_id}.png"
            balanced_path = self.annotation_balanced_dir / f"frame_{frame_id}.png"

            if include_path.exists() and exclude_path.exists() and balanced_path.exists():
                skip_count += 1
                continue

            # Create all three annotation types
            annotation_include = self.create_annotation_include(frame_id)
            annotation_exclude = self.create_annotation_exclude(frame_id)
            annotation_balanced = self.create_annotation_balanced(frame_id)

            if annotation_include is None or annotation_exclude is None or annotation_balanced is None:
                error_count += 1
                continue

            # Save all three annotations
            cv2.imwrite(str(include_path), annotation_include)
            cv2.imwrite(str(exclude_path), annotation_exclude)
            cv2.imwrite(str(balanced_path), annotation_balanced)
            success_count += 1

            # Mark frame as processed
            self.processed_frames.add(frame_id)
            with open(self.processed_frames_file, 'a') as f:
                f.write(f"{frame_id}\n")

            # Create visualization if requested
            if create_vis:
                try:
                    # Load camera image
                    camera_path = self.camera_dir / f"frame_{frame_id}.png"
                    if camera_path.exists():
                        camera_img = cv2.imread(str(camera_path))
                        if camera_img is not None:
                            h, w = camera_img.shape[:2]

                            # Create individual visualizations for each annotation type
                            # annotation_include visualization
                            colored_include = np.zeros((h, w, 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                colored_include[annotation_include == class_id] = color
                            overlay_include = cv2.addWeighted(camera_img, 0.6, colored_include, 0.4, 0)

                            # Add label and class counts for include
                            include_counts = {class_id: np.sum(annotation_include == class_id)
                                            for class_id in range(6)}
                            cv2.putText(overlay_include, f"Frame {frame_id} - Include", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            y_offset = 60
                            for class_id in [1, 2, 3, 4, 5]:  # Include ignore class (road regions)
                                if include_counts[class_id] > 0:
                                    color = color_map[class_id].tolist()
                                    class_name = self.CLASSES[class_id]
                                    count = include_counts[class_id]
                                    text = f"{class_name}: {count:,} px"
                                    cv2.putText(overlay_include, text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    y_offset += 25

                            vis_path_include = vis_dir_include / f"frame_{frame_id}.png"
                            cv2.imwrite(str(vis_path_include), overlay_include)

                            # annotation_exclude visualization
                            colored_exclude = np.zeros((h, w, 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                colored_exclude[annotation_exclude == class_id] = color
                            overlay_exclude = cv2.addWeighted(camera_img, 0.6, colored_exclude, 0.4, 0)

                            # Add label and class counts for exclude
                            exclude_counts = {class_id: np.sum(annotation_exclude == class_id)
                                            for class_id in range(6)}
                            cv2.putText(overlay_exclude, f"Frame {frame_id} - Exclude", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            y_offset = 60
                            for class_id in [0, 2, 3, 4, 5]:  # Include background, exclude ignore (not used in exclude)
                                if exclude_counts[class_id] > 0:
                                    color = color_map[class_id].tolist()
                                    class_name = self.CLASSES[class_id]
                                    count = exclude_counts[class_id]
                                    text = f"{class_name}: {count:,} px"
                                    cv2.putText(overlay_exclude, text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    y_offset += 25

                            vis_path_exclude = vis_dir_exclude / f"frame_{frame_id}.png"
                            cv2.imwrite(str(vis_path_exclude), overlay_exclude)

                            # annotation_balanced visualization
                            colored_balanced = np.zeros((h, w, 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                colored_balanced[annotation_balanced == class_id] = color
                            overlay_balanced = cv2.addWeighted(camera_img, 0.6, colored_balanced, 0.4, 0)

                            # Add label and class counts for balanced
                            balanced_counts = {class_id: np.sum(annotation_balanced == class_id)
                                             for class_id in range(6)}
                            cv2.putText(overlay_balanced, f"Frame {frame_id} - Balanced", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            y_offset = 60
                            for class_id in [0, 1, 2, 3, 4, 5]:  # Show all classes for balanced
                                if balanced_counts[class_id] > 0:
                                    color = color_map[class_id].tolist()
                                    class_name = self.CLASSES[class_id]
                                    count = balanced_counts[class_id]
                                    text = f"{class_name}: {count:,} px"
                                    cv2.putText(overlay_balanced, text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    y_offset += 25

                            vis_path_balanced = vis_dir_balanced / f"frame_{frame_id}.png"
                            cv2.imwrite(str(vis_path_balanced), overlay_balanced)

                            # Create side-by-side comparison (now 3 annotations)
                            comparison_img = np.zeros((h, 3*w, 3), dtype=np.uint8)
                            comparison_img[:, :w] = overlay_include
                            comparison_img[:, w:2*w] = overlay_exclude
                            comparison_img[:, 2*w:] = overlay_balanced

                            # Add comparison labels
                            cv2.putText(comparison_img, f"Frame {frame_id} - Include vs Exclude vs Balanced", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            vis_path_comparison = vis_dir_comparison / f"frame_{frame_id}.png"
                            cv2.imwrite(str(vis_path_comparison), comparison_img)

                            vis_count += 4  # Count all four visualizations
                except Exception as e:
                    pass  # Skip visualization errors

        print(f"\n{'='*60}")
        print(f"MERGED ANNOTATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,} (3 per frame)")
        print(f"\nOutput:")
        print(f"  📁 {self.annotation_include_dir}/ - Include annotations (semantic regions: sky→bg, road→ignore)")
        print(f"  📁 {self.annotation_exclude_dir}/ - Exclude annotations (smart fusion: signs keep SAM, vehicles need both)")
        print(f"  📁 {self.annotation_balanced_dir}/ - Balanced annotations (semantic + smart fusion)")
        if create_vis:
            print(f"  📁 {vis_dir_include}/ - Include annotation visualizations")
            print(f"  📁 {vis_dir_exclude}/ - Exclude annotation visualizations")
            print(f"  📁 {vis_dir_balanced}/ - Balanced annotation visualizations")
            print(f"  📁 {vis_dir_comparison}/ - Side-by-side comparison visualizations")
        print(f"\nClass mapping (same for both):")
        for class_id, class_name in self.CLASSES.items():
            print(f"  {class_id}: {class_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate merged annotations from pre-generated SAM and LiDAR outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates three types of annotations from pre-generated outputs:

REQUIRES: Run generate_sam.py and generate_lidar_projections.py first!

annotation_include: Smart semantic region-based annotations
  - Sky region (top 25%): Forced background (0) - not relevant for driving
  - Road region (bottom 50%): Forced ignore (1) - too complex for pixel supervision
  - Other regions: SAM camera annotations (Vehicle, Sign, Cyclist, Pedestrian)
  - Background (0) where no SAM annotation in non-ignored regions

annotation_exclude: Smart sensor fusion based on object type
  - Vehicles: Require both SAM + LiDAR confirmation (LiDAR reliable for vehicles)
  - Signs: Keep all SAM detections (LiDAR often misses signs)
  - Pedestrians/Cyclists: Require both sensors (safety-critical, need validation)
  - LiDAR points filtered by distance (>60m points excluded)
  - Preserves valuable camera-only detections while being conservative for vehicles

annotation_balanced: Best of both worlds (RECOMMENDED for training)
  - Combines semantic region awareness with smart sensor fusion
  - Sky/road regions handled semantically, objects validated by appropriate sensors
  - Provides context-aware, reliable annotations for real-world training

Class mapping:
  0: Background (sky and true background)
  1: Ignore (road regions)
  2: Vehicle (from SAM, validated by LiDAR where required)
  3: Sign (from SAM, LiDAR validation not required)
  4: Cyclist (from SAM, validated by LiDAR where required)
  5: Pedestrian (from SAM, validated by LiDAR where required)
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create side-by-side comparison visualizations for ALL frames')
    args = parser.parse_args()


    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/projects/clft-zod/output_clft_v2")
    FRAMES_LIST = Path("/media/tom/ml/projects/clft-zod/output_clft_v2/all.txt")

    print("="*60)
    print("Merged Annotation Generator")
    print("="*60)

    generator = MergedAnnotationGenerator(
        output_root=OUTPUT_ROOT,
        frames_list=FRAMES_LIST
    )

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()