#!/usr/bin/env python3
"""
Generate LiDAR-only annotations from enhanced lidar_png using SAM as ground truth.

This script creates LiDAR-native segmentation annotations for LiDAR-only model training by:
1. Using enhanced lidar_png geometric projections for LiDAR coverage analysis
2. Using SAM annotations as ground truth for object locations and types
3. Mapping SAM-identified objects onto actual LiDAR points for LiDAR-native annotations
4. Applying dilation to provide better supervision signal for sparse LiDAR data

Class mapping (LiDAR-native):
  0: Background (no LiDAR coverage)
  1: Ignore (sparse LiDAR regions, distant points, low-confidence areas)
  2: Vehicle (LiDAR points within SAM vehicle regions)
  3: Sign (LiDAR points within SAM sign regions)
  4: Cyclist (LiDAR points within SAM cyclist regions)
  5: Pedestrian (LiDAR points within SAM pedestrian regions)

REQUIRES: Run generate_sam.py and generate_lidar_png.py first!
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from scipy import ndimage
import json


class LiDARPNGAnnotationGenerator:
    """Generate LiDAR-only annotations from enhanced lidar_png using SAM guidance"""

    def __init__(self, output_root):
        """Initialize LiDAR annotation generator

        Args:
            output_root: Path to output_clft_v2 directory
        """
        self.output_root = Path(output_root)

        # Input directories
        self.lidar_png_dir = self.output_root / "lidar_png"  # Use enhanced version
        self.sam_annotation_dir = self.output_root / "annotation_sam"

        # Output directory for LiDAR annotations
        self.lidar_annotation_dir = self.output_root / "annotation_lidar_only"
        self.lidar_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.processed_frames_file = self.output_root / "processed_lidar_png_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # Find frames that have both enhanced lidar_png and SAM annotation
        print("Scanning for available input files...")
        
        # Get frame IDs from enhanced lidar_png files
        lidar_files = list(self.lidar_png_dir.glob("frame_*.png"))
        lidar_frame_ids = set()
        for lidar_file in lidar_files:
            frame_id = lidar_file.stem.replace("frame_", "")
            lidar_frame_ids.add(frame_id)
        
        # Get frame IDs from SAM annotation files
        sam_files = list(self.sam_annotation_dir.glob("frame_*.png"))
        sam_frame_ids = set()
        for sam_file in sam_files:
            frame_id = sam_file.stem.replace("frame_", "")
            sam_frame_ids.add(frame_id)
        
        # Find intersection - frames with both inputs
        available_frame_ids = lidar_frame_ids & sam_frame_ids
        
        print(f"✓ Found {len(lidar_frame_ids):,} lidar_png files")
        print(f"✓ Found {len(sam_frame_ids):,} SAM annotation files")
        print(f"✓ Found {len(available_frame_ids):,} frames with both inputs")
        
        # Convert to sorted list for consistent processing
        self.frame_ids = sorted(list(available_frame_ids))
        
        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

        # LiDAR annotation parameters
        self.density_threshold = 0.30  # 30% density threshold for sparse regions (was 5%, too low for enhanced data)
        self.distance_threshold = 60.0  # Points beyond 60m become ignore
        self.dilation_iterations = 3  # More dilation for enhanced lidar_png

    def load_enhanced_lidar_png(self, frame_id):
        """Load enhanced LiDAR geometric projection"""
        lidar_path = self.lidar_png_dir / f"frame_{frame_id}.png"
        if not lidar_path.exists():
            return None

        # Load 3-channel enhanced geometric projection
        lidar_img = cv2.imread(str(lidar_path), cv2.IMREAD_UNCHANGED)
        return lidar_img

    def load_sam_annotation(self, frame_id):
        """Load SAM segmentation mask"""
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None

        annotation = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)
        return annotation

    def create_lidar_native_annotation(self, enhanced_lidar_img, sam_annotation):
        """Create LiDAR-native annotation using lidar_png and SAM guidance

        Strategy:
        1. Use lidar_png for LiDAR coverage and geometric analysis
        2. Use SAM as ground truth for object locations and types
        3. Map SAM-identified objects onto actual LiDAR points
        4. Create LiDAR-native segmentation with proper class distinctions

        Args:
            enhanced_lidar_img: 3-channel enhanced geometric projection
            sam_annotation: SAM segmentation mask (ground truth)

        Returns:
            annotation: LiDAR-native segmentation mask
        """
        if enhanced_lidar_img is None or sam_annotation is None:
            return None

        h, w, c = enhanced_lidar_img.shape

        # Initialize annotation with background (0)
        annotation = np.zeros((h, w), dtype=np.uint8)

        # Extract geometric information from enhanced lidar_png
        x_channel = enhanced_lidar_img[:, :, 0].astype(float)  # X coordinates (normalized)
        y_channel = enhanced_lidar_img[:, :, 1].astype(float)  # Y coordinates (normalized)
        z_channel = enhanced_lidar_img[:, :, 2].astype(float)  # Z coordinates (depth, normalized)

        # Create LiDAR coverage mask (any non-zero channel indicates LiDAR presence)
        lidar_coverage = np.any(enhanced_lidar_img > 0, axis=2)

        # Decode distance from enhanced normalization
        # From enhanced method: z_norm = np.clip(z_log / np.log(101.0), 0, 1) * 255
        # This is more complex due to logarithmic scaling, approximate inverse
        z_norm = z_channel / 255.0
        # Approximate inverse log scaling (not perfect but good enough for thresholding)
        distance_map = np.exp(z_norm * np.log(101.0))

        # 1. Background class (0): Areas with no LiDAR coverage
        # (already initialized to 0)

        # 2. Ignore class (1): Sparse LiDAR regions, distant points, AND edge regions
        # Strategy A: Sparse/distant detection (quality-based ignore)
        # Create density map using gaussian filter for local coverage analysis
        lidar_density = ndimage.gaussian_filter(
            lidar_coverage.astype(float),
            sigma=3,  # Smaller sigma for enhanced data
            mode='constant',
            cval=0
        )

        # Areas with sparse LiDAR coverage (< 30% local density)
        # BUT only where we actually have some LiDAR points
        sparse_regions = (lidar_density < self.density_threshold) & lidar_coverage

        # Areas beyond distance threshold (approximate)
        distant_regions = (distance_map > self.distance_threshold) & lidar_coverage

        # Strategy B: Dynamic vertical edge regions (consistency with camera)
        # Find topmost and bottommost objects from SAM to define usable region
        has_any_object = np.isin(sam_annotation, [2, 3, 4, 5])
        vertical_edge_regions = np.zeros_like(sam_annotation, dtype=bool)
        
        if np.any(has_any_object):
            rows_with_objects = np.any(has_any_object, axis=1)
            object_rows = np.where(rows_with_objects)[0]
            
            if len(object_rows) > 0:
                topmost_object = object_rows[0]
                bottommost_object = object_rows[-1]
                
                # Add margin beyond objects (2% of height as safety margin)
                margin = max(int(h * 0.02), 10)
                
                # Mark regions BEYOND objects as ignore
                top_ignore_line = max(0, topmost_object - margin)
                bottom_ignore_line = min(h, bottommost_object + margin)
                
                vertical_edge_regions[:top_ignore_line, :] = True  # Above objects
                vertical_edge_regions[bottom_ignore_line:, :] = True  # Below objects
        
        # Combine all ignore criteria
        ignore_regions = sparse_regions | distant_regions | vertical_edge_regions
        annotation[ignore_regions] = 1

        # 3. Objects from SAM guidance: Map onto LiDAR points
        # Only mark LiDAR points that fall within SAM object regions
        for sam_class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            sam_class_mask = (sam_annotation == sam_class_id)
            # Only apply where we have LiDAR coverage AND it's not an ignore region
            lidar_in_class = sam_class_mask & lidar_coverage & ~ignore_regions
            annotation[lidar_in_class] = sam_class_id  # Preserve SAM class

        # 4. Final cleanup: Ensure areas without LiDAR coverage are background
        # But preserve ignore regions that have sparse LiDAR coverage
        no_lidar = ~lidar_coverage
        annotation[no_lidar] = 0  # Force background only where there's NO LiDAR at all
        # Do NOT overwrite ignore regions that were set in step 2

        return annotation

    def dilate_object_regions(self, annotation):
        """Dilate object regions to provide more supervision signal for enhanced lidar data"""
        if annotation is None:
            return None

        dilated = annotation.copy()

        # Dilate each object class separately to avoid conflicts
        for class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            obj_mask = (annotation == class_id)

            if np.any(obj_mask):
                # Use morphological dilation to expand objects (more iterations for enhanced data)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_mask = cv2.dilate(obj_mask.astype(np.uint8),
                                        kernel, iterations=self.dilation_iterations)

                # Only expand into background/ignore regions (don't overwrite other objects)
                valid_expansion = dilated_mask & ((annotation == 0) | (annotation == 1))
                dilated[valid_expansion] = class_id

        # Force edge rows to background to prevent dilation artifacts
        # Dynamic edge clearing based on dilation radius
        edge_rows = self.dilation_iterations * 2 + 1  # Kernel radius * 2 + 1 for safety
        
        # But check if we have objects near edges - if so, use minimal clearing
        h = dilated.shape[0]
        has_objects_top = np.any(np.isin(dilated[edge_rows:edge_rows*2, :], [2, 3, 4, 5]))
        has_objects_bottom = np.any(np.isin(dilated[-edge_rows*2:-edge_rows, :], [2, 3, 4, 5]))
        
        # Clear fewer rows if objects are near edges (to preserve them)
        top_clear = 2 if has_objects_top else edge_rows
        bottom_clear = 2 if has_objects_bottom else edge_rows
        
        dilated[:top_clear, :] = 0  # Top rows
        dilated[-bottom_clear:, :] = 0  # Bottom rows

        return dilated

    def create_lidar_png_annotation(self, frame_id):
        """Create LiDAR-only annotation for a frame using enhanced lidar_png"""
        try:
            # Load enhanced LiDAR geometric projection
            enhanced_lidar_img = self.load_enhanced_lidar_png(frame_id)
            if enhanced_lidar_img is None:
                print(f"  ⚠️  No enhanced lidar_png for {frame_id}")
                return None

            # Load SAM annotation for object locations
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            # Create LiDAR-native annotation using enhanced lidar_png and SAM guidance
            annotation = self.create_lidar_native_annotation(enhanced_lidar_img, sam_annotation)

            # Dilate object regions for better supervision with enhanced data
            annotation = self.dilate_object_regions(annotation)

            return annotation

        except Exception as e:
            print(f"  ❌ Error creating LiDAR annotation for {frame_id}: {e}")
            return None

    def process_all_frames(self, create_vis=False):
        """Process all frames and create LiDAR annotations

        Args:
            create_vis: Whether to create visualizations
        """
        print(f"\n🎯 Generating LiDAR-only annotations from enhanced lidar_png")
        print(f"Input: {self.lidar_png_dir} (enhanced)")
        print(f"SAM guidance: {self.sam_annotation_dir}")
        print(f"Output: {self.lidar_annotation_dir}")
        print(f"Frames to process: {len(self.frame_ids):,}")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory if needed
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "lidar_only_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Color map for visualization (consistent with SAM)
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([255, 0, 0], dtype=np.uint8),        # Vehicle - Red
                3: np.array([255, 255, 0], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
            }

        for frame_id in tqdm(self.frame_ids, desc="Processing frames"):
            # Check if already exists
            output_path = self.lidar_annotation_dir / f"frame_{frame_id}.png"

            if output_path.exists():
                skip_count += 1
                continue

            # Create LiDAR annotation
            annotation = self.create_lidar_png_annotation(frame_id)

            if annotation is None:
                error_count += 1
                continue

            # Save annotation
            cv2.imwrite(str(output_path), annotation)
            success_count += 1

            # Mark frame as processed
            self.processed_frames.add(frame_id)
            with open(self.processed_frames_file, 'a') as f:
                f.write(f"{frame_id}\n")

            # Create visualization if requested
            if create_vis:
                try:
                    # Create colored visualization showing all classes
                    # Background (0) = Black, Ignore (1) = Gray, Objects (2-5) = Colors
                    colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                    
                    # Color map for all classes (BGR format for OpenCV)
                    full_color_map = {
                        0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                        1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                        2: np.array([0, 0, 255], dtype=np.uint8),        # Vehicle - Red (BGR)
                        3: np.array([0, 255, 255], dtype=np.uint8),      # Sign - Yellow (BGR)
                        4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta (BGR)
                        5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green (BGR)
                    }
                    
                    for class_id, color in full_color_map.items():
                        mask = annotation == class_id
                        colored_annotation[mask] = color

                    vis_path = vis_dir / f"frame_{frame_id}.png"
                    cv2.imwrite(str(vis_path), colored_annotation)
                    vis_count += 1
                except Exception as e:
                    pass  # Skip visualization errors

        print(f"\n{'='*60}")
        print(f"LIDAR PNG ANNOTATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,}")
        print(f"\n📁 Output: {self.lidar_annotation_dir}/")
        print(f"\n🎯 LiDAR-Only Training Annotations:")
        print(f"   📥 Input: Enhanced lidar_png + SAM guidance")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 Features:")
        print(f"      - Dynamic vertical ignore (adapts to object positions)")
        print(f"      - Sparse LiDAR detection (< 30% density)")
        print(f"      - Distance-based ignore (>60m)")
        print(f"      - Object dilation for better supervision")
        print(f"      - Edge rows cleared (prevents dilation artifacts)")
        print(f"   📊 Background vs Ignore:")
        print(f"      - Background (0): No LiDAR coverage (black in viz)")
        print(f"      - Ignore (1): Edges beyond objects, sparse/distant LiDAR (gray in viz)")
        print(f"   ✓ Consistent with camera-only ignore strategy")
        print(f"   📊 Visualization: All classes visible")
        print(f"      - Black = Background, Gray = Ignore, Colors = Objects")
        print(f"   🎯 Purpose: LiDAR-native annotations for LiDAR-only model training")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LiDAR-only annotations from enhanced lidar_png using SAM guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates LiDAR-native segmentation annotations for LiDAR-only model training.

REQUIRES: Run generate_sam.py and generate_lidar_png.py (enhanced) first!

Strategy:
1. Automatically finds frames with both enhanced lidar_png and SAM annotations
2. Uses enhanced lidar_png geometric projections for LiDAR coverage analysis
3. Uses SAM annotations as ground truth for object locations and types
4. Maps SAM-identified objects onto actual LiDAR points for LiDAR-native annotations
5. Applies dilation to provide better supervision signal

Class mapping:
  0: Background (no LiDAR coverage)
  1: Ignore (sparse LiDAR regions, distant points >60m)
  2: Vehicle (LiDAR points within SAM vehicle regions)
  3: Sign (LiDAR points within SAM sign regions)
  4: Cyclist (LiDAR points within SAM cyclist regions)
  5: Pedestrian (LiDAR points within SAM pedestrian regions)

Usage examples:
  # Generate annotations for all frames with both inputs available
  python generate_lidar_png_annotation.py

  # Generate with visualizations for all processed frames
  python generate_lidar_png_annotation.py --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations showing LiDAR-detected objects for all processed frames')
    parser.add_argument('--dilation', type=int, default=3,
                       help='Dilation iterations for object regions (default: 3)')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/projects/clft-zod/output")

    print("="*60)
    print("LiDAR PNG Annotation Generator")
    print("="*60)
    print(f"Input: {OUTPUT_ROOT / 'lidar_png'} (enhanced)")
    print(f"SAM: {OUTPUT_ROOT / 'annotation_sam'}")
    print(f"Output: {OUTPUT_ROOT / 'annotation_lidar_only'}")

    generator = LiDARPNGAnnotationGenerator(
        output_root=OUTPUT_ROOT
    )

    # Override dilation if specified
    if hasattr(args, 'dilation') and args.dilation != 3:
        generator.dilation_iterations = args.dilation
        print(f"Using custom dilation: {args.dilation} iterations")

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()