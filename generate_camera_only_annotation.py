#!/usr/bin/env python3
"""
Generate camera-only annotations with ignore regions for camera-only model training.

This script takes SAM annotations and adds ignore regions (class 1) for areas that
should not be trained on in camera-only models, such as areas with poor visibility,
motion blur, or peripheral regions.

REQUIRES: Run generate_sam.py first!
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse


class CameraOnlyAnnotationGenerator:
    """Generate camera-only annotations with ignore regions"""

    def __init__(self, output_root):
        """Initialize camera-only annotation generator

        Args:
            output_root: Path to output_clft_v2 directory
        """
        self.output_root = Path(output_root)

        # Input directory
        self.sam_annotation_dir = self.output_root / "annotation_sam"

        # Output directory for camera-only annotations
        self.camera_annotation_dir = self.output_root / "annotation_camera_only"
        self.camera_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.processed_frames_file = self.output_root / "processed_camera_only_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # Find frames with SAM annotations
        print("Scanning for SAM annotation files...")
        sam_files = list(self.sam_annotation_dir.glob("frame_*.png"))
        self.frame_ids = []
        for sam_file in sam_files:
            frame_id = sam_file.stem.replace("frame_", "")
            self.frame_ids.append(frame_id)

        print(f"✓ Found {len(self.frame_ids):,} SAM annotation files")

        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

    def load_sam_annotation(self, frame_id):
        """Load SAM segmentation mask"""
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None
        return cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)

    def load_camera_image(self, frame_id):
        """Load camera image for quality analysis"""
        camera_path = self.output_root / "camera" / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return None
        return cv2.imread(str(camera_path), cv2.IMREAD_COLOR)

    def create_ignore_regions(self, sam_annotation, camera_img=None):
        """Create ignore regions (class 1) for camera-only training

        Strategies for ignore regions:
        1. Peripheral regions (outside central field of view)
        2. Areas with poor contrast/low visibility
        3. Areas with motion blur (if detectable)
        4. Over/under exposed regions
        """
        ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        h, w = sam_annotation.shape

        # Strategy 1: Peripheral regions (outside central 80% of image)
        # Create elliptical mask for central region
        center_y, center_x = h // 2, w // 2
        # Use 80% of dimensions for central region
        axis_y = int(h * 0.8) // 2
        axis_x = int(w * 0.8) // 2

        y_coords, x_coords = np.ogrid[:h, :w]
        # Elliptical distance from center
        dist_from_center = ((x_coords - center_x) / axis_x) ** 2 + ((y_coords - center_y) / axis_y) ** 2
        peripheral_mask = dist_from_center > 1.0
        ignore_mask |= peripheral_mask

        # Strategy 2: Areas with poor contrast (low local variance)
        if camera_img is not None:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY).astype(float)

            # Use simpler contrast detection - blur and compare
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)

            # Calculate local contrast as difference from blurred version
            local_contrast = np.abs(gray - blurred)

            # Areas with very low local contrast (< 3) are likely poor visibility
            low_contrast = local_contrast < 3
            ignore_mask |= low_contrast

            # Strategy 3: Over/under exposed regions
            # Convert to HSV for brightness analysis
            hsv = cv2.cvtColor(camera_img, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2].astype(float)

            # Over-exposed (too bright, > 240)
            over_exposed = brightness > 240
            ignore_mask |= over_exposed

            # Under-exposed (too dark, < 15)
            under_exposed = brightness < 15
            ignore_mask |= under_exposed

        # Strategy 4: Very small isolated regions in SAM annotations
        # (similar to noise reduction in other generators)
        for class_id in [2, 3, 4, 5]:  # object classes
            obj_mask = (sam_annotation == class_id)
            if np.any(obj_mask):
                # Use OpenCV connected components instead of scipy
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask.astype(np.uint8), connectivity=8)

                # Skip background (label 0)
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]

                    # Mark very small objects (< 20 pixels) as ignore
                    if area < 20:
                        region_mask = (labels == label_id)
                        ignore_mask |= region_mask

        return ignore_mask

    def create_camera_only_annotation(self, frame_id):
        """Create camera-only annotation with ignore regions"""
        try:
            # Load SAM annotation (required)
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            # Load camera image (optional, for quality analysis)
            camera_img = self.load_camera_image(frame_id)

            # Start with SAM annotation
            annotation = sam_annotation.copy()

            # Create ignore regions
            ignore_mask = self.create_ignore_regions(sam_annotation, camera_img)

            # Apply ignore regions (class 1)
            # Only override background/unknown regions, don't overwrite existing objects
            background_mask = (annotation == 0) & ignore_mask
            annotation[background_mask] = 1

            return annotation

        except Exception as e:
            print(f"  ❌ Error creating camera-only annotation for {frame_id}: {e}")
            return None

    def process_all_frames(self, create_vis=False):
        """Process all frames and create camera-only annotations"""
        print(f"\n🎯 Generating camera-only annotations with ignore regions")
        print(f"Input: SAM annotations + camera images")
        print(f"Output: {self.camera_annotation_dir}")
        print(f"Frames to process: {len(self.frame_ids):,}")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory if needed
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "camera_only_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Color map for visualization
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
            output_path = self.camera_annotation_dir / f"frame_{frame_id}.png"

            if output_path.exists():
                skip_count += 1
                continue

            # Create camera-only annotation
            annotation = self.create_camera_only_annotation(frame_id)

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
                    # Create colored visualization
                    colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                    for class_id, color in color_map.items():
                        colored_annotation[annotation == class_id] = color

                    vis_path = vis_dir / f"frame_{frame_id}.png"
                    cv2.imwrite(str(vis_path), colored_annotation)
                    vis_count += 1
                except Exception as e:
                    pass  # Skip visualization errors

        print(f"\n{'='*60}")
        print(f"CAMERA-ONLY ANNOTATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,}")
        print(f"\n📁 Output: {self.camera_annotation_dir}/")
        print(f"\n🎯 Camera-Only Training Annotations:")
        print(f"   📥 Input: SAM annotations + camera quality analysis")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 Ignore regions: Peripheral areas, low contrast, over/under exposure, tiny objects")
        print(f"   🎯 Purpose: Camera-only model training with quality-aware ignore regions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate camera-only annotations with ignore regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates camera-only segmentation annotations with ignore regions for camera-only model training.

REQUIRES: Run generate_sam.py first!

Strategy:
1. Start with SAM annotations (camera-based object segmentation)
2. Add ignore regions (class 1) for areas unsuitable for training:
   - Peripheral regions (outside central 80% field of view)
   - Low contrast areas (poor visibility)
   - Over/under exposed regions
   - Very small isolated objects (< 20 pixels)

Class mapping:
  0: Background (trainable)
  1: Ignore (not trained on)
  2: Vehicle
  3: Sign
  4: Cyclist
  5: Pedestrian

Usage examples:
  # Generate camera-only annotations
  python generate_camera_only_annotation.py

  # Generate with visualizations
  python generate_camera_only_annotation.py --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations for all processed frames')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/projects/clft-zod/output")

    print("="*60)
    print("Camera-Only Annotation Generator")
    print("="*60)
    print(f"Input: {OUTPUT_ROOT / 'annotation_sam'} + {OUTPUT_ROOT / 'camera'}")
    print(f"Output: {OUTPUT_ROOT / 'annotation_camera_only'}")

    generator = CameraOnlyAnnotationGenerator(
        output_root=OUTPUT_ROOT
    )

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()