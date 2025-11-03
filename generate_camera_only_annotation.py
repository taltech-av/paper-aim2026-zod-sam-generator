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

        MINIMAL ignore strategy for camera-only models:
        1. Camera sensors provide clear, undistorted views
        2. Only mark tiny edge strips (1% of height) as ignore for safety
        3. Remove very small objects (< 25 pixels) that are likely noise
        4. Keep all other areas as trainable background
        
        Purpose of classes:
        - Background (0): Empty areas, safe negative samples for training
        - Ignore (1): Don't compute loss here - truly uncertain/distorted areas
        """
        ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        h, w = sam_annotation.shape

        # Strategy 1: Minimal edge strips only (camera has clear view)
        # Mark only 1% of top/bottom edges as ignore (much more conservative)
        edge_fraction = 0.01  # Reduced from 0.1/0.02
        edge_height = max(int(h * edge_fraction), 5)  # At least 5 pixels
        
        peripheral_mask = np.zeros_like(sam_annotation, dtype=bool)
        peripheral_mask[:edge_height, :] = True  # Top 1%
        peripheral_mask[-edge_height:, :] = True  # Bottom 1%

        # Strategy 2: Mark very small objects as ignore (noise reduction)
        objects_to_ignore_mask = np.zeros_like(sam_annotation, dtype=bool)
        
        for class_id in [2, 3, 4, 5]:  # object classes
            obj_mask = (sam_annotation == class_id)
            if np.any(obj_mask):
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    obj_mask.astype(np.uint8), connectivity=8)

                # Check each object
                for label_id in range(1, num_labels):
                    object_region = (labels == label_id)
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    
                    # More aggressive small object removal (reduced threshold)
                    is_too_small = area < 25  # Reduced from 50
                    
                    # Mark entire object for ignore if too small
                    if is_too_small:
                        objects_to_ignore_mask |= object_region

        # Combine: minimal edge strips + tiny noise objects
        ignore_mask = peripheral_mask | objects_to_ignore_mask

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
            # IMPORTANT: Only apply ignore to background regions (class 0)
            # Do NOT override object classes (2, 3, 4, 5) with ignore
            can_be_ignored = (annotation == 0)
            annotation[ignore_mask & can_be_ignored] = 1

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

            # Color map for visualization (BGR format for OpenCV) - used for overlay
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Transparent in overlay
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([0, 0, 255], dtype=np.uint8),        # Vehicle - Red
                3: np.array([0, 255, 255], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
            }

        for frame_id in tqdm(self.frame_ids, desc="Processing frames"):
            output_path = self.camera_annotation_dir / f"frame_{frame_id}.png"

            # Check if annotation already exists
            if output_path.exists():
                skip_count += 1
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            else:
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

            # Create visualization if requested and it doesn't exist
            if create_vis and annotation is not None:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        # Load original camera image
                        camera_img = self.load_camera_image(frame_id)
                        if camera_img is None:
                            # Fallback to colored mask only if camera image not found
                            colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                colored_annotation[annotation == class_id] = color
                            cv2.imwrite(str(vis_path), colored_annotation)
                        else:
                            # Create colored mask
                            colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                if class_id == 0:  # Skip background for overlay
                                    continue
                                colored_mask[annotation == class_id] = color
                            
                            # Overlay mask on camera image with transparency
                            alpha = 0.6  # Transparency level for mask
                            overlay = camera_img.copy()
                            
                            # Apply mask only where there are annotations (non-background)
                            mask_pixels = (annotation > 0)
                            overlay[mask_pixels] = cv2.addWeighted(
                                camera_img[mask_pixels], 1-alpha, 
                                colored_mask[mask_pixels], alpha, 0
                            )
                            
                            cv2.imwrite(str(vis_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        
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
        print(f"   📥 Input: SAM annotations")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 Minimal Ignore Strategy for Camera:")
        print(f"      - Camera provides clear, undistorted views")
        print(f"      - Only 1% top/bottom edges marked as ignore")
        print(f"      - Very small objects (< 25 pixels) removed as noise")
        print(f"      - Maximize trainable background areas")
        print(f"   ✓ Full horizontal width preserved")
        print(f"   ✓ Minimal ignore regions per frame")
        print(f"   📊 Class Purpose:")
        print(f"      - Background (0): Empty areas, negative samples (LOSS APPLIED)")
        print(f"      - Ignore (1): Uncertain areas, distortion zones (NO LOSS)")
        print(f"   🎯 Purpose: Scene-adaptive annotations for optimal training")


def main():
    parser = argparse.ArgumentParser(
        description="Generate camera-only annotations with ignore regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates camera-only segmentation annotations with ignore regions for camera-only model training.

REQUIRES: Run generate_sam.py first!

Strategy:
1. Start with SAM annotations (camera-based object segmentation)
2. **MINIMAL** ignore regions for camera-only training:
   - Camera sensors provide clear, undistorted views
   - Mark only 1% of top/bottom edges as ignore (safety margin)
   - Remove very small objects (< 25 pixels) that are likely noise
   - Keep 98%+ of image as trainable background
3. Preserve full horizontal width
4. Maximize training data for optimal camera-only performance

**Why Background vs Ignore?**
  Background (0): Definite negative samples - model learns "not an object"
                  Loss is calculated and backpropagated
  Ignore (1):     Uncertain/ambiguous areas - model skips these
                  Loss is NOT calculated (masked out in training)

Class mapping:
  0: Background (trainable negative samples)
  1: Ignore (loss masked, not trained on)
  2: Vehicle
  3: Sign
  4: Cyclist
  5: Pedestrian

Usage examples:
  # Generate camera-only annotations
  python generate_camera_only_annotation.py

  # Generate with visualizations (masks overlaid on camera images)
  python generate_camera_only_annotation.py --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations overlaying masks on original camera images for visual verification')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/zod_temp")

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