#!/usr/bin/env python3
"""
Generate camera-only annotations with ignore regions for camera-only model training.

This script takes SAM annotations and adds minimal ignore regions (class 1) for areas that
should not be trained on in camera-only models. Camera sensors provide clear, undistorted views,
so only tiny edge strips and noise objects are marked as ignore to maximize trainable data.

REQUIRES: Run generate_sam.py first!

Key Features:
- Minimal ignore strategy optimized for camera sensors
- Progress tracking and resume capability
- Optional visualization with mask overlays
- Quality filtering to remove noise objects
- Preserves maximum training data for optimal performance
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse


class CameraOnlyAnnotationGenerator:
    """
    Generate camera-only annotations with minimal ignore regions.

    This class implements a camera-optimized annotation strategy that:
    - Starts with SAM segmentation masks as ground truth
    - Adds only minimal ignore regions for camera-only training
    - Preserves clear camera views with maximum trainable background
    - Removes noise objects while maintaining scene understanding
    - Supports progress tracking and visualization
    """

    def __init__(self, output_root):
        """
        Initialize camera-only annotation generator with directory structure.

        Sets up input/output directories, discovers available SAM annotations,
        and configures processing parameters for efficient batch generation.

        Args:
            output_root: Path to output_clft_v2 directory containing SAM annotations
        """
        self.output_root = Path(output_root)

        # ===== INPUT DIRECTORY =====
        # SAM annotations provide the base segmentation for camera objects
        self.sam_annotation_dir = self.output_root / "annotation_sam"

        # ===== OUTPUT DIRECTORY =====
        # Camera-only annotations with minimal ignore regions
        self.camera_annotation_dir = self.output_root / "annotation_camera_only"
        self.camera_annotation_dir.mkdir(parents=True, exist_ok=True)

        # ===== PROGRESS TRACKING =====
        # Resume capability - track already processed frames
        self.processed_frames_file = self.output_root / "processed_camera_only_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # ===== FRAME DISCOVERY =====
        # Automatically find frames with SAM annotations available
        print("🔍 Scanning for SAM annotation files...")
        sam_files = list(self.sam_annotation_dir.glob("frame_*.png"))
        self.frame_ids = []
        for sam_file in sam_files:
            frame_id = sam_file.stem.replace("frame_", "")
            self.frame_ids.append(frame_id)

        print(f"✓ Found {len(self.frame_ids):,} SAM annotation files")

        # Filter out already processed frames for resume capability
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

    def load_sam_annotation(self, frame_id):
        """
        Load SAM segmentation mask as the base for camera-only annotations.

        SAM provides high-quality object segmentation that serves as the foundation
        for camera-only training. These masks identify object locations and types
        that will be preserved in the final annotations.

        Args:
            frame_id: Frame identifier string

        Returns:
            np.ndarray or None: Grayscale segmentation mask, or None if not found
        """
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None
        return cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)

    def load_camera_image(self, frame_id):
        """
        Load camera image for optional quality analysis and visualization.

        Camera images are used for creating visualizations that overlay annotations
        on the original camera view, helping verify annotation quality and alignment.

        Args:
            frame_id: Frame identifier string

        Returns:
            np.ndarray or None: Color camera image, or None if not found
        """
        camera_path = self.output_root / "camera" / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return None
        return cv2.imread(str(camera_path), cv2.IMREAD_COLOR)

    def create_ignore_regions(self, sam_annotation, camera_img=None):
        """
        Create minimal ignore regions (class 1) optimized for camera-only training.

        This method implements a conservative ignore strategy specifically designed
        for camera sensors, which provide clear, undistorted views. Only tiny edge
        strips and noise objects are marked as ignore to maximize trainable data.

        Strategy rationale:
        - Camera sensors have excellent visibility and minimal distortion
        - Peripheral regions are still clear and useful for training
        - Only mark minimal safety margins to avoid edge artifacts
        - Remove noise while preserving scene understanding

        Args:
            sam_annotation: SAM segmentation mask with object classes
            camera_img: Optional camera image for quality analysis (not used in current implementation)

        Returns:
            ignore_mask: Boolean mask where True indicates ignore regions (class 1)
        """
        ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        h, w = sam_annotation.shape

        # ===== STRATEGY 1: MINIMAL EDGE STRIPS =====
        # Camera sensors provide clear views, so only tiny edge strips needed for safety
        # Much more conservative than LiDAR (which has blind spots and distortion)
        edge_fraction = 0.01  # Only 1% of image height (vs 5%+ for LiDAR)
        edge_height = max(int(h * edge_fraction), 5)  # At least 5 pixels minimum

        peripheral_mask = np.zeros_like(sam_annotation, dtype=bool)
        peripheral_mask[:edge_height, :] = True  # Top 1% - safety margin only
        peripheral_mask[-edge_height:, :] = True  # Bottom 1% - safety margin only

        # ===== STRATEGY 2: NOISE OBJECT REMOVAL =====
        # Remove very small objects that are likely segmentation noise or artifacts
        # This improves training quality by eliminating false positive training targets
        objects_to_ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        # Process each object class to identify noise objects
        for class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            obj_mask = (sam_annotation == class_id)
            if np.any(obj_mask):
                # Find connected components to identify individual objects
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    obj_mask.astype(np.uint8), connectivity=8)

                # Evaluate each detected object
                for label_id in range(1, num_labels):
                    object_region = (labels == label_id)
                    area = stats[label_id, cv2.CC_STAT_AREA]

                    # Mark small objects as ignore (likely noise/artifacts)
                    # Threshold chosen to remove obvious noise while preserving small valid objects
                    is_too_small = area < 25  # Very conservative threshold

                    if is_too_small:
                        objects_to_ignore_mask |= object_region

        # ===== COMBINE IGNORE CRITERIA =====
        # Final ignore mask: minimal edge strips + noise objects
        # This preserves 98%+ of the image as trainable data
        ignore_mask = peripheral_mask | objects_to_ignore_mask

        return ignore_mask

    def create_camera_only_annotation(self, frame_id):
        """
        Create camera-only annotation with minimal ignore regions for a single frame.

        This method orchestrates the complete annotation pipeline for one frame:
        1. Load SAM annotation as the base segmentation
        2. Load camera image for optional quality analysis
        3. Generate minimal ignore regions optimized for camera sensors
        4. Apply ignore regions only to background areas (preserve object classes)

        The key principle is to maximize trainable data by using minimal ignore regions,
        since camera sensors provide clear, undistorted views.

        Args:
            frame_id: Frame identifier string to process

        Returns:
            np.ndarray or None: Camera-only segmentation mask, or None if processing failed
        """
        try:
            # ===== LOAD REQUIRED SAM ANNOTATION =====
            # SAM provides the base object segmentation for camera training
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            # ===== LOAD OPTIONAL CAMERA IMAGE =====
            # Used for visualization, not currently for ignore region creation
            camera_img = self.load_camera_image(frame_id)

            # ===== START WITH SAM ANNOTATION =====
            # Preserve all SAM object classes and background
            annotation = sam_annotation.copy()

            # ===== CREATE MINIMAL IGNORE REGIONS =====
            # Generate ignore mask optimized for camera sensors
            ignore_mask = self.create_ignore_regions(sam_annotation, camera_img)

            # ===== APPLY IGNORE REGIONS SELECTIVELY =====
            # CRITICAL: Only apply ignore to background regions (class 0)
            # NEVER override object classes (2, 3, 4, 5) with ignore
            # This preserves all detected objects while marking uncertain background areas
            can_be_ignored = (annotation == 0)  # Only background can become ignore
            annotation[ignore_mask & can_be_ignored] = 1  # Apply ignore class

            return annotation

        except Exception as e:
            print(f"  ❌ Error creating camera-only annotation for {frame_id}: {e}")
            return None

    def process_all_frames(self, create_vis=False):
        """
        Process all frames and create camera-only annotations with optional visualization.

        This method orchestrates the complete batch processing pipeline:
        1. Sets up visualization infrastructure if requested
        2. Processes each frame with progress tracking
        3. Generates overlay visualizations for quality verification
        4. Reports comprehensive statistics and processing summary

        Args:
            create_vis: Whether to create overlay visualizations on camera images
        """
        print(f"\n🎯 Generating camera-only annotations with ignore regions")
        print(f"Input: SAM annotations + camera images")
        print(f"Output: {self.camera_annotation_dir}")
        print(f"Frames to process: {len(self.frame_ids):,}")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # ===== VISUALIZATION SETUP =====
        # Prepare color mapping and directories for optional visualizations
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "camera_only_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Color map for visualization (BGR format for OpenCV overlay)
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Transparent in overlay
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([0, 0, 255], dtype=np.uint8),        # Vehicle - Red
                3: np.array([0, 255, 255], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
            }

        # ===== FRAME PROCESSING LOOP =====
        for frame_id in tqdm(self.frame_ids, desc="Processing frames"):
            output_path = self.camera_annotation_dir / f"frame_{frame_id}.png"

            # Check for existing annotation (resume capability)
            if output_path.exists():
                skip_count += 1
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            else:
                # ===== CREATE NEW ANNOTATION =====
                annotation = self.create_camera_only_annotation(frame_id)

                if annotation is None:
                    error_count += 1
                    continue

                # Save annotation to disk
                cv2.imwrite(str(output_path), annotation)
                success_count += 1

                # Mark frame as processed for resume capability
                self.processed_frames.add(frame_id)
                with open(self.processed_frames_file, 'a') as f:
                    f.write(f"{frame_id}\n")

            # ===== CREATE VISUALIZATION =====
            # Generate overlay visualization if requested and annotation exists
            if create_vis and annotation is not None:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        # Load original camera image for overlay base
                        camera_img = self.load_camera_image(frame_id)
                        if camera_img is None:
                            # Fallback: Create colored mask visualization only
                            colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                colored_annotation[annotation == class_id] = color
                            cv2.imwrite(str(vis_path), colored_annotation)
                        else:
                            # ===== CREATE OVERLAY VISUALIZATION =====
                            # Generate colored mask for overlay (skip background for transparency)
                            colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in color_map.items():
                                if class_id == 0:  # Skip background for cleaner overlay
                                    continue
                                colored_mask[annotation == class_id] = color

                            # Apply transparent overlay on camera image
                            alpha = 0.6  # Transparency level for annotations
                            overlay = camera_img.copy()

                            # Only overlay where there are annotations (non-background)
                            mask_pixels = (annotation > 0)
                            overlay[mask_pixels] = cv2.addWeighted(
                                camera_img[mask_pixels], 1-alpha,  # Original image reduced opacity
                                colored_mask[mask_pixels], alpha, 0  # Colored mask with transparency
                            )

                            cv2.imwrite(str(vis_path), overlay)

                        vis_count += 1
                    except Exception as e:
                        # Skip visualization errors silently
                        pass

        # ===== FINAL REPORT =====
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