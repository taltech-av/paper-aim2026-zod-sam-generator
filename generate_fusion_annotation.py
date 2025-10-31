#!/usr/bin/env python3
"""
Generate fusion-only annotations combining camera and LiDAR data for fusion model training.

This script creates fusion annotations that leverage both camera and LiDAR modalities
to produce more accurate and robust segmentation masks. It combines SAM camera-based
segmentation with LiDAR geometric information for enhanced object detection and
quality-aware ignore regions.

REQUIRES: Run generate_sam.py and generate_lidar_png.py first!
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from scipy import ndimage


class FusionAnnotationGenerator:
    """Generate fusion annotations combining camera and LiDAR data"""

    def __init__(self, output_root, sam_dir='annotation_sam', lidar_dir='lidar_png_enhanced',
                 camera_dir='camera', fusion_dir='annotation_fusion'):
        """Initialize fusion annotation generator

        Args:
            output_root: Path to output root directory
            sam_dir: Directory name for SAM annotations (relative to output_root)
            lidar_dir: Directory name for enhanced LiDAR PNGs (relative to output_root)
            camera_dir: Directory name for camera images (relative to output_root)
            fusion_dir: Directory name for fusion annotations output (relative to output_root)
        """
        self.output_root = Path(output_root)

        # Input directories
        self.sam_annotation_dir = self.output_root / sam_dir
        self.lidar_png_dir = self.output_root / lidar_dir
        self.camera_dir = self.output_root / camera_dir

        # Output directory for fusion annotations
        self.fusion_annotation_dir = self.output_root / fusion_dir
        self.fusion_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.processed_frames_file = self.output_root / "processed_fusion_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # Find frames with all required inputs
        print("Scanning for input files...")

        # Get frame IDs from different sources
        sam_files = list(self.sam_annotation_dir.glob("frame_*.png"))
        sam_frame_ids = {f.stem.replace("frame_", "") for f in sam_files}

        lidar_files = list(self.lidar_png_dir.glob("frame_*.png"))
        lidar_frame_ids = {f.stem.replace("frame_", "") for f in lidar_files}

        camera_files = list(self.camera_dir.glob("frame_*.png"))
        camera_frame_ids = {f.stem.replace("frame_", "") for f in camera_files}

        # Find intersection - frames with all inputs
        available_frame_ids = sam_frame_ids & lidar_frame_ids & camera_frame_ids

        print(f"✓ Found {len(sam_frame_ids):,} SAM annotation files")
        print(f"✓ Found {len(lidar_frame_ids):,} enhanced lidar_png files")
        print(f"✓ Found {len(camera_frame_ids):,} camera image files")
        print(f"✓ Found {len(available_frame_ids):,} frames with all inputs")

        # Convert to sorted list for consistent processing
        self.frame_ids = sorted(list(available_frame_ids))

        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

        # Fusion parameters
        self.lidar_confidence_threshold = 0.3  # Minimum LiDAR density for confident regions
        self.depth_consistency_threshold = 0.8  # Depth consistency check
        self.fusion_dilation_iterations = 2  # Moderate dilation for fusion

    def load_sam_annotation(self, frame_id):
        """Load SAM segmentation mask"""
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None
        return cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)

    def load_enhanced_lidar_png(self, frame_id):
        """Load enhanced LiDAR geometric projection"""
        lidar_path = self.lidar_png_dir / f"frame_{frame_id}.png"
        if not lidar_path.exists():
            return None
        return cv2.imread(str(lidar_path), cv2.IMREAD_UNCHANGED)

    def load_camera_image(self, frame_id):
        """Load camera image for quality analysis"""
        camera_path = self.camera_dir / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return None
        return cv2.imread(str(camera_path), cv2.IMREAD_COLOR)

    def create_fusion_confidence_mask(self, enhanced_lidar_img, sam_annotation):
        """Create confidence mask based on LiDAR-camera fusion analysis

        Returns:
            confidence_mask: Boolean mask where True indicates high confidence regions
        """
        h, w, c = enhanced_lidar_img.shape

        # LiDAR coverage and density analysis
        lidar_coverage = np.any(enhanced_lidar_img > 0, axis=2)

        # Calculate LiDAR density using gaussian filter
        lidar_density = ndimage.gaussian_filter(
            lidar_coverage.astype(float),
            sigma=2,
            mode='constant',
            cval=0
        )

        # High confidence regions: good LiDAR coverage
        high_density_regions = lidar_density > self.lidar_confidence_threshold

        # Depth consistency check (simplified)
        z_channel = enhanced_lidar_img[:, :, 2].astype(float) / 255.0
        # Areas with reasonable depth values (not extreme)
        reasonable_depth = (z_channel > 0.1) & (z_channel < 0.9)
        depth_confident = reasonable_depth & lidar_coverage

        # Combine confidence indicators
        confidence_mask = high_density_regions & depth_confident

        return confidence_mask

    def fuse_camera_lidar_annotations(self, sam_annotation, enhanced_lidar_img, confidence_mask):
        """Fuse camera (SAM) and LiDAR information for improved annotations

        Strategy:
        1. Start with SAM annotations (camera-based)
        2. Use LiDAR to validate and refine object boundaries
        3. Enhance object detection where LiDAR confirms camera detections
        4. Create ignore regions where modalities disagree or have low confidence
        """
        h, w = sam_annotation.shape

        # Initialize fusion annotation with SAM base
        fusion_annotation = sam_annotation.copy()

        # Extract LiDAR geometric information
        x_channel = enhanced_lidar_img[:, :, 0].astype(float)
        y_channel = enhanced_lidar_img[:, :, 1].astype(float)
        z_channel = enhanced_lidar_img[:, :, 2].astype(float)

        # LiDAR coverage mask
        lidar_coverage = np.any(enhanced_lidar_img > 0, axis=2)

        # Strategy 1: Enhance object boundaries using LiDAR confirmation
        for class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            sam_obj_mask = (sam_annotation == class_id)
            lidar_in_obj = sam_obj_mask & lidar_coverage & confidence_mask

            if np.any(lidar_in_obj):
                # Strengthen objects where LiDAR confirms SAM detection
                # Dilate confirmed objects slightly
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                confirmed_dilated = cv2.dilate(lidar_in_obj.astype(np.uint8),
                                             kernel, iterations=1)

                # Only expand into background (don't overwrite other objects)
                valid_expansion = confirmed_dilated & (fusion_annotation == 0)
                fusion_annotation[valid_expansion] = class_id

        # Strategy 2: Create ignore regions for low-confidence areas
        # Areas with LiDAR coverage but low confidence
        low_confidence_lidar = lidar_coverage & ~confidence_mask

        # Areas where SAM has objects but LiDAR doesn't confirm
        sam_objects = np.isin(sam_annotation, [2, 3, 4, 5])
        unconfirmed_objects = sam_objects & ~lidar_coverage

        # Combine low-confidence regions
        ignore_candidates = low_confidence_lidar | unconfirmed_objects

        # Apply ignore regions (class 1) - only on background, don't overwrite objects
        background_ignore = (fusion_annotation == 0) & ignore_candidates
        fusion_annotation[background_ignore] = 1

        # Strategy 3: Handle depth-based ignore regions
        # Very distant objects (> 80m approximate)
        z_norm = z_channel / 255.0
        # Approximate distance thresholding
        distant_regions = (z_norm > 0.85) & lidar_coverage  # Rough approximation

        # Mark distant regions as ignore if they're not confirmed objects
        distant_ignore = distant_regions & (fusion_annotation == 0)
        fusion_annotation[distant_ignore] = 1

        return fusion_annotation

    def create_ignore_regions_fusion(self, sam_annotation, enhanced_lidar_img, camera_img):
        """Create ignore regions specific to fusion training

        Dynamic fusion-specific strategy (aligned with camera/LiDAR):
        1. Dynamic vertical edge strips based on object positions
        2. Sparse LiDAR regions (low confidence)
        3. Extreme depth regions
        4. Do NOT mark object interiors as ignore
        """
        ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        h, w = sam_annotation.shape

        # Strategy 1: Dynamic vertical edge regions (consistent with camera/LiDAR)
        # Find topmost and bottommost objects to define usable region
        has_any_object = np.isin(sam_annotation, [2, 3, 4, 5])
        
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
                
                ignore_mask[:top_ignore_line, :] = True  # Above objects
                ignore_mask[bottom_ignore_line:, :] = True  # Below objects
        else:
            # No objects - mark larger edge strips
            edge_height = int(h * 0.1)
            ignore_mask[:edge_height, :] = True
            ignore_mask[-edge_height:, :] = True

        # Strategy 2: Sparse LiDAR regions (low geometric confidence)
        if enhanced_lidar_img is not None:
            lidar_coverage = np.any(enhanced_lidar_img > 0, axis=2)
            
            # Calculate LiDAR density
            lidar_density = ndimage.gaussian_filter(
                lidar_coverage.astype(float),
                sigma=3,
                mode='constant',
                cval=0
            )
            
            # Mark sparse regions as ignore (< 30% density, consistent with LiDAR-only)
            sparse_lidar = (lidar_density < 0.30) & lidar_coverage
            ignore_mask |= sparse_lidar
            
            # Strategy 3: Extreme depth regions
            z_channel = enhanced_lidar_img[:, :, 2].astype(float)
            # Very close or very far (potential artifacts)
            extreme_depth = ((z_channel < 5) | (z_channel > 250)) & lidar_coverage
            ignore_mask |= extreme_depth

        return ignore_mask

    def create_fusion_annotation(self, frame_id):
        """Create fusion annotation combining camera and LiDAR data"""
        try:
            # Load all required inputs
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            enhanced_lidar_img = self.load_enhanced_lidar_png(frame_id)
            if enhanced_lidar_img is None:
                print(f"  ⚠️  No enhanced lidar_png for {frame_id}")
                return None

            camera_img = self.load_camera_image(frame_id)
            # Camera image is optional for quality analysis

            # Create confidence mask from fusion analysis
            confidence_mask = self.create_fusion_confidence_mask(enhanced_lidar_img, sam_annotation)

            # Fuse camera and LiDAR annotations
            fusion_annotation = self.fuse_camera_lidar_annotations(
                sam_annotation, enhanced_lidar_img, confidence_mask
            )

            # Add fusion-specific ignore regions
            fusion_ignore_mask = self.create_ignore_regions_fusion(
                sam_annotation, enhanced_lidar_img, camera_img
            )

            # Apply ignore regions - ONLY to background, preserve object annotations
            can_be_ignored = (fusion_annotation == 0)
            fusion_annotation[fusion_ignore_mask & can_be_ignored] = 1

            # Apply moderate dilation to confirmed objects
            dilated_annotation = self.apply_fusion_dilation(fusion_annotation, confidence_mask)

            # Clear edge rows to prevent dilation artifacts (consistent with LiDAR)
            edge_rows = self.fusion_dilation_iterations * 2 + 1
            h = dilated_annotation.shape[0]
            
            # Check if objects near edges
            has_objects_top = np.any(np.isin(dilated_annotation[edge_rows:edge_rows*2, :], [2, 3, 4, 5]))
            has_objects_bottom = np.any(np.isin(dilated_annotation[-edge_rows*2:-edge_rows, :], [2, 3, 4, 5]))
            
            top_clear = 2 if has_objects_top else edge_rows
            bottom_clear = 2 if has_objects_bottom else edge_rows
            
            dilated_annotation[:top_clear, :] = 0
            dilated_annotation[-bottom_clear:, :] = 0

            return dilated_annotation

        except Exception as e:
            print(f"  ❌ Error creating fusion annotation for {frame_id}: {e}")
            return None

    def apply_fusion_dilation(self, annotation, confidence_mask):
        """Apply moderate dilation to high-confidence object regions"""
        dilated = annotation.copy()

        # Dilate each object class separately, but only in high-confidence regions
        for class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            obj_mask = (annotation == class_id)
            confident_obj = obj_mask & confidence_mask

            if np.any(confident_obj):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_mask = cv2.dilate(confident_obj.astype(np.uint8),
                                        kernel, iterations=self.fusion_dilation_iterations)

                # Only expand into background regions (not ignore regions)
                valid_expansion = dilated_mask & (annotation == 0)
                dilated[valid_expansion] = class_id

        return dilated

    def process_all_frames(self, create_vis=False):
        """Process all frames and create fusion annotations"""
        print(f"\n🎯 Generating fusion annotations combining camera and LiDAR")
        print(f"Input: SAM + enhanced lidar_png + camera images")
        print(f"Output: {self.fusion_annotation_dir}")
        print(f"Frames to process: {len(self.frame_ids):,}")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory if needed
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "fusion_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Color map for visualization (BGR format for OpenCV)
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([0, 0, 255], dtype=np.uint8),        # Vehicle - Red (BGR)
                3: np.array([0, 255, 255], dtype=np.uint8),      # Sign - Yellow (BGR)
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta (BGR)
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green (BGR)
            }

        for frame_id in tqdm(self.frame_ids, desc="Processing frames"):
            # Check if already exists
            output_path = self.fusion_annotation_dir / f"frame_{frame_id}.png"

            if output_path.exists():
                skip_count += 1
                continue

            # Create fusion annotation
            annotation = self.create_fusion_annotation(frame_id)

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
        print(f"FUSION ANNOTATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,}")
        print(f"\n📁 Output: {self.fusion_annotation_dir}/")
        print(f"\n🎯 Fusion Training Annotations:")
        print(f"   📥 Input: SAM + enhanced lidar_png + camera images")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 Fusion features:")
        print(f"      - Dynamic vertical ignore (adapts to object positions)")
        print(f"      - LiDAR confirmation for enhanced accuracy")
        print(f"      - Sparse LiDAR detection (< 30% density)")
        print(f"      - Depth consistency validation")
        print(f"      - Adaptive edge clearing (preserves near-edge objects)")
        print(f"   📊 Ignore strategy:")
        print(f"      - Dynamic top/bottom strips beyond objects")
        print(f"      - Sparse LiDAR regions")
        print(f"      - Extreme depth regions")
        print(f"   ✓ Consistent with camera/LiDAR ignore strategies")
        print(f"   ✓ Objects preserved: Full annotations, no interior holes")
        print(f"   🎯 Purpose: Multi-modal fusion training with quality-aware regions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fusion annotations combining camera and LiDAR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates fusion segmentation annotations that combine camera and LiDAR modalities
for enhanced object detection and segmentation accuracy.

REQUIRES: Run generate_sam.py and generate_lidar_png.py first!

Fusion Strategy (UPDATED - Scene-Adaptive):
1. Start with SAM camera-based segmentation as foundation
2. Use enhanced LiDAR geometric projections for validation and refinement
3. Enhance object boundaries where LiDAR confirms camera detections
4. **DYNAMIC** ignore regions based on object positions:
   - Vertical edges beyond objects (with 2% margin)
   - Sparse LiDAR regions (< 30% density)
   - Extreme depth regions (< 5 or > 250 normalized)
5. Preserve full object annotations (no interior holes)
6. Adaptive edge clearing after dilation

**Why Background vs Ignore?**
  Background (0): Definite negative samples - model learns "not an object"
                  Loss is calculated and backpropagated
  Ignore (1):     Uncertain/ambiguous areas - model skips these
                  Loss is NOT calculated (masked out in training)

Class mapping:
  0: Background (trainable negative samples)
  1: Ignore (loss masked, not trained on)
  2: Vehicle (fusion-enhanced with LiDAR confirmation)
  3: Sign (fusion-enhanced with LiDAR confirmation)
  4: Cyclist (fusion-enhanced with LiDAR confirmation)
  5: Pedestrian (fusion-enhanced with LiDAR confirmation)

Fusion Advantages:
- Better object boundary refinement using LiDAR geometry
- Dynamic, scene-adaptive ignore regions (consistent with camera/LiDAR)
- Depth-aware filtering for distant or problematic regions
- Multi-modal validation for improved accuracy
- No object interior holes from over-aggressive ignore masking

Usage examples:
  # Generate fusion annotations
  python generate_fusion_annotation.py

  # Generate with visualizations
  python generate_fusion_annotation.py --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations for all processed frames')
    parser.add_argument('--output-root', type=str, default='/media/tom/ml/projects/clft-zod/output',
                       help='Root directory for all outputs (default: /media/tom/ml/projects/clft-zod/output)')
    parser.add_argument('--sam-dir', type=str, default='annotation_sam',
                       help='Directory containing SAM annotations relative to output-root (default: annotation_sam)')
    parser.add_argument('--lidar-dir', type=str, default='lidar_png',
                       help='Directory containing enhanced LiDAR PNGs relative to output-root (default: lidar_png)')
    parser.add_argument('--camera-dir', type=str, default='camera',
                       help='Directory containing camera images relative to output-root (default: camera)')
    parser.add_argument('--fusion-dir', type=str, default='annotation_fusion',
                       help='Output directory for fusion annotations relative to output-root (default: annotation_fusion)')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path(args.output_root)
    SAM_DIR = args.sam_dir
    LIDAR_DIR = args.lidar_dir
    CAMERA_DIR = args.camera_dir
    FUSION_DIR = args.fusion_dir

    print("="*60)
    print("Fusion Annotation Generator")
    print("="*60)
    print(f"Input: SAM + enhanced lidar_png + camera images")
    print(f"Output: {OUTPUT_ROOT / FUSION_DIR}")

    generator = FusionAnnotationGenerator(
        output_root=OUTPUT_ROOT,
        sam_dir=SAM_DIR,
        lidar_dir=LIDAR_DIR,
        camera_dir=CAMERA_DIR,
        fusion_dir=FUSION_DIR
    )

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()