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
import concurrent.futures
import multiprocessing
from functools import partial
import time


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

        # Performance optimization parameters
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers max
        self.batch_size = 50  # Process frames in batches to manage memory

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

        # Calculate LiDAR density and distance for quality assessment
        lidar_density = ndimage.gaussian_filter(
            lidar_coverage.astype(float),
            sigma=3,
            mode='constant',
            cval=0
        )

        # Calculate distance map
        z_norm = z_channel / 255.0
        distance_map = np.exp(z_norm * np.log(101.0))

        # Strategy 1: Enhance object boundaries using LiDAR confirmation
        # Conservative approach: Only enhance where LiDAR points actually exist
        # This prevents unrealistic training targets in fusion training
        for class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            sam_obj_mask = (sam_annotation == class_id)

            # Only work with SAM-LiDAR overlap regions
            sam_lidar_overlap = sam_obj_mask & lidar_coverage

            if np.any(sam_lidar_overlap):
                # Calculate quality metrics for the overlapping regions
                overlap_density = lidar_density[sam_lidar_overlap]
                overlap_distances = distance_map[sam_lidar_overlap]

                # Quality criteria for fusion enhancement
                min_density_for_fusion = 0.15  # Lower threshold for fusion (combines modalities)
                max_distance_for_fusion = 50.0  # Higher distance limit for fusion

                # Find high-quality LiDAR-confirmed regions
                quality_overlap = np.zeros_like(sam_lidar_overlap)
                quality_overlap[sam_lidar_overlap] = (
                    (overlap_density >= min_density_for_fusion) &
                    (overlap_distances <= max_distance_for_fusion)
                )

                # Apply quality filtering and confidence check
                confirmed_regions = quality_overlap & confidence_mask

                if np.any(confirmed_regions):
                    # Dilate confirmed objects slightly for boundary enhancement
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    confirmed_dilated = cv2.dilate(confirmed_regions.astype(np.uint8),
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
        # Approximate distance thresholding
        distant_regions = (z_norm > 0.85) & lidar_coverage  # Rough approximation

        # Mark distant regions as ignore if they're not confirmed objects
        distant_ignore = distant_regions & (fusion_annotation == 0)
        fusion_annotation[distant_ignore] = 1

        return fusion_annotation

    def create_ignore_regions_fusion(self, sam_annotation, enhanced_lidar_img, camera_img):
        """Create minimal ignore regions for fusion training

        Conservative fusion-specific strategy:
        1. Minimal vertical edge strips (fusion combines modalities well)
        2. Sparse LiDAR regions (low confidence)
        3. Extreme depth regions
        4. Do NOT mark large areas beyond objects as ignore
        """
        ignore_mask = np.zeros_like(sam_annotation, dtype=bool)

        h, w = sam_annotation.shape

        # Strategy 1: Minimal vertical edge regions (conservative for fusion)
        # Fusion combines camera + LiDAR, so minimal edge strips
        edge_fraction = 0.01  # Only 1% edges
        edge_height = max(int(h * edge_fraction), 5)  # At least 5 pixels
        
        ignore_mask[:edge_height, :] = True  # Top 1%
        ignore_mask[-edge_height:, :] = True  # Bottom 1%

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
            
            # Mark sparse regions as ignore (< 30% density)
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

    def create_overlay_visualization(self, base_image, colored_mask, title=""):
        """Create overlay visualization with title
        
        Args:
            base_image: Base image (camera image)
            colored_mask: Colored annotation mask
            title: Title text to add
            
        Returns:
            visualization: Image with overlay and title
        """
        h, w = base_image.shape[:2]
        
        # Prepare base image for overlay
        if len(base_image.shape) == 3 and base_image.shape[2] == 3:
            # Color image (camera)
            overlay_base = base_image.copy()
        else:
            # Grayscale image - convert to RGB
            if len(base_image.shape) == 2:
                overlay_base = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
            else:
                overlay_base = base_image.copy()
        
        # Overlay mask with transparency
        alpha = 0.6
        overlay = overlay_base.copy()
        
        # Apply mask only where there are annotations
        mask_pixels = np.any(colored_mask > 0, axis=2)
        overlay[mask_pixels] = cv2.addWeighted(
            overlay_base[mask_pixels], 1-alpha, 
            colored_mask[mask_pixels], alpha, 0
        )
        
        # Add title if provided
        if title:
            # Add black background for text
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(overlay, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return overlay

    def process_single_frame(self, frame_id, create_vis=False, vis_dir=None, color_map=None):
        """Process a single frame and return results for parallel processing
        
        Returns:
            tuple: (frame_id, status, annotation, visualization_data)
        """
        try:
            # Check if annotation already exists
            output_path = self.fusion_annotation_dir / f"frame_{frame_id}.png"
            if output_path.exists():
                # Load existing annotation for potential visualization
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                if annotation is None:
                    return (frame_id, "error", None, None)
            else:
                # Create fusion annotation
                annotation = self.create_fusion_annotation(frame_id)
                
                if annotation is None:
                    return (frame_id, "error", None, None)

                # Save annotation
                cv2.imwrite(str(output_path), annotation)

                # Mark frame as processed
                self.processed_frames.add(frame_id)
                with open(self.processed_frames_file, 'a') as f:
                    f.write(f"{frame_id}\n")

            # Create visualization data if requested and it doesn't exist
            vis_data = None
            if create_vis and vis_dir and color_map:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        # Load camera image for visualization
                        camera_img = self.load_camera_image(frame_id)
                        
                        # Create colored mask for overlay
                        colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                        for class_id, color in color_map.items():
                            if class_id == 0:  # Skip background for overlay
                                continue
                            colored_mask[annotation == class_id] = color
                        
                        if camera_img is not None:
                            # Resize camera image to match annotation dimensions if needed
                            if camera_img.shape[:2] != annotation.shape:
                                camera_img = cv2.resize(camera_img, (annotation.shape[1], annotation.shape[0]), interpolation=cv2.INTER_LINEAR)
                            
                            # Create overlay visualization: Camera image with fusion annotations
                            overlay = self.create_overlay_visualization(camera_img, colored_mask, "Fusion: Camera + LiDAR Annotations")
                            vis_data = overlay
                        else:
                            # Fallback to colored mask only if no camera available
                            vis_data = colored_mask
                            
                    except Exception as e:
                        vis_data = None

            # Return appropriate status
            if output_path.exists() and 'annotation' not in locals():
                return (frame_id, "skip", annotation, vis_data)
            else:
                return (frame_id, "success", annotation, vis_data)
            
        except Exception as e:
            return (frame_id, "error", None, None)

    def process_all_frames(self, create_vis=False):
        """Process all frames and create fusion annotations using parallel processing"""
        start_time = time.time()
        
        print(f"\n🎯 Generating fusion annotations combining camera and LiDAR")
        print(f"Input: SAM + enhanced lidar_png + camera images")
        print(f"Output: {self.fusion_annotation_dir}")
        if create_vis:
            print(f"Visualizations: Camera images with fusion annotation overlays")
        print(f"Frames to process: {len(self.frame_ids):,}")
        print(f"Using {self.max_workers} parallel workers")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory and color map if needed
        vis_dir = None
        color_map = None
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

        # Process frames in batches to manage memory
        frame_batches = [self.frame_ids[i:i + self.batch_size] 
                        for i in range(0, len(self.frame_ids), self.batch_size)]
        
        with tqdm(total=len(self.frame_ids), desc="Processing frames") as pbar:
            for batch in frame_batches:
                # Process batch in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Create partial function with fixed parameters
                    process_func = partial(self.process_single_frame, 
                                         create_vis=create_vis, 
                                         vis_dir=vis_dir, 
                                         color_map=color_map)
                    
                    # Submit all frames in batch
                    futures = [executor.submit(process_func, frame_id) for frame_id in batch]
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            frame_id, status, annotation, vis_data = future.result()
                            
                            if status == "skip":
                                skip_count += 1
                            elif status == "error":
                                error_count += 1
                            elif status == "success":
                                success_count += 1
                                
                                # Mark frame as processed
                                self.processed_frames.add(frame_id)
                                with open(self.processed_frames_file, 'a') as f:
                                    f.write(f"{frame_id}\n")
                            
                            # Save visualization if available (for both existing and new annotations)
                            if vis_data is not None and vis_dir:
                                vis_path = vis_dir / f"frame_{frame_id}.png"
                                cv2.imwrite(str(vis_path), vis_data)
                                vis_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            print(f"  ❌ Error processing frame result: {e}")
                        
                        pbar.update(1)

        # Calculate and display performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        processed_frames = success_count + error_count
        
        if processed_frames > 0:
            time_per_frame = total_time / processed_frames
            frames_per_second = processed_frames / total_time
            
            print(f"\n⏱️  Performance Metrics:")
            print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"   Time per frame: {time_per_frame:.3f}s")
            print(f"   Processing rate: {frames_per_second:.2f} frames/sec")
            print(f"   Parallel efficiency: {self.max_workers} workers")

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
        print(f"   🔧 Minimal Ignore Strategy for Fusion:")
        print(f"      - Fusion combines modalities, minimal ignore regions")
        print(f"      - Only 1% top/bottom edges marked as ignore")
        print(f"      - Sparse LiDAR detection (< 30% density)")
        print(f"      - Extreme depth regions")
        print(f"      - Maximize trainable fusion regions")
        print(f"   📊 Ignore strategy:")
        print(f"      - Dynamic top/bottom strips beyond objects")
        print(f"      - Sparse LiDAR regions")
        print(f"      - Extreme depth regions")
        print(f"   ✓ Consistent with camera/LiDAR ignore strategies")
        print(f"   ✓ Objects preserved: Full annotations, no interior holes")
        if create_vis:
            print(f"   📸 Visualizations: Camera images with fusion annotation overlays")
        print(f"   🎯 Purpose: Multi-modal fusion training with quality-aware regions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fusion annotations combining camera and LiDAR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates fusion segmentation annotations that combine camera and LiDAR modalities
for enhanced object detection and segmentation accuracy.

PERFORMANCE OPTIMIZATIONS:
- Parallel processing using multiple CPU cores
- Batch processing to manage memory usage
- Efficient vectorized operations for faster computation

REQUIRES: Run generate_sam.py and generate_lidar_png.py first!

Fusion Strategy (UPDATED - Minimal Ignore):
1. Start with SAM camera-based segmentation as foundation
2. Use enhanced LiDAR geometric projections for validation and refinement
3. Enhance object boundaries where LiDAR confirms camera detections
4. **MINIMAL** ignore regions for fusion training:
   - Only 1% top/bottom edges as ignore (conservative)
   - Sparse LiDAR regions (< 30% density)
   - Extreme depth regions (< 5 or > 250 normalized)
5. Preserve full object annotations (no interior holes)
6. Adaptive edge clearing after dilation
7. **Camera-based visualizations**: Overlay annotations on camera images

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
- **Camera-based visualization**: See annotations in context of actual scene

Usage examples:
  # Generate fusion annotations (optimized)
  python generate_fusion_annotation.py

  # Generate with camera-based visualizations
  python generate_fusion_annotation.py --visualize

  # Use custom number of workers
  python generate_fusion_annotation.py --workers 4 --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations for all processed frames')
    parser.add_argument('--output-root', type=str, default='/media/tom/ml/zod_temp',
                       help='Root directory for all outputs (default: /media/tom/ml/zod_temp)')
    parser.add_argument('--sam-dir', type=str, default='annotation_sam',
                       help='Directory containing SAM annotations relative to output-root (default: annotation_sam)')
    parser.add_argument('--lidar-dir', type=str, default='lidar_png',
                       help='Directory containing enhanced LiDAR PNGs relative to output-root (default: lidar_png)')
    parser.add_argument('--camera-dir', type=str, default='camera',
                       help='Directory containing camera images relative to output-root (default: camera)')
    parser.add_argument('--fusion-dir', type=str, default='annotation_fusion',
                       help='Output directory for fusion annotations relative to output-root (default: annotation_fusion)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect, max 8)')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path(args.output_root)
    SAM_DIR = args.sam_dir
    LIDAR_DIR = args.lidar_dir
    CAMERA_DIR = args.camera_dir
    FUSION_DIR = args.fusion_dir

    print("="*60)
    print("Fusion Annotation Generator (Optimized)")
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

    # Override parameters if specified
    if hasattr(args, 'workers') and args.workers:
        generator.max_workers = min(args.workers, multiprocessing.cpu_count())
        print(f"Using {generator.max_workers} workers")

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()