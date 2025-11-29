#!/usr/bin/env python3
"""
Generate LiDAR-only annotations from basic lidar_png using SAM as ground truth.

This script creates LiDAR-native segmentation annotations for LiDAR-only model training by:
1. Using basic lidar_png geometric projections for LiDAR coverage analysis
2. Using SAM annotations as ground truth for object locations and types
3. Mapping SAM-identified objects onto actual LiDAR points for LiDAR-native annotations
4. Maintaining exact geometric correspondence with LiDAR PNG coordinates (no dilation)

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
import concurrent.futures
import multiprocessing
from functools import partial
import time


class LiDARPNGAnnotationGenerator:
    """Generate LiDAR-only annotations from lidar_png using SAM guidance"""

    def __init__(self, output_root):
        """Initialize LiDAR annotation generator

        Args:
            output_root: Path to output_clft_v2 directory
        """
        self.output_root = Path(output_root)

        # Input directories
        self.lidar_png_dir = self.output_root / "lidar_png"  # Use basic version
        self.sam_annotation_dir = self.output_root / "annotation_sam"
        self.camera_dir = self.output_root / "camera"  # Add camera images for comparison

        # Output directory for LiDAR annotations
        self.lidar_annotation_dir = self.output_root / "annotation_lidar_only"
        self.lidar_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.processed_frames_file = self.output_root / "processed_lidar_png_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # Find frames that have both lidar_png and SAM annotation
        print("Scanning for available input files...")
        
        # Get frame IDs from lidar_png files
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

        # LiDAR annotation parameters - RELAXED for better training data
        self.density_threshold = 0.10  # REDUCED from 0.15 - More permissive for training
        self.distance_threshold = 90.0  # INCREASED from 70.0 - Include more distant objects
        
        # Performance optimization parameters
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers max
        self.batch_size = 50  # Process frames in batches to manage memory

    def compute_density_map_efficient(self, lidar_coverage, window_size=7):
        """Compute density map using efficient sliding window approach
        
        Args:
            lidar_coverage: Boolean array indicating LiDAR coverage
            window_size: Size of sliding window for density calculation
            
        Returns:
            density_map: Float array with local density values
        """
        # Use uniform filter for faster density calculation
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
        density_map = cv2.filter2D(lidar_coverage.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Clip to [0, 1] range
        density_map = np.clip(density_map, 0, 1)
        
        return density_map

    def load_lidar_png(self, frame_id):
        """Load LiDAR geometric projection"""
        lidar_path = self.lidar_png_dir / f"frame_{frame_id}.png"
        if not lidar_path.exists():
            return None

        # Load 3-channel geometric projection
        lidar_img = cv2.imread(str(lidar_path), cv2.IMREAD_UNCHANGED)
        return lidar_img

    def load_sam_annotation(self, frame_id):
        """Load SAM segmentation mask"""
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None

        annotation = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)
        return annotation

    def load_camera_image(self, frame_id):
        """Load camera image for visualization comparison"""
        camera_path = self.camera_dir / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return None

        camera_img = cv2.imread(str(camera_path), cv2.IMREAD_COLOR)
        return camera_img

    def create_lidar_native_annotation(self, lidar_img, sam_annotation):
        """Create LiDAR-native annotation using lidar_png and SAM guidance

        Strategy:
        1. Use lidar_png for LiDAR coverage (permissive to match PNG content)
        2. Use SAM as ground truth for object locations and types
        3. Map SAM-identified objects onto LiDAR-covered regions
        4. Create LiDAR-native segmentation with proper class distinctions

        Args:
            lidar_img: 3-channel geometric projection
            sam_annotation: SAM segmentation mask (ground truth)

        Returns:
            annotation: LiDAR-native segmentation mask
        """
        if lidar_img is None or sam_annotation is None:
            return None

        h, w, c = lidar_img.shape

        # Initialize annotation with background (0)
        annotation = np.zeros((h, w), dtype=np.uint8)

        # Extract geometric information from lidar_png
        x_channel = lidar_img[:, :, 0].astype(float)  # X coordinates (normalized)
        y_channel = lidar_img[:, :, 1].astype(float)  # Y coordinates (normalized)
        z_channel = lidar_img[:, :, 2].astype(float)  # Z coordinates (depth, normalized)

        # Create LiDAR coverage mask that aligns with LiDAR PNG
        # The LiDAR PNG uses basic format with interpretable coordinates, so we should accept
        # regions where ANY channel has meaningful data (not require ALL channels)
        
        # MORE PERMISSIVE coverage: accept regions where LiDAR PNG has data
        # This ensures annotations align with actual LiDAR PNG content
        lidar_coverage = np.any(lidar_img > 3, axis=2)  # REDUCED threshold from 5
        
        # For stricter validation, also check that we don't have extremely distant points
        # that are likely just smoothing artifacts - LESS conservative for training
        z_channel_norm = z_channel / 255.0
        distance_approx = z_channel_norm * 100.0  # Linear distance calculation
        reasonable_distance = distance_approx < 100.0  # INCREASED from 80.0 - More permissive
        
        # Final coverage: has LiDAR data AND not extremely distant
        lidar_coverage = lidar_coverage & reasonable_distance

        # Decode distance from normalization for density calculations
        z_norm = z_channel / 255.0
        distance_map = z_norm * 100.0

        # 1. Background class (0): Areas with no LiDAR coverage
        # (already initialized to 0)

        # 2. Ignore class (1): Sparse LiDAR regions, distant points, AND edge regions
        # Strategy A: Sparse/distant detection (quality-based ignore) - LESS aggressive
        # Create density map using efficient sliding window approach
        lidar_density = self.compute_density_map_efficient(lidar_coverage.astype(np.uint8))

        # Areas with sparse LiDAR coverage (< 25% local density) - INCREASED from 30%
        # BUT only where we actually have some LiDAR points
        sparse_regions = (lidar_density < 0.25) & lidar_coverage

        # Areas beyond distance threshold (approximate) - INCREASED threshold
        distant_regions = (distance_map > self.distance_threshold) & lidar_coverage

        # Strategy B: Minimal vertical edge regions (conservative for LiDAR)
        # LiDAR covers the scene well, so use minimal edge strips
        vertical_edge_regions = np.zeros_like(sam_annotation, dtype=bool)
        
        # Only mark tiny edge strips (0.5% of height) as ignore for safety - REDUCED from 1%
        edge_fraction = 0.005  # REDUCED from 0.01
        edge_height = max(int(h * edge_fraction), 3)  # At least 3 pixels - REDUCED from 5
        
        vertical_edge_regions[:edge_height, :] = True  # Top 0.5%
        vertical_edge_regions[-edge_height:, :] = True  # Bottom 0.5%
        
        # Combine all ignore criteria
        ignore_regions = sparse_regions | distant_regions | vertical_edge_regions
        annotation[ignore_regions] = 1

        # 3. Objects from SAM guidance: Map onto LiDAR points
        # LiDAR-native approach: Only annotate where LiDAR points actually exist
        # This ensures training targets match available input data
        
        # Quality thresholds for LiDAR-native object detection - RELAXED
        min_density_for_object = 0.12  # REDUCED from 0.20 - More permissive
        min_region_size = 5  # REDUCED from 8 - Allow smaller objects
        max_distance_for_full_mask = 60.0  # INCREASED from 40.0 - Include more distant objects
        
        # For each SAM object class, only annotate LiDAR points that:
        # 1. Have sufficient local density
        # 2. Are at reasonable distance  
        # 3. Actually exist (have LiDAR coverage)
        # 4. Are not in ignore regions
        for sam_class_id in [2, 3, 4, 5]:  # vehicle, sign, cyclist, pedestrian
            sam_class_mask = (sam_annotation == sam_class_id)
            
            # Only annotate where LiDAR points actually exist
            sam_lidar_overlap = sam_class_mask & lidar_coverage
            
            if np.sum(sam_lidar_overlap) >= min_region_size:
                # Within the LiDAR-covered SAM regions, apply quality filtering
                overlap_pixels = np.sum(sam_lidar_overlap)
                
                # Check quality criteria for the LiDAR-covered regions
                overlap_density = lidar_density[sam_lidar_overlap]
                overlap_distances = distance_map[sam_lidar_overlap]
                
                # Quality mask: only high-quality LiDAR points get object labels
                quality_mask = np.zeros_like(sam_lidar_overlap, dtype=bool)
                quality_mask[sam_lidar_overlap] = (
                    (overlap_density >= min_density_for_object) & 
                    (overlap_distances <= max_distance_for_full_mask)
                )
                
                # Apply to non-ignore regions
                final_mask = quality_mask & ~ignore_regions
                
                if np.sum(final_mask) >= min_region_size:
                    annotation[final_mask] = sam_class_id

        # 4. Final cleanup: Ensure areas without LiDAR coverage are background
        # But preserve ignore regions that have sparse LiDAR coverage
        no_lidar = ~lidar_coverage
        annotation[no_lidar] = 0  # Force background only where there's NO LiDAR at all
        # Do NOT overwrite ignore regions that were set in step 2

        return annotation

    def create_overlay_visualization(self, base_image, colored_mask, title=""):
        """Create overlay visualization with title
        
        Args:
            base_image: Base image (LiDAR or camera)
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
            # LiDAR image - normalize for visualization
            if len(base_image.shape) == 3:
                base_channel = base_image[:, :, 0].astype(float)
            else:
                base_channel = base_image.astype(float)
            
            base_channel = (base_channel - base_channel.min()) / (base_channel.max() - base_channel.min() + 1e-6)
            base_channel = (base_channel * 255).astype(np.uint8)
            overlay_base = cv2.cvtColor(base_channel, cv2.COLOR_GRAY2RGB)
        
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

    def create_lidar_png_annotation(self, frame_id):
        """Create LiDAR-only annotation for a frame using lidar_png"""
        try:
            # Load LiDAR geometric projection
            lidar_img = self.load_lidar_png(frame_id)
            if lidar_img is None:
                # print(f"  ⚠️  No lidar_png for {frame_id}")
                return None

            # Load SAM annotation for object locations
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                # print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            # Create LiDAR-native annotation using lidar_png and SAM guidance
            annotation = self.create_lidar_native_annotation(lidar_img, sam_annotation)

            return annotation

        except Exception as e:
            # print(f"  ❌ Error creating LiDAR annotation for {frame_id}: {e}")
            return None

    def process_single_frame(self, frame_id, create_vis=False, vis_dir=None, full_color_map=None):
        """Process a single frame and return results for parallel processing
        
        Returns:
            tuple: (frame_id, status, annotation, visualization_data)
        """
        try:
            # Check if annotation already exists
            output_path = self.lidar_annotation_dir / f"frame_{frame_id}.png"
            if output_path.exists():
                # Load existing annotation for potential visualization
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                if annotation is None:
                    return (frame_id, "error", None, None)
            else:
                # Create LiDAR annotation
                annotation = self.create_lidar_png_annotation(frame_id)
                
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
            if create_vis and vis_dir and full_color_map:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        # Load images for comparison
                        lidar_img = self.load_lidar_png(frame_id)
                        camera_img = self.load_camera_image(frame_id)
                        
                        if lidar_img is not None:
                            # Create colored mask for overlay
                            colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in full_color_map.items():
                                if class_id == 0:  # Skip background for overlay
                                    continue
                                colored_mask[annotation == class_id] = color
                            
                            # Create stacked visualization: LiDAR top, Camera bottom
                            h, w = annotation.shape
                            
                            # Top half: LiDAR annotation on LiDAR PNG
                            lidar_overlay = self.create_overlay_visualization(lidar_img, colored_mask, "LiDAR + Annotations")
                            
                            # Bottom half: LiDAR annotation on Camera image (if available)
                            if camera_img is not None:
                                # Resize camera image to match annotation dimensions if needed
                                if camera_img.shape[:2] != (h, w):
                                    camera_img = cv2.resize(camera_img, (w, h), interpolation=cv2.INTER_LINEAR)
                                
                                camera_overlay = self.create_overlay_visualization(camera_img, colored_mask, "Camera + LiDAR Annotations")
                                
                                # Ensure consistent color space (BGR) before stacking
                                if lidar_overlay.shape[2] == 3:
                                    lidar_overlay = cv2.cvtColor(lidar_overlay, cv2.COLOR_RGB2BGR)
                                # camera_overlay is already BGR from create_overlay_visualization
                                
                                # Stack vertically: LiDAR on top, Camera on bottom
                                vis_data = np.vstack([lidar_overlay, camera_overlay])
                            else:
                                # Fallback: Just LiDAR overlay if no camera available
                                vis_data = cv2.cvtColor(lidar_overlay, cv2.COLOR_RGB2BGR) if lidar_overlay.shape[2] == 3 else lidar_overlay
                        else:
                            # Fallback to colored mask only
                            colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in full_color_map.items():
                                colored_annotation[annotation == class_id] = color
                            vis_data = colored_annotation
                            
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
        """Process all frames and create LiDAR annotations using parallel processing

        Args:
            create_vis: Whether to create visualizations
        """
        start_time = time.time()
        
        print(f"\n🎯 Generating LiDAR-only annotations from lidar_png")
        print(f"Input: {self.lidar_png_dir}")
        print(f"SAM guidance: {self.sam_annotation_dir}")
        print(f"Output: {self.lidar_annotation_dir}")
        print(f"Frames to process: {len(self.frame_ids):,}")
        print(f"Using {self.max_workers} parallel workers")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        # Prepare visualization directory and color map if needed
        vis_dir = None
        full_color_map = None
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "lidar_only_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Color map for visualization (consistent with SAM)
            full_color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([255, 0, 0], dtype=np.uint8),        # Vehicle - Red
                3: np.array([255, 255, 0], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
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
                                         full_color_map=full_color_map)
                    
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
                                
                                # Mark frame as processed (only for newly created annotations)
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
        print(f"LIDAR PNG ANNOTATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,}")
        print(f"\n📁 Output: {self.lidar_annotation_dir}/")
        print(f"\n🎯 LiDAR-Only Training Annotations:")
        print(f"   📥 Input: lidar_png + SAM guidance")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 LiDAR-Aligned Strategy (DISABLED DILATION for basic PNG compatibility):")
        print(f"      - RELAXED density threshold: {self.density_threshold} (from 0.15)")
        print(f"      - EXTENDED distance threshold: {self.distance_threshold}m (from 70.0m)")
        print(f"      - MORE permissive coverage detection (>3 vs >5)")
        print(f"      - REDUCED edge ignore regions (0.5% vs 1%)")
        print(f"      - NO dilation applied - exact geometric correspondence")
        print(f"      - Annotations align with LiDAR PNG coverage")
        print(f"      - Accepts smoothed/interpolated regions from LiDAR PNG")
        print(f"      - Prevents SAM masks from appearing in LiDAR-free areas")
        print(f"   📊 Background vs Ignore:")
        print(f"      - Background (0): No LiDAR coverage (black in viz)")
        print(f"      - Ignore (1): Edges beyond objects, sparse/distant LiDAR (gray in viz)")
        print(f"   ✓ Consistent with camera-only ignore strategy")
        print(f"   📊 Visualization: Stacked comparison view")
        print(f"      - Top: LiDAR PNG with annotation overlay")
        print(f"      - Bottom: Camera image with LiDAR annotation overlay")
        print(f"      - Verify geometric alignment between LiDAR and camera views")
        print(f"      - Black = Background, Gray = Ignore, Colors = Objects")
        print(f"   🎯 Purpose: LiDAR-native annotations for LiDAR-only model training")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LiDAR-only annotations from lidar_png using SAM guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates LiDAR-native segmentation annotations for LiDAR-only model training.

PERFORMANCE OPTIMIZATIONS:
- Parallel processing using multiple CPU cores
- Efficient density calculation using sliding window instead of Gaussian filter
- Batch processing to manage memory usage
- Vectorized operations for faster computation

REQUIRES: Run generate_sam.py and generate_lidar_png.py first!

Strategy:
1. Automatically finds frames with both lidar_png and SAM annotations
2. Uses lidar_png geometric projections for LiDAR coverage analysis
3. Uses SAM annotations as ground truth for object locations and types
4. Maps SAM-identified objects onto actual LiDAR points for LiDAR-native annotations
5. Maintains exact geometric correspondence with LiDAR PNG coordinates (no dilation)

Class mapping:
  0: Background (no LiDAR coverage)
  1: Ignore (sparse LiDAR regions, distant points >60m)
  2: Vehicle (LiDAR points within SAM vehicle regions)
  3: Sign (LiDAR points within SAM sign regions)
  4: Cyclist (LiDAR points within SAM cyclist regions)
  5: Pedestrian (LiDAR points within SAM pedestrian regions)

Usage examples:
  # Generate annotations for all frames with both inputs available (optimized)
  python generate_lidar_png_annotation.py

  # Generate with stacked visualizations comparing LiDAR vs camera alignment
  python generate_lidar_png_annotation.py --visualize

  # Use custom number of workers
  python generate_lidar_png_annotation.py --workers 4 --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create stacked PNG visualizations comparing LiDAR and camera views with annotation overlays')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect, max 8)')

    args = parser.parse_args()

    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/zod_temp")

    print("="*60)
    print("LiDAR PNG Annotation Generator (Optimized)")
    print("="*60)
    print(f"Input: {OUTPUT_ROOT / 'lidar_png'}")
    print(f"SAM: {OUTPUT_ROOT / 'annotation_sam'}")
    print(f"Output: {OUTPUT_ROOT / 'annotation_lidar_only'}")

    generator = LiDARPNGAnnotationGenerator(
        output_root=OUTPUT_ROOT
    )

    # Override parameters if specified
    if hasattr(args, 'workers') and args.workers:
        generator.max_workers = min(args.workers, multiprocessing.cpu_count())
        print(f"Using {generator.max_workers} workers")

    # Process all frames with optional visualization
    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()