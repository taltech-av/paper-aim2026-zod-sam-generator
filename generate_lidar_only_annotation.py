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

Key Features:
- Parallel processing for performance optimization
- LiDAR-native annotation strategy (no dilation for geometric accuracy)
- Quality filtering based on LiDAR density and distance
- Automatic frame discovery from available inputs
- Progress tracking and resume capability
- Optional visualization with stacked LiDAR/camera comparison
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
    """
    Generate LiDAR-only annotations from lidar_png using SAM guidance.

    This class implements a LiDAR-native annotation strategy that:
    - Uses basic lidar_png geometric projections to identify LiDAR coverage areas
    - Leverages SAM segmentation masks as ground truth for object locations
    - Creates annotations that align exactly with available LiDAR data
    - Applies quality filtering to ensure training data reliability
    - Supports parallel processing for efficient batch generation
    """

    def __init__(self, output_root):
        """
        Initialize LiDAR annotation generator with directory structure and parameters.

        Sets up input/output directories, discovers available frames, and configures
        processing parameters for optimal LiDAR-native annotation generation.

        Args:
            output_root: Path to output_clft_v2 directory containing processed data
        """
        self.output_root = Path(output_root)

        # ===== INPUT DIRECTORIES =====
        # Basic lidar_png for LiDAR coverage analysis (geometric projections)
        self.lidar_png_dir = self.output_root / "lidar_png"
        # SAM annotations provide ground truth object locations and types
        self.sam_annotation_dir = self.output_root / "annotation_sam"
        # Camera images for optional visualization comparison
        self.camera_dir = self.output_root / "camera"

        # ===== OUTPUT DIRECTORY =====
        # LiDAR-native annotations for LiDAR-only model training
        self.lidar_annotation_dir = self.output_root / "annotation_lidar_only"
        self.lidar_annotation_dir.mkdir(parents=True, exist_ok=True)

        # ===== PROGRESS TRACKING =====
        # Resume capability - track already processed frames
        self.processed_frames_file = self.output_root / "processed_lidar_png_annotations.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())

        # ===== FRAME DISCOVERY =====
        # Automatically find frames that have both lidar_png and SAM annotation
        print("🔍 Scanning for available input files...")

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

        # Find intersection - frames with both inputs available
        available_frame_ids = lidar_frame_ids & sam_frame_ids

        print(f"✓ Found {len(lidar_frame_ids):,} lidar_png files")
        print(f"✓ Found {len(sam_frame_ids):,} SAM annotation files")
        print(f"✓ Found {len(available_frame_ids):,} frames with both inputs")

        # Convert to sorted list for consistent processing order
        self.frame_ids = sorted(list(available_frame_ids))

        # Filter out already processed frames for resume capability
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

        # ===== LIDAR ANNOTATION PARAMETERS =====
        # RELAXED thresholds for better training data coverage
        self.density_threshold = 0.10  # Minimum local LiDAR density (reduced from 0.15)
        self.distance_threshold = 90.0  # Maximum distance for reliable annotations (increased from 70.0)

        # ===== PERFORMANCE OPTIMIZATION =====
        # Parallel processing configuration
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers maximum
        self.batch_size = 50  # Process frames in batches to manage memory usage

    def compute_density_map_efficient(self, lidar_coverage, window_size=7):
        """
        Compute local LiDAR density map using efficient sliding window approach.

        This method calculates the local density of LiDAR points within a sliding window
        to identify sparse vs dense regions. Uses uniform filter for better performance
        compared to Gaussian filtering, which is important for large-scale processing.

        Args:
            lidar_coverage: Boolean array indicating LiDAR coverage (True where LiDAR exists)
            window_size: Size of sliding window for density calculation (default: 7x7)

        Returns:
            density_map: Float array with local density values (0.0 to 1.0)
                         where 1.0 means all pixels in window have LiDAR coverage
        """
        # Use uniform filter (box filter) for efficient density calculation
        # This is faster than Gaussian filtering and suitable for density estimation
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)

        # Apply 2D filter to get local density (fraction of LiDAR points in each window)
        density_map = cv2.filter2D(lidar_coverage.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Clip to valid range [0, 1] to handle edge effects
        density_map = np.clip(density_map, 0, 1)

        return density_map

    def load_lidar_png(self, frame_id):
        """
        Load LiDAR geometric projection for coverage analysis.

        The lidar_png contains 3-channel geometric information (X, Y, Z coordinates)
        that indicates where LiDAR data exists in the scene. This is used to ensure
        annotations only appear where actual LiDAR measurements are available.

        Args:
            frame_id: Frame identifier string

        Returns:
            np.ndarray or None: 3-channel geometric projection image, or None if not found
        """
        lidar_path = self.lidar_png_dir / f"frame_{frame_id}.png"
        if not lidar_path.exists():
            return None

        # Load 3-channel geometric projection (X, Y, Z coordinates as RGB)
        lidar_img = cv2.imread(str(lidar_path), cv2.IMREAD_UNCHANGED)
        return lidar_img

    def load_sam_annotation(self, frame_id):
        """
        Load SAM segmentation mask for ground truth object locations.

        SAM provides high-quality segmentation masks that serve as ground truth
        for object locations and types. These masks guide where objects should
        be annotated in the LiDAR-native annotations.

        Args:
            frame_id: Frame identifier string

        Returns:
            np.ndarray or None: Grayscale segmentation mask, or None if not found
        """
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None

        # Load grayscale segmentation mask with class IDs
        annotation = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)
        return annotation

    def load_camera_image(self, frame_id):
        """
        Load camera image for optional visualization comparison.

        Camera images are used for creating stacked visualizations that show
        how LiDAR annotations align with the camera view, helping verify
        geometric accuracy of the annotation process.

        Args:
            frame_id: Frame identifier string

        Returns:
            np.ndarray or None: Color camera image, or None if not found
        """
        camera_path = self.camera_dir / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return None

        # Load color camera image for visualization
        camera_img = cv2.imread(str(camera_path), cv2.IMREAD_COLOR)
        return camera_img

    def create_lidar_native_annotation(self, lidar_img, sam_annotation):
        """
        Create LiDAR-native annotation using lidar_png and SAM guidance.

        This is the core algorithm that implements LiDAR-native annotation strategy:
        1. Use lidar_png for LiDAR coverage (permissive to match PNG content)
        2. Use SAM as ground truth for object locations and types
        3. Map SAM-identified objects onto LiDAR-covered regions
        4. Create LiDAR-native segmentation with proper class distinctions

        The key insight is that annotations should only exist where LiDAR data
        is actually available, ensuring training targets match input data.

        Args:
            lidar_img: 3-channel geometric projection from lidar_png
            sam_annotation: SAM segmentation mask (ground truth)

        Returns:
            annotation: LiDAR-native segmentation mask with class IDs
        """
        if lidar_img is None or sam_annotation is None:
            return None

        h, w, c = lidar_img.shape

        # Initialize annotation with background (0) - no LiDAR coverage areas
        annotation = np.zeros((h, w), dtype=np.uint8)

        # ===== EXTRACT GEOMETRIC INFORMATION FROM LIDAR_PNG =====
        # Decode the 3-channel geometric projection back to coordinate information
        x_channel = lidar_img[:, :, 0].astype(float)  # X coordinates (normalized)
        y_channel = lidar_img[:, :, 1].astype(float)  # Y coordinates (normalized)
        z_channel = lidar_img[:, :, 2].astype(float)  # Z coordinates (depth, normalized)

        # ===== DETERMINE LIDAR COVERAGE =====
        # Create LiDAR coverage mask that aligns with LiDAR PNG content
        # Strategy: Accept regions where ANY channel has meaningful data
        # This ensures annotations align with actual LiDAR PNG content
        lidar_coverage = np.any(lidar_img > 3, axis=2)  # Lower threshold for permissiveness

        # Additional quality filter: exclude extremely distant points that might be artifacts
        z_channel_norm = z_channel / 255.0
        distance_approx = z_channel_norm * 100.0  # Convert normalized Z back to approximate distance
        reasonable_distance = distance_approx < 100.0  # Filter out very distant points

        # Final coverage: has LiDAR data AND not extremely distant
        lidar_coverage = lidar_coverage & reasonable_distance

        # ===== DISTANCE CALCULATION FOR QUALITY FILTERING =====
        # Decode distance from normalization for density-based quality assessment
        z_norm = z_channel / 255.0
        distance_map = z_norm * 100.0  # Convert to approximate distance in meters

        # ===== STEP 1: BACKGROUND CLASS (0) =====
        # Areas with no LiDAR coverage are automatically background (already initialized to 0)

        # ===== STEP 2: IGNORE CLASS (1) =====
        # Mark regions that should be ignored during training due to low quality
        # Strategy combines multiple quality criteria for robust ignore region detection

        # Quality criterion A: Sparse LiDAR regions (density-based)
        # Compute local density map to identify areas with insufficient LiDAR coverage
        lidar_density = self.compute_density_map_efficient(lidar_coverage.astype(np.uint8))

        # Areas with sparse LiDAR coverage (< 25% local density) - only where LiDAR exists
        sparse_regions = (lidar_density < 0.25) & lidar_coverage

        # Quality criterion B: Distant regions beyond reliable range
        distant_regions = (distance_map > self.distance_threshold) & lidar_coverage

        # Quality criterion C: Minimal vertical edge regions (conservative for LiDAR)
        # LiDAR covers the scene well, so use minimal edge strips for safety
        vertical_edge_regions = np.zeros_like(sam_annotation, dtype=bool)

        # Only mark tiny edge strips (0.5% of height) as ignore regions
        edge_fraction = 0.005  # Conservative edge fraction
        edge_height = max(int(h * edge_fraction), 3)  # At least 3 pixels minimum

        vertical_edge_regions[:edge_height, :] = True  # Top edge strip
        vertical_edge_regions[-edge_height:, :] = True  # Bottom edge strip

        # Combine all ignore criteria: sparse OR distant OR edge regions
        ignore_regions = sparse_regions | distant_regions | vertical_edge_regions
        annotation[ignore_regions] = 1  # Mark as ignore class

        # ===== STEP 3: OBJECT CLASSES FROM SAM GUIDANCE =====
        # Map SAM-identified objects onto LiDAR-covered regions for LiDAR-native annotations
        # Only annotate where LiDAR points actually exist - ensures training data validity

        # Quality thresholds for LiDAR-native object detection - RELAXED for better coverage
        min_density_for_object = 0.12  # Minimum local density for reliable object detection
        min_region_size = 5  # Minimum region size to avoid noise
        max_distance_for_full_mask = 60.0  # Maximum distance for full object mask inclusion

        # Process each SAM object class (vehicle, sign, cyclist, pedestrian)
        for sam_class_id in [2, 3, 4, 5]:  # SAM class IDs for objects
            sam_class_mask = (sam_annotation == sam_class_id)

            # Find overlap between SAM object regions and actual LiDAR coverage
            sam_lidar_overlap = sam_class_mask & lidar_coverage

            if np.sum(sam_lidar_overlap) >= min_region_size:
                # Within the LiDAR-covered SAM regions, apply quality filtering
                overlap_pixels = np.sum(sam_lidar_overlap)

                # Extract quality metrics for the overlapping regions
                overlap_density = lidar_density[sam_lidar_overlap]  # Local density values
                overlap_distances = distance_map[sam_lidar_overlap]  # Distance values

                # Create quality mask: only high-quality LiDAR points get object labels
                quality_mask = np.zeros_like(sam_lidar_overlap, dtype=bool)
                quality_mask[sam_lidar_overlap] = (
                    (overlap_density >= min_density_for_object) &  # Sufficient density
                    (overlap_distances <= max_distance_for_full_mask)  # Reasonable distance
                )

                # Apply to non-ignore regions only
                final_mask = quality_mask & ~ignore_regions

                if np.sum(final_mask) >= min_region_size:
                    # Assign the SAM class ID to qualifying LiDAR regions
                    annotation[final_mask] = sam_class_id

        # ===== STEP 4: FINAL CLEANUP =====
        # Ensure areas without LiDAR coverage remain background (no false positives)
        # But preserve ignore regions that have sparse LiDAR coverage
        no_lidar = ~lidar_coverage
        annotation[no_lidar] = 0  # Force background only where there's NO LiDAR at all
        # Note: Do NOT overwrite ignore regions that were set in step 2

        return annotation

    def create_overlay_visualization(self, base_image, colored_mask, title=""):
        """
        Create overlay visualization with colored annotations and title.

        Generates a visualization that overlays colored segmentation masks on base images
        with transparency, making it easy to see how annotations align with the data.
        Used for both LiDAR and camera image comparisons.

        Args:
            base_image: Base image (LiDAR geometric projection or camera image)
            colored_mask: Colored annotation mask with class-specific colors
            title: Title text to add to the visualization

        Returns:
            visualization: Image with overlay and title for analysis
        """
        h, w = base_image.shape[:2]

        # ===== PREPARE BASE IMAGE FOR OVERLAY =====
        if len(base_image.shape) == 3 and base_image.shape[2] == 3:
            # Color image (camera) - use as-is for overlay
            overlay_base = base_image.copy()
        else:
            # LiDAR image (grayscale or 3-channel) - normalize for visualization
            if len(base_image.shape) == 3:
                # Use first channel for grayscale conversion
                base_channel = base_image[:, :, 0].astype(float)
            else:
                base_channel = base_image.astype(float)

            # Normalize to 0-255 range for proper visualization
            base_channel = (base_channel - base_channel.min()) / (base_channel.max() - base_channel.min() + 1e-6)
            base_channel = (base_channel * 255).astype(np.uint8)
            # Convert to RGB for consistent overlay processing
            overlay_base = cv2.cvtColor(base_channel, cv2.COLOR_GRAY2RGB)

        # ===== CREATE TRANSPARENT OVERLAY =====
        # Start with base image
        overlay = overlay_base.copy()
        alpha = 0.6  # Transparency level for annotations

        # Find pixels that have annotations (non-black in colored_mask)
        mask_pixels = np.any(colored_mask > 0, axis=2)

        # Apply weighted blend where annotations exist
        overlay[mask_pixels] = cv2.addWeighted(
            overlay_base[mask_pixels], 1-alpha,  # Base image with reduced opacity
            colored_mask[mask_pixels], alpha, 0  # Colored mask with transparency
        )

        # ===== ADD TITLE =====
        if title:
            # Add black background rectangle for text readability
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            # Add white title text
            cv2.putText(overlay, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay

    def create_lidar_png_annotation(self, frame_id):
        """
        Create LiDAR-only annotation for a single frame using lidar_png guidance.

        Orchestrates the complete annotation pipeline for one frame:
        1. Load LiDAR geometric projection for coverage analysis
        2. Load SAM annotation for ground truth object locations
        3. Generate LiDAR-native annotation using quality-filtered mapping
        4. Return annotation mask for saving and visualization

        Args:
            frame_id: Frame identifier string to process

        Returns:
            np.ndarray or None: LiDAR-native segmentation mask, or None if processing failed
        """
        try:
            # ===== LOAD LIDAR GEOMETRIC PROJECTION =====
            # Provides LiDAR coverage information and geometric context
            lidar_img = self.load_lidar_png(frame_id)
            if lidar_img is None:
                # Silently skip frames without lidar_png (handled by caller)
                return None

            # ===== LOAD SAM GROUND TRUTH =====
            # Provides object locations and types as annotation guidance
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                # Silently skip frames without SAM annotation (handled by caller)
                return None

            # ===== GENERATE LIDAR-NATIVE ANNOTATION =====
            # Core algorithm: Map SAM objects onto LiDAR-covered regions
            annotation = self.create_lidar_native_annotation(lidar_img, sam_annotation)

            return annotation

        except Exception as e:
            # Exception handling delegated to caller for proper error reporting
            return None

    def process_single_frame(self, frame_id, create_vis=False, vis_dir=None, full_color_map=None):
        """
        Process a single frame and return results for parallel processing.

        This method handles the complete processing pipeline for one frame in a way
        that's compatible with parallel execution. It manages file I/O, annotation
        creation, visualization generation, and progress tracking.

        Args:
            frame_id: Frame identifier string to process
            create_vis: Whether to create visualization images
            vis_dir: Directory path for saving visualizations
            full_color_map: Color mapping dictionary for visualization

        Returns:
            tuple: (frame_id, status, annotation, visualization_data)
                   status can be "success", "error", or "skip"
        """
        try:
            # ===== CHECK FOR EXISTING ANNOTATION =====
            # Resume capability - skip if already processed
            output_path = self.lidar_annotation_dir / f"frame_{frame_id}.png"
            if output_path.exists():
                # Load existing annotation for potential visualization reuse
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                if annotation is None:
                    return (frame_id, "error", None, None)
            else:
                # ===== CREATE NEW ANNOTATION =====
                annotation = self.create_lidar_png_annotation(frame_id)

                if annotation is None:
                    return (frame_id, "error", None, None)

                # Save annotation to disk
                cv2.imwrite(str(output_path), annotation)

                # Mark frame as processed for resume capability
                self.processed_frames.add(frame_id)
                with open(self.processed_frames_file, 'a') as f:
                    f.write(f"{frame_id}\n")

            # ===== CREATE VISUALIZATION DATA =====
            # Generate stacked comparison view if requested and not already cached
            vis_data = None
            if create_vis and vis_dir and full_color_map:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        # Load base images for visualization
                        lidar_img = self.load_lidar_png(frame_id)
                        camera_img = self.load_camera_image(frame_id)

                        if lidar_img is not None:
                            # Create colored mask for overlay visualization
                            colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in full_color_map.items():
                                if class_id == 0:  # Skip background for cleaner overlay
                                    continue
                                colored_mask[annotation == class_id] = color

                            # ===== CREATE STACKED VISUALIZATION =====
                            h, w = annotation.shape

                            # Top half: LiDAR PNG with annotation overlay
                            lidar_overlay = self.create_overlay_visualization(
                                lidar_img, colored_mask, "LiDAR + Annotations"
                            )

                            # Bottom half: Camera image with annotation overlay (if available)
                            if camera_img is not None:
                                # Resize camera to match annotation dimensions if needed
                                if camera_img.shape[:2] != (h, w):
                                    camera_img = cv2.resize(camera_img, (w, h), interpolation=cv2.INTER_LINEAR)

                                camera_overlay = self.create_overlay_visualization(
                                    camera_img, colored_mask, "Camera + LiDAR Annotations"
                                )

                                # Ensure consistent BGR color space for stacking
                                if lidar_overlay.shape[2] == 3:
                                    lidar_overlay = cv2.cvtColor(lidar_overlay, cv2.COLOR_RGB2BGR)
                                # camera_overlay is already BGR from create_overlay_visualization

                                # Stack vertically: LiDAR on top, Camera on bottom
                                vis_data = np.vstack([lidar_overlay, camera_overlay])
                            else:
                                # Fallback: Just LiDAR overlay if no camera available
                                vis_data = cv2.cvtColor(lidar_overlay, cv2.COLOR_RGB2BGR) if lidar_overlay.shape[2] == 3 else lidar_overlay
                        else:
                            # Ultimate fallback: Just colored annotation mask
                            colored_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                            for class_id, color in full_color_map.items():
                                colored_annotation[annotation == class_id] = color
                            vis_data = colored_annotation

                    except Exception as e:
                        # Visualization failure doesn't stop annotation processing
                        vis_data = None

            # ===== RETURN RESULTS =====
            # Determine appropriate status based on processing outcome
            if output_path.exists() and 'annotation' not in locals():
                return (frame_id, "skip", annotation, vis_data)  # Existing annotation loaded
            else:
                return (frame_id, "success", annotation, vis_data)  # New annotation created

        except Exception as e:
            # Return error status for any unexpected failures
            return (frame_id, "error", None, None)

    def process_all_frames(self, create_vis=False):
        """
        Process all frames and create LiDAR annotations using parallel processing.

        This method orchestrates the complete batch processing pipeline:
        1. Sets up parallel processing infrastructure
        2. Processes frames in optimized batches
        3. Tracks progress and performance metrics
        4. Generates optional visualizations
        5. Reports comprehensive statistics

        Args:
            create_vis: Whether to create stacked visualization images for quality verification
        """
        start_time = time.time()

        # ===== INITIALIZATION AND STATUS REPORT =====
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

        # ===== VISUALIZATION SETUP =====
        # Prepare directories and color mapping for optional visualizations
        vis_dir = None
        full_color_map = None
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "lidar_only_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Define color mapping for visualization consistency
            full_color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
                2: np.array([255, 0, 0], dtype=np.uint8),        # Vehicle - Red
                3: np.array([255, 255, 0], dtype=np.uint8),      # Sign - Yellow
                4: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                5: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
            }

        # ===== BATCH PROCESSING SETUP =====
        # Divide work into batches to manage memory and provide intermediate progress
        frame_batches = [self.frame_ids[i:i + self.batch_size]
                        for i in range(0, len(self.frame_ids), self.batch_size)]

        # ===== PARALLEL PROCESSING EXECUTION =====
        with tqdm(total=len(self.frame_ids), desc="Processing frames") as pbar:
            for batch in frame_batches:
                # Process batch in parallel using process pool
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Create partial function with fixed visualization parameters
                    process_func = partial(self.process_single_frame,
                                         create_vis=create_vis,
                                         vis_dir=vis_dir,
                                         full_color_map=full_color_map)

                    # Submit all frames in batch for parallel execution
                    futures = [executor.submit(process_func, frame_id) for frame_id in batch]

                    # Process results as they complete (order doesn't matter for progress)
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            frame_id, status, annotation, vis_data = future.result()

                            # Update counters based on processing status
                            if status == "skip":
                                skip_count += 1  # Already processed frame
                            elif status == "error":
                                error_count += 1  # Processing failed
                            elif status == "success":
                                success_count += 1  # New annotation created
                                # Mark as processed (only for newly created annotations)
                                self.processed_frames.add(frame_id)
                                with open(self.processed_frames_file, 'a') as f:
                                    f.write(f"{frame_id}\n")

                            # Save visualization if generated (for both new and existing annotations)
                            if vis_data is not None and vis_dir:
                                vis_path = vis_dir / f"frame_{frame_id}.png"
                                cv2.imwrite(str(vis_path), vis_data)
                                vis_count += 1

                        except Exception as e:
                            # Handle any unexpected errors in result processing
                            error_count += 1
                            print(f"  ❌ Error processing frame result: {e}")

                        # Update progress bar for each completed frame
                        pbar.update(1)

        # ===== PERFORMANCE ANALYSIS =====
        end_time = time.time()
        total_time = end_time - start_time
        processed_frames = success_count + error_count

        if processed_frames > 0:
            # Calculate and display performance metrics
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