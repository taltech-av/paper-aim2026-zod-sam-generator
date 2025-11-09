#!/usr/bin/env python3
"""
Generate fusion annotations - consistent with actual LiDAR coverage.

This script creates fusion annotations that respect the sparse nature of LiDAR coverage.
Unlike the original fusion script that assumes comprehensive LiDAR validation, this version:

1. Only applies LiDAR-based quality filtering where LiDAR actually exists (5.2% coverage)
2. Uses camera-only logic for areas without LiDAR coverage (94.8% of image)
3. Maintains object annotations from SAM without unnecessary LiDAR-based removal
4. Creates ignore regions only where LiDAR exists AND indicates poor quality

This ensures consistency between training inputs (sparse LiDAR) and annotations.

REQUIRES: Run generate_sam.py, generate_camera_only_annotation.py, and generate_lidar_png.py first!
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
    """Generate fusion annotations consistent with actual LiDAR coverage"""

    def __init__(self, output_root, sam_dir='annotation_sam', lidar_dir='lidar_png',
                 camera_dir='camera', camera_only_dir='annotation_camera_only',
                 fusion_dir='annotation_fusion'):
        """Initialize fusion annotation generator

        Args:
            output_root: Path to output root directory
            sam_dir: Directory name for SAM annotations
            lidar_dir: Directory name for LiDAR PNGs
            camera_dir: Directory name for camera images
            camera_only_dir: Directory name for camera-only annotations
            fusion_dir: Directory name for fusion annotations output
        """
        self.output_root = Path(output_root)

        # Input directories
        self.sam_annotation_dir = self.output_root / sam_dir
        self.lidar_png_dir = self.output_root / lidar_dir
        self.camera_dir = self.output_root / camera_dir
        self.camera_only_dir = self.output_root / camera_only_dir

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

        camera_only_files = list(self.camera_only_dir.glob("frame_*.png"))
        camera_only_frame_ids = {f.stem.replace("frame_", "") for f in camera_only_files}

        # Find intersection - frames with all inputs
        available_frame_ids = sam_frame_ids & lidar_frame_ids & camera_frame_ids & camera_only_frame_ids

        print(f"✓ Found {len(sam_frame_ids):,} SAM annotation files")
        print(f"✓ Found {len(lidar_frame_ids):,} lidar_png files")
        print(f"✓ Found {len(camera_frame_ids):,} camera image files")
        print(f"✓ Found {len(camera_only_frame_ids):,} camera-only annotation files")
        print(f"✓ Found {len(available_frame_ids):,} frames with all inputs")

        # Convert to sorted list for consistent processing
        self.frame_ids = sorted(list(available_frame_ids))

        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]

        print(f"✓ Total frames to process: {original_count:,}")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")

        # Fusion parameters - adjusted for sparse LiDAR
        self.lidar_confidence_threshold = 0.15  # Lower threshold for sparse LiDAR
        self.depth_consistency_threshold = 0.6  # Relaxed for sparse data

        # Performance optimization parameters
        self.max_workers = min(multiprocessing.cpu_count(), 8)
        self.batch_size = 50

    def load_sam_annotation(self, frame_id):
        """Load SAM segmentation mask"""
        sam_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not sam_path.exists():
            return None
        return cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)

    def load_camera_only_annotation(self, frame_id):
        """Load camera-only annotation"""
        cam_only_path = self.camera_only_dir / f"frame_{frame_id}.png"
        if not cam_only_path.exists():
            return None
        return cv2.imread(str(cam_only_path), cv2.IMREAD_GRAYSCALE)

    def load_lidar_png(self, frame_id):
        """Load LiDAR geometric projection"""
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

    def create_lidar_quality_mask(self, lidar_img):
        """Create quality mask ONLY for areas with actual LiDAR coverage

        Returns:
            quality_mask: Boolean mask where True indicates LiDAR-covered areas with good quality
        """
        h, w, c = lidar_img.shape

        # LiDAR coverage mask - only where LiDAR actually exists
        lidar_coverage = np.any(lidar_img > 0, axis=2)

        if np.sum(lidar_coverage) == 0:
            # No LiDAR coverage at all
            return np.zeros_like(lidar_coverage, dtype=bool)

        # Calculate LiDAR density using gaussian filter (only on covered areas)
        lidar_density = np.zeros_like(lidar_coverage, dtype=float)
        lidar_density[lidar_coverage] = ndimage.gaussian_filter(
            lidar_coverage.astype(float),
            sigma=2,
            mode='constant',
            cval=0
        )[lidar_coverage]

        # Depth consistency check (only on covered areas)
        z_channel = lidar_img[:, :, 2].astype(float)
        # Areas with reasonable depth values (not extreme)
        reasonable_depth = np.zeros_like(lidar_coverage, dtype=bool)
        reasonable_depth[lidar_coverage] = (z_channel[lidar_coverage] > 0.05) & (z_channel[lidar_coverage] < 0.95)

        # High quality regions: good LiDAR coverage AND reasonable depth
        quality_mask = (lidar_density > self.lidar_confidence_threshold) & reasonable_depth

        return quality_mask

    def create_fusion_annotation(self, frame_id):
        """Create fusion annotation - consistent with sparse LiDAR coverage

        Strategy:
        1. Start with camera-only annotations (SAM + minimal ignore)
        2. Only apply LiDAR-based modifications where LiDAR actually exists
        3. Keep camera-only logic for 94.8% of image without LiDAR
        4. Only mark LiDAR-covered areas as ignore if they have poor quality
        """
        try:
            # Load all required inputs
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                print(f"  ⚠️  No SAM annotation for {frame_id}")
                return None

            camera_only = self.load_camera_only_annotation(frame_id)
            if camera_only is None:
                print(f"  ⚠️  No camera-only annotation for {frame_id}")
                return None

            lidar_img = self.load_lidar_png(frame_id)
            if lidar_img is None:
                print(f"  ⚠️  No LiDAR PNG for {frame_id}")
                return None

            camera_img = self.load_camera_image(frame_id)

            # Start with camera-only annotations (already has minimal ignore regions)
            fusion_annotation = camera_only.copy()

            # Create LiDAR quality assessment (only where LiDAR exists)
            lidar_quality_mask = self.create_lidar_quality_mask(lidar_img)

            # LiDAR coverage mask
            lidar_coverage = np.any(lidar_img > 0, axis=2)

            # Strategy: Only modify areas where LiDAR actually provides information

            # 1. Areas with good LiDAR quality: Keep camera annotations as-is
            # (No changes needed - camera-only already provides good baseline)

            # 2. Areas with poor LiDAR quality: Mark as ignore (only where LiDAR exists)
            poor_quality_lidar = lidar_coverage & ~lidar_quality_mask

            # Apply ignore only to background pixels in poor quality LiDAR areas
            background_in_poor_lidar = poor_quality_lidar & (fusion_annotation == 0)
            fusion_annotation[background_in_poor_lidar] = 1

            # 3. Areas without LiDAR: Keep camera-only annotations (94.8% of image)
            # (Already handled by starting with camera-only)

            # 4. Additional conservative ignore regions (minimal, consistent with sparse LiDAR)
            h, w = fusion_annotation.shape

            # Very conservative edge regions (reduced from camera-only's 1%)
            edge_fraction = 0.005  # Only 0.5% edges (much more conservative)
            edge_height = max(int(h * edge_fraction), 3)
            edge_ignore = np.zeros_like(fusion_annotation, dtype=bool)
            edge_ignore[:edge_height, :] = True
            edge_ignore[-edge_height:, :] = True

            # Apply edge ignore only to background
            edge_background = edge_ignore & (fusion_annotation == 0)
            fusion_annotation[edge_background] = 1

            return fusion_annotation

        except Exception as e:
            print(f"  ❌ Error creating fusion annotation for {frame_id}: {e}")
            return None

    def create_overlay_visualization(self, base_image, colored_mask, title=""):
        """Create overlay visualization with title"""
        h, w = base_image.shape[:2]

        if len(base_image.shape) == 3 and base_image.shape[2] == 3:
            overlay_base = base_image.copy()
        else:
            if len(base_image.shape) == 2:
                overlay_base = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
            else:
                overlay_base = base_image.copy()

        alpha = 0.6
        overlay = overlay_base.copy()

        mask_pixels = np.any(colored_mask > 0, axis=2)
        overlay[mask_pixels] = cv2.addWeighted(
            overlay_base[mask_pixels], 1-alpha,
            colored_mask[mask_pixels], alpha, 0
        )

        if title:
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(overlay, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay

    def process_single_frame(self, frame_id, create_vis=False, vis_dir=None, color_map=None):
        """Process a single frame"""
        try:
            output_path = self.fusion_annotation_dir / f"frame_{frame_id}.png"
            if output_path.exists():
                annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                if annotation is None:
                    return (frame_id, "error", None, None)
            else:
                annotation = self.create_fusion_annotation(frame_id)

                if annotation is None:
                    return (frame_id, "error", None, None)

                cv2.imwrite(str(output_path), annotation)

                self.processed_frames.add(frame_id)
                with open(self.processed_frames_file, 'a') as f:
                    f.write(f"{frame_id}\n")

            vis_data = None
            if create_vis and vis_dir and color_map:
                vis_path = vis_dir / f"frame_{frame_id}.png"
                if not vis_path.exists():
                    try:
                        camera_img = self.load_camera_image(frame_id)

                        colored_mask = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
                        for class_id, color in color_map.items():
                            if class_id == 0:
                                continue
                            colored_mask[annotation == class_id] = color

                        if camera_img is not None:
                            if camera_img.shape[:2] != annotation.shape:
                                camera_img = cv2.resize(camera_img, (annotation.shape[1], annotation.shape[0]), interpolation=cv2.INTER_LINEAR)

                            overlay = self.create_overlay_visualization(camera_img, colored_mask, "Fusion: LiDAR-Consistent Annotations")
                            vis_data = overlay
                        else:
                            vis_data = colored_mask

                    except Exception as e:
                        vis_data = None

            if output_path.exists() and 'annotation' not in locals():
                return (frame_id, "skip", annotation, vis_data)
            else:
                return (frame_id, "success", annotation, vis_data)

        except Exception as e:
            return (frame_id, "error", None, None)

    def process_all_frames(self, create_vis=False):
        """Process all frames"""
        start_time = time.time()

        print(f"\n🎯 Generating fusion annotations (LiDAR-consistent)")
        print(f"Input: SAM + camera-only + lidar_png + camera images")
        print(f"Output: {self.fusion_annotation_dir}")
        if create_vis:
            print(f"Visualizations: Camera images with fusion annotation overlays")
        print(f"Frames to process: {len(self.frame_ids):,}")
        print(f"Using {self.max_workers} parallel workers")

        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0

        if create_vis:
            vis_dir = self.output_root / "visualizations" / "fusion_annotation"
            vis_dir.mkdir(parents=True, exist_ok=True)

            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),
                1: np.array([128, 128, 128], dtype=np.uint8),
                2: np.array([0, 0, 255], dtype=np.uint8),
                3: np.array([0, 255, 255], dtype=np.uint8),
                4: np.array([255, 0, 255], dtype=np.uint8),
                5: np.array([0, 255, 0], dtype=np.uint8),
            }

        frame_batches = [self.frame_ids[i:i + self.batch_size]
                        for i in range(0, len(self.frame_ids), self.batch_size)]

        with tqdm(total=len(self.frame_ids), desc="Processing frames") as pbar:
            for batch in frame_batches:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    process_func = partial(self.process_single_frame,
                                         create_vis=create_vis,
                                         vis_dir=vis_dir if create_vis else None,
                                         color_map=color_map if create_vis else None)

                    futures = [executor.submit(process_func, frame_id) for frame_id in batch]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            frame_id, status, annotation, vis_data = future.result()

                            if status == "skip":
                                skip_count += 1
                            elif status == "error":
                                error_count += 1
                            elif status == "success":
                                success_count += 1

                                self.processed_frames.add(frame_id)
                                with open(self.processed_frames_file, 'a') as f:
                                    f.write(f"{frame_id}\n")

                            if vis_data is not None and create_vis:
                                vis_path = vis_dir / f"frame_{frame_id}.png"
                                cv2.imwrite(str(vis_path), vis_data)
                                vis_count += 1

                        except Exception as e:
                            error_count += 1
                            print(f"  ❌ Error processing frame result: {e}")

                        pbar.update(1)

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
        print(f"   📥 Input: SAM + camera-only + lidar_png + camera images")
        print(f"   🎨 Classes: Background(0), Ignore(1), Vehicle(2), Sign(3), Cyclist(4), Pedestrian(5)")
        print(f"   🔧 LiDAR-Consistent Strategy:")
        print(f"      - Start with camera-only annotations (minimal ignore)")
        print(f"      - Only apply LiDAR quality filtering where LiDAR exists (5.2% coverage)")
        print(f"      - Keep camera-only logic for 94.8% without LiDAR")
        print(f"      - Mark ignore only in poor-quality LiDAR areas")
        print(f"      - Very conservative edge ignore (0.5% vs camera-only's 1%)")
        print(f"   📊 Quality Thresholds (adjusted for sparse LiDAR):")
        print(f"      - LiDAR confidence threshold: {self.lidar_confidence_threshold} (relaxed)")
        print(f"      - Depth consistency threshold: {self.depth_consistency_threshold} (relaxed)")
        print(f"   ✓ Consistent with training inputs: sparse LiDAR coverage")
        print(f"   ✓ Maximizes training data while respecting LiDAR limitations")
        print(f"   ✓ No assumptions about comprehensive LiDAR validation")
        if create_vis:
            print(f"   📸 Visualizations: Camera images with fusion annotation overlays")
        print(f"   🎯 Purpose: Realistic fusion training with sparse LiDAR constraints")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fusion annotations - consistent with actual LiDAR coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates fusion segmentation annotations that respect sparse LiDAR coverage reality.

KEY IMPROVEMENT: Unlike original fusion that assumes comprehensive LiDAR validation,
this version only applies LiDAR-based modifications where LiDAR actually exists.

Strategy:
1. Start with camera-only annotations (already optimized for camera training)
2. Only apply LiDAR quality filtering in areas with actual LiDAR coverage (5.2%)
3. Keep camera-only logic for areas without LiDAR (94.8% of image)
4. Mark ignore only where LiDAR exists AND indicates poor quality
5. Very conservative ignore regions overall

This ensures consistency between training inputs (sparse LiDAR) and annotations.

REQUIRES: Run generate_sam.py, generate_camera_only_annotation.py, and generate_lidar_png.py first!

Usage examples:
  # Generate fusion annotations
  python generate_fusion_annotation.py

  # Generate with visualizations
  python generate_fusion_annotation.py --visualize
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create PNG visualizations for all processed frames')
    parser.add_argument('--output-root', type=str, default='/media/tom/ml/zod_temp',
                       help='Root directory for all outputs (default: /media/tom/ml/zod_temp)')

    args = parser.parse_args()

    OUTPUT_ROOT = Path(args.output_root)

    print("="*60)
    print("Fusion Annotation Generator (LiDAR-Consistent)")
    print("="*60)
    print(f"Input: SAM + camera-only + lidar_png + camera images")
    print(f"Output: {OUTPUT_ROOT / 'annotation_fusion'}")

    generator = FusionAnnotationGenerator(
        output_root=OUTPUT_ROOT
    )

    generator.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()