#!/usr/bin/env python3
"""
Generate 2D LiDAR projections as PNG images
- Projects LiDAR points to camera coordinates
- Uses distance-based coloring for visualization
- Creates standalone projection images
"""

import numpy as np
import pickle
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# Module-level functions for ProcessPoolExecutor
def preload_lidar_data_worker(args):
    """Worker function for preloading LiDAR data"""
    frame_id, lidar_dir = args
    try:
        lidar_points, distances = load_lidar_pkl_worker(frame_id, lidar_dir)
        return frame_id, (lidar_points, distances)
    except Exception as e:
        print(f"  Error preloading {frame_id}: {e}")
        return frame_id, (None, None)

def load_lidar_pkl_worker(frame_id, lidar_dir):
    """Load LiDAR pkl file (module-level function for multiprocessing)"""
    from pathlib import Path
    import pickle
    import numpy as np
    
    pkl_path = Path(lidar_dir) / f"frame_{frame_id}.pkl"
    if not pkl_path.exists():
        return None, None
    
    # Check if file is empty (corrupted/incomplete)
    if pkl_path.stat().st_size == 0:
        return None, None
    
    try:
        with open(pkl_path, 'rb') as f:
            clft_data = pickle.load(f)
        
        # Extract camera coordinates and filter for front camera (ID=1)
        camera_coords = clft_data['camera_coordinates']
        front_mask = camera_coords[:, 0] == 1
        
        if not np.any(front_mask):
            return None, None
        
        # Get pixel coordinates for front camera (in original ZOD resolution)
        front_coords = camera_coords[front_mask]
        x_coords = front_coords[:, 1].astype(np.float32)  # X pixel
        y_coords = front_coords[:, 2].astype(np.float32)  # Y pixel
        
        lidar_points = np.column_stack([x_coords, y_coords])
        
        # Extract 3D points for distance calculation
        points_3d = clft_data['3d_points'][front_mask]
        distances = np.sqrt(np.sum(points_3d[:, :3]**2, axis=1))  # Euclidean distance
        
        return lidar_points, distances
        
    except Exception as e:
        print(f"  Error loading LiDAR pkl for {frame_id}: {e}")
        return None, None

def process_single_frame_worker(args):
    """Worker function for processing a single frame"""
    frame_id, lidar_points, distances, projection_dir = args
    
    if lidar_points is None or distances is None:
        return False
    
    try:
        # Create projection image
        projection_img = create_projection_image_from_data_worker(frame_id, lidar_points, distances)
        
        if projection_img is None:
            return False
        
        # Save projection
        from pathlib import Path
        output_path = Path(projection_dir) / f"frame_{frame_id}.png"
        import cv2
        cv2.imwrite(str(output_path), projection_img)
        return True
        
    except Exception as e:
        print(f"  Error processing {frame_id}: {e}")
        return False

def create_projection_image_from_data_worker(frame_id, lidar_points, distances):
    """Create projection image (module-level function for multiprocessing)"""
    import numpy as np
    import cv2
    
    if lidar_points is None or distances is None or len(lidar_points) == 0:
        return None
    
    # Target dimensions: 1363x768 pixels
    target_w = 1363
    target_h = 768
    
    # Create black background image
    projection_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Scale coordinates to target shape
    # ZOD original resolution for front camera is 3848x2168
    orig_width = 3848
    orig_height = 2168
    
    scale_x = target_w / orig_width
    scale_y = target_h / orig_height
    
    scaled_points = lidar_points.copy()
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y
    
    # Draw each LiDAR point with distance-based color
    # Use numpy for faster pixel setting
    valid_mask = (
        (scaled_points[:, 0] >= 0) & (scaled_points[:, 0] < target_w) &
        (scaled_points[:, 1] >= 0) & (scaled_points[:, 1] < target_h)
    )
    valid_points = scaled_points[valid_mask].astype(int)
    valid_distances = distances[valid_mask]
    
    # Set pixels directly for speed (single pixels instead of circles)
    for point, dist in zip(valid_points, valid_distances):
        x, y = point[0], point[1]
        color = get_distance_color_worker(dist)
        projection_img[y, x] = color
    
    return projection_img

def get_distance_color_worker(distance, min_dist=0, max_dist=80):
    """Get color for distance (module-level function for multiprocessing)"""
    import numpy as np
    
    # Clamp distance to range
    distance = np.clip(distance, min_dist, max_dist)
    
    # Normalize to 0-1
    normalized = (distance - min_dist) / (max_dist - min_dist)
    
    # Cyan (close) to Blue (far) gradient
    # Close: Cyan (0, 255, 255) - BGR format
    # Far: Dark Blue (139, 0, 0) - BGR format
    r = int(255 * (1 - normalized))
    g = int(255 * (1 - normalized * 0.6))  # Gradual decrease
    b = int(139 + 116 * (1 - normalized))  # 139 to 255
    
    return (b, g, r)

class LidarProjectionGenerator:
    def __init__(self, output_root, frames_list, target_resolution=768, num_workers=16, batch_size=4):
        """Initialize LiDAR projection generator
        
        Args:
            output_root: Path to output_clft_v2 directory
            frames_list: Path to frames_to_process.txt or all.txt
            target_resolution: Target resolution for output images (shorter dimension)
            num_workers: Number of parallel workers (default: 16)
            batch_size: Number of frames to process in parallel
        """
        self.output_root = Path(output_root)
        self.frames_list = Path(frames_list)
        self.target_resolution = target_resolution
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        self.batch_size = batch_size
        
        # Input directory
        self.lidar_dir = self.output_root / "lidar"
        
        # Output directory
        self.projection_dir = self.output_root / "lidar_projection"
        self.projection_dir.mkdir(parents=True, exist_ok=True)
        
        # Load frames to process
        try:
            with open(self.frames_list) as f:
                self.frame_ids = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(self.frame_ids):,} frames from {self.frames_list}")
        except FileNotFoundError:
            print(f"⚠️  Frames file not found: {self.frames_list}")
            print(f"🔍 Scanning lidar directory for .pkl files...")
            # Scan lidar directory for .pkl files
            pkl_files = list(self.lidar_dir.glob("frame_*.pkl"))
            self.frame_ids = []
            for pkl_file in pkl_files:
                frame_id = pkl_file.stem.replace("frame_", "")
                self.frame_ids.append(frame_id)
            self.frame_ids.sort()
            print(f"✓ Found {len(self.frame_ids):,} frames from lidar files")
    
    def get_distance_color(self, distance, min_dist=0, max_dist=80):
        """Get color for a distance value using cyan-to-blue gradient
        
        Args:
            distance: Distance in meters
            min_dist: Minimum distance for color mapping
            max_dist: Maximum distance for color mapping
        
        Returns:
            (B, G, R) color tuple for OpenCV
        """
        # Clamp distance to range
        distance = np.clip(distance, min_dist, max_dist)
        
        # Normalize to 0-1
        normalized = (distance - min_dist) / (max_dist - min_dist)
        
        # Cyan (close) to Blue (far) gradient
        # Close: Cyan (0, 255, 255) - BGR format
        # Far: Dark Blue (139, 0, 0) - BGR format
        r = int(255 * (1 - normalized))
        g = int(255 * (1 - normalized * 0.6))  # Gradual decrease
        b = int(139 + 116 * (1 - normalized))  # 139 to 255
        
        return (b, g, r)
    
    def load_lidar_pkl(self, frame_id):
        """Load LiDAR pkl file and extract front camera projections with distances
        
        Args:
            frame_id: Frame identifier
        
        Returns:
            tuple of ((N, 2) coordinates, (N,) distances) or (None, None)
        """
        pkl_path = self.lidar_dir / f"frame_{frame_id}.pkl"
        if not pkl_path.exists():
            return None, None
        
        # Check if file is empty (corrupted/incomplete)
        if pkl_path.stat().st_size == 0:
            return None, None
        
        try:
            with open(pkl_path, 'rb') as f:
                clft_data = pickle.load(f)
            
            # Extract camera coordinates and filter for front camera (ID=1)
            camera_coords = clft_data['camera_coordinates']
            front_mask = camera_coords[:, 0] == 1
            
            if not np.any(front_mask):
                return None, None
            
            # Get pixel coordinates for front camera (in original ZOD resolution)
            front_coords = camera_coords[front_mask]
            x_coords = front_coords[:, 1].astype(np.float32)  # X pixel
            y_coords = front_coords[:, 2].astype(np.float32)  # Y pixel
            
            lidar_points = np.column_stack([x_coords, y_coords])
            
            # Extract 3D points for distance calculation
            points_3d = clft_data['3d_points'][front_mask]
            distances = np.sqrt(np.sum(points_3d[:, :3]**2, axis=1))  # Euclidean distance
            
            return lidar_points, distances
            
        except Exception as e:
            print(f"  Error loading LiDAR pkl for {frame_id}: {e}")
            return None, None
    
    def create_projection_image_from_data(self, frame_id, lidar_points, distances):
        """Create 2D projection image from preloaded LiDAR data
        
        Args:
            frame_id: Frame identifier
            lidar_points: Preloaded LiDAR points (N, 2)
            distances: Preloaded distances (N,)
        
        Returns:
            OpenCV image array or None if failed
        """
        if lidar_points is None or distances is None or len(lidar_points) == 0:
            return None
        
        # Target dimensions: 1363x768 pixels
        target_w = 1363
        target_h = 768
        
        # Create black background image
        projection_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Scale coordinates to target shape
        # ZOD original resolution for front camera is 3848x2168
        orig_width = 3848
        orig_height = 2168
        
        scale_x = target_w / orig_width
        scale_y = target_h / orig_height
        
        scaled_points = lidar_points.copy()
        scaled_points[:, 0] *= scale_x
        scaled_points[:, 1] *= scale_y
        
        # Draw each LiDAR point with distance-based color
        # Use numpy for faster pixel setting
        valid_mask = (
            (scaled_points[:, 0] >= 0) & (scaled_points[:, 0] < target_w) &
            (scaled_points[:, 1] >= 0) & (scaled_points[:, 1] < target_h)
        )
        valid_points = scaled_points[valid_mask].astype(int)
        valid_distances = distances[valid_mask]
        
        # Set pixels directly for speed (single pixels instead of circles)
        for point, dist in zip(valid_points, valid_distances):
            x, y = point[0], point[1]
            color = self.get_distance_color(dist)
            projection_img[y, x] = color
        
        return projection_img
    
    def process_all_frames_parallel(self):
        """Process all frames in parallel with preloading"""
        print(f"\n🚀 Parallel LiDAR projection generation")
        print(f"Input: {self.lidar_dir}")
        print(f"Output: {self.projection_dir}")
        print(f"Workers: {self.num_workers} | Batch size: {self.batch_size}")
        print(f"Frames: {len(self.frame_ids):,}")
        
        # Filter out already processed frames
        remaining_frames = []
        for frame_id in self.frame_ids:
            output_path = self.projection_dir / f"frame_{frame_id}.png"
            if not output_path.exists():
                remaining_frames.append(frame_id)
        
        if not remaining_frames:
            print("✅ All frames already processed!")
            return
        
        print(f"Processing: {len(remaining_frames):,} frames")
        
        success_count = 0
        error_count = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Process in batches to manage memory
            for i in tqdm(range(0, len(remaining_frames), self.batch_size), 
                         desc="Processing batches", unit="batch"):
                
                batch_frames = remaining_frames[i:i + self.batch_size]
                
                # Preload batch data in parallel using module-level functions
                preload_args = [(frame_id, str(self.lidar_dir)) for frame_id in batch_frames]
                preload_futures = [executor.submit(preload_lidar_data_worker, args) 
                                 for args in preload_args]
                
                # Wait for preloading to complete
                preloaded_data = []
                for future in preload_futures:
                    preloaded_data.append(future.result())
                
                # Process batch in parallel using module-level functions
                process_args = [(frame_id, lidar_points, distances, str(self.projection_dir)) 
                              for frame_id, (lidar_points, distances) in preloaded_data]
                process_futures = [executor.submit(process_single_frame_worker, args) 
                                 for args in process_args]
                
                # Collect results
                for future in process_futures:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
        
        total_time = time.time() - start_time
        avg_time = total_time / len(remaining_frames) if remaining_frames else 0
        
        print(f"\n{'='*60}")
        print(f"PARALLEL LIDAR PROJECTION GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully created: {success_count:,}")
        print(f"✗ Errors: {error_count:,}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"⏱️  Average per frame: {avg_time:.3f}s")
        print(f"🚀 Speedup: ~{1.5/avg_time:.1f}x faster than sequential")
        print(f"\nOutput:")
        print(f"  📁 {self.projection_dir}/ - LiDAR projection images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D LiDAR projection images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates 2D projection images from LiDAR point clouds using distance-based coloring.
- Cyan: Close points (< 5m)
- Blue: Far points (> 80m)
- Gradient in between

Each projection shows LiDAR points colored by distance.

PERFORMANCE:
- Uses parallel processing with ProcessPoolExecutor
- Preloads LiDAR data in batches
- True multiprocessing for CPU-bound operations
        """
    )
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers (default: 16)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of frames to process in parallel (default: 4)')
    args = parser.parse_args()
    
    # Configuration
    OUTPUT_ROOT = Path("output_clft_v2")
    FRAMES_LIST = OUTPUT_ROOT / "all.txt"  # Use all.txt for all frames
    
    print("="*60)
    print("Parallel LiDAR Projection Generator")
    print("="*60)
    
    generator = LidarProjectionGenerator(
        output_root=OUTPUT_ROOT,
        frames_list=FRAMES_LIST,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    # Process all frames in parallel
    generator.process_all_frames_parallel()


if __name__ == "__main__":
    main()