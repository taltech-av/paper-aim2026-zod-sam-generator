#!/usr/bin/env python3
"""
Generate Proper 3D-Geometric LiDAR Projections
Creates 3-channel images where each channel contains normalized X, Y, Z coordinates
CRITICAL: Preserves complete geometric information for fusion training
"""

import numpy as np
import cv2
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def place_weighted_points_numba(x_valid, y_valid, x_norm, y_norm, z_norm, distance_weights, target_shape):
    """JIT-compiled function to place weighted points in sparse arrays"""
    x_sparse = np.zeros(target_shape, dtype=np.float32)
    y_sparse = np.zeros(target_shape, dtype=np.float32)
    z_sparse = np.zeros(target_shape, dtype=np.float32)
    
    for i in range(len(x_valid)):
        x, y = x_valid[i], y_valid[i]
        weight = distance_weights[i]
        x_sparse[y, x] = x_norm[i] * weight
        y_sparse[y, x] = y_norm[i] * weight
        z_sparse[y, x] = z_norm[i] * weight
    
    return x_sparse, y_sparse, z_sparse

class GeometricLiDARProjector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_lidar_data(self, frame_id):
        """Load LiDAR pickle data"""
        pkl_path = self.input_dir / "lidar_pickle" / f"frame_{frame_id}.pkl"
        if not pkl_path.exists():
            return None

        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def create_geometric_projection(self, frame_id, target_shape=(768, 1363)):
        """
        Create 3-channel geometric projection:
        - Channel 0 (Blue): Normalized X coordinates (-1 to 1)
        - Channel 1 (Green): Normalized Y coordinates (-1 to 1)
        - Channel 2 (Red): Normalized Z coordinates (0 to 1, distance)
        """
        data = self.load_lidar_data(frame_id)
        if data is None:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Initialize geometric projection (3 channels)
        geometric_proj = np.zeros((*target_shape, 3), dtype=np.float32)

        # Get front camera coordinates and 3D points
        camera_coords = data['camera_coordinates']
        points_3d = data['3d_points']
        front_mask = camera_coords[:, 0] == 1

        front_coords = camera_coords[front_mask]
        front_points = points_3d[front_mask]

        if len(front_coords) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Extract pixel coordinates and scale to target resolution
        x_coords = front_coords[:, 1].astype(np.float32)
        y_coords = front_coords[:, 2].astype(np.float32)

        # Scale coordinates (same as original projection)
        orig_width = 3848
        orig_height = 2168
        scale_x = target_shape[1] / orig_width
        scale_y = target_shape[0] / orig_height

        scaled_x = x_coords * scale_x
        scaled_y = y_coords * scale_y

        # Filter valid coordinates after scaling
        valid_mask = (scaled_x >= 0) & (scaled_x < target_shape[1]) & \
                    (scaled_y >= 0) & (scaled_y < target_shape[0])

        x_valid = scaled_x[valid_mask].astype(int)
        y_valid = scaled_y[valid_mask].astype(int)
        points_valid = front_points[valid_mask]

        if len(points_valid) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Normalize 3D coordinates to 0-255 range
        # X and Y: Normalize to -1 to 1, then scale to 0-255
        x_3d = points_valid[:, 0]
        y_3d = points_valid[:, 1]
        z_3d = points_valid[:, 2]

        # Normalize X (left/right) - typically -50 to 50 meters
        x_norm = np.clip((x_3d + 50) / 100, 0, 1) * 255

        # Normalize Y (up/down) - typically -50 to 50 meters
        y_norm = np.clip((y_3d + 50) / 100, 0, 1) * 255

        # Normalize Z (depth) - typically 0 to 100+ meters
        z_norm = np.clip(z_3d / 100, 0, 1) * 255

        # Set geometric values at each pixel location
        for i, (x, y) in enumerate(zip(x_valid, y_valid)):
            geometric_proj[y, x, 0] = x_norm[i]  # X coordinate (Blue)
            geometric_proj[y, x, 1] = y_norm[i]  # Y coordinate (Green)
            geometric_proj[y, x, 2] = z_norm[i]  # Z coordinate (Red)

        return geometric_proj.astype(np.uint8)

    def create_enhanced_geometric_projection(self, frame_id, target_shape=(768, 1363), sigma=1.0):
        """
        Create enhanced 3-channel geometric projection with:
        - Better channel weighting (emphasize distance for object distinctness)
        - Gaussian smoothing (increase point density)
        - Improved normalization for object detection
        """
        data = self.load_lidar_data(frame_id)
        if data is None:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Initialize geometric projection (3 channels) - higher precision for smoothing
        geometric_proj = np.zeros((*target_shape, 3), dtype=np.float32)

        # Get front camera coordinates and 3D points
        camera_coords = data['camera_coordinates']
        points_3d = data['3d_points']
        front_mask = camera_coords[:, 0] == 1

        front_coords = camera_coords[front_mask]
        front_points = points_3d[front_mask]

        if len(front_coords) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Extract pixel coordinates and scale to target resolution
        x_coords = front_coords[:, 1].astype(np.float32)
        y_coords = front_coords[:, 2].astype(np.float32)

        # Scale coordinates (same as original projection)
        orig_width = 3848
        orig_height = 2168
        scale_x = target_shape[1] / orig_width
        scale_y = target_shape[0] / orig_height

        scaled_x = x_coords * scale_x
        scaled_y = y_coords * scale_y

        # Filter valid coordinates after scaling
        valid_mask = (scaled_x >= 0) & (scaled_x < target_shape[1]) & \
                    (scaled_y >= 0) & (scaled_y < target_shape[0])

        x_valid = scaled_x[valid_mask].astype(int)
        y_valid = scaled_y[valid_mask].astype(int)
        points_valid = front_points[valid_mask]

        if len(points_valid) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Extract 3D coordinates
        x_3d = points_valid[:, 0]
        y_3d = points_valid[:, 1]
        z_3d = points_valid[:, 2]

        # ===== ENHANCED NORMALIZATION =====
        # 1. Distance-based weighting: closer objects get higher intensity
        distance_weights = 1.0 / (z_3d + 1.0)  # Inverse distance weighting
        distance_weights = np.clip(distance_weights * 2.0, 0.1, 2.0)  # Scale and clip

        # 2. Improved normalization with better dynamic range
        # X (left/right): Use smaller range for better precision near objects
        x_norm = np.clip((x_3d + 20) / 40, 0, 1) * 255  # -20m to +20m range

        # Y (up/down): Use smaller range for better precision
        y_norm = np.clip((y_3d + 10) / 20, 0, 1) * 255  # -10m to +10m range

        # Z (depth): Emphasize closer distances with non-linear scaling
        # Use logarithmic scaling to make close objects more distinct
        z_log = np.log(z_3d + 1.0)  # Logarithmic distance
        z_norm = np.clip(z_log / np.log(101.0), 0, 1) * 255  # 0-100m log-scaled

        # Apply distance weighting to make closer objects more prominent
        x_norm = np.clip(x_norm * distance_weights, 0, 255)
        y_norm = np.clip(y_norm * distance_weights, 0, 255)
        z_norm = np.clip(z_norm * distance_weights, 0, 255)

        # ===== POINT DENSITY ENHANCEMENT (OPTIMIZED) =====
        # Create sparse arrays for each channel, then apply gaussian filtering
        from scipy import ndimage
        
        # Use JIT-compiled function for fast point placement
        if NUMBA_AVAILABLE:
            x_sparse, y_sparse, z_sparse = place_weighted_points_numba(
                x_valid, y_valid, x_norm, y_norm, z_norm, distance_weights, target_shape
            )
        else:
            # Fallback to Python loop if numba not available
            x_sparse = np.zeros(target_shape, dtype=np.float32)
            y_sparse = np.zeros(target_shape, dtype=np.float32) 
            z_sparse = np.zeros(target_shape, dtype=np.float32)
            
            # Place weighted values at pixel locations
            for i, (x, y) in enumerate(zip(x_valid, y_valid)):
                x_sparse[y, x] = x_norm[i] * distance_weights[i]
                y_sparse[y, x] = y_norm[i] * distance_weights[i]
                z_sparse[y, x] = z_norm[i] * distance_weights[i]
        
        # Apply gaussian smoothing to entire images at once
        # Try cv2.GaussianBlur for potentially better performance
        kernel_size = int(6 * sigma) + 1  # Rule of thumb: kernel_size = 6*sigma + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        
        try:
            # Use cv2.GaussianBlur if available and appropriate
            geometric_proj[:, :, 0] = cv2.GaussianBlur(x_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
            geometric_proj[:, :, 1] = cv2.GaussianBlur(y_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
            geometric_proj[:, :, 2] = cv2.GaussianBlur(z_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
        except:
            # Fallback to scipy if cv2 fails
            geometric_proj[:, :, 0] = ndimage.gaussian_filter(x_sparse, sigma=sigma, mode='constant', cval=0)
            geometric_proj[:, :, 1] = ndimage.gaussian_filter(y_sparse, sigma=sigma, mode='constant', cval=0)
            geometric_proj[:, :, 2] = ndimage.gaussian_filter(z_sparse, sigma=sigma, mode='constant', cval=0)

        # ===== EFFICIENT DISTANCE-BASED GAP FILLING =====
        # Fill gaps using distance-based interpolation for smoother object boundaries
        # Vectorized implementation for better performance
        for c in range(3):
            channel = geometric_proj[:, :, c]
            mask = (channel > 0).astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Skip if no objects or only background
            if num_labels <= 1:
                continue

            # Create distance transform to find gaps near objects
            dist_transform = cv2.distanceTransform(255 - mask * 255, cv2.DIST_L2, 5)

            # Identify gap pixels (close to objects but not part of them)
            gap_mask = (dist_transform > 0) & (dist_transform <= 2.0) & (mask == 0)

            if np.any(gap_mask):
                # Efficient vectorized approach: use dilation with distance weighting
                # Create a weighted dilation that considers distance
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

                # Dilate the channel and mask
                dilated_channel = cv2.dilate(channel.astype(np.float32), kernel, iterations=1)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)

                # Create distance-weighted interpolation
                # Closer original pixels contribute more
                distance_weights = 1.0 / (dist_transform + 1.0)  # Avoid division by zero
                distance_weights = np.clip(distance_weights, 0, 1)  # Normalize

                # Combine original channel with dilated version using distance weighting
                interpolated = channel * (1 - distance_weights * gap_mask) + dilated_channel * (distance_weights * gap_mask)

                # Only fill gap regions
                geometric_proj[:, :, c] = np.where(gap_mask, interpolated, geometric_proj[:, :, c])

        # ===== MORPHOLOGICAL DILATION =====
        # Dilate object regions to provide more supervision signal for training
        # This expands the object boundaries to help with object detection
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel
        for c in range(3):
            # Create binary mask of non-zero pixels
            mask = (geometric_proj[:, :, c] > 0).astype(np.uint8)
            # Dilate the mask with more iterations
            dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=2)  # More iterations
            # Apply dilation by expanding non-zero regions
            dilated_channel = cv2.dilate(geometric_proj[:, :, c], dilation_kernel, iterations=2)
            # Only keep dilated values where mask was expanded
            geometric_proj[:, :, c] = np.where(dilated_mask > mask, dilated_channel, geometric_proj[:, :, c])

        # ===== POST-PROCESSING =====
        # Boost brightness using faster scaling method for better contrast
        for c in range(3):
            channel = geometric_proj[:, :, c]
            if np.any(channel > 0):
                # Use max value for scaling (faster than percentile)
                max_val = np.max(channel)
                if max_val > 0:
                    # Scale so that max value becomes 255 (full brightness for best contrast)
                    scale_factor = 255.0 / max_val
                    geometric_proj[:, :, c] = np.clip(channel * scale_factor, 0, 255)

        # Ensure we have some minimum brightness for visibility
        geometric_proj = np.clip(geometric_proj, 0, 255)

        return geometric_proj.astype(np.uint8)

    def verify_geometric_projection(self, frame_id, use_enhanced=True):
        """Verify that geometric projection contains proper X, Y, Z information"""
        if use_enhanced:
            geom_proj = self.create_enhanced_geometric_projection(frame_id)
        else:
            geom_proj = self.create_geometric_projection(frame_id)

        print(f"\n🔍 Geometric Projection Verification for {frame_id}")
        print(f"Shape: {geom_proj.shape}, Dtype: {geom_proj.dtype}")
        print(f"Method: {'Enhanced' if use_enhanced else 'Basic'}")

        # Check each channel
        channel_names = ['X (Blue)', 'Y (Green)', 'Z (Red)']
        coord_ranges = [(-20, 20), (-10, 10), (0, 100)]  # Enhanced ranges

        for i, (name, coord_range) in enumerate(zip(channel_names, coord_ranges)):
            channel = geom_proj[:, :, i]
            nonzero = np.count_nonzero(channel)
            unique_vals = len(np.unique(channel))

            print(f"\n{name} Channel:")
            print(f"  Non-zero pixels: {nonzero:,}")
            print(f"  Unique values: {unique_vals}")
            print(f"  Value range: {channel.min()} - {channel.max()}")
            print(f"  Expected 3D range: {coord_range[0]}m - {coord_range[1]}m")

            if nonzero > 0:
                # For enhanced method, values are accumulated and normalized differently
                if use_enhanced:
                    print(f"  Enhanced: Distance-weighted with gaussian smoothing")
                else:
                    # Convert back to 3D coordinates to verify
                    coord_values = (channel[channel > 0].astype(float) / 255)
                    if i < 2:  # X, Y channels
                        coord_3d = coord_values * 40 - 20  # Enhanced X range
                        if i == 1:  # Y channel
                            coord_3d = coord_values * 20 - 10  # Enhanced Y range
                    else:  # Z channel
                        coord_3d = coord_values * 100

                    print(f"  Sample 3D values: {coord_3d[:5]}")
                    print(f"  3D range in data: {coord_3d.min():.1f}m - {coord_3d.max():.1f}m")

        # Overall statistics
        total_pixels = geom_proj.shape[0] * geom_proj.shape[1] * geom_proj.shape[2]
        nonzero_pixels = np.count_nonzero(geom_proj)
        sparsity = 100 * nonzero_pixels / total_pixels
        print(f"\nOverall Statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Non-zero pixels: {nonzero_pixels:,}")
        print(f"  Sparsity: {sparsity:.2f}%")
        
        if use_enhanced:
            print(f"  Enhancement: Distance weighting + gaussian smoothing applied")

        return geom_proj

    def process_frames(self, frame_list, use_enhanced=True):
        """Process all frames and create geometric projections"""
        method_name = "enhanced geometric" if use_enhanced else "basic geometric"
        print(f"🔄 Creating {method_name} LiDAR projections for {len(frame_list)} frames...")

        for frame_id in tqdm(frame_list):
            try:
                # Create geometric projection
                if use_enhanced:
                    geom_proj = self.create_enhanced_geometric_projection(frame_id)
                else:
                    geom_proj = self.create_geometric_projection(frame_id)

                if geom_proj is not None:
                    # Save geometric projection
                    output_path = self.output_dir / f"frame_{frame_id}.png"
                    cv2.imwrite(str(output_path), geom_proj)

            except Exception as e:
                print(f"❌ Error processing {frame_id}: {e}")
                continue

        print("✅ Geometric LiDAR projections complete!")

def main():
    parser = argparse.ArgumentParser(description="Create geometric LiDAR projections")
    parser.add_argument("--input_dir", default="/media/tom/ml/zod_temp", help="Input directory")
    parser.add_argument("--output_dir", default="/media/tom/ml/zod_temp/lidar_png", help="Output directory")
    parser.add_argument("--frames_file", help="Frames file to process (optional - if not provided, processes all frames from metadata)")
    parser.add_argument("--verify", action="store_true", help="Verify geometric content")
    parser.add_argument("--enhanced", action="store_true", default=True, 
                       help="Use enhanced projections with better channel weighting and gaussian smoothing (default: True)")

    args = parser.parse_args()

    # Get frame IDs
    if args.frames_file:
        # Load from specified file
        frame_ids = []
        with open(args.frames_file) as f:
            for line in f:
                if line.strip():
                    frame_id = line.strip().split('/')[-1].replace('frame_', '').replace('.png', '')
                    frame_ids.append(frame_id)
    else:
        # Get all frame IDs from lidar pickle files
        lidar_dir = Path(args.input_dir) / "lidar_pickle"
        frame_ids = []
        if lidar_dir.exists():
            for pkl_file in lidar_dir.glob("frame_*.pkl"):
                frame_id = pkl_file.stem.replace('frame_', '')
                frame_ids.append(frame_id)
        frame_ids.sort()  # Ensure consistent ordering

    print(f"📊 Processing {len(frame_ids)} frames")

    # Create projector
    projector = GeometricLiDARProjector(args.input_dir, args.output_dir)

    if args.verify:
        # Verify on first frame
        if frame_ids:
            projector.verify_geometric_projection(frame_ids[0], use_enhanced=args.enhanced)
    else:
        # Process all frames
        projector.process_frames(frame_ids, use_enhanced=args.enhanced)

if __name__ == "__main__":
    main()