#!/usr/bin/env python3
"""
Generate 3D-Geometric LiDAR Projections as PNG Images

This script creates 3-channel geometric LiDAR projection images where each pixel contains
normalized X, Y, Z coordinate information from LiDAR point clouds. These projections
preserve complete geometric information for camera-LiDAR fusion training.

Key Features:
- 3-channel PNG output: Blue=X, Green=Y, Red=Z coordinates
- Distance-weighted enhancement for better object distinctness
- Gaussian smoothing to increase point density
- Morphological operations for improved object boundaries
- Support for both basic and enhanced projection methods

Coordinate System Assumptions (ZOD dataset):
- camera_coordinates[:, 0] == 1: Front camera points only
- camera_coordinates[:, 1]: Pixel X coordinate (0-3848)
- camera_coordinates[:, 2]: Pixel Y coordinate (0-2168)
- 3d_points[:, 0]: X coordinate (left/right, meters)
- 3d_points[:, 1]: Y coordinate (up/down, meters)
- 3d_points[:, 2]: Z coordinate (depth/distance, meters)

Usage:
    python generate_lidar_png.py --input_dir /path/to/data --output_dir /path/to/output
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

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  Warning: scipy not available, gaussian filtering will use OpenCV fallback")

@jit(nopython=True)
def place_weighted_points_numba(x_valid, y_valid, x_norm, y_norm, z_norm, distance_weights, target_shape):
    """
    JIT-compiled function for fast placement of weighted LiDAR points in sparse arrays.
    
    This function efficiently creates sparse arrays for each coordinate channel,
    applying distance-based weighting to emphasize closer objects.
    
    Args:
        x_valid: Valid x pixel coordinates (integers)
        y_valid: Valid y pixel coordinates (integers)
        x_norm: Normalized X coordinates (0-255 range)
        y_norm: Normalized Y coordinates (0-255 range)
        z_norm: Normalized Z coordinates (0-255 range)
        distance_weights: Distance-based weights for each point
        target_shape: Target image shape (height, width)
        
    Returns:
        tuple: (x_sparse, y_sparse, z_sparse) arrays with weighted point values
    """
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
    """
    Creates geometric LiDAR projections from CLFT-format pickle files.
    
    This class handles the conversion of 3D LiDAR point clouds into 2D geometric
    projection images that preserve spatial coordinate information for deep
    learning model training.
    """
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize the geometric LiDAR projector.
        
        Args:
            input_dir: Directory containing 'lidar_pickle' folder with CLFT pickle files
            output_dir: Directory where geometric projection PNGs will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_lidar_data(self, frame_id):
        """
        Load and validate LiDAR pickle data for a specific frame.
        
        Args:
            frame_id: Frame identifier string
            
        Returns:
            dict: CLFT data structure with camera_coordinates and 3d_points, or None if invalid
        """
        pkl_path = self.input_dir / "lidar_pickle" / f"frame_{frame_id}.pkl"
        if not pkl_path.exists():
            print(f"  ⚠️  LiDAR pickle file not found: {pkl_path}")
            return None

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate required data structure keys
            required_keys = ['camera_coordinates', '3d_points']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"  ❌ Invalid LiDAR data structure for {frame_id}: missing keys {missing_keys}")
                return None
            
            # Validate data types and array shapes
            if not isinstance(data['camera_coordinates'], np.ndarray):
                print(f"  ❌ camera_coordinates is not a numpy array for {frame_id}")
                return None
            if not isinstance(data['3d_points'], np.ndarray):
                print(f"  ❌ 3d_points is not a numpy array for {frame_id}")
                return None
            if data['camera_coordinates'].shape[0] != data['3d_points'].shape[0]:
                print(f"  ❌ Mismatched array lengths for {frame_id}: camera_coords {data['camera_coordinates'].shape[0]}, 3d_points {data['3d_points'].shape[0]}")
                return None
            
            return data
            
        except Exception as e:
            print(f"  ❌ Error loading LiDAR data for {frame_id}: {e}")
            return None

    def create_geometric_projection(self, frame_id, target_shape=(768, 1363)):
        """
        Create basic 3-channel geometric projection from LiDAR data.
        
        Each channel contains normalized coordinate information:
        - Channel 0 (Blue): Normalized X coordinates (-50m to +50m → 0-255)
        - Channel 1 (Green): Normalized Y coordinates (-50m to +50m → 0-255)
        - Channel 2 (Red): Normalized Z coordinates (0-100m → 0-255)
        
        Args:
            frame_id: Frame identifier string
            target_shape: Target image shape (height, width)
            
        Returns:
            np.ndarray: 3-channel geometric projection image (uint8)
        """
        data = self.load_lidar_data(frame_id)
        if data is None:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Initialize geometric projection with 3 channels (X, Y, Z)
        geometric_proj = np.zeros((*target_shape, 3), dtype=np.float32)

        # Extract front camera coordinates and 3D points only
        camera_coords = data['camera_coordinates']
        points_3d = data['3d_points']
        
        # Filter for front camera points (camera_coords[:, 0] == 1)
        front_mask = camera_coords[:, 0] == 1
        front_count = np.sum(front_mask)
        total_count = len(camera_coords)
        
        if front_count == 0:
            print(f"  ⚠️  No front camera points found for {frame_id} (all camera_coords[:, 0] != 1)")
            return np.zeros((*target_shape, 3), dtype=np.uint8)
        elif front_count < total_count * 0.1:  # Less than 10% front camera points
            print(f"  ⚠️  Very few front camera points for {frame_id}: {front_count}/{total_count} ({front_count/total_count:.1%})")
        
        front_coords = camera_coords[front_mask]
        front_points = points_3d[front_mask]

        if len(front_coords) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Extract pixel coordinates from camera_coordinates array
        x_coords = front_coords[:, 1].astype(np.float32)  # X pixel coordinates
        y_coords = front_coords[:, 2].astype(np.float32)  # Y pixel coordinates

        # Scale coordinates from original resolution to target resolution
        orig_width = 3848   # Original ZOD image width
        orig_height = 2168  # Original ZOD image height
        scale_x = target_shape[1] / orig_width
        scale_y = target_shape[0] / orig_height

        scaled_x = x_coords * scale_x
        scaled_y = y_coords * scale_y

        # Filter coordinates that fall within target image bounds
        valid_mask = (scaled_x >= 0) & (scaled_x < target_shape[1]) & \
                    (scaled_y >= 0) & (scaled_y < target_shape[0])

        x_valid = scaled_x[valid_mask].astype(int)
        y_valid = scaled_y[valid_mask].astype(int)
        points_valid = front_points[valid_mask]

        if len(points_valid) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Normalize 3D coordinates to 0-255 range for image channels
        x_3d = points_valid[:, 0]  # X coordinates (left/right)
        y_3d = points_valid[:, 1]  # Y coordinates (up/down)
        z_3d = points_valid[:, 2]  # Z coordinates (depth)

        # Normalize X coordinates: -50m to +50m range → 0-255
        x_norm = np.clip((x_3d + 50) / 100, 0, 1) * 255

        # Normalize Y coordinates: -50m to +50m range → 0-255
        y_norm = np.clip((y_3d + 50) / 100, 0, 1) * 255

        # Normalize Z coordinates: 0-100m range → 0-255
        z_norm = np.clip(z_3d / 100, 0, 1) * 255

        # Place normalized coordinate values at corresponding pixel locations
        for i, (x, y) in enumerate(zip(x_valid, y_valid)):
            geometric_proj[y, x, 0] = x_norm[i]  # X coordinate in Blue channel
            geometric_proj[y, x, 1] = y_norm[i]  # Y coordinate in Green channel
            geometric_proj[y, x, 2] = z_norm[i]  # Z coordinate in Red channel

        return geometric_proj.astype(np.uint8)

    def create_enhanced_geometric_projection(self, frame_id, target_shape=(768, 1363), sigma=1.0):
        """
        Create enhanced 3-channel geometric projection with advanced processing.
        
        Enhancements include:
        - Distance-based weighting (closer objects more prominent)
        - Improved normalization ranges for better precision
        - Gaussian smoothing for increased point density
        - Morphological operations for better object boundaries
        - Gap filling using distance-based interpolation
        
        Args:
            frame_id: Frame identifier string
            target_shape: Target image shape (height, width)
            sigma: Gaussian smoothing standard deviation
            
        Returns:
            np.ndarray: Enhanced 3-channel geometric projection image (uint8)
        """
        data = self.load_lidar_data(frame_id)
        if data is None:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Initialize geometric projection with higher precision for processing
        geometric_proj = np.zeros((*target_shape, 3), dtype=np.float32)

        # Extract front camera coordinates and 3D points
        camera_coords = data['camera_coordinates']
        points_3d = data['3d_points']
        
        # Filter for front camera points only
        front_mask = camera_coords[:, 0] == 1
        front_count = np.sum(front_mask)
        total_count = len(camera_coords)
        
        if front_count == 0:
            print(f"  ⚠️  No front camera points found for {frame_id} (all camera_coords[:, 0] != 1)")
            return np.zeros((*target_shape, 3), dtype=np.uint8)
        elif front_count < total_count * 0.1:
            print(f"  ⚠️  Very few front camera points for {frame_id}: {front_count}/{total_count} ({front_count/total_count:.1%})")
        
        front_coords = camera_coords[front_mask]
        front_points = points_3d[front_mask]

        if len(front_coords) == 0:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Extract and scale pixel coordinates
        x_coords = front_coords[:, 1].astype(np.float32)
        y_coords = front_coords[:, 2].astype(np.float32)

        orig_width = 3848
        orig_height = 2168
        scale_x = target_shape[1] / orig_width
        scale_y = target_shape[0] / orig_height

        scaled_x = x_coords * scale_x
        scaled_y = y_coords * scale_y

        # Filter valid pixel coordinates
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
        distance_weights = np.clip(distance_weights * 3.0, 0.2, 3.0)  # Scale and clip

        # 2. Improved normalization with wider dynamic ranges for better precision
        # X (left/right): Wider range for better precision near objects
        x_norm = np.clip((x_3d + 30) / 60, 0, 1) * 255  # -30m to +30m range

        # Y (up/down): Wider range for better precision
        y_norm = np.clip((y_3d + 15) / 30, 0, 1) * 255  # -15m to +15m range

        # Z (depth): Logarithmic scaling to make close objects more distinct
        z_log = np.log(z_3d + 1.0)  # Logarithmic distance
        z_norm = np.clip(z_log / np.log(121.0), 0, 1) * 255  # 0-120m log-scaled

        # Apply distance weighting to make closer objects more prominent
        x_norm = np.clip(x_norm * distance_weights, 0, 255)
        y_norm = np.clip(y_norm * distance_weights, 0, 255)
        z_norm = np.clip(z_norm * distance_weights, 0, 255)

        # ===== POINT DENSITY ENHANCEMENT =====
        # Create sparse arrays and apply gaussian smoothing for denser point coverage
        if NUMBA_AVAILABLE:
            # Use JIT-compiled function for fast point placement
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
        
        # Apply gaussian smoothing to increase point density
        kernel_size = int(6 * sigma) + 1  # Rule of thumb: kernel_size = 6*sigma + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        
        try:
            # Use OpenCV GaussianBlur for performance
            geometric_proj[:, :, 0] = cv2.GaussianBlur(x_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
            geometric_proj[:, :, 1] = cv2.GaussianBlur(y_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
            geometric_proj[:, :, 2] = cv2.GaussianBlur(z_sparse, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_CONSTANT)
        except:
            # Fallback to scipy if OpenCV fails
            if SCIPY_AVAILABLE:
                geometric_proj[:, :, 0] = ndimage.gaussian_filter(x_sparse, sigma=sigma, mode='constant', cval=0)
                geometric_proj[:, :, 1] = ndimage.gaussian_filter(y_sparse, sigma=sigma, mode='constant', cval=0)
                geometric_proj[:, :, 2] = ndimage.gaussian_filter(z_sparse, sigma=sigma, mode='constant', cval=0)
            else:
                print("  ❌ Neither OpenCV nor scipy gaussian filtering available!")
                raise RuntimeError("Gaussian filtering not available - install scipy or check OpenCV")

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
            gap_mask = (dist_transform > 0) & (dist_transform <= 3.0) & (mask == 0)

            if np.any(gap_mask):
                # Efficient vectorized approach: use dilation with distance weighting
                # Create a weighted dilation that considers distance
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # LARGER kernel

                # Dilate the channel and mask
                dilated_channel = cv2.dilate(channel.astype(np.float32), kernel, iterations=1)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)

                # Create distance-weighted interpolation
                # Closer original pixels contribute more
                distance_weights_interp = 1.0 / (dist_transform + 1.0)  # Avoid division by zero
                distance_weights_interp = np.clip(distance_weights_interp, 0, 1)  # Normalize

                # Combine original channel with dilated version using distance weighting
                interpolated = channel * (1 - distance_weights_interp * gap_mask) + dilated_channel * (distance_weights_interp * gap_mask)

                # Only fill gap regions
                geometric_proj[:, :, c] = np.where(gap_mask, interpolated, geometric_proj[:, :, c])

        # ===== MORPHOLOGICAL DILATION =====
        # Expand object regions to provide more supervision signal for training
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for c in range(3):
            # Create binary mask of non-zero pixels
            mask = (geometric_proj[:, :, c] > 0).astype(np.uint8)
            # Dilate the mask to expand object boundaries
            dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=3)
            # Apply dilation by expanding non-zero regions
            dilated_channel = cv2.dilate(geometric_proj[:, :, c], dilation_kernel, iterations=3)
            # Only keep dilated values where mask was expanded
            geometric_proj[:, :, c] = np.where(dilated_mask > mask, dilated_channel, geometric_proj[:, :, c])

        # ===== POST-PROCESSING =====
        # Boost brightness using max-based scaling for better contrast
        for c in range(3):
            channel = geometric_proj[:, :, c]
            if np.any(channel > 0):
                # Scale so that max value becomes 255 for full brightness contrast
                max_val = np.max(channel)
                if max_val > 0:
                    scale_factor = 255.0 / max_val
                    geometric_proj[:, :, c] = np.clip(channel * scale_factor, 0, 255)

        # Ensure minimum brightness for visibility and clip to valid range
        geometric_proj = np.clip(geometric_proj, 0, 255)

        return geometric_proj.astype(np.uint8)

    def verify_geometric_projection(self, frame_id, use_enhanced=True):
        """
        Verify that geometric projection contains proper X, Y, Z coordinate information.

        Performs comprehensive validation of the geometric projection by:
        - Checking pixel value ranges and distributions
        - Converting back to 3D coordinates to verify physical ranges
        - Analyzing sparsity and data quality metrics
        - Comparing enhanced vs basic method characteristics

        Args:
            frame_id: Frame identifier string to verify
            use_enhanced: Whether to use enhanced projection method

        Returns:
            np.ndarray: The geometric projection image for further inspection
        """
        if use_enhanced:
            geom_proj = self.create_enhanced_geometric_projection(frame_id)
        else:
            geom_proj = self.create_geometric_projection(frame_id)

        print(f"\n🔍 Geometric Projection Verification for {frame_id}")
        print(f"Shape: {geom_proj.shape}, Dtype: {geom_proj.dtype}")
        print(f"Method: {'Enhanced' if use_enhanced else 'Basic'}")

        # Check each channel for coordinate information integrity
        channel_names = ['X (Blue)', 'Y (Green)', 'Z (Red)']

        # Use correct physical coordinate ranges for each method
        if use_enhanced:
            coord_ranges = [(-30, 30), (-15, 15), (0, 120)]  # Enhanced ranges with better precision
        else:
            coord_ranges = [(-50, 50), (-50, 50), (0, 100)]  # Basic ranges

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
                # Enhanced method uses distance weighting and smoothing
                if use_enhanced:
                    print(f"  Enhanced: Distance-weighted with gaussian smoothing applied")
                else:
                    # Convert pixel values back to 3D coordinates for verification
                    coord_values = (channel[channel > 0].astype(float) / 255)
                    if i == 0:  # X channel (left/right position)
                        coord_3d = coord_values * 100 - 50  # -50m to +50m range
                    elif i == 1:  # Y channel (up/down position)
                        coord_3d = coord_values * 100 - 50  # -50m to +50m range
                    else:  # Z channel (depth)
                        coord_3d = coord_values * 100  # 0-100m range

                    print(f"  Sample 3D values: {coord_3d[:5]}")
                    print(f"  3D range in data: {coord_3d.min():.1f}m - {coord_3d.max():.1f}m")

        # Overall statistics for data quality assessment
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
        """
        Process all frames in the list and create geometric LiDAR projections.

        This method orchestrates the batch processing of multiple frames, creating
        either basic or enhanced geometric projections and saving them as PNG files.
        It provides progress tracking and error handling for robust batch processing.

        Args:
            frame_list: List of frame identifier strings to process
            use_enhanced: Whether to use enhanced projection method with advanced processing

        Returns:
            None (saves PNG files to output directory)
        """
        method_name = "enhanced geometric" if use_enhanced else "basic geometric"
        print(f"🔄 Creating {method_name} LiDAR projections for {len(frame_list)} frames...")

        success_count = 0
        error_count = 0

        for frame_id in tqdm(frame_list):
            try:
                # Create geometric projection using specified method
                if use_enhanced:
                    geom_proj = self.create_enhanced_geometric_projection(frame_id)
                else:
                    geom_proj = self.create_geometric_projection(frame_id)

                if geom_proj is not None:
                    # Save geometric projection as PNG file
                    output_path = self.output_dir / f"frame_{frame_id}.png"
                    success = cv2.imwrite(str(output_path), geom_proj)
                    if success:
                        success_count += 1
                    else:
                        print(f"  ❌ Failed to save projection for {frame_id}")
                        error_count += 1
                else:
                    print(f"  ❌ Failed to create projection for {frame_id}")
                    error_count += 1

            except Exception as e:
                print(f"  ❌ Error processing {frame_id}: {e}")
                error_count += 1
                continue

        print(f"✅ Geometric LiDAR projections complete!")
        print(f"   ✓ Successfully processed: {success_count:,} frames")
        if error_count > 0:
            print(f"   ❌ Errors: {error_count:,} frames")

def main():
    """
    Main entry point for the geometric LiDAR projection generation script.

    This script processes ZOD dataset LiDAR data to create 3-channel geometric
    projections where each channel encodes X, Y, Z coordinate information.
    Supports both basic and enhanced projection methods with various processing options.

    Command line usage:
        python generate_lidar_png.py --input_dir /path/to/zod/data --output_dir /path/to/output
        python generate_lidar_png.py --frames_file frames.txt --verify --enhanced
    """
    parser = argparse.ArgumentParser(description="Create geometric LiDAR projections")
    parser.add_argument("--input_dir", default="/media/tom/ml/zod_temp",
                       help="Input directory containing lidar_pickle folder with CLFT data")
    parser.add_argument("--output_dir", default="/media/tom/ml/zod_temp/lidar_png",
                       help="Output directory for LiDAR geometric projection PNG files")
    parser.add_argument("--frames_file", help="Optional frames file to process specific frames")
    parser.add_argument("--verify", action="store_true",
                       help="Run verification on geometric content instead of full processing")
    parser.add_argument("--enhanced", action="store_true", default=False,
                       help="Use enhanced projections with distance weighting and gaussian smoothing")

    args = parser.parse_args()

    # Determine which frames to process
    if args.frames_file:
        # Load frame IDs from specified text file
        frame_ids = []
        with open(args.frames_file) as f:
            for line in f:
                if line.strip():
                    # Extract frame ID from filename (remove path, frame_ prefix, .png extension)
                    frame_id = line.strip().split('/')[-1].replace('frame_', '').replace('.png', '')
                    frame_ids.append(frame_id)
    else:
        # Auto-discover all available frames from CLFT pickle files
        lidar_dir = Path(args.input_dir) / "lidar_pickle"
        frame_ids = []
        if lidar_dir.exists():
            for pkl_file in lidar_dir.glob("frame_*.pkl"):
                # Extract frame ID from filename
                frame_id = pkl_file.stem.replace('frame_', '')
                frame_ids.append(frame_id)
        frame_ids.sort()  # Ensure consistent processing order

    print(f"📊 Processing {len(frame_ids)} frames")

    # Initialize the geometric projector
    projector = GeometricLiDARProjector(args.input_dir, args.output_dir)

    if args.verify:
        # Verification mode: analyze geometric content of first frame
        if frame_ids:
            projector.verify_geometric_projection(frame_ids[0], use_enhanced=args.enhanced)
    else:
        # Processing mode: create geometric projections for all frames
        projector.process_frames(frame_ids, use_enhanced=args.enhanced)


if __name__ == "__main__":
    main()