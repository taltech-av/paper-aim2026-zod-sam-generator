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

class GeometricLiDARProjector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_lidar_data(self, frame_id):
        """Load LiDAR pickle data"""
        pkl_path = self.input_dir / "lidar" / f"frame_{frame_id}.pkl"
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

    def verify_geometric_projection(self, frame_id):
        """Verify that geometric projection contains proper X, Y, Z information"""
        geom_proj = self.create_geometric_projection(frame_id)

        print(f"\n🔍 Geometric Projection Verification for {frame_id}")
        print(f"Shape: {geom_proj.shape}, Dtype: {geom_proj.dtype}")

        # Check each channel
        channel_names = ['X (Blue)', 'Y (Green)', 'Z (Red)']
        coord_ranges = [(-50, 50), (-50, 50), (0, 100)]  # Expected 3D ranges

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
                # Convert back to 3D coordinates to verify
                coord_values = (channel[channel > 0].astype(float) / 255)
                if i < 2:  # X, Y channels
                    coord_3d = coord_values * 100 - 50
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

        return geom_proj

    def process_frames(self, frame_list):
        """Process all frames and create geometric projections"""
        print(f"🔄 Creating geometric LiDAR projections for {len(frame_list)} frames...")

        for frame_id in tqdm(frame_list):
            try:
                # Create geometric projection
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
    parser.add_argument("--input_dir", default="output_clft_v2", help="Input directory")
    parser.add_argument("--output_dir", default="output_clft_v2/lidar_geometric", help="Output directory")
    parser.add_argument("--frames_file", default="output_clft_v2/splits/test.txt", help="Frames to process")
    parser.add_argument("--verify", action="store_true", help="Verify geometric content")

    args = parser.parse_args()

    # Load frame list
    frame_ids = []
    with open(args.frames_file) as f:
        for line in f:
            if line.strip():
                frame_id = line.strip().split('/')[-1].replace('frame_', '').replace('.png', '')
                frame_ids.append(frame_id)

    print(f"📊 Processing {len(frame_ids)} frames")

    # Create projector
    projector = GeometricLiDARProjector(args.input_dir, args.output_dir)

    if args.verify:
        # Verify on first frame
        if frame_ids:
            projector.verify_geometric_projection(frame_ids[0])
    else:
        # Process all frames
        projector.process_frames(frame_ids)

if __name__ == "__main__":
    main()