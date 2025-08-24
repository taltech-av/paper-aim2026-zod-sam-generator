#!/usr/bin/env python3
"""
SAM Example: Use Segment Anything Model to convert ZOD 2D/3D annotations 
to precise semantic segmentation masks
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List
import urllib.request
import shutil
import sys

# ZOD imports
from zod import ZodFrames
from zod import ObjectAnnotation
from zod.constants import Camera, Anonymization, AnnotationProject

# SAM imports
try:
    from segment_anything import SamPredictor, sam_model_registry
    import torch
    SAM_AVAILABLE = True
    print("SAM available")
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not available. Install with: pip install segment-anything")

# URL for the SAM checkpoint
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

def _download_file(url: str, dest: str) -> bool:
    """Download url to dest (streaming). Returns True on success."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM checkpoint from {url} to {dest_path} ...")
    try:
        with urllib.request.urlopen(url) as r:
            total = r.getheader('Content-Length')
            if total is None:
                # Unknown size
                with open(dest_path, 'wb') as f:
                    shutil.copyfileobj(r, f)
            else:
                total = int(total.strip())
                downloaded = 0
                chunk_size = 1024 * 1024
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = r.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded * 100 / total
                        sys.stdout.write(f"\r  {pct:.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)")
                        sys.stdout.flush()
                print()
        print("Download complete")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

class SAMZODProcessor:
    """Process ZOD dataset using SAM for semantic segmentation"""
    
    def __init__(self, dataset_root, sam_checkpoint="sam_vit_h_4b8939.pth"):
        self.dataset_root = dataset_root
        self.sam_checkpoint = sam_checkpoint
        
        # Class mapping for semantic segmentation
        self.class_mapping = {
            'Vehicle': 1,
            'Pedestrian': 2,
            'Cyclist': 3,
            'TrafficSign': 4,
            'TrafficLight': 4,
            'Animal': 5,
            'VulnerableVehicle': 3,
            'Truck': 1,
            'Bus': 1,
            'Motorcycle': 1,
            'Bicycle': 2,
        }
        
        self.background_class = 0
        self.max_classes = max(self.class_mapping.values()) + 1
        
        # Color mapping for visualization
        self.class_colors = {
            0: (0, 0, 0),       # Background - Black
            1: (100, 0, 0),     # Vehicles
            2: (0, 100, 0),     # Pedestrians
            3: (0, 0, 100),     # Traffic Signs
            4: (0, 100, 100),   # Cyclists
            5: (100, 0, 100),   # Animal
        }
        
        # Initialize ZOD dataset
        self.zod_frames = None
        self.sam_predictor = None
        self._initialize_zod()
        self._initialize_sam()
    
    def _initialize_zod(self):
        """Initialize ZOD dataset"""
        try:
            self.zod_frames = ZodFrames(dataset_root=self.dataset_root, version="mini")
            print(f"ZOD dataset loaded: {len(self.zod_frames)} frames")
        except Exception as e:
            print(f"Failed to load ZOD dataset: {e}")
    
    def _initialize_sam(self):
        """Initialize SAM model"""
        if not SAM_AVAILABLE:
            return
        # Ensure checkpoint exists, download if missing
        if not Path(self.sam_checkpoint).exists():
            print(f"SAM checkpoint not found: {self.sam_checkpoint}")
            if not _download_file(SAM_DOWNLOAD_URL, self.sam_checkpoint):
                print("Failed to obtain SAM checkpoint — SAM will be disabled.")
                return
        
        try:
            # Determine model type from checkpoint name
            if "vit_h" in self.sam_checkpoint:
                model_type = "vit_h"
            elif "vit_l" in self.sam_checkpoint:
                model_type = "vit_l"
            elif "vit_b" in self.sam_checkpoint:
                model_type = "vit_b"
            else:
                model_type = "vit_h"  # default
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            
            print(f"SAM model loaded: {model_type} on {device}")
            
        except Exception as e:
            print(f"SAM loading failed: {e}")
    def get_2d_boxes_from_annotations(self, annotations: List[ObjectAnnotation], calibration, image_shape):
        """Extract 2D bounding boxes from ZOD annotations (3D -> 2D projection if needed)"""
        boxes_2d = []
        
        print(f"  Processing {len(annotations)} annotations...")
        
        for i, annotation in enumerate(annotations):
            category = annotation.name
            class_id = self.class_mapping.get(category, 0)
            print(f"    Annotation {i+1}: category='{category}', class_id={class_id}")
            
            if class_id == 0:  # Skip unknown classes
                print(f"      Skipping unknown class: {category}")
                continue
            
            # Try to get 2D box directly
            if hasattr(annotation, 'box2d') and annotation.box2d is not None:
                try:
                    # Direct 2D box - use corners attribute
                    box2d = annotation.box2d
                    corners = box2d.corners
                    # corners[0] is top-left, corners[2] is bottom-right
                    x1, y1 = int(corners[0][0]), int(corners[0][1])
                    x2, y2 = int(corners[2][0]), int(corners[2][1])
                    print(f"      Found 2D box: ({x1},{y1}) to ({x2},{y2})")
                    
                except Exception as e:
                    print(f"      Error processing 2D box: {e}")
                    continue
                
            elif hasattr(annotation, 'box3d') and annotation.box3d is not None:
                # Project 3D box to 2D
                print(f"      Attempting 3D to 2D projection...")
                try:
                    box3d = annotation.box3d
                    # Get 3D box corners
                    corners_3d = box3d.corners
                    
                    # Project to camera coordinates
                    if hasattr(calibration, 'camera_front'):
                        camera_matrix = calibration.camera_front.camera_matrix
                        dist_coeffs = calibration.camera_front.distortion_coefficients
                        
                        # Project 3D points to 2D
                        corners_2d, _ = cv2.projectPoints(
                            corners_3d.reshape(-1, 1, 3), 
                            np.zeros(3), np.zeros(3), 
                            camera_matrix, dist_coeffs
                        )
                        corners_2d = corners_2d.reshape(-1, 2)
                        
                        # Get bounding rectangle
                        x1, y1 = np.min(corners_2d, axis=0).astype(int)
                        x2, y2 = np.max(corners_2d, axis=0).astype(int)
                        print(f"      Projected 3D->2D: ({x1},{y1}) to ({x2},{y2})")
                        
                    else:
                        print(f"      No camera calibration available")
                        continue  # Skip if no calibration
                        
                except Exception as e:
                    print(f"      Failed to project 3D box: {e}")
                    continue
            else:
                print(f"      No 2D or 3D box found")
                continue  # Skip if no box info
            
            # Ensure box is within image bounds
            height, width = image_shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            boxes_2d.append({
                'bbox': [x1, y1, x2, y2],
                'category': category,
                'class_id': class_id
            })
        
        return boxes_2d
    
    def create_sam_mask(self, image, boxes_2d):
        """Create semantic segmentation mask using SAM"""
        if not self.sam_predictor:
            return None, {'success': 0, 'failed': len(boxes_2d)}
        
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Set image for SAM
        self.sam_predictor.set_image(image)
        
        stats = {'success': 0, 'failed': 0}
        
        for box_info in boxes_2d:
            try:
                bbox = box_info['bbox']
                class_id = box_info['class_id']
                
                # Create input box for SAM
                input_box = np.array(bbox)
                
                # Generate mask
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Use the generated mask
                if len(masks) > 0:
                    sam_mask = masks[0].astype(np.uint8)
                    mask[sam_mask == 1] = class_id
                    stats['success'] += 1
                else:
                    # Fallback to simple box fill
                    x1, y1, x2, y2 = bbox
                    mask[y1:y2, x1:x2] = class_id
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"SAM processing failed for box: {e}")
                # Fallback to simple box fill
                x1, y1, x2, y2 = box_info['bbox']
                mask[y1:y2, x1:x2] = box_info['class_id']
                stats['failed'] += 1
        
        return mask, stats
    
    def process_frame(self, frame_id, output_dir):
        """Process a single frame with SAM"""
        try:
            # Get frame
            zod_frame = self.zod_frames[frame_id]
            
            # Get image
            image_pil = zod_frame.get_image(Anonymization.DNAT)
            image_np = np.array(image_pil)
            
            # Get object annotations
            object_annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            
            # Convert annotations to 2D boxes
            boxes_2d = self.get_2d_boxes_from_annotations(
                object_annotations, 
                zod_frame.calibration, 
                image_np.shape
            )
            
            if not boxes_2d:
                print(f"  No valid boxes found for frame {frame_id}")
                return False
            
            print(f"  Found {len(boxes_2d)} objects")
            
            # Create SAM segmentation mask
            seg_mask, stats = self.create_sam_mask(image_np, boxes_2d)
            
            if seg_mask is None:
                print(f"  SAM processing failed for frame {frame_id}")
                return False
            
            print(f"  SAM - Success: {stats['success']}, Failed: {stats['failed']}")
            
            # Save results
            self._save_results(image_np, seg_mask, boxes_2d, frame_id, output_dir, stats)
            
            return True
            
        except Exception as e:
            print(f"  Error processing frame {frame_id}: {e}")
            return False
    
    def _save_results(self, image, mask, boxes_2d, frame_id, output_dir, stats):
        """Save processing results"""
        
        # Create output directories
        for subdir in ['camera', 'annotation', 'annotation_rgb', 'visualizations']:
            Path(output_dir, subdir).mkdir(parents=True, exist_ok=True)
        
        frame_name = f"frame_{frame_id}"
        
        # Save original image
        cv2.imwrite(str(Path(output_dir, 'camera', f'{frame_name}.png')), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save grayscale mask
        cv2.imwrite(str(Path(output_dir, 'annotation', f'{frame_name}.png')), mask)
        
        # Save RGB mask (CLFT format)
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(self.max_classes):
            class_pixels = mask == class_id
            rgb_mask[class_pixels] = [class_id, 0, 0]
        
        cv2.imwrite(str(Path(output_dir, 'annotation_rgb', f'{frame_name}.png')), 
                   cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
        
        # Create visualization
        self._create_visualization(image, mask, boxes_2d, frame_name, output_dir, stats)
    def _create_visualization(self, image, mask, boxes_2d, frame_name, output_dir, stats):
        """Create SAM overlay visualization only"""
        
        # Create SAM overlay
        overlay = image.copy().astype(float)
        colored_mask = plt.cm.tab10(mask / (self.max_classes-1))[:,:,:3]
        mask_alpha = (mask > 0).astype(float) * 0.6
        
        for c in range(3):
            overlay[:,:,c] = (overlay[:,:,c] * (1 - mask_alpha) + 
                            colored_mask[:,:,c] * 255 * mask_alpha)
          # Create single plot with just the overlay
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(overlay.astype(np.uint8))
        ax.axis('off')
        
        plt.tight_layout(pad=0.5)
        
        # Save visualization
        viz_path = Path(output_dir, 'visualizations', f'{frame_name}_sam_overlay.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def process_dataset(self, num_frames=5, output_dir="output_clft"):
        """Process dataset with SAM"""
        
        print(f"\n=== SAM ZOD PROCESSING ===")
        
        if not self.zod_frames:
            print("ZOD dataset not available!")
            return
        
        if not self.sam_predictor:
            print("SAM model not available!")
            return
          # Get frame IDs
        train_ids = list(self.zod_frames.get_split("train"))
        val_ids = list(self.zod_frames.get_split("val"))
        all_ids = train_ids + val_ids
        print(f"Available frames: train={len(train_ids)}, val={len(val_ids)}, total={len(all_ids)}")
        print(f"All frame IDs: {sorted(all_ids)}")
        
        # Check for specific frame
        if "007674" in all_ids:
            print(f"Frame 007674 is available for processing")
        else:
            print(f"Frame 007674 not found in train/val splits")
            print(f"  This frame may be in a different split or have no annotations")
        
        if num_frames:
            all_ids = all_ids[:num_frames]
            print(f"Limited to first {num_frames} frames: {all_ids}")
        
        print(f"Processing {len(all_ids)} frames...")
        
        success_count = 0
        for i, frame_id in enumerate(all_ids):
            print(f"\nProcessing {i+1}/{len(all_ids)}: Frame {frame_id}")
            
            if self.process_frame(frame_id, output_dir):
                success_count += 1
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Successfully processed: {success_count}/{len(all_ids)} frames")
        print(f"Output directory: {output_dir}")
        print(f"\nGenerated files:")
        print(f"  📁 camera/ - Original images")
        print(f"  📁 annotation/ - SAM grayscale masks")
        print(f"  📁 annotation_rgb/ - RGB masks (CLFT format)")
        print(f"  📁 visualizations/ - Quality comparisons")

def main():
    if not SAM_AVAILABLE:
        print("Please install SAM: pip install segment-anything")
        return

    dataset_root = "./data"  # Adjust path as needed
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # Make sure this file exists
    output_dir = "output_clft"
    num_frames = None  # Process all frames (set to number to limit, e.g., 5 for testing)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_root}")
    print(f"  SAM model: {sam_checkpoint}")
    print(f"  Output: {output_dir}")
    print(f"  Frames to process: {'All available' if num_frames is None else num_frames}")
    
    # Check if SAM checkpoint exists and attempt download
    if not Path(sam_checkpoint).exists():
        print(f"\nSAM checkpoint not found: {sam_checkpoint}")
        if not _download_file(SAM_DOWNLOAD_URL, sam_checkpoint):
            print("Could not download SAM checkpoint. Aborting.")
            return
    
    # Initialize processor
    processor = SAMZODProcessor(dataset_root, sam_checkpoint)
    
    # Process dataset
    processor.process_dataset(num_frames=num_frames, output_dir=output_dir)
    
    print(f"\nSAM processing complete!")
    print("Check the visualizations to see high-quality segmentation results.")

if __name__ == "__main__":
    main()
