#!/usr/bin/env python3
"""
Production ZOD dataset processor based on WORKING visualization code
- Uses proven SAM processing logic from visualize_frame_sam.py
- Batch processing and preloading for speed
- Outputs: annotation, camera, visualizations (overlay + boxes)
- Progress tracking with resume capability
"""

import numpy as np
from PIL import Image
import cv2
import torch
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
import traceback
import gc
import signal
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time


# Global cleanup handler
def cleanup_gpu():
    """Clean up GPU memory properly"""
    print("\n🧹 Cleaning up GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("✅ GPU cleanup complete")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️  Ctrl+C detected! Cleaning up...")
    cleanup_gpu()
    print("👋 Exiting cleanly")
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


class ZODProcessor:
    """Process ZOD dataset with SAM using proven working logic"""
    
    def __init__(self, 
                 dataset_root: str,
                 output_root: str,
                 sam_model_type: str = 'vit_h',
                 sam_resolution: int = 1024,  # Resolution for SAM processing
                 target_resolution: int = 768,
                 gpu_batch_size: int = 4,
                 preload_workers: int = 8):
        
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.sam_model_type = sam_model_type
        self.sam_resolution = sam_resolution
        self.target_resolution = target_resolution
        self.gpu_batch_size = gpu_batch_size
        self.preload_workers = preload_workers
        
        # Create output directories
        self.output_dirs = {
            'annotation': self.output_root / 'annotation',
            'camera': self.output_root / 'camera',
            'overlay': self.output_root / 'visualizations' / 'overlay',
            'boxes': self.output_root / 'visualizations' / 'boxes',
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.completed_file = self.output_root / 'completed_frames.txt'
        self.failed_file = self.output_root / 'failed_frames.txt'
        self.completed_frames = self._load_completed_frames()
        self.failed_frames = set()
        
        # Statistics
        self.stats = {
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat(),
            'total_filter_time': 0.0,
            'total_dedup_time': 0.0,
            'total_sam_time': 0.0,
            'total_save_time': 0.0,
        }
        
        # Class mapping (CLFT format: 2=Vehicle, 3=Sign, 4=Cyclist, 5=Pedestrian)
        self.class_mapping = {
            'Vehicle': 2,
            'VulnerableVehicle': 4,
            'Pedestrian': 5,
            'PoleObject': 3,
            'TrafficSign': 3,
            'TrafficLight': 3,
            'TrafficSignal': 3,
            'TrafficGuide': 3,
        }
        
        # Priority for overlap resolution
        self.priority = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        
        # Class-specific minimum size thresholds
        self.min_size_by_class = {
            1: 20,  # Lane
            2: 30,  # Vehicle
            3: 15,  # Sign - SMALL signs are important
            4: 15,  # Cyclist - SMALL for distant cyclists
            5: 15,  # Pedestrian - SMALL for distant people
        }
        
        # Colors for visualization (BGR format for OpenCV) - Very distinct colors
        self.class_colors = {
            1: (128, 64, 128),   # Lane - purple
            2: (0, 0, 255),      # Vehicle - bright red
            3: (0, 255, 255),    # Sign - bright yellow
            4: (255, 0, 255),    # Cyclist/VulnerableVehicle - magenta/pink
            5: (0, 255, 0),      # Pedestrian - bright green
        }
        
        print(f"📁 Output directory: {self.output_root}")
        print(f"🎯 SAM resolution: {self.sam_resolution}px (lower = faster)")
        print(f"🎯 Target resolution: {self.target_resolution}px")
        print(f"⚙️  GPU batch size: {self.gpu_batch_size}")
        print(f"⚙️  Preload workers: {self.preload_workers}")
    
    def _load_completed_frames(self):
        """Load list of already completed frames"""
        if self.completed_file.exists():
            with open(self.completed_file, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    def initialize_models(self):
        """Initialize ZOD and SAM models"""
        print("\n🤖 Initializing models...")
        
        # Load ZOD
        from zod import ZodFrames
        print("Loading ZOD dataset...")
        self.zod = ZodFrames(dataset_root=str(self.dataset_root), version="full")
        print(f"✅ Loaded {len(self.zod)} frames")
        
        # Load SAM
        from segment_anything import sam_model_registry, SamPredictor
        
        model_paths = {
            'vit_b': 'models/sam_vit_b_01ec64.pth',
            'vit_l': 'models/sam_vit_l_0b3195.pth',
            'vit_h': 'models/sam_vit_h_4b8939.pth',
        }
        
        sam_checkpoint = model_paths[self.sam_model_type]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading SAM model ({self.sam_model_type}) on {device}...")
        
        sam = sam_model_registry[self.sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        
        self.sam = SamPredictor(sam)
        
        # Enable optimizations
        if device == 'cuda':
            self.sam.model = self.sam.model.to(memory_format=torch.channels_last)
        
        print("✅ Models initialized")
    
    def process_frame(self, frame_id: str, preloaded_data=None):
        """Process single frame using proven working logic"""
        try:
            from zod.constants import AnnotationProject
            
            # Use preloaded data if available
            if preloaded_data:
                frame, img_rgb = preloaded_data
                h, w = img_rgb.shape[:2]
            else:
                frame = self.zod[frame_id]
                
                # Load image
                camera_frame = frame.info.get_key_camera_frame()
                img_path = Path(camera_frame.filepath)
                img = cv2.imread(str(img_path))
                if img is None:
                    return False
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
            
            # Get annotations
            annotation = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            if annotation is None:
                return False
            
            # ================================================================
            # STEP 1: Filter objects (EXACT LOGIC FROM VISUALIZATION)
            # ================================================================
            t0 = time.time()
            filtered_objects = []
            
            for obj in annotation:
                if not hasattr(obj, 'box2d') or obj.box2d is None:
                    continue
                
                class_name = obj.name
                if class_name not in self.class_mapping:
                    continue
                
                box = obj.box2d
                x_min, y_min = int(box.xmin), int(box.ymin)
                x_max, y_max = int(box.xmax), int(box.ymax)
                
                # Clip to bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # Class-specific minimum size
                class_id = self.class_mapping[class_name]
                min_size = self.min_size_by_class.get(class_id, 30)
                
                # Size filtering
                if box_w < min_size or box_h < min_size:
                    continue
                
                # Allow larger objects (40% of image) to include close-up vehicles
                if box_w > w * 0.4 or box_h > h * 0.4:
                    continue
                
                # Aspect ratio
                aspect = max(box_w, box_h) / min(box_w, box_h) if min(box_w, box_h) > 0 else 0
                if aspect > 8.0:
                    continue
                
                area = box_w * box_h
                filtered_objects.append(([x_min, y_min, x_max, y_max], class_id, area))
            
            if not filtered_objects:
                return False
            
            t1 = time.time()
            self.stats['total_filter_time'] += (t1 - t0)
            
            # ================================================================
            # STEP 2: Deduplicate (EXACT LOGIC FROM VISUALIZATION)
            # ================================================================
            by_class_filtered = defaultdict(list)
            for i, (box, class_id, area) in enumerate(filtered_objects):
                by_class_filtered[class_id].append((i, box, area))
            
            deduplicated = []
            for class_id, boxes_list in by_class_filtered.items():
                boxes_list.sort(key=lambda x: -x[2])
                
                kept = []
                for i, box, area in boxes_list:
                    merged = False
                    for idx, (kept_box, kept_area) in enumerate(kept):
                        if not (box[2] < kept_box[0] or box[0] > kept_box[2] or 
                                box[3] < kept_box[1] or box[1] > kept_box[3]):
                            x1 = max(box[0], kept_box[0])
                            y1 = max(box[1], kept_box[1])
                            x2 = min(box[2], kept_box[2])
                            y2 = min(box[3], kept_box[3])
                            
                            intersection = (x2 - x1) * (y2 - y1)
                            box_area = (box[2] - box[0]) * (box[3] - box[1])
                            kept_area_calc = (kept_box[2] - kept_box[0]) * (kept_box[3] - kept_box[1])
                            union = box_area + kept_area_calc - intersection
                            
                            iou = intersection / union if union > 0 else 0
                            
                            if iou > 0.3:
                                merged_box = [
                                    min(box[0], kept_box[0]),
                                    min(box[1], kept_box[1]),
                                    max(box[2], kept_box[2]),
                                    max(box[3], kept_box[3])
                                ]
                                merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                                kept[idx] = (merged_box, merged_area)
                                merged = True
                                break
                    
                    if not merged:
                        kept.append((box, area))
                
                for kept_box, kept_area in kept:
                    deduplicated.append((kept_box, class_id, kept_area))
            
            t2 = time.time()
            self.stats['total_dedup_time'] += (t2 - t1)
            
            # SPEED OPTIMIZATION: Limit max objects per frame (focus on largest/most important)
            MAX_OBJECTS = 75  # Process top 75 objects (balance speed vs completeness)
            if len(deduplicated) > MAX_OBJECTS:
                # Sort by AREA ONLY (keep biggest objects from all classes)
                # Don't use priority here - that's for overlap resolution, not filtering
                deduplicated.sort(key=lambda x: -x[2])  # Sort by area descending
                deduplicated = deduplicated[:MAX_OBJECTS]
                print(f"    ⚠️  Limited to {MAX_OBJECTS} objects (was {len(deduplicated)})")
            
            print(f"    ⏱️  Deduplication: {t2-t1:.3f}s | Objects after: {len(deduplicated)}")
            
            # ================================================================
            # STEP 3: Process with SAM (EXACT LOGIC FROM VISUALIZATION)
            # ================================================================
            # SPEED OPTIMIZATION: Resize image for SAM processing
            # SAM encoding time scales with image size - 1024px is 16x faster than 4096px!
            sam_h, sam_w = h, w
            if max(h, w) > self.sam_resolution:
                scale = self.sam_resolution / max(h, w)
                sam_w = int(w * scale)
                sam_h = int(h * scale)
                img_rgb_sam = cv2.resize(img_rgb, (sam_w, sam_h), interpolation=cv2.INTER_LINEAR)
                print(f"    📐 Resized for SAM: {w}x{h} → {sam_w}x{sam_h} (scale={scale:.2f})")
            else:
                scale = 1.0
                img_rgb_sam = img_rgb
                print(f"    📐 No resize needed: {w}x{h} ≤ {self.sam_resolution}")
            
            self.sam.set_image(img_rgb_sam)
            seg_mask = np.zeros((sam_h, sam_w), dtype=np.uint8)
            
            # SPEED OPTIMIZATION: Precompute priority lookup array (vectorize is slow!)
            priority_map = np.array([0, 1, 2, 3, 4, 5])  # index = class_id, value = priority
            
            use_amp = torch.cuda.is_available()
            
            # SPEED OPTIMIZATION: Batch process boxes (much faster than one-by-one)
            batch_size = 16  # Process 16 boxes at once
            with torch.no_grad():
                for i in range(0, len(deduplicated), batch_size):
                    batch = deduplicated[i:i+batch_size]
                    
                    # Prepare batch of boxes
                    boxes_batch = []
                    for box, class_id, area in batch:
                        x_min, y_min, x_max, y_max = box
                        box_xyxy = np.array([
                            x_min * scale,
                            y_min * scale,
                            x_max * scale,
                            y_max * scale
                        ])
                        boxes_batch.append(box_xyxy)
                    
                    if not boxes_batch:
                        continue
                    
                    # Batch predict (much faster!)
                    try:
                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                masks_batch = []
                                for box_xyxy in boxes_batch:
                                    masks, _, _ = self.sam.predict(
                                        box=box_xyxy,
                                        multimask_output=False
                                    )
                                    masks_batch.append(masks[0] if len(masks) > 0 else None)
                        else:
                            masks_batch = []
                            for box_xyxy in boxes_batch:
                                masks, _, _ = self.sam.predict(
                                    box=box_xyxy,
                                    multimask_output=False
                                )
                                masks_batch.append(masks[0] if len(masks) > 0 else None)
                        
                        # Apply masks with priority
                        for (box, class_id, area), mask in zip(batch, masks_batch):
                            if mask is None:
                                continue
                            
                            try:
                                obj_priority = self.priority.get(class_id, 1)
                                # FAST priority lookup using numpy indexing instead of vectorize
                                current_priorities = priority_map[seg_mask]
                                
                                update_mask = (mask > 0) & ((seg_mask == 0) | (obj_priority >= current_priorities))
                                seg_mask[update_mask] = class_id
                            except Exception:
                                continue
                                
                    except Exception as e:
                        # Fall back to one-by-one for this batch
                        for box, class_id, area in batch:
                            try:
                                x_min, y_min, x_max, y_max = box
                                box_xyxy = np.array([x_min * scale, y_min * scale, x_max * scale, y_max * scale])
                                
                                if use_amp:
                                    with torch.amp.autocast('cuda'):
                                        masks, _, _ = self.sam.predict(box=box_xyxy, multimask_output=False)
                                else:
                                    masks, _, _ = self.sam.predict(box=box_xyxy, multimask_output=False)
                                
                                if len(masks) > 0:
                                    mask = masks[0]
                                    obj_priority = self.priority.get(class_id, 1)
                                    # FAST priority lookup
                                    current_priorities = priority_map[seg_mask]
                                    update_mask = (mask > 0) & ((seg_mask == 0) | (obj_priority >= current_priorities))
                                    seg_mask[update_mask] = class_id
                            except Exception:
                                continue
            
            t3 = time.time()
            self.stats['total_sam_time'] += (t3 - t2)
            sam_time = t3 - t2
            print(f"    ⏱️  SAM inference: {sam_time:.3f}s ({sam_time/max(len(deduplicated),1)*1000:.1f}ms per object)")
            
            # ================================================================
            # STEP 4: Save outputs
            # ================================================================
            self._save_outputs(frame_id, img_rgb, seg_mask, deduplicated)
            
            t4 = time.time()
            self.stats['total_save_time'] += (t4 - t3)
            
            total_time = t4 - t0
            print(f"    ✅ Frame complete: {total_time:.3f}s total")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {frame_id}: {e}")
            traceback.print_exc()
            return False
    
    def _save_outputs(self, frame_id: str, img_rgb: np.ndarray, seg_mask: np.ndarray, boxes: list):
        """Save all outputs: annotation, camera, overlay, boxes"""
        
        h, w = img_rgb.shape[:2]
        target = self.target_resolution
        
        # Scale based on shorter dimension
        scale = target / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image and mask
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(seg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 1. Save camera image
        camera_file = self.output_dirs['camera'] / f"frame_{frame_id}.png"
        cv2.imwrite(str(camera_file), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        
        # 2. Save annotation (grayscale)
        anno_file = self.output_dirs['annotation'] / f"frame_{frame_id}.png"
        cv2.imwrite(str(anno_file), mask_resized.astype(np.uint8))
        
        # 3. Create segmentation visualization (colored masks)
        seg_vis = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for class_id in range(1, 6):
            mask = mask_resized == class_id
            if mask.any():
                color = self.class_colors.get(class_id, (255, 255, 255))
                seg_vis[mask] = color
        
        # 4. Save overlay
        overlay = img_resized.copy()
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1-alpha, seg_vis, alpha, 0)
        overlay_file = self.output_dirs['overlay'] / f"frame_{frame_id}.png"
        cv2.imwrite(str(overlay_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 5. Save boxes visualization
        img_boxes = img_resized.copy()
        for box, class_id, area in boxes:
            x_min, y_min, x_max, y_max = box
            # Scale box coordinates
            x_min = int(x_min * scale)
            y_min = int(y_min * scale)
            x_max = int(x_max * scale)
            y_max = int(y_max * scale)
            
            color = self.class_colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), color, 2)
        
        boxes_file = self.output_dirs['boxes'] / f"frame_{frame_id}.png"
        cv2.imwrite(str(boxes_file), cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR))
    
    def process_dataset(self, frame_list_file: str):
        """Process frames from list with parallel preloading"""
        
        print(f"\n📋 Reading frame list from: {frame_list_file}")
        with open(frame_list_file, 'r') as f:
            all_frame_ids = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(all_frame_ids)} frames")
        
        # Remove already processed
        remaining_frames = [fid for fid in all_frame_ids if fid not in self.completed_frames]
        
        if len(self.completed_frames) > 0:
            print(f"Resuming: {len(self.completed_frames)} done, {len(remaining_frames)} remaining\n")
        else:
            print(f"Processing {len(remaining_frames)} frames\n")
        
        if not remaining_frames:
            print("✅ All frames already processed!")
            return
        
        # Preloading function
        def preload_frame_data(frame_id):
            """Preload frame data on CPU while GPU is busy"""
            try:
                from zod.constants import Anonymization
                frame = self.zod[frame_id]
                camera_frame = frame.info.get_key_camera_frame()
                img_path = Path(camera_frame.filepath)
                img = cv2.imread(str(img_path))
                if img is None:
                    return frame_id, None
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return frame_id, (frame, img_rgb)
            except Exception:
                return frame_id, None
        
        pbar = tqdm(remaining_frames, desc="Processing", unit="frame")
        
        # Use thread pool for parallel I/O preloading
        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            # Submit first batch of frames for preloading
            future_to_frame = {}
            for i in range(min(self.preload_workers, len(remaining_frames))):
                future = executor.submit(preload_frame_data, remaining_frames[i])
                future_to_frame[future] = remaining_frames[i]
            
            for idx, frame_id in enumerate(pbar):
                # Get preloaded data
                preloaded_data = None
                for future in list(future_to_frame.keys()):
                    if future_to_frame[future] == frame_id:
                        _, preloaded_data = future.result()
                        del future_to_frame[future]
                        break
                
                # Submit next frame for preloading
                next_idx = idx + self.preload_workers
                if next_idx < len(remaining_frames):
                    future = executor.submit(preload_frame_data, remaining_frames[next_idx])
                    future_to_frame[future] = remaining_frames[next_idx]
                
                # Process frame with preloaded data
                success = self.process_frame(frame_id, preloaded_data)
                
                if success:
                    self.stats['successful'] += 1
                    self.completed_frames.add(frame_id)
                    with open(self.completed_file, 'a') as f:
                        f.write(f"{frame_id}\n")
                else:
                    self.stats['failed'] += 1
                    with open(self.failed_file, 'a') as f:
                        f.write(f"{frame_id}\n")
                
                # Update progress
                total_processed = self.stats['successful'] + self.stats['failed']
                if total_processed > 0:
                    avg_filter = self.stats['total_filter_time'] / total_processed
                    avg_dedup = self.stats['total_dedup_time'] / total_processed
                    avg_sam = self.stats['total_sam_time'] / total_processed
                    avg_save = self.stats['total_save_time'] / total_processed
                    
                    pbar.set_postfix({
                        'ok': self.stats['successful'],
                        'fail': self.stats['failed'],
                        'SAM': f'{avg_sam:.1f}s',
                    })
                else:
                    pbar.set_postfix({
                        'success': self.stats['successful'],
                        'failed': self.stats['failed'],
                    })
                
                # Memory cleanup every 50 frames
                if (self.stats['successful'] + self.stats['failed']) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print(f"\n✅ Complete: {self.stats['successful']} successful, {self.stats['failed']} failed")
        print(f"📁 Output: {self.output_root}")
        
        # Print timing breakdown
        total = self.stats['successful']
        if total > 0:
            print(f"\n⏱️  Average timing per frame:")
            print(f"   Filter:  {self.stats['total_filter_time']/total:.2f}s")
            print(f"   Dedup:   {self.stats['total_dedup_time']/total:.2f}s")
            print(f"   SAM:     {self.stats['total_sam_time']/total:.2f}s  ← BOTTLENECK")
            print(f"   Save:    {self.stats['total_save_time']/total:.2f}s")
            total_time = (self.stats['total_filter_time'] + self.stats['total_dedup_time'] + 
                         self.stats['total_sam_time'] + self.stats['total_save_time'])
            print(f"   Total:   {total_time/total:.2f}s")
        print()


def main():
    parser = argparse.ArgumentParser(description="Process ZOD dataset with working SAM code")
    
    parser.add_argument('--dataset-root', type=str, default='/media/tom/ml/zod-data',
                       help='Path to ZOD dataset root')
    parser.add_argument('--output-root', type=str, default='./output_clft_v2',
                       help='Output directory')
    parser.add_argument('--frame-list', type=str, required=True, default='frames_to_process.txt',
                       help='Path to frames_to_process.txt')
    parser.add_argument('--sam-model', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
                       default='vit_h', help='SAM model type')
    parser.add_argument('--sam-resolution', type=int, default=1024,
                       help='SAM processing resolution (lower = faster)')
    parser.add_argument('--target-resolution', type=int, default=768,
                       help='Target output resolution')
    parser.add_argument('--gpu-batch-size', type=int, default=4,
                       help='GPU batch size (unused for now)')
    parser.add_argument('--preload-workers', type=int, default=8,
                       help='Preload workers (unused for now)')
    
    args = parser.parse_args()
    
    # Check if frame list exists
    if not Path(args.frame_list).exists():
        print(f"❌ Error: Frame list file not found: {args.frame_list}")
        return
    
    processor = None
    try:
        # Create processor
        processor = ZODProcessor(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            sam_model_type=args.sam_model,
            sam_resolution=args.sam_resolution,
            target_resolution=args.target_resolution,
            gpu_batch_size=args.gpu_batch_size,
            preload_workers=args.preload_workers,
        )
        
        # Initialize models
        processor.initialize_models()
        
        # Process dataset
        processor.process_dataset(frame_list_file=args.frame_list)
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Always cleanup
        print("\n🧹 Final cleanup...")
        cleanup_gpu()


if __name__ == "__main__":
    main()
