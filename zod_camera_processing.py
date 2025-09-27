import os
from typing import List
import numpy as np
from PIL import Image
from zod import ZodFrames
from zod import ObjectAnnotation
from zod import LaneAnnotation
from zod import EgoRoadAnnotation
from zod.constants import Camera, Anonymization, AnnotationProject
from zod.visualization.object_visualization import overlay_object_2d_box_on_image
from zod.visualization.object_visualization import overlay_object_3d_box_on_image
from zod.utils.polygon_transformations import polygons_to_binary_mask
from zod.visualization.polygon_utils import overlay_mask_on_image
from zod.visualization.oxts_on_image import visualize_oxts_on_image

def initialize_zod_frames(dataset_root: str = "./data", version: str = "mini") -> ZodFrames:
    """
    Initialize ZodFrames object with specified dataset root and version.
    
    Args:
        dataset_root: Path to the ZOD dataset root directory
        version: Dataset version to use ("mini" or "full")
        
    Returns:
        Initialized ZodFrames object
    """
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
    print(f"Initialized ZOD dataset from: {dataset_root}")
    print(f"Version: {version}")
    print(f"Total frames available: {len(zod_frames)}")
    return zod_frames

def get_all_frame_ids(zod_frames: ZodFrames) -> List[str]:
    """
    Get all available frame IDs from training and validation splits.
    
    Args:
        zod_frames: ZodFrames object
        
    Returns:
        List of all frame IDs (train + validation)
    """
    train_frame_ids = list(zod_frames.get_split("train"))
    val_frame_ids = list(zod_frames.get_split("val"))
    all_frame_ids = train_frame_ids + val_frame_ids
    
    print(f"Found {len(all_frame_ids)} total frames ({len(train_frame_ids)} train + {len(val_frame_ids)} val)")
    return all_frame_ids

def process_frame_with_annotations(zod_frames: ZodFrames, frame_id: str) -> Image.Image:
    """
    Process a single frame and return an annotated image.
    
    Args:
        zod_frames: ZodFrames object for accessing frame data
        frame_id: The frame ID to process
        
    Returns:
        PIL Image with annotations applied
    """
    # Get the frame object
    zod_frame = zod_frames[frame_id]
    
    # Get camera image with DNAT anonymization
    image = zod_frame.get_image(Anonymization.DNAT)
    
    # Apply object annotations
    object_annotations: List[ObjectAnnotation] = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
    
    for annotation in object_annotations:
        if annotation.box2d:
            if annotation.name == 'Vehicle':
                image = overlay_object_2d_box_on_image(image, annotation.box2d, color=(100, 0, 0), line_thickness=5)
            elif annotation.name == 'Pedestrian':
                image = overlay_object_2d_box_on_image(image, annotation.box2d, color=(0, 100, 0), line_thickness=5)
            elif annotation.name == 'TrafficSign':
                image = overlay_object_2d_box_on_image(image, annotation.box2d, color=(0, 0, 100), line_thickness=5)
            elif annotation.name == 'VulnerableVehicle':
                image = overlay_object_2d_box_on_image(image, annotation.box2d, color=(0, 100, 100), line_thickness=5)
            else:
                image = overlay_object_2d_box_on_image(image, annotation.box2d, color=(0, 0, 0), line_thickness=5)

    # Try to get lane annotations (skip if not available)
    try:
        lane_annotations: List[LaneAnnotation] = zod_frame.get_annotation(AnnotationProject.LANE_MARKINGS)
        lane_mask = polygons_to_binary_mask([anno.geometry for anno in lane_annotations])
        image = overlay_mask_on_image(lane_mask, image, fill_color=(100, 100, 0), alpha=0.5)
        print(f"  Applied lane annotations")
    except Exception as lane_error:
        print(f"  Lane annotations not available: {lane_error}")

    # Try to get ego road annotations (skip if not available)
    try:
        ego_road_annotations: List[EgoRoadAnnotation] = zod_frame.get_annotation(AnnotationProject.EGO_ROAD)
        ego_road_mask = polygons_to_binary_mask([anno.geometry for anno in ego_road_annotations])
        image = overlay_mask_on_image(ego_road_mask, image, fill_color=(100, 0, 100), alpha=0.5)
        print(f"  Applied ego road annotations")
    except Exception as ego_error:
        print(f"  Ego road annotations not available: {ego_error}")

    # Try to get OXTS data (skip if not available)
    try:
        oxts = zod_frame.oxts
        if oxts is not None:
            key_timestamp = zod_frame.info.keyframe_time.timestamp()
            calibrations = zod_frame.calibration
            image = visualize_oxts_on_image(oxts, key_timestamp, calibrations, image, camera=Camera.FRONT)
            print(f"  Applied OXTS visualization")
    except Exception as oxts_error:
        print(f"  OXTS data not available: {oxts_error}")

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # If image is in RGB format (height, width, 3)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            # Handle other formats if needed
            pil_image = Image.fromarray(image)
    else:
        # If image is already a PIL Image
        pil_image = image    
    return pil_image


def main():
    dataset_root = "./data"  # your local path to zod
    version = "mini"  # "mini" or "full"
    output_dir = './output/camera'
    
    # Initialize ZodFrames
    zod_frames = initialize_zod_frames(dataset_root, version)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all available frame IDs
    all_frame_ids = get_all_frame_ids(zod_frames)

    print(f"Processing {len(all_frame_ids)} frames...")

    for i, frame_id in enumerate(all_frame_ids):
        try:
            print(f"Processing frame {i+1}/{len(all_frame_ids)}: {frame_id}")
            
            # Process frame and get annotated image
            annotated_image = process_frame_with_annotations(zod_frames, frame_id)
            
            # Use iterator frame_id for filename
            frame_id_for_filename = int(frame_id)
            
            # Save the image
            output_path = f'{output_dir}/camera_{frame_id_for_filename:06d}.png'
            annotated_image.save(output_path, 'PNG', optimize=True, quality=95)
            
            if i % 10 == 0:  # Print progress every 10 frames
                print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error processing frame {frame_id}: {e}")
            continue

    print(f"Finished processing all frames. Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
