#!/usr/bin/env python3
"""Convert the full ZOD dataset into CLFT-compatible artifacts.

This script orchestrates the existing camera/LiDAR/SAM pipelines against the
full 100k-frame ZOD release and writes the outputs in the CLFT directory
structure that was prototyped on the mini dataset.

Features
--------
* Works with either the mini or full dataset (defaults to full).
* Supports selective generation of camera images, LiDAR CLFT tensors,
  SAM-based segmentation masks, fused visualisations, and metadata copies.
* Skips frames that already exist on disk (resume friendly).
* Emits a conversion manifest summarising successes/failures.
* Provides optional batching (`--limit` / `--start-index`) to split the
  100k-frame job into smaller runs.

Example
-------
```bash
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --components camera mask lidar metadata \
  --enable-sam \
  --workers 1
```
"""

from __future__ import annotations

import argparse
import json
import pickle
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from zod import ZodFrames
from zod.constants import Anonymization, Camera, AnnotationProject

# Local tooling that we reuse
from zod_fusion_processing import process_frame_with_camera_lidar_fusion
from zod_pickle_processing import ZODToCLFT

# SAM is optional and only imported when the user requests mask generation.
SAM_MODULE_IMPORT_ERROR = None
try:  # pragma: no cover - best effort, we only need this for CLI checks
    from zod_sam_processing import (
        SAMZODProcessor,
        SAM_CHECKPOINT_SPECS,
        DEFAULT_SAM_MODEL_TYPE,
    )
except Exception as exc:  # noqa: BLE001 - we really want the full error for UX
    SAM_MODULE_IMPORT_ERROR = exc
    SAMZODProcessor = None  # type: ignore
    SAM_CHECKPOINT_SPECS = {
        "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": ""},
        "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": ""},
        "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": ""},
    }
    DEFAULT_SAM_MODEL_TYPE = "vit_h"


AVAILABLE_COMPONENTS = {"camera", "lidar", "mask", "fusion", "metadata"}
DEFAULT_COMPONENTS = ["camera", "lidar", "metadata"]

# Target spatial resolution (width x height) for exported RGB / mask / fusion artefacts.
TARGET_IMAGE_SIZE = 512


@dataclass
class ConversionConfig:
    dataset_root: Path
    output_root: Path
    version: str = "full"
    components: Sequence[str] = field(default_factory=lambda: DEFAULT_COMPONENTS)
    splits: Sequence[str] = ("train", "val")
    start_index: int = 0
    limit: Optional[int] = None
    workers: int = 1
    enable_sam: bool = False
    sam_checkpoint: Path = Path("models/sam_vit_h_4b8939.pth")
    sam_model_type: str = DEFAULT_SAM_MODEL_TYPE
    skip_existing: bool = True
    dry_run: bool = False
    manifest_path: Optional[Path] = None
    lidar_overlays: bool = False
    lidar_overlay_only: bool = False
    batch_size: Optional[int] = None
    progress_log: Optional[Path] = None
    resume_progress: bool = False
    sam_overlay_only: bool = False
    sam_only: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.components) - AVAILABLE_COMPONENTS
        if unknown:
            raise ValueError(f"Unknown component(s): {sorted(unknown)}")


@dataclass
class FrameResult:
    frame_id: str
    camera: bool = False
    lidar: bool = False
    mask: bool = False
    fusion: bool = False
    metadata: bool = False
    sam_overlay: bool = False
    error: Optional[str] = None

    def success(self) -> bool:
        return self.error is None


class MissingLiDARData(RuntimeError):
    """Raised when no LiDAR data is available for a frame."""


class CLFTFullDatasetConverter:
    """Coordinator that executes the conversion pipeline for many frames."""

    def __init__(self, config: ConversionConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("clft_converter")
        self.logger.setLevel(logging.INFO)

        # Ensure output root exists up-front to catch permission issues early.
        self.config.output_root.mkdir(parents=True, exist_ok=True)

        self.progress_log_path = config.progress_log
        self.progress_lock = threading.Lock()
        self.completed_frames: Set[str] = set()
        self.frame_status: Dict[str, str] = {}
        if self.progress_log_path is not None:
            self.progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.progress_log_path.exists():
                with self.progress_log_path.open("r", encoding="utf-8") as log_file:
                    for line in log_file:
                        entry = line.strip()
                        if not entry:
                            continue
                        parts = [part.strip() for part in entry.split(",") if part.strip()]
                        if not parts:
                            continue
                        frame = parts[0]
                        status = parts[1] if len(parts) > 1 else "success"
                        self.completed_frames.add(frame)
                        self.frame_status[frame] = status
                if self.completed_frames:
                    self.logger.info(
                        "Loaded %d processed frame(s) from %s",
                        len(self.completed_frames),
                        self.progress_log_path,
                    )

        self.logger.info("Loading ZOD dataset from %s (version=%s)",
                         config.dataset_root, config.version)
        self.zod_frames = ZodFrames(dataset_root=str(config.dataset_root),
                                    version=config.version)

        self.frame_ids = self._collect_frame_ids(config.splits)
        self.logger.info("Found %d frames across splits %s", len(self.frame_ids),
                         ",".join(config.splits))

        # Pre-build CLFT converter (LiDAR to CLFT tensors)
        if "lidar" in config.components or config.lidar_overlays or config.lidar_overlay_only:
            self.clft_converter = ZODToCLFT(
                str(config.dataset_root),
                output_dir=str(config.output_root),
                version=config.version,
                zod_frames=self.zod_frames,
                visualization_subdir="visualizations_lidar",
                target_image_size=TARGET_IMAGE_SIZE,
            )
            # Reuse the already-loaded full dataset (ZODToCLFT defaults to mini)
            self.clft_converter.zod_frames = self.zod_frames
            self.logger.info("Prepared LiDAR -> CLFT converter")
        else:
            self.clft_converter = None

        # Optional SAM processor for segmentation masks
        self.sam_processor = None
        self.sam_lock = threading.Lock()
        if "mask" in config.components or config.enable_sam:
            if SAM_MODULE_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "segment-anything pipeline requested but the import failed"
                ) from SAM_MODULE_IMPORT_ERROR
            self.sam_processor = SAMZODProcessor(
                str(config.dataset_root),
                sam_checkpoint=str(config.sam_checkpoint),
                target_image_size=TARGET_IMAGE_SIZE,
                save_camera_image="camera" not in config.components,
                sam_model_type=config.sam_model_type,
            )
            self.sam_processor.zod_frames = self.zod_frames
            self.logger.info(
                "SAM processor ready (model=%s, checkpoint=%s)",
                config.sam_model_type,
                config.sam_checkpoint,
            )

    @staticmethod
    def _resize_image(image: Image.Image, target_size: int, resample: int) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if image.size == (target_size, target_size):
            return image
        return image.resize((target_size, target_size), resample=resample)

    @staticmethod
    def _image_matches_size(path: Path, target_size: int) -> bool:
        try:
            with Image.open(path) as existing:
                return existing.size == (target_size, target_size)
        except Exception:
            return False

    def _ensure_mask_size(self, mask_path: Path) -> bool:
        if not mask_path.exists():
            return False
        with Image.open(mask_path) as mask_image:
            if mask_image.size == (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE):
                return True
            resized = mask_image.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), resample=Image.NEAREST)
            resized.save(mask_path, format="PNG", optimize=True)
        return True

    # ------------------------------------------------------------------
    # Frame enumeration & manifest helpers
    # ------------------------------------------------------------------
    def _collect_frame_ids(self, splits: Sequence[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for split in splits:
            try:
                split_ids = list(self.zod_frames.get_split(split))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Split '%s' unavailable: %s", split, exc)
                continue
            for fid in split_ids:
                if fid not in seen:
                    seen.add(fid)
                    ordered.append(fid)
        return ordered

    # ------------------------------------------------------------------
    # Export helpers for individual components
    # ------------------------------------------------------------------
    def _camera_exists(self, frame_id: str) -> bool:
        return (self.config.output_root / "camera" / f"frame_{frame_id}.png").exists()

    def _load_camera_image(self, frame_id: str) -> Image.Image:
        frame = self.zod_frames[frame_id]
        attempts = [
            (Camera.FRONT, Anonymization.DNAT),
            (Camera.FRONT, Anonymization.BLUR),
        ]
        errors = []

        for camera, anonymization in attempts:
            try:
                return frame.get_image(anonymization)
            except FileNotFoundError as exc:
                errors.append(f"{camera.name}-{anonymization.name}: {exc}")
            except Exception as exc:  # noqa: BLE001 - want full context for manifest
                errors.append(f"{camera.name}-{anonymization.name}: {exc}")

        try:
            return frame.get_image()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"DEFAULT: {exc}")

        error_msg = "; ".join(errors) if errors else "no attempts made"
        raise FileNotFoundError(
            f"Unable to load camera image for frame {frame_id}. Attempts: {error_msg}"
        )

    def _lidar_exists(self, frame_id: str) -> bool:
        return (self.config.output_root / "lidar" / f"frame_{frame_id}.pkl").exists()

    def _mask_exists(self, frame_id: str) -> bool:
        return (self.config.output_root / "annotation" / f"frame_{frame_id}.png").exists()

    def _fusion_exists(self, frame_id: str) -> bool:
        return (self.config.output_root / "fusion" / f"frame_{frame_id}.png").exists()

    def _metadata_exists(self, frame_id: str) -> bool:
        return (self.config.output_root / "metadata" / f"frame_{frame_id}.json").exists()

    def _lidar_overlay_exists(self, frame_id: str) -> bool:
        return (
            self.config.output_root
            / "visualizations_lidar"
            / f"frame_{frame_id}_lidar_overlay.png"
        ).exists()

    def _process_frame(self, frame_id: str) -> FrameResult:
        result = FrameResult(frame_id=frame_id)
        try:
            if self.config.lidar_overlay_only:
                result.lidar = self._export_lidar_overlay(frame_id)
                if not result.lidar:
                    result.error = "missing_lidar_overlay_inputs"
                return result
            if "lidar" in self.config.components or self.config.lidar_overlays:
                try:
                    result.lidar = self._export_lidar(frame_id)
                except MissingLiDARData as exc:
                    result.error = "missing_lidar"
                    overlay_path = self.config.output_root / "visualizations_lidar" / f"frame_{frame_id}_lidar_overlay.png"
                    try:
                        overlay_path.unlink()
                    except FileNotFoundError:
                        pass
                    lidar_path = self.config.output_root / "lidar" / f"frame_{frame_id}.pkl"
                    try:
                        lidar_path.unlink()
                    except FileNotFoundError:
                        pass
                    cleanup_targets = [
                        self.config.output_root / "camera" / f"frame_{frame_id}.png",
                        self.config.output_root / "annotation" / f"frame_{frame_id}.png",
                        self.config.output_root / "annotation_rgb" / f"frame_{frame_id}.png",
                        self.config.output_root / "fusion" / f"frame_{frame_id}.png",
                        self.config.output_root / "metadata" / f"frame_{frame_id}.json",
                    ]
                    for target in cleanup_targets:
                        try:
                            target.unlink()
                        except FileNotFoundError:
                            pass
                    self.logger.warning("Skipping frame %s: %s", frame_id, exc)
                    return result
            if "camera" in self.config.components:
                result.camera = self._export_camera(frame_id)
            if "mask" in self.config.components or self.config.enable_sam:
                result.mask = self._export_mask(frame_id)
            if "fusion" in self.config.components:
                result.fusion = self._export_fusion(frame_id)
            if "metadata" in self.config.components:
                result.metadata = self._export_metadata(frame_id)
            if self.config.sam_overlay_only:
                result.sam_overlay = self._export_sam_overlay(frame_id)
                if not result.sam_overlay and result.error is None:
                    result.error = "missing_sam_assets"
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            self.logger.exception("Failed to process frame %s", frame_id)
        if self.config.sam_overlay_only and not result.sam_overlay and result.error == "missing_sam_assets":
            self.logger.warning(
                "SAM overlay skipped for frame %s: required camera or mask artefacts missing",
                frame_id,
            )
        return result

    def _record_progress(self, frame_id: str, status: str) -> None:
        if self.progress_log_path is None:
            return
        if frame_id in self.completed_frames:
            return
        entry = f"{frame_id},{status}"
        with self.progress_lock:
            with self.progress_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{entry}\n")
            self.completed_frames.add(frame_id)
            self.frame_status[frame_id] = status

    def _process_batch(self, frame_ids: Sequence[str], progress: tqdm) -> Dict[str, FrameResult]:
        batch_results: Dict[str, FrameResult] = {}
        if not frame_ids:
            return batch_results

        if self.config.workers <= 1:
            for frame_id in frame_ids:
                result = self._process_frame(frame_id)
                batch_results[frame_id] = result
                status = "success" if result.success() else (result.error or "error")
                status = status.replace(" ", "_")
                if status in {"success", "missing_lidar", "missing_sam_assets", "missing_lidar_overlay_inputs"}:
                    self._record_progress(frame_id, status)
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                future_to_frame = {
                    executor.submit(self._process_frame, frame_id): frame_id for frame_id in frame_ids
                }
                for future in as_completed(future_to_frame):
                    frame_id = future_to_frame[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        self.logger.exception("Worker crashed while processing frame %s", frame_id)
                        result = FrameResult(frame_id=frame_id, error=str(exc))
                    batch_results[frame_id] = result
                    status = "success" if result.success() else (result.error or "error")
                    status = status.replace(" ", "_")
                    if status in {"success", "missing_lidar", "missing_sam_assets", "missing_lidar_overlay_inputs"}:
                        self._record_progress(frame_id, status)
                    progress.update(1)

        return batch_results

    # -- camera --------------------------------------------------------
    def _export_camera(self, frame_id: str) -> bool:
        output_path = self.config.output_root / "camera" / f"frame_{frame_id}.png"
        if self.config.skip_existing and output_path.exists() and self._image_matches_size(output_path, TARGET_IMAGE_SIZE):
            return True
        image = self._load_camera_image(frame_id)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        image = self._resize_image(image, TARGET_IMAGE_SIZE, Image.BILINEAR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        return True

    # -- lidar ---------------------------------------------------------
    def _export_lidar(self, frame_id: str) -> bool:
        if self.clft_converter is None:
            return False
        if self.config.skip_existing and self._lidar_exists(frame_id):
            return True
        clft_data = self.clft_converter.create_clft_data(frame_id)
        if not clft_data:
            raise MissingLiDARData("no LiDAR point cloud available")
        zod_frame = self.zod_frames[frame_id] if self.config.lidar_overlays else None
        success = self.clft_converter.save_clft_data(clft_data, frame_id, zod_frame=zod_frame)
        if not success:
            raise MissingLiDARData("failed to save LiDAR CLFT artefact")
        return True

    def _export_lidar_overlay(self, frame_id: str) -> bool:
        if self.clft_converter is None:
            return False

        overlay_dir = self.config.output_root / "visualizations_lidar"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"frame_{frame_id}_lidar_overlay.png"

        if self.config.skip_existing and overlay_path.exists() and self._image_matches_size(overlay_path, TARGET_IMAGE_SIZE):
            return True

        lidar_path = self.config.output_root / "lidar" / f"frame_{frame_id}.pkl"
        if not lidar_path.exists():
            return False

        try:
            with lidar_path.open("rb") as handle:
                clft_data = pickle.load(handle)
        except Exception:
            return False

        zod_frame = None
        if self.clft_converter.zod_frames is not None:
            try:
                zod_frame = self.zod_frames[frame_id]
            except Exception:  # noqa: BLE001
                zod_frame = None

        if zod_frame is not None:
            return self.clft_converter.create_visualization(
                frame_id,
                clft_data,
                zod_frame=zod_frame,
            )

        camera_path = self.config.output_root / "camera" / f"frame_{frame_id}.png"
        if not camera_path.exists():
            return False

        return self.clft_converter.create_visualization(
            frame_id,
            clft_data,
            zod_frame=None,
            camera_image_path=camera_path,
        )

    # -- mask (SAM) ----------------------------------------------------
    def _export_mask(self, frame_id: str) -> bool:
        if self.sam_processor is None:
            return False
        mask_path = self.config.output_root / "annotation" / f"frame_{frame_id}.png"
        if self.config.skip_existing and mask_path.exists():
            if self._image_matches_size(mask_path, TARGET_IMAGE_SIZE):
                return True
            return self._ensure_mask_size(mask_path)

        with self.sam_lock:
            success = self.sam_processor.process_frame(frame_id, str(self.config.output_root))
        if success:
            self._ensure_mask_size(mask_path)
        return success

    def _export_sam_overlay(self, frame_id: str) -> bool:
        overlay_dir = self.config.output_root / "visualizations"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"frame_{frame_id}_sam_overlay.png"

        if self.config.skip_existing and overlay_path.exists() and self._image_matches_size(overlay_path, TARGET_IMAGE_SIZE):
            return True

        mask_path = self.config.output_root / "annotation" / f"frame_{frame_id}.png"
        if not mask_path.exists():
            return False

        camera_path = self.config.output_root / "camera" / f"frame_{frame_id}.png"

        image: Optional[Image.Image]
        target_size = TARGET_IMAGE_SIZE

        if camera_path.exists():
            with Image.open(camera_path) as cam_img:
                image = cam_img.convert("RGB")
                if target_size is not None:
                    image = self._resize_image(image, target_size, Image.BILINEAR)
        else:
            try:
                image = self._load_camera_image(frame_id)
                if target_size is not None:
                    image = self._resize_image(image, target_size, Image.BILINEAR)
            except FileNotFoundError:
                return False
            except Exception:
                return False

        with Image.open(mask_path) as mask_img:
            resize_target = image.size if target_size is not None else image.size
            if mask_img.size != resize_target:
                mask_img = mask_img.resize(resize_target, resample=Image.NEAREST)
            mask_np = np.array(mask_img, dtype=np.uint8)

        image_np = np.array(image).astype(np.float32)

        max_label = int(mask_np.max())
        if max_label > 0:
            denom = max(1, max_label)
            colored_mask = plt.cm.tab10(mask_np.astype(np.float32) / denom)[:, :, :3]
            mask_alpha = (mask_np > 0).astype(np.float32) * 0.6
            image_np = (
                image_np * (1.0 - mask_alpha[..., None])
                + colored_mask * 255.0 * mask_alpha[..., None]
            )

        overlay_image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        overlay_image.save(overlay_path, format="PNG", optimize=True)
        return True

    # -- fusion --------------------------------------------------------
    def _export_fusion(self, frame_id: str) -> bool:
        output_path = self.config.output_root / "fusion" / f"frame_{frame_id}.png"
        if self.config.skip_existing and output_path.exists() and self._image_matches_size(output_path, TARGET_IMAGE_SIZE):
            return True
        frame = process_frame_with_camera_lidar_fusion(self.zod_frames, frame_id)
        frame = self._resize_image(frame, TARGET_IMAGE_SIZE, Image.BILINEAR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(output_path, format="PNG", optimize=True)
        return True

    # -- metadata ------------------------------------------------------
    def _export_metadata(self, frame_id: str) -> bool:
        if self.config.skip_existing and self._metadata_exists(frame_id):
            return True
        frame_dir = Path(self.config.dataset_root) / "single_frames" / frame_id
        if not frame_dir.exists():
            return False
        output_dir = self.config.output_root / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / f"frame_{frame_id}.json"
        metadata_file = frame_dir / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
            except json.JSONDecodeError:
                metadata = {}

            metadata.setdefault("frame_id", frame_id)

            # Weather enrichment -------------------------------------------------
            scraped_weather = str(metadata.get("scraped_weather", "") or "").replace("_", "-")
            weather_tags = [part for part in scraped_weather.replace(" ", "-").lower().split("-") if part]
            metadata["weather_tags"] = weather_tags
            if weather_tags:
                metadata["weather_condition"] = weather_tags[0]
                if len(weather_tags) > 1:
                    metadata["weather_time_context"] = weather_tags[1]

            precipitation_tokens = {"rain", "snow", "sleet", "hail", "storm", "fog", "mist", "drizzle"}
            metadata["weather_precipitation"] = any(tag in precipitation_tokens for tag in weather_tags)
            time_of_day = str(metadata.get("time_of_day", ""))
            metadata["is_daylight"] = time_of_day.lower() in {"day", "daytime", "afternoon", "morning"}

            # Object annotation enrichment --------------------------------------
            class_counts: Counter[str] = Counter()
            occlusion_counts: Counter[str] = Counter()
            objects_details = []
            
            try:
                annotations = self.zod_frames[frame_id].get_annotation(AnnotationProject.OBJECT_DETECTION)
            except Exception:
                annotations = []

            for annotation in annotations:
                name = getattr(annotation, "name", None) or "unknown"
                class_counts[name] += 1
                
                # Extract occlusion information - try different methods
                occlusion = None
                if hasattr(annotation, 'occlusion'):
                    occlusion = annotation.occlusion
                elif hasattr(annotation, 'visibility'):
                    occlusion = annotation.visibility
                elif hasattr(annotation, 'occlusion_level'):
                    occlusion = annotation.occlusion_level
                
                if occlusion is not None:
                    occlusion_str = str(occlusion).replace("Occlusion.", "").replace("Visibility.", "").lower()
                    occlusion_counts[occlusion_str] += 1
                else:
                    occlusion_counts['unknown'] += 1
                
                # Extract detailed object information
                object_info = {
                    "name": name,
                    "occlusion": occlusion_str if occlusion is not None else "unknown",
                }
                
                objects_details.append(object_info)

            metadata["object_class_counts"] = dict(sorted(class_counts.items()))
            metadata["object_classes_present"] = sorted(class_counts.keys())
            metadata["num_objects_total"] = int(sum(class_counts.values()))
            if class_counts:
                metadata["dominant_object_class"] = max(class_counts.items(), key=lambda item: item[1])[0]

            # Detailed object information
            metadata["objects"] = objects_details
            metadata["num_objects_detailed"] = len(objects_details)

            # Occlusion enrichment ---------------------------------------------
            # Define all possible occlusion levels
            all_occlusion_levels = {"none", "light", "medium", "heavy", "veryheavy", "unknown"}
            
            # Initialize counts for all levels
            full_occlusion_counts = {level: 0 for level in all_occlusion_levels}
            full_occlusion_counts.update(occlusion_counts)  # Update with actual counts
            
        metadata["occlusion_counts"] = dict(sorted(full_occlusion_counts.items()))
        metadata["occlusion_levels_present"] = sorted([level for level, count in full_occlusion_counts.items() if count > 0])
        
        # Calculate occlusion percentages
        total_objects = metadata["num_objects_total"]
        if total_objects > 0:
            occlusion_percentages = {level: (count / total_objects) * 100 for level, count in full_occlusion_counts.items()}
            metadata["occlusion_percentages"] = dict(sorted(occlusion_percentages.items()))
        else:
            metadata["occlusion_percentages"] = {level: 0.0 for level in all_occlusion_levels}
        
        # Check for every occlusion level
        metadata["has_none_occlusion"] = full_occlusion_counts.get("none", 0) > 0
        metadata["has_light_occlusion"] = full_occlusion_counts.get("light", 0) > 0
        metadata["has_medium_occlusion"] = full_occlusion_counts.get("medium", 0) > 0
        metadata["has_heavy_occlusion"] = full_occlusion_counts.get("heavy", 0) > 0
        metadata["has_veryheavy_occlusion"] = full_occlusion_counts.get("veryheavy", 0) > 0
        metadata["has_unknown_occlusion"] = full_occlusion_counts.get("unknown", 0) > 0
        
        # Count objects for each occlusion level
        metadata["num_objects_none_occlusion"] = full_occlusion_counts.get("none", 0)
        metadata["num_objects_light_occlusion"] = full_occlusion_counts.get("light", 0)
        metadata["num_objects_medium_occlusion"] = full_occlusion_counts.get("medium", 0)
        metadata["num_objects_heavy_occlusion"] = full_occlusion_counts.get("heavy", 0)
        metadata["num_objects_veryheavy_occlusion"] = full_occlusion_counts.get("veryheavy", 0)
        metadata["num_objects_unknown_occlusion"] = full_occlusion_counts.get("unknown", 0)
        
        # Occlusion summary statistics
        total_objects_with_occlusion = sum(full_occlusion_counts.values())
        if total_objects_with_occlusion > 0:
            metadata["occlusion_summary"] = {
                "total_objects_with_occlusion": total_objects_with_occlusion,
                "none_occlusion_percentage": (full_occlusion_counts.get("none", 0) / total_objects_with_occlusion) * 100,
                "light_occlusion_percentage": (full_occlusion_counts.get("light", 0) / total_objects_with_occlusion) * 100,
                "medium_occlusion_percentage": (full_occlusion_counts.get("medium", 0) / total_objects_with_occlusion) * 100,
                "heavy_occlusion_percentage": (full_occlusion_counts.get("heavy", 0) / total_objects_with_occlusion) * 100,
                "veryheavy_occlusion_percentage": (full_occlusion_counts.get("veryheavy", 0) / total_objects_with_occlusion) * 100,
                "unknown_occlusion_percentage": (full_occlusion_counts.get("unknown", 0) / total_objects_with_occlusion) * 100,
                "most_common_occlusion": max(full_occlusion_counts.items(), key=lambda x: x[1])[0] if any(full_occlusion_counts.values()) else None
            }
        else:
            metadata["occlusion_summary"] = {
                "total_objects_with_occlusion": 0,
                "none_occlusion_percentage": 0.0,
                "light_occlusion_percentage": 0.0,
                "medium_occlusion_percentage": 0.0,
                "heavy_occlusion_percentage": 0.0,
                "veryheavy_occlusion_percentage": 0.0,
                "unknown_occlusion_percentage": 0.0,
                "most_common_occlusion": None
            }

        metadata["num_pedestrians"] = metadata.get("num_pedestrians", 0)
        metadata["num_vulnerable_road_users"] = metadata.get("num_vulnerable_vehicles", 0) + metadata.get("num_pedestrians", 0)
        metadata["num_traffic_control"] = metadata.get("num_traffic_signs", 0) + metadata.get("num_traffic_lights", 0)

        target_path.write_text(json.dumps(metadata, indent=2))
        return True
        return False

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def convert(self) -> Dict[str, FrameResult]:
        start = self.config.start_index
        end = len(self.frame_ids) if self.config.limit is None else min(
            len(self.frame_ids), start + self.config.limit
        )
        selected = self.frame_ids[start:end]

        if not selected:
            self.logger.warning("No frames selected for conversion")
            return {}

        frames_to_process = selected
        if self.config.resume_progress and self.completed_frames:
            before = len(frames_to_process)
            frames_to_process = [fid for fid in frames_to_process if fid not in self.completed_frames]
            skipped = before - len(frames_to_process)
            if skipped:
                self.logger.info("Skipping %d frame(s) already recorded in progress log", skipped)

        if not frames_to_process:
            self.logger.warning("No frames left to process after applying resume filters")
            return {}

        total_frames = len(frames_to_process)
        self.logger.info("Processing %d frame(s) [%d:%d])", total_frames, start, end)
        results: Dict[str, FrameResult] = {}

        batch_size = self.config.batch_size if self.config.batch_size and self.config.batch_size > 0 else None
        if batch_size:
            batches = [frames_to_process[i:i + batch_size] for i in range(0, len(frames_to_process), batch_size)]
        else:
            batches = [frames_to_process]

        progress = tqdm(total=total_frames, desc="Converting", unit="frame")
        try:
            total_batches = len(batches)
            for batch_index, batch in enumerate(batches, start=1):
                if not batch:
                    continue
                if total_batches > 1:
                    self.logger.info(
                        "Processing batch %d/%d (%d frame(s))",
                        batch_index,
                        total_batches,
                        len(batch),
                    )
                batch_results = self._process_batch(batch, progress)
                results.update(batch_results)
        finally:
            progress.close()

        ordered_results = {frame_id: results[frame_id] for frame_id in frames_to_process if frame_id in results}

        if self.config.manifest_path:
            manifest = {
                "dataset_root": str(self.config.dataset_root),
                "output_root": str(self.config.output_root),
                "version": self.config.version,
                "components": list(self.config.components),
                "frames": {
                    fid: {
                        "success": res.success(),
                        "camera": res.camera,
                        "lidar": res.lidar,
                        "mask": res.mask,
                        "fusion": res.fusion,
                        "metadata": res.metadata,
                        "sam_overlay": res.sam_overlay,
                        "error": res.error,
                    }
                    for fid, res in ordered_results.items()
                },
            }
            self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.manifest_path.write_text(json.dumps(manifest, indent=2))
            self.logger.info("Wrote manifest to %s", self.config.manifest_path)

        return ordered_results


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root directory of the extracted ZOD dataset")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Destination directory for CLFT artefacts")
    parser.add_argument("--version", choices=["mini", "full"], default="full",
                        help="ZOD dataset variant to use")
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS,
                        choices=sorted(AVAILABLE_COMPONENTS),
                        help="Which artefacts to create")
    parser.add_argument("--with-mask", action="store_true",
                        help="Shortcut to add SAM masks and overlays to the selected components")
    parser.add_argument("--with-fusion", action="store_true",
                        help="Shortcut to add camera/LiDAR fusion visualisations")
    parser.add_argument("--with-lidar-overlay", action="store_true",
                        help="Generate LiDAR point projections on the camera image (stored separately)")
    parser.add_argument("--lidar-overlay-only", action="store_true",
                        help="Render LiDAR overlays from existing camera and LiDAR artefacts without recomputing point clouds")
    parser.add_argument("--sam-only", action="store_true",
                        help="Generate SAM annotations and overlays without exporting other artefacts")
    parser.add_argument("--sam-overlay-only", action="store_true",
                        help="Render SAM overlays from existing camera images and SAM masks without running SAM")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Dataset split(s) to process")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip this many frames before processing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker threads for per-frame processing")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Process frames in batches of this many before yielding control")
    parser.add_argument("--enable-sam", action="store_true",
                        help="Force-load SAM even if 'mask' is not in components")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("models/sam_vit_h_4b8939.pth"),
                        help="Path to SAM checkpoint")
    parser.add_argument("--sam-model-type", choices=sorted(SAM_CHECKPOINT_SPECS.keys()), default=None,
                        help="Which SAM backbone to use (vit_h, vit_l, vit_b)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Recreate outputs even if files exist")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional JSON manifest path for run summary")
    parser.add_argument("--dry-run", action="store_true",
                        help="List selected frames without writing any files")
    parser.add_argument("--progress-log", type=Path, default=None,
                        help="Append successful frame IDs to this log file for resumable runs")
    parser.add_argument("--resume-progress", action="store_true",
                        help="Skip frames already listed in the progress log")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    components = list(dict.fromkeys(args.components))
    if args.with_mask and "mask" not in components:
        components.append("mask")
    if args.with_fusion and "fusion" not in components:
        components.append("fusion")
    if args.sam_only:
        components = ["mask"]
        args.with_mask = True
        args.enable_sam = True
    if args.sam_overlay_only:
        components = []
        args.with_mask = False
        args.enable_sam = False
    if args.lidar_overlay_only:
        args.with_lidar_overlay = True
        components = []
    if args.with_lidar_overlay and not args.lidar_overlay_only and "lidar" not in components:
        components.append("lidar")

    sam_model_type = args.sam_model_type or DEFAULT_SAM_MODEL_TYPE
    sam_checkpoint = args.sam_checkpoint
    default_checkpoint = Path("models") / SAM_CHECKPOINT_SPECS[DEFAULT_SAM_MODEL_TYPE]["filename"]
    if (
        args.sam_model_type
        and args.sam_checkpoint == default_checkpoint
    ):
        target_spec = SAM_CHECKPOINT_SPECS.get(sam_model_type, SAM_CHECKPOINT_SPECS[DEFAULT_SAM_MODEL_TYPE])
        sam_checkpoint = args.sam_checkpoint.parent / target_spec["filename"]

    config = ConversionConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        version=args.version,
        components=components,
        splits=args.splits,
        start_index=args.start_index,
        limit=args.limit,
        workers=max(1, args.workers),
        enable_sam=args.enable_sam or args.with_mask or args.sam_only,
    sam_checkpoint=sam_checkpoint,
    sam_model_type=sam_model_type,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
        manifest_path=args.manifest,
        lidar_overlays=args.with_lidar_overlay,
        lidar_overlay_only=args.lidar_overlay_only,
        sam_overlay_only=args.sam_overlay_only,
        sam_only=args.sam_only,
        batch_size=args.batch_size if args.batch_size and args.batch_size > 0 else None,
        progress_log=args.progress_log,
        resume_progress=args.resume_progress,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    converter = CLFTFullDatasetConverter(config)

    if config.dry_run:
        print("Selected frames:")
        for frame_id in converter.frame_ids[config.start_index:config.start_index + (config.limit or len(converter.frame_ids))]:
            print(frame_id)
        return

    converter.convert()


if __name__ == "__main__":
    main()
