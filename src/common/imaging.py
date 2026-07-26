"""Deterministic image, ROI, placement, and blending helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImagingError(RuntimeError):
    """An image operation could not satisfy its invariant."""


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``x, y, width, height`` for a non-empty boolean mask."""

    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ImagingError("Mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    retained = np.zeros_like(mask, dtype=np.uint8)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            retained[labels == component] = 1
    return retained.astype(bool)


def detect_legal_roi(
    image: np.ndarray,
    *,
    mode: str = "union",
    min_component_area_ratio: float,
    close_kernel: int,
    open_kernel: int,
    erosion_px: int,
    border_fraction: float,
) -> np.ndarray:
    """Estimate foreground objects from border-color distance and saturation."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ImagingError(f"Expected RGB image, observed shape {image.shape}")
    height, width = image.shape[:2]
    border_y = max(1, round(height * border_fraction))
    border_x = max(1, round(width * border_fraction))
    border_pixels = np.concatenate(
        (
            image[:border_y].reshape(-1, 3),
            image[-border_y:].reshape(-1, 3),
            image[:, :border_x].reshape(-1, 3),
            image[:, -border_x:].reshape(-1, 3),
        ),
        axis=0,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_lab = cv2.cvtColor(
        border_pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    background = np.median(border_lab, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_u8 = np.clip(distance / max(float(distance.max()), 1.0) * 255, 0, 255).astype(np.uint8)
    _, distance_mask = cv2.threshold(
        distance_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    _, saturation_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    use_largest_bbox = mode == "saturation_bbox"
    if mode == "union":
        combined = np.maximum(distance_mask, saturation_mask)
    elif mode in {"saturation", "saturation_bbox"}:
        combined = saturation_mask
    elif mode == "distance":
        combined = distance_mask
    else:
        raise ImagingError(f"Unknown ROI mode: {mode}")
    close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel, close_kernel),
    )
    opened = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel, open_kernel),
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, opened)
    minimum_area = max(32, round(height * width * min_component_area_ratio))
    legal = _remove_small_components(combined > 0, minimum_area)
    if use_largest_bbox:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            legal.astype(np.uint8),
            connectivity=8,
        )
        if count <= 1:
            raise ImagingError("Could not find a saturated foreground component")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x = int(stats[largest, cv2.CC_STAT_LEFT])
        y = int(stats[largest, cv2.CC_STAT_TOP])
        box_width = int(stats[largest, cv2.CC_STAT_WIDTH])
        box_height = int(stats[largest, cv2.CC_STAT_HEIGHT])
        legal = np.zeros((height, width), dtype=bool)
        legal[y : y + box_height, x : x + box_width] = True
    if erosion_px:
        erosion = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_px * 2 + 1, erosion_px * 2 + 1),
        )
        legal = cv2.erode(legal.astype(np.uint8), erosion).astype(bool)
    legal[:erosion_px, :] = False
    legal[-erosion_px:, :] = False
    legal[:, :erosion_px] = False
    legal[:, -erosion_px:] = False
    if int(legal.sum()) < minimum_area:
        raise ImagingError("Legal ROI is empty after foreground cleanup")
    return legal


def transform_component(
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    rotation_deg: float,
    scale: float,
    flip: bool,
    brightness: float,
    contrast: float,
    saturation: float,
    hue_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic affine and color transforms to one component patch."""

    if flip:
        patch = np.ascontiguousarray(np.fliplr(patch))
        mask = np.ascontiguousarray(np.fliplr(mask))
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, scale)
    transformed_patch = cv2.warpAffine(
        patch,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if not transformed_mask.any():
        raise ImagingError("Affine transform produced an empty mask")

    enhanced = Image.fromarray(transformed_patch)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    enhanced = ImageEnhance.Color(enhanced).enhance(saturation)
    hsv = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2HSV)
    hue_delta = round(hue_shift * 180)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_delta) % 180
    transformed_patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    x, y, box_width, box_height = mask_bbox(transformed_mask)
    padding = 6
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_width + padding)
    y1 = min(height, y + box_height + padding)
    return (
        transformed_patch[y0:y1, x0:x1],
        transformed_mask[y0:y1, x0:x1],
    )


def place_on_legal_roi(
    component_mask: np.ndarray,
    legal_roi: np.ndarray,
    rng: np.random.Generator,
    *,
    max_tries: int,
) -> tuple[int, int]:
    """Return a top-left placement whose non-zero mask is fully legal."""

    patch_height, patch_width = component_mask.shape
    image_height, image_width = legal_roi.shape
    if patch_height > image_height or patch_width > image_width:
        raise ImagingError("Component patch is larger than the destination image")

    illegal_roi = (~legal_roi).astype(np.uint8)
    overlap = cv2.matchTemplate(
        illegal_roi,
        component_mask.astype(np.uint8),
        cv2.TM_CCORR,
    )
    legal_y, legal_x = np.nonzero(overlap < 0.5)
    if not len(legal_y):
        raise ImagingError("No legal top-left position exists for this transformed mask")
    tries = min(max_tries, len(legal_y))
    candidate_order = rng.choice(len(legal_y), size=tries, replace=False)
    for candidate in np.atleast_1d(candidate_order):
        x0 = int(legal_x[int(candidate)])
        y0 = int(legal_y[int(candidate)])
        destination_roi = legal_roi[
            y0 : y0 + patch_height,
            x0 : x0 + patch_width,
        ]
        if bool(np.all(destination_roi[component_mask])):
            return x0, y0
    raise ImagingError(f"No legal placement found in {max_tries} attempts")


def blend_component(
    background: np.ndarray,
    patch: np.ndarray,
    component_mask: np.ndarray,
    *,
    x0: int,
    y0: int,
    method: str,
    opacity: float,
    feather_radius: float,
) -> tuple[np.ndarray, str]:
    """Blend a patch and return the image plus the actual method used."""

    height, width = component_mask.shape
    if method == "poisson":
        source_mask = component_mask.astype(np.uint8) * 255
        center = (x0 + width // 2, y0 + height // 2)
        try:
            blended_bgr = cv2.seamlessClone(
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(background, cv2.COLOR_RGB2BGR),
                source_mask,
                center,
                cv2.MIXED_CLONE,
            )
            return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB), "poisson"
        except cv2.error:
            method = "feather"

    if method != "feather":
        raise ImagingError(f"Unknown blend method: {method}")
    sigma = max(feather_radius, 0.01)
    alpha = cv2.GaussianBlur(
        component_mask.astype(np.float32),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    alpha = np.clip(alpha * opacity, 0.0, 1.0)[..., None]
    result = background.astype(np.float32).copy()
    destination = result[y0 : y0 + height, x0 : x0 + width]
    destination[:] = patch.astype(np.float32) * alpha + destination * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8), "feather_alpha"
