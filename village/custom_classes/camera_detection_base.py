from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from village.custom_classes.task_base import TaskBase
from village.settings import Color

if TYPE_CHECKING:
    from village.devices.camera import Camera


@dataclass
class CustomDetectionParam:
    """Custom param a CameraDetectionBase subclass wants
    exposed and live-editable in the GUI."""

    name: str  # attribute name on the detector instance
    type_: type  # for now supports float, int, or bool
    default: float
    label: str  # UI label text
    min_val: float = 0.0
    max_val: float = 1.0
    tooltip: str = ""

    def clamp(self, val):
        if self.type_ is bool:
            return bool(val)
        return max(self.min_val, min(self.max_val, self.type_(val)))


class CameraDetectionBase:
    """Shared default per-frame detection logic for all cameras.
    Reimplementation of previous behavior written in camera.py here.

    Not meant to be subclassed directly: subclass CorridorDetectionBase or
    BoxDetectionBase below to replace one camera's detection only. Each is
    its own independent plugin slot, so you can have a custom corridor
    detector and a custom box detector loaded at the same time, or either
    one on its own; whichever one you don't customize keeps this class's
    default behavior.

    Declare PARAMS to get tunable values rendered as editable fields in the GUI
    automatically so you can use them in your custom detection class, e.g.:

        PARAMS = [
            CustomDetectionParam(
                "erosion_size", int, 0, "Erosion kernel size", 0, 15,
                "Shrinks the detection mask by this many pixels before "
                "contour extraction, to strip speckle noise (0 = off).",
            ),
        ]

    This exposes an editable "Erosion kernel size" field in the GUI and sets
    self.erosion_size = 0 by default; editing it calls update_params().

    Declare ACTIONS to also get a GUI button per named method, e.g.
    ACTIONS = ["flush", "save"] renders "Flush" and "Save" buttons that
    call self.flush() / self.save() when clicked.
    """

    PARAMS: list[CustomDetectionParam] = []
    ACTIONS: list[str] = []

    def __init__(self) -> None:
        self.name = "Camera Detection"
        self.task = TaskBase()
        for param in self.PARAMS:
            setattr(self, param.name, param.default)

    def update_params(self, **kwargs) -> None:
        """Update param attributes from the GUI.."""
        for param in self.PARAMS:
            if param.name in kwargs:
                setattr(self, param.name, param.clamp(kwargs[param.name]))

    def detect(self, cam: Camera) -> None:
        """Runs detection for one frame.
        Called on every frame for both cameras."""
        if cam.color == Color.BLACK:
            if cam.tracking:
                self._detect_black_position_contours(cam)
            else:
                self._detect_black(cam)
        else:
            if cam.tracking:
                self._detect_white_position_contours(cam)
            else:
                self._detect_white(cam)

    def _detect_black(self, cam: Camera) -> None:
        """Detects black objects in defined areas using thresholding."""
        for index, (x1, y1, x2, y2) in enumerate(cam.areas):
            if cam.areas_active[index]:
                roi = cam.gray_frame[y1:y2, x1:x2]
                threshold = cam.thresholds[index]
                _, thresh = cv2.threshold(roi, threshold, 255,
                                          cv2.THRESH_BINARY_INV)
                cam.masks[index] = thresh
                cam.counts[index] = cv2.countNonZero(thresh)
            else:
                cam.masks[index] = -1
                cam.counts[index] = -1

    def _detect_white(self, cam: Camera) -> None:
        """Detects white objects in defined areas using thresholding."""
        for index, (x1, y1, x2, y2) in enumerate(cam.areas):
            if cam.areas_active[index]:
                roi = cam.gray_frame[y1:y2, x1:x2]
                threshold = cam.thresholds[index]
                _, thresh = cv2.threshold(roi, threshold, 255,
                                          cv2.THRESH_BINARY)
                cam.masks[index] = thresh
                cam.counts[index] = cv2.countNonZero(thresh)
            else:
                cam.masks[index] = -1
                cam.counts[index] = -1

    def _detect_black_position_contours(self, cam: Camera) -> None:
        """Detects position of black mouse using contours."""
        mask = np.zeros_like(cam.gray_frame, dtype=np.uint8)
        for index, (x1, y1, x2, y2) in enumerate(cam.areas):
            if cam.areas_active[index]:
                roi = cam.gray_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    cam.masks[index] = -1
                    cam.counts[index] = -1
                    continue
                threshold = cam.thresholds[index]
                _, roi_bin = cv2.threshold(roi, threshold, 255,
                                           cv2.THRESH_BINARY_INV)
                roi_bin = np.asarray(roi_bin, dtype=np.uint8)
                sub = mask[y1:y2, x1:x2]
                np.maximum(sub, roi_bin, out=sub)
                cam.masks[index] = roi_bin
                cam.counts[index] = cv2.countNonZero(roi_bin)
            else:
                cam.masks[index] = -1
                cam.counts[index] = -1

        cam._add_custom_area_mask(mask, invert=True)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cam.x_position = -1
            cam.y_position = -1
            return

        best_c = None
        best_area = 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a >= cam.zero_or_one_mouse and a > best_area:
                best_area = a
                best_c = c

        if best_c is None:
            cam.x_position = -1
            cam.y_position = -1
            return

        M = cv2.moments(best_c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cam.x_position = cx
            cam.y_position = cy
        else:
            cam.x_position = -1
            cam.y_position = -1

    def _detect_white_position_contours(self, cam: Camera) -> None:
        """Detects position of white mouse using contours."""
        mask = np.zeros_like(cam.gray_frame, dtype=np.uint8)
        for index, (x1, y1, x2, y2) in enumerate(cam.areas):
            if cam.areas_active[index]:
                roi = cam.gray_frame[y1:y2, x1:x2]
                threshold = cam.thresholds[index]
                _, roi_bin = cv2.threshold(roi, threshold, 255,
                                           cv2.THRESH_BINARY)
                sub = mask[y1:y2, x1:x2]
                np.maximum(sub, roi_bin, out=sub)
                cam.masks[index] = roi_bin
                cam.counts[index] = cv2.countNonZero(roi_bin)
            else:
                cam.masks[index] = -1
                cam.counts[index] = -1

        cam._add_custom_area_mask(mask, invert=False)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cam.x_position = -1
            cam.y_position = -1
            return

        best_c = None
        best_area = 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a >= cam.zero_or_one_mouse and a > best_area:
                best_area = a
                best_c = c

        if best_c is None:
            cam.x_position = -1
            cam.y_position = -1
            return

        M = cv2.moments(best_c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cam.x_position = cx
            cam.y_position = cy
        else:
            cam.x_position = -1
            cam.y_position = -1


class CorridorDetectionBase(CameraDetectionBase):
    """Subclass this in your project's CODE_DIRECTORY to replace CORRIDOR
    detection only; BOX keeps CameraDetectionBase's default behavior.
    Picked up automatically by import_all."""


class BoxDetectionBase(CameraDetectionBase):
    """Subclass this in your project's CODE_DIRECTORY to replace BOX
    detection only; CORRIDOR keeps CameraDetectionBase's default behavior.
    Picked up automatically by import_all."""
