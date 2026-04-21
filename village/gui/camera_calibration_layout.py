from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QLabel, QMessageBox

from village.calibration.camera_calibration import CameraCalibration, \
    default_capture_dir, default_result_path
from village.calibration.camera_calibration_grid import make_circle_grid

from village.classes.enums import State
from village.devices.camera import cam_box
from village.gui.layout import Layout
from village.manager import manager
from village.scripts.utils import create_pixmap
from village.settings import settings

if TYPE_CHECKING:
    from village.gui.gui_window import GuiWindow


def _make_distortion_plot(result: dict, width_in: float, height_in: float):
    """Return a matplotlib Figure showing the distortion magnitude map."""
    k = np.array(result["camera_matrix"])
    d = np.array(result["dist_coeffs"])
    w, h = result["image_size_wh"]

    step = max(w, h) // 14
    xs = np.arange(step // 2, w, step, dtype=np.float32)
    ys = np.arange(step // 2, h, step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).reshape(-1, 1, 2)

    distorted = cv2.undistortPoints(pts, k, d, P=k).reshape(-1, 2)
    pts_flat = pts.reshape(-1, 2)
    magnitude = np.hypot(distorted[:, 0] - pts_flat[:, 0],
                         distorted[:, 1] - pts_flat[:, 1])

    mag_grid = magnitude.reshape(len(ys), len(xs))
    k1 = result["dist_coeffs"][0]
    dtype = "barrel" if k1 < 0 else "pincushion"

    dpi = int(settings.get("MATPLOTLIB_DPI"))
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(mag_grid, cmap="hot", origin="upper",
                   extent=[0, w, h, 0], aspect="auto")
    plt.colorbar(im, ax=ax, label="displacement (px)")
    err = result["reprojection_error_px"]
    ax.set_title(f"{dtype}  k1={k1:+.4f}  |  reproj. error: {err:.3f} px")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.tight_layout()
    return fig


class DistortionPlotLayout(Layout):
    """Stacked sub-layout that renders the distortion heatmap."""

    def __init__(self, window: GuiWindow, rows: int, columns: int) -> None:
        super().__init__(window, stacked=True, rows=rows, columns=columns)
        self.rows = rows
        self.columns = columns
        self._draw()

    def _draw(self) -> None:
        self.plot_label = QLabel()
        self.plot_label.setStyleSheet(
            "QLabel {border: 1px solid gray; background-color: white;}"
        )
        self.addWidget(self.plot_label, 0, 0, self.rows, self.columns)
        dpi = int(settings.get("MATPLOTLIB_DPI"))
        self.plot_width = (self.columns * self.column_width - 10) / dpi
        self.plot_height = (self.rows * self.row_height - 5) / dpi

    def update(self, result: dict) -> None:
        try:
            fig = _make_distortion_plot(result, self.plot_width,
                                        self.plot_height)
            pixmap = create_pixmap(fig)
            self.plot_label.setPixmap(pixmap)
        except Exception:
            pass


class CameraCalibrationLayout(Layout):
    """Village GUI tab for camera distortion calibration."""

    def __init__(self, window: GuiWindow) -> None:
        super().__init__(window)
        manager.state = State.MANUAL_MODE
        manager.changing_settings = False
        self._draw()

    def _draw(self) -> None:
        self.camera_calibration_button.setDisabled(True)

        self._calib: CameraCalibration | None = None
        self._capture_dir: Path = default_capture_dir()
        self._result_path: Path = default_result_path()
        self._result: dict | None = None

        descr = "Generate a printable symmetric circle grid. \n" \
                "Print it and verify measure with a ruler."
        self.create_and_add_label("GRID GENERATION", 5, 2, 40, 2, "black",
                                  description=descr)

        descr = "Page width in mm (e.g. 210 for A4)"
        self.create_and_add_label("PAGE W (mm)", 8, 2, 14, 2, "black", bold=False,
                                  description=descr)
        self.page_w_edit = self.create_and_add_line_edit("210", 10, 2, 8, 2,
                                                         self._params_changed)

        descr = "Page height in mm (e.g. 297 for A4)"
        self.create_and_add_label("PAGE H (mm)", 8, 12, 14, 2, "black", bold=False,
                                  description=descr)
        self.page_h_edit = self.create_and_add_line_edit("297", 10, 12, 8, 2,
                                                         self._params_changed)

        descr = "Centre-to-centre distance between circles in mm"
        self.create_and_add_label("SPACING (mm)", 8, 22, 16, 2, "black", bold=False,
                                  description=descr)
        self.spacing_edit = self.create_and_add_line_edit("10", 10, 22, 8, 2,
                                                          self._params_changed)

        descr = "Radius of each printed circle in mm"
        self.create_and_add_label("DOT RADIUS (mm)", 8, 32, 18, 2, "black", bold=False,
                                  description=descr)
        self.dot_radius_edit = self.create_and_add_line_edit("1.5", 10, 32, 8, 2,
                                                             self._params_changed)

        descr = "Empty border around the grid in mm"
        self.create_and_add_label("MARGIN (mm)", 8, 42, 14, 2, "black", bold=False,
                                  description=descr)
        self.margin_edit = self.create_and_add_line_edit("10", 10, 42, 8, 2,
                                                         self._params_changed)

        self.generate_button = self.create_and_add_button(
            "GENERATE GRID PDF",
            13, 2, 30, 2,
            self._generate_grid_clicked,
            "Generate a printable circle grid PDF",
            "powderblue",
        )

        self.grid_status_label = self.create_and_add_label(
            "", 16, 2, 50, 2, "gray", bold=False,
        )

        # descr = "Place the printed grid in corridor and grab ~15 frames."

        # self.create_and_add_label("IMAGE CAPTURE", 19, 2, 40, 2, "black",
        #                           description=descr)

        # self.create_and_add_label("CAMERA", 22, 2, 10, 2, "black", bold=False)
        # self.camera_combo = self.create_and_add_combo_box(
        #     "camera", 24, 2, 20, 2,
        #     ["corridor"],
        #     0,
        #     self._params_changed,
        # )

        self.capture_button = self.create_and_add_button(
            "CAPTURE IMAGE",
            22, 24, 22, 2,
            self._capture_clicked,
            "Save the current camera frame as a calibration image",
            "powderblue",
        )

        self.capture_count_label = self.create_and_add_label(
            "Captured: 0", 25, 2, 30, 2, "black", bold=False,
        )

        self.clear_button = self.create_and_add_button(
            "CLEAR ALL",
            28, 2, 20, 2,
            self._clear_clicked,
            "Delete all captured calibration images",
            "lightcoral",
        )

        self.create_and_add_label(
            "CALIBRATION", 32, 2, 30, 2, "black",
            description="Run calibration once you have >= 10 images.",
        )

        self.run_button = self.create_and_add_button(
            "RUN CALIBRATION",
            35, 2, 30, 2,
            self._run_calibration_clicked,
            "Run OpenCV circle-grid calibration on captured images",
            "powderblue",
        )
        self.run_button.setDisabled(True)

        self.calib_status_label = self.create_and_add_label(
            "", 38, 2, 95, 2, "gray", bold=False,
        )

        self.result_label = self.create_and_add_label(
            "", 41, 2, 95, 6, "black", bold=False,
        )

        self.save_button = self.create_and_add_button(
            "SAVE JSON",
            48, 2, 20, 2,
            self._save_clicked,
            "Save calibration result to camera_calibration.json",
            "powderblue",
        )
        self.save_button.setDisabled(True)

        self.plot_layout = DistortionPlotLayout(self.window, 44, 95)
        self.addLayout(self.plot_layout, 5, 103, 44, 95)

        self._refresh_capture_count()

    def _params_changed(self, *_) -> None:
        pass

    def _generate_grid_clicked(self) -> None:
        try:
            page_w = float(self.page_w_edit.text())
            page_h = float(self.page_h_edit.text())
            spacing = float(self.spacing_edit.text())
            out = Path(settings.get("DATA_DIRECTORY")) / "calibration_grid.pdf"
            dot_radius = float(self.dot_radius_edit.text())
            margin = float(self.margin_edit.text())
            make_circle_grid(page_w_mm=page_w, page_h_mm=page_h,
                             spacing_mm=spacing,
                             circle_radius_mm=dot_radius,
                             margin_mm=margin,
                             out_path=str(out))
            self.grid_status_label.setText(f"Saved: {out}")
            self.grid_status_label.setStyleSheet(
                "QLabel {color: green; font-weight: normal}"
            )
        except Exception:
            self.grid_status_label.setText("Error generating grid")
            self.grid_status_label.setStyleSheet(
                "QLabel {color: red; font-weight: normal}"
            )

    def _capture_clicked(self) -> None:
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(self._capture_dir.glob("*.png")))
            dest = self._capture_dir / f"calib_{n:04d}.png"
            cam_box.take_picture()
            frame = cv2.imread(cam_box.path_picture)
            if frame is None:
                self.calib_status_label.setText("No frame available")
                return
            cv2.imwrite(str(dest), frame)
            self._refresh_capture_count()
        except Exception:
            self.calib_status_label.setText("Calibration capture error?")

    def _clear_clicked(self) -> None:
        reply = QMessageBox.question(
            self.window,
            "CLEAR",
            "Delete all captured calibration images?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            for f in self._capture_dir.glob("*.png"):
                f.unlink()
            self._refresh_capture_count()

    def _run_calibration_clicked(self) -> None:
        if self._calib is not None and self._calib.running:
            return
        try:
            spacing = float(self.spacing_edit.text())
            dot_radius = float(self.dot_radius_edit.text())
        except ValueError:
            self.calib_status_label.setText("Calibration invalid parameters")
            return

        self._calib = CameraCalibration(self._capture_dir, spacing, dot_radius)
        self._calib.run_in_thread()
        self.calib_status_label.setText("Running calibration...")
        self.run_button.setDisabled(True)

    def _save_clicked(self) -> None:
        if self._result is None:
            return
        try:
            self._calib.save(self._result_path)
            self.calib_status_label.setText(f"Saved: {self._result_path}")
        except Exception:
            self.calib_status_label.setText("Error saving result")

    def _refresh_capture_count(self) -> None:
        n = len(list(self._capture_dir.glob("*.png"))) if (
            self._capture_dir.exists()) else 0
        self.capture_count_label.setText(f"Captured: {n}")
        self.run_button.setEnabled(n >= 4)

    def _show_result(self, result: dict) -> None:
        d = result["dist_coeffs"]
        k1 = d[0]
        dtype = "barrel" if k1 < 0 else "pincushion"
        level = ("minimal" if abs(k1) < 0.05
                 else "moderate" if abs(k1) < 0.15
                 else "strong")
        err = result["reprojection_error_px"]
        quality = "good" if err < 1.0 else "high — retake images"
        text = (f"Images used: {result['n_images_used']}  "
                f"failed: {result['n_images_failed']}\n"
                f"Reprojection error: {err:.4f} px  ({quality})\n"
                f"Distortion: {level} {dtype}  (k1={k1:+.4f})\n"
                f"k1={d[0]:+.4f}  k2={d[1]:+.4f}  "
                f"p1={d[2]:+.4f}  p2={d[3]:+.4f}")
        self.result_label.setText(text)
        self.calib_status_label.setText("Calibration complete")
        self.calib_status_label.setStyleSheet(
            "QLabel {color: green; font-weight: normal}")
        self.save_button.setEnabled(True)
        self.plot_layout.update(result)

    def update_gui(self) -> None:
        self.update_status_label_buttons()
        if self._calib is None:
            return
        if self._calib.error:
            self.calib_status_label.setText("Calibration failed, check log")
            self.calib_status_label.setStyleSheet(
                "QLabel {color: red; font-weight: normal}")
            self.run_button.setEnabled(True)
            self._calib = None
            return
        if not self._calib.running and self._calib.result is not None:
            self._result = self._calib.result
            self._show_result(self._result)
            self._calib = None
            self.run_button.setEnabled(True)
