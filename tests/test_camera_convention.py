# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Lock the wrist-camera frame convention (ada_assets#6).

MuJoCo cameras view down local -z with +y up (OpenGL); the D415
``color_optical_frame`` is the OpenCV convention (+z forward, +y down). The
``<camera>`` elements must bake in the 180°-about-x flip between the two, or the
cameras point backwards (and render upside-down) relative to the real optics.

These tests compile the assembled ADA model and assert the rendered cameras look
along the real optical axis — a regression here means the convention flip in
``models/camera.xml`` was lost.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from ada_assets.assembly import build_spec, compile_and_init

OPTICAL_SITE = "camera/color_optical_frame"
CAMERAS = ("camera/d415_color", "camera/d415_depth")


@pytest.fixture(scope="module")
def model_data():
    spec = build_spec(with_human=False, with_camera=True, tool="articutool")
    return compile_and_init(spec, tool="articutool")


def _rot(mat9: np.ndarray) -> np.ndarray:
    return np.asarray(mat9).reshape(3, 3)


@pytest.mark.parametrize("camera", CAMERAS)
def test_camera_looks_along_optical_axis(model_data, camera):
    """MuJoCo view axis (-z) aligns with the optical frame forward (+z)."""
    model, data = model_data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    opt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, OPTICAL_SITE)
    assert cam >= 0 and opt >= 0

    view_dir = -_rot(data.cam_xmat[cam])[:, 2]  # MuJoCo camera looks along -z
    opt_fwd = _rot(data.site_xmat[opt])[:, 2]  # OpenCV optical frame: +z forward

    # +1 = looking along the real optical axis; -1 = backwards (the bug).
    assert float(view_dir @ opt_fwd) > 0.999


@pytest.mark.parametrize("camera", CAMERAS)
def test_camera_image_is_upright(model_data, camera):
    """Image-up (+y_cam) opposes optical-down (+y_opt) — not rolled or flipped.

    Together with the view-axis test this pins the full OpenCV->OpenGL flip
    (Rx 180°): -z_cam = +z_opt, +y_cam = -y_opt, +x_cam = +x_opt.
    """
    model, data = model_data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    opt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, OPTICAL_SITE)

    cam_up = _rot(data.cam_xmat[cam])[:, 1]  # MuJoCo camera +y is up
    opt_down = _rot(data.site_xmat[opt])[:, 1]  # OpenCV optical frame +y is down

    assert float(cam_up @ opt_down) < -0.999
