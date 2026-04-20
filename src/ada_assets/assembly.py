# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Assemble the ADA robot from components via MjSpec.

Composes wheelchair + JACO2 arm (+ optionally seated human and tool)
into a single MuJoCo model. The JACO2 attaches at the wheelchair's
arm_attachment_site. The human is added to worldbody as static scenery.
Tools (forque or articutool) are freejoint objects welded to link_6.

Usage::

    from ada_assets.assembly import assemble_ada

    model, data = assemble_ada()                        # wheelchair + arm + human + articutool
    model, data = assemble_ada(tool="forque")           # swap to forque (rigid fork)
    model, data = assemble_ada(tool_tip="spoon")        # articutool with spoon
    model, data = assemble_ada(tool=None)               # no tool

Or generate the XML::

    uv run python -m ada_assets.assembly --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

from ada_assets import ASSETS_DIR, MODELS_DIR

TOOLS = {
    "forque": "forque_attachment_site",
    "articutool": "articutool_attachment_site",
}

# Articutool tool tip configs (from articutool_ros2 tool_tip_configurations.yaml).
# xyz in meters (source is in mm, pre-scaled here).
TOOL_TIPS = {
    "fork": {"mesh": "fork_tool", "tip_pos": "0.007 0 0.07"},
    "spoon": {"mesh": "spoon_tool", "tip_pos": "-0.0055 0 0.0805"},
}


def _init_tool_pose(
    model: mujoco.MjModel, data: mujoco.MjData, prefix: str, tool: str,
) -> None:
    """Set a tool's freejoint qpos so it starts at the grasp site."""
    attach_site = TOOLS[tool]
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, attach_site,
    )
    jnt_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/fork_freejoint",
    )
    adr = model.jnt_qposadr[jnt_id]
    # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[adr:adr + 3] = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    data.qpos[adr + 3:adr + 7] = quat


def assemble_ada(
    *,
    with_human: bool = True,
    with_camera: bool = True,
    tool: str = "articutool",
    tool_tip: str = "fork",
    with_floor: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Assemble the ADA robot and return (model, data).

    Args:
        with_human: Include seated human (body collision, head, mouth).
        with_camera: Include wrist camera (D415 + Jetson Nano enclosure).
        tool: Tool to attach — "articutool" (2-DOF, default), "forque" (rigid),
              or None for no tool.
        tool_tip: For articutool only — "fork" (default) or "spoon".
        with_floor: Include floor plane and lighting.

    Returns:
        Compiled MuJoCo model and data, with the JACO2 at stow keyframe.
    """
    if tool is not None and tool not in TOOLS:
        raise ValueError(f"Unknown tool '{tool}'. Available: {sorted(TOOLS.keys())}")
    if tool_tip not in TOOL_TIPS:
        raise ValueError(f"Unknown tool_tip '{tool_tip}'. Available: {sorted(TOOL_TIPS.keys())}")

    spec = _build_spec(with_human=with_human, with_camera=with_camera, tool=tool, tool_tip=tool_tip, with_floor=with_floor)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Apply stow keyframe if it exists
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stow")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # Initialize tool freejoint to the grasp site on link_6.
    if tool is not None:
        _init_tool_pose(model, data, tool, tool)
        mujoco.mj_forward(model, data)

    return model, data


def _attach_tool(spec: mujoco.MjSpec, tool: str, tool_tip: str = "fork") -> None:
    """Attach a tool as a freejoint object with a weld constraint."""
    tool_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / f"{tool}.xml"))
    tool_spec.meshdir = str(ASSETS_DIR)

    # For articutool, swap the tool tip mesh and position if not fork.
    if tool == "articutool" and tool_tip != "fork":
        tip_cfg = TOOL_TIPS[tool_tip]
        # Swap the visual and collision geoms on fork_tine body
        tine_body = tool_spec.body("fork_tine")
        for geom in tine_body.geoms:
            geom.meshname = tip_cfg["mesh"]
        # Update fork_tip site position
        for site in tine_body.sites:
            if site.name == "fork_tip":
                site.pos = [float(x) for x in tip_cfg["tip_pos"].split()]

    spec.attach(tool_spec, prefix=f"{tool}/", frame=spec.worldbody.add_frame())

    # Weld: tool grasp_site → attachment site on link_6.
    # Disable at runtime to release the tool.
    attach_site = TOOLS[tool]
    weld = spec.add_equality()
    weld.name = f"{tool}_grasp_weld"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_SITE
    weld.name1 = f"{tool}/grasp_site"
    weld.name2 = attach_site
    weld.active = True


def _build_spec(
    *,
    with_human: bool = True,
    with_camera: bool = True,
    tool: str | None = None,
    tool_tip: str = "fork",
    with_floor: bool = True,
) -> mujoco.MjSpec:
    """Build the MjSpec for the ADA assembly."""
    # Start with the wheelchair as the base
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "wheelchair.xml"))
    spec.meshdir = str(ASSETS_DIR)
    spec.compiler.degree = False  # use radians
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    # Attach JACO2 at the wheelchair's arm_attachment_site
    jaco2_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "jaco2.xml"))
    jaco2_spec.meshdir = str(ASSETS_DIR)
    arm_site = spec.site("arm_attachment_site")
    spec.attach(jaco2_spec, prefix="", site=arm_site)

    # Attach wrist camera at camera_attachment_site on link_6
    if with_camera:
        camera_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "camera.xml"))
        camera_spec.meshdir = str(ASSETS_DIR)
        cam_site = spec.site("camera_attachment_site")
        spec.attach(camera_spec, prefix="camera/", site=cam_site)

    # Add seated human (static scenery — no joints, no physics interaction)
    if with_human:
        human_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "seated.xml"))
        human_spec.meshdir = str(ASSETS_DIR)
        spec.attach(human_spec, prefix="human/", frame=spec.worldbody.add_frame())

    # Attach tool (forque or articutool) as freejoint + weld
    if tool is not None:
        _attach_tool(spec, tool, tool_tip)

    # Floor + lighting
    if with_floor:
        _add_floor(spec)

    return spec


def _add_floor(spec: mujoco.MjSpec) -> None:
    """Add floor plane and lighting."""
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3, 3, 0.05]
    floor.rgba = [0.9, 0.9, 0.9, 1]
    floor.contype = 1
    floor.conaffinity = 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the ADA robot model.")
    parser.add_argument("--no-human", action="store_true", help="Exclude seated human.")
    parser.add_argument("--no-camera", action="store_true", help="Exclude wrist camera.")
    parser.add_argument("--tool", choices=sorted(TOOLS.keys()), default="articutool",
                        help="Tool to attach (default: articutool). Use --no-tool for none.")
    parser.add_argument("--no-tool", action="store_true", help="Exclude tool.")
    parser.add_argument("--tool-tip", choices=sorted(TOOL_TIPS.keys()), default="fork",
                        help="Articutool tip: fork or spoon (default: fork).")
    parser.add_argument("--save", type=Path, help="Save assembled XML to this path.")
    parser.add_argument("--view", action="store_true", help="Launch mj_viser viewer.")
    args = parser.parse_args()

    tool = None if args.no_tool else args.tool
    model, data = assemble_ada(
        with_human=not args.no_human,
        with_camera=not args.no_camera,
        tool=tool,
        tool_tip=args.tool_tip,
    )
    print(f"ADA assembled: nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt} nu={model.nu}")

    if args.save:
        spec = _build_spec(
            with_human=not args.no_human,
            with_camera=not args.no_camera,
            tool=tool,
            tool_tip=args.tool_tip,
        )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        spec.to_file(str(args.save))
        print(f"Saved to {args.save}")

    if args.view:
        from mj_viser import MujocoViewer

        viewer = MujocoViewer(model, data, label="ADA")
        viewer.launch()

    return 0


if __name__ == "__main__":
    sys.exit(main())
