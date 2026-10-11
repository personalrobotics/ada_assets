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
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

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
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prefix: str,
    tool: str,
) -> None:
    """Set a tool's freejoint qpos so it starts at the grasp site."""
    attach_site = TOOLS[tool]
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        attach_site,
    )
    jnt_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{prefix}/fork_freejoint",
    )
    adr = model.jnt_qposadr[jnt_id]
    # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[adr : adr + 3] = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    data.qpos[adr + 3 : adr + 7] = quat


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

    spec = build_spec(
        with_human=with_human, with_camera=with_camera, tool=tool, tool_tip=tool_tip, with_floor=with_floor
    )
    return compile_and_init(spec, tool=tool)


def _place_tool_in_keyframes(model: mujoco.MjModel, tool: str) -> None:
    """Write each keyframe's tool freejoint pose: at its grasp site for that arm pose."""
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{tool}/fork_freejoint")
    address = model.jnt_qposadr[joint]
    for key in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        _init_tool_pose(model, data, tool, tool)
        model.key_qpos[key][address : address + 7] = data.qpos[address : address + 7]


def compile_and_init(
    spec: mujoco.MjSpec,
    *,
    tool: str | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a spec, apply stow keyframe, and initialize the tool freejoint.

    Scenes that extend the base ADA spec (e.g. by adding a table or plate)
    should call this instead of compiling directly so the post-build init
    (keyframe + tool pose) stays consistent.

    Args:
        spec: A spec returned by ``build_spec`` (possibly with extra bodies added).
        tool: Tool name passed to ``build_spec`` — ``"articutool"``, ``"forque"``,
              or ``None``. Used to initialize the tool freejoint to its grasp site.

    Returns:
        Compiled MuJoCo model and data, with the JACO2 at stow keyframe.
    """
    model = spec.compile()
    data = mujoco.MjData(model)

    # Apply stow keyframe if it exists
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stow")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # Initialize tool freejoint to the grasp site on link_6, in the initial state and in
    # every keyframe. MuJoCo pads keyframes with zeros for joints they don't mention, which
    # would put the tool at the world origin.
    if tool is not None:
        _place_tool_in_keyframes(model, tool)
        _init_tool_pose(model, data, tool, tool)
        mujoco.mj_forward(model, data)

    return model, data


def _collect_body_names(body) -> list[str]:
    """Recursively collect all body names under a parent."""
    names = []
    for child in body.bodies:
        if child.name:
            names.append(child.name)
        names.extend(_collect_body_names(child))
    return names


def _load_tool(tool: str, tool_tip: str) -> tuple[mujoco.MjSpec, list[str]]:
    """A tool's spec, with the articutool tip chosen, and its body names."""
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

    # Collect tool body names BEFORE attach (attach modifies the spec in-place)
    return tool_spec, _collect_body_names(tool_spec.worldbody)


def _attach_tool(spec: mujoco.MjSpec, tool: str, tool_tip: str = "fork") -> None:
    """Attach a tool as a freejoint object with a weld constraint, so it can be released."""
    tool_spec, tool_body_names = _load_tool(tool, tool_tip)
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
    # Stiff weld — the tool is rigidly held by the fingers.
    # Small timeconst + high damping makes the constraint nearly rigid.
    weld.solref = [0.005, 1.0]

    _exclude_hand_from_tool(spec, tool, tool_body_names)


def _mount_tool(spec: mujoco.MjSpec, tool: str, tool_tip: str = "fork") -> None:
    """Mount a tool rigidly in the hand: its body is a child of the attachment site, with
    no free joint or weld. Its grasp site is its root body's origin, so it sits exactly
    where the weld would hold it."""
    tool_spec, tool_body_names = _load_tool(tool, tool_tip)
    for joint in list(tool_spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            tool_spec.delete(joint)
    spec.attach(tool_spec, prefix=f"{tool}/", site=spec.site(TOOLS[tool]))
    _exclude_hand_from_tool(spec, tool, tool_body_names)


def _exclude_hand_from_tool(spec: mujoco.MjSpec, tool: str, tool_body_names: list[str]) -> None:
    """The fingers grip the tool, so their collision meshes and the tool's overlap by
    design; exclude those pairs."""
    hand_bodies = [
        "j2n6s200_link_2",
        "j2n6s200_link_3",
        "j2n6s200_link_4",
        "j2n6s200_link_5",
        "j2n6s200_link_6",
        "j2n6s200_link_finger_1",
        "j2n6s200_link_finger_tip_1",
        "j2n6s200_link_finger_2",
        "j2n6s200_link_finger_tip_2",
    ]
    tool_bodies = [f"{tool}/{n}" for n in tool_body_names]

    for hb in hand_bodies:
        for tb in tool_bodies:
            ex = spec.add_exclude()
            ex.bodyname1 = hb
            ex.bodyname2 = tb


def build_spec(
    *,
    with_human: bool = True,
    with_camera: bool = True,
    tool: str | None = None,
    tool_tip: str = "fork",
    with_floor: bool = True,
    tool_mount: str = "weld",
) -> mujoco.MjSpec:
    """Build the MjSpec for the ADA assembly.

    ``tool_mount`` is "weld" (a free tool held by a weld, which can be released, as the
    real tool is for charging) or "rigid" (the tool mounted in the hand, as in the
    canonical model ssrobot loads).
    """
    if tool_mount not in ("weld", "rigid"):
        raise ValueError(f"tool_mount must be 'weld' or 'rigid', not {tool_mount!r}")
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
    # Attached without a prefix, the arm's main default class would lose its name, and
    # MuJoCo's XML writer would emit an unnamed nested class it cannot read back.
    jaco2_spec.default.name = "jaco2_main"
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

    # Attach the tool (forque or articutool): free and welded, or rigidly mounted.
    if tool is not None:
        (_attach_tool if tool_mount == "weld" else _mount_tool)(spec, tool, tool_tip)

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


GENERATED_HEADER = (
    "<!-- Generated by ada_assets.assembly from the component models; do not edit.\n"
    "     Regenerate with: uv run python -c 'from ada_assets.assembly import generate; generate()' -->\n"
)

GENERATED = {
    # The robot ssrobot loads: wheelchair, JACO2, wrist camera, and the articutool
    # mounted in the hand. No person or floor: those are a scene.
    "ada.xml": dict(with_human=False, with_camera=True, tool="articutool", with_floor=False),
    # Everything, for viewing and simulation: the robot, the seated user, and the floor.
    "scene.xml": dict(with_human=True, with_camera=True, tool="articutool", with_floor=True),
}


def _generated_xml(spec: mujoco.MjSpec, name: str) -> str:
    spec.modelname = name
    spec.meshdir = "assets"
    # Make the output the same however and wherever it was produced:
    # - MuJoCo writes a mesh's content_type only the first time a process writes a spec;
    #   the type follows from the file extension.
    # - Whether a body or geom's zero position or identity orientation is written out
    #   differs by platform. Both are the defaults, so drop them from the worldbody's
    #   bodies, geoms, sites, cameras, and frames, where they are optional. (An inertial
    #   requires its pos.) No default class sets a pose these could be overriding.
    xml = re.sub(r' content_type="[^"]*"', "", spec.to_xml())
    head, body = xml.split("<worldbody>", 1)
    body = re.sub(
        r"<(body|geom|site|camera|frame)\b[^>]*>",
        lambda tag: re.sub(r' (pos="0 0 0"|quat="1 0 0 0")', "", tag.group(0)),
        body,
    )
    return GENERATED_HEADER + head + "<worldbody>" + body


def seated_user_spec() -> mujoco.MjSpec:
    """The seated user and the floor, without the robot: a task scene for the robot."""
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "seated.xml"))
    spec.meshdir = str(ASSETS_DIR)
    _add_floor(spec)
    return spec


def generate(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    """Write the generated models from the component models; return their paths.

    ``ada.xml`` is the canonical robot, ``scene.xml`` the full scene, and
    ``seated_user_scene.xml`` the user and floor alone.
    """
    texts = {
        name: _generated_xml(build_spec(tool_mount="rigid", **options), Path(name).stem)
        for name, options in GENERATED.items()
    }
    texts["seated_user_scene.xml"] = _generated_xml(seated_user_spec(), "seated_user_scene")
    paths = {}
    for name, text in texts.items():
        paths[name] = models_dir / name
        paths[name].write_text(text)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the ADA robot model.")
    parser.add_argument("--no-human", action="store_true", help="Exclude seated human.")
    parser.add_argument("--no-camera", action="store_true", help="Exclude wrist camera.")
    parser.add_argument(
        "--tool",
        choices=sorted(TOOLS.keys()),
        default="articutool",
        help="Tool to attach (default: articutool). Use --no-tool for none.",
    )
    parser.add_argument("--no-tool", action="store_true", help="Exclude tool.")
    parser.add_argument(
        "--tool-tip",
        choices=sorted(TOOL_TIPS.keys()),
        default="fork",
        help="Articutool tip: fork or spoon (default: fork).",
    )
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
        spec = build_spec(
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
