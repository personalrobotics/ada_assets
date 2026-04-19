# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Assemble the ADA robot from components via MjSpec.

Composes wheelchair + JACO2 arm (+ optionally seated human and forque)
into a single MuJoCo model. The JACO2 attaches at the wheelchair's
arm_attachment_site. The human is added to worldbody as static scenery.
The forque is welded to link_6 at the xacro FTArmMount transform.

Usage::

    from ada_assets.assembly import assemble_ada

    model, data = assemble_ada()                    # wheelchair + arm + human
    model, data = assemble_ada(with_human=False)    # wheelchair + arm only
    model, data = assemble_ada(with_forque=True)      # + forque fork

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


def _init_forque_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the forque freejoint qpos so it starts at the grasp site."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "forque_attachment_site")
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "forque/fork_freejoint")
    adr = model.jnt_qposadr[jnt_id]
    # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[adr:adr + 3] = data.site_xpos[site_id]
    # Site orientation as quaternion
    mat = data.site_xmat[site_id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    data.qpos[adr + 3:adr + 7] = quat


def assemble_ada(
    *,
    with_human: bool = True,
    with_forque: bool = False,
    with_floor: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Assemble the ADA robot and return (model, data).

    Args:
        with_human: Include seated human (body collision, head, mouth).
        with_forque: Include Articutool (2-DOF) on the arm flange.
        with_floor: Include floor plane and lighting.

    Returns:
        Compiled MuJoCo model and data, with the JACO2 at above_plate keyframe.
    """
    spec = _build_spec(
        with_human=with_human,
        with_forque=with_forque,
        with_floor=with_floor,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    # Apply above_plate keyframe if it exists
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "above_plate")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # Initialize forque freejoint to match the hand's forque_attachment_site
    # so it starts in the correct grasped position (not at origin).
    if with_forque:
        _init_forque_pose(model, data)
        mujoco.mj_forward(model, data)

    return model, data


def _build_spec(
    *,
    with_human: bool = True,
    with_forque: bool = False,
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

    # Add seated human (static scenery — no joints, no physics interaction)
    if with_human:
        human_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "seated.xml"))
        human_spec.meshdir = str(ASSETS_DIR)
        spec.attach(human_spec, prefix="human/", frame=spec.worldbody.add_frame())

    # Forque as a graspable freejoint object on worldbody.
    # A weld equality constraint connects grasp_site to forque_attachment_site,
    # starting enabled. Disable the constraint at runtime to release the tool.
    if with_forque:
        forque_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "forque.xml"))
        forque_spec.meshdir = str(ASSETS_DIR)
        spec.attach(forque_spec, prefix="forque/", frame=spec.worldbody.add_frame())

        # Weld: forque/grasp_site → forque_attachment_site (on link_6)
        weld = spec.add_equality()
        weld.name = "forque_grasp_weld"
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_SITE
        weld.name1 = "forque/grasp_site"
        weld.name2 = "forque_attachment_site"
        weld.active = True

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
    parser.add_argument("--with-forque", action="store_true", help="Include Articutool.")
    parser.add_argument("--save", type=Path, help="Save assembled XML to this path.")
    parser.add_argument("--view", action="store_true", help="Launch mj_viser viewer.")
    args = parser.parse_args()

    model, data = assemble_ada(
        with_human=not args.no_human,
        with_forque=args.with_forque,
    )
    print(f"ADA assembled: nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt} nu={model.nu}")

    if args.save:
        spec = _build_spec(
            with_human=not args.no_human,
            with_forque=args.with_forque,
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
