"""The bare-hand package's end effector is the hand's canonical grasp frame (#34).

``palm_grasp_site`` must be the frame sstsr's grasp templates constrain: z along the
approach, y along the fingers' opening, x the palm normal, at the palm. This test checks
that against the fingers, as ssrobot runs them:

- the site is where sstsr 4.2.0's ``KinovaJaco2TwoFinger`` puts the palm, in link_6;
- with the fingers open, as the package starts, the two finger pads lie on either side
  of the site's z axis along its y, ahead of the palm;
- a cylinder of ADA's size (r 25 mm, h 100 mm), fixed where the hand's own side grasp
  template puts it relative to ``palm_grasp_site``, is clear of the open hand. Opened
  with ssrobot's ``MujocoRuntime`` and closed with a gripper command, both fingers touch
  the cylinder and the palm does not, and the close stops short of the fingers' target.

A wrong orientation or offset would put the cylinder into the open hand, against the
palm, or outside the fingers' closing plane. The run is made twice and its record,
``artifacts/test_hand_package/hand_grasp.json``, must be byte-identical. Reproduce with
``uv run pytest tests/test_hand_package.py``; ``ADA_ASSETS_ARTIFACTS`` moves the directory.
"""

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from tsr.hands import KinovaJaco2TwoFinger

ssrobot = pytest.importorskip("ssrobot")  # Python 3.11+
pytest.importorskip("ssrobot.mujoco")

SITE = "palm_grasp_site"
PALM = "j2n6s200_link_6"
FINGERS = (
    ("j2n6s200_link_finger_1", "j2n6s200_link_finger_tip_1"),
    ("j2n6s200_link_finger_2", "j2n6s200_link_finger_tip_2"),
)
RADIUS, HEIGHT = 0.025, 0.10  # ADA's cylinder
TEMPLATE = "Side Grasp — mid, roll 0°"
DEADLINE_TICKS = 5000  # 10 s: a bound for failing, not a wait
ARTIFACTS = Path(os.environ.get("ADA_ASSETS_ARTIFACTS") or Path(__file__).parents[1] / "artifacts")


def _transform(position, rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation).reshape(3, 3)
    T[:3, 3] = position
    return T


def _start(package) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The package's model in its profile's start keyframe, as MujocoRuntime opens it."""
    model = mujoco.MjModel.from_xml_path(str(package.root / "ada_hand.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("resting").id)
    mujoco.mj_forward(model, data)
    return model, data


def _grasp_in_palm() -> np.ndarray:
    """The cylinder's pose in the palm frame: the hand's side grasp at its Bw midpoint,
    the cylinder's axis +z from its base, as sstsr's tools/grasp_closure.py poses it."""
    (template,) = [t for t in KinovaJaco2TwoFinger().grasp_cylinder(RADIUS, HEIGHT) if t.name.endswith(TEMPLATE)]
    tsr = template.instantiate(np.eye(4))
    mid = tsr.Bw.mean(axis=1)
    return np.linalg.inv(tsr.to_transform(mid))


def _open_hand(model, data) -> dict:
    """Each finger pad's centre in the site frame, with the fingers open."""
    site = model.site(SITE).id
    T_site = _transform(data.site_xpos[site], data.site_xmat[site])
    pads = {}
    for _, tip in FINGERS:
        body = model.body(tip).id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]
        vertices = []
        for g in geoms:
            mesh = model.geom_dataid[g]
            local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            vertices.append((np.linalg.inv(T_site) @ np.c_[world, np.ones(len(world))].T)[:3].T)
        pads[tip] = np.vstack(vertices).mean(axis=0)
    return pads


def _scene(directory: Path, T_world_cylinder: np.ndarray) -> Path:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T_world_cylinder[:3, :3].flatten())
    position = " ".join(f"{v:.9f}" for v in T_world_cylinder[:3, 3])
    orientation = " ".join(f"{v:.9f}" for v in quat)
    scene = directory / "hand_grasp_scene.xml"
    scene.write_text(
        f"""<mujoco model="hand_grasp">
  <worldbody>
    <body name="cylinder" pos="{position}" quat="{orientation}">
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" pos="0 0 {HEIGHT / 2}"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return scene


def _run(directory: Path) -> dict:
    """Open the package with the cylinder in the open hand, close the gripper, and record
    the contacts and the gripper's result."""
    from ssrobot import GripperCommand, ObservationRequest, RobotContext
    from ssrobot.mujoco import MujocoRuntime

    directory.mkdir(parents=True, exist_ok=True)
    package = ssrobot.load_installed_package("ada_assets.models")
    description = package.description
    hand = description.end_effector("hand")
    model, data = _start(package)
    site = model.site(SITE).id
    T_world_site = _transform(data.site_xpos[site], data.site_xmat[site])
    scene = _scene(directory, T_world_site @ _grasp_in_palm())

    bodies = [PALM, *(b for finger in FINGERS for b in finger)]
    contacts = ObservationRequest(channels=tuple(f"contact.cylinder.{b}" for b in bodies))
    opening = ObservationRequest(channels=("gripper_opening",))

    def touching(ctx) -> dict:
        readings = {r.channel.removeprefix("contact.cylinder."): r.value for r in ctx.observe(contacts).readings}
        return {b: readings[b][0] > 0 for b in bodies}

    runtime = MujocoRuntime(package, scene=scene, contacts=[("cylinder", b) for b in bodies])
    with RobotContext(description, runtime) as ctx:
        open_opening = float(ctx.observe(opening).readings[0].value[0])
        at_start = touching(ctx)
        execution = ctx.submit(GripperCommand(gripper=hand.gripper, opening=0.0))
        ticks = 0
        while not execution.done and ticks < DEADLINE_TICKS:
            ctx.step()
            ticks += 1
        result = execution.status.gripper
        closed = touching(ctx)

    record = {
        "package": description.name,
        "end_effector": {"name": hand.name, "frame": hand.frame},
        "site_in_link_6": {
            "position_m": [round(float(v), 6) for v in model.site_pos[site]],
            "quat_wxyz": [round(float(v), 6) for v in model.site_quat[site]],
        },
        "open_pads_in_site_mm": {k: [round(float(v) * 1e3, 1) for v in p] for k, p in _open_hand(model, data).items()},
        "cylinder": {"radius_m": RADIUS, "height_m": HEIGHT, "template": TEMPLATE},
        "opening_at_start": round(open_opening, 4),
        "touching_at_start": at_start,
        "close": {
            "state": execution.status.state.value,
            "termination": None if result is None else result.termination.value,
            "final_opening": None if result is None else round(result.final_opening, 4),
            "ticks": ticks,
        },
        "touching_when_closed": closed,
    }
    path = directory / "hand_grasp.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def test_palm_grasp_site_is_the_hands_grasp_frame():
    hand = KinovaJaco2TwoFinger()
    first = ARTIFACTS / "test_hand_package" / "run-1"
    record = _run(first)

    # Where sstsr's hand puts the palm, in its canonical orientation.
    site = record["site_in_link_6"]
    assert site["position_m"] == pytest.approx([hand.PALM_X_IN_LINK_6, 0.0, -hand.PALM_OFFSET_FROM_LINK_6], abs=1e-9)
    assert site["quat_wxyz"] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-9)

    # Open: the pads straddle the z axis along y, ahead of the palm. The fingers are mirror
    # images closing in the plane x = 0, so their centres agree with that within 1 mm.
    (x1, y1, z1), (x2, y2, z2) = record["open_pads_in_site_mm"].values()
    assert record["opening_at_start"] == pytest.approx(1.0, abs=0.01)
    assert y1 * y2 < 0 and abs(y1 + y2) < 1.0
    assert abs(x1) < 1.0 and abs(x2) < 1.0
    assert z1 > 0 and z2 > 0

    # The cylinder is clear of the open hand; closed, both fingers hold it and the palm does not.
    assert not any(record["touching_at_start"].values())
    closed = record["touching_when_closed"]
    assert all(any(closed[b] for b in finger) for finger in FINGERS), closed
    assert not closed[PALM]
    assert record["close"]["state"] == "succeeded"
    assert record["close"]["termination"] == "stopped_short"

    second = _run(ARTIFACTS / "test_hand_package" / "run-2")
    assert second == record
    assert (first / "hand_grasp.json").read_bytes() == (
        ARTIFACTS / "test_hand_package" / "run-2" / "hand_grasp.json"
    ).read_bytes()
