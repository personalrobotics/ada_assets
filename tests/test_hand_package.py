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
  The cylinder is fixed, so this is the hand before any roll-in: a free object pinched
  at the fingertips is drawn into the palm (``TestGrip``, #46).

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


class TestGrip:
    """The bare hand grips as the real JACO2 hand does (#46): at its rated force, and,
    being underactuated, by rolling a pinched object into an enveloping hold.

    A free 100 g cylinder of ADA's size (r 25 mm, h 100 mm) waits, gravity-compensated and
    held still, at the hand's side grasp template relative to ``palm_grasp_site``, with
    the arm held at ``resting`` by its servos. Then:

    - **close:** the fingers close on it until their motion ends, by the gripper's own
      completion rule (each tendon varies by no more than ``stall_delta`` of the stroke
      over ``stall_dwell_ns``, mujoco.toml). Both fingertips press it and the palm does
      not touch it: the palm invariant holds here, before the roll-in. Each finger presses
      with the force its drive's limit gives at its contacts' moment arm about the
      proximal joint, F / r, and that is the rated 25 N per finger within
      ``_FORCE_TOLERANCE``. jaco2.xml derives F from 25 N at the 72.3 mm arm of the
      fingers' first contact; pressing at full force the distal pads fold and the arm
      grows, so the grip is a little under 25 N;
    - **roll-in:** the cylinder is released (gravity on). The fingers draw it into the
      palm, as the real hand's underactuated fingers do (#46), until the hold settles:
      the cylinder's pose relative to the palm varies by no more than ``_SETTLED_M`` and
      ``_SETTLED_RAD``, 1% of ssrobot's attachment tolerances, over the gripper's 0.5 s
      stall interval: an observed event, bounded by a deadline only for failing. Settled,
      the palm and both fingers touch it;
    - **held:** a torque of ``_TORQUE`` about the line between the two fingers' contact
      centroids, for ``_TORQUE_S``, leaves the settled cylinder within ssrobot's
      attachment tolerances relative to the palm, 10 mm and 0.1 rad. 0.0028 Nm is the
      peak a rigid grip needs during ssrobot's rest calibration motions
      (personalrobotics/ssrobot#239);
    - **overload:** from the same settled hold, a torque about that line ramped at
      ``_RAMP`` turns it more than 0.1 rad before reaching ``_CAP`` times the hold's
      predicted resistance about the line (each contact's torsional coefficient times its
      normal force, plus its sliding coefficient times its normal force times its distance
      from the line): the grip is finite, not a weld. The breakaway torque is recorded.

    The run is made twice and its record, ``artifacts/test_hand_package/grip.json``, must
    be byte-identical.
    """

    _MASS = 0.1  # kg
    _STALL_DELTA, _DWELL_S = 0.00015625, 0.5  # mujoco.toml's completion detector
    _DEADLINE_S = 120.0  # a bound for failing, not a wait
    _STROKE = 1.35  # tendon length at "closed" (mujoco.toml)
    _GRIP_N, _FORCE_TOLERANCE = 25.0, 0.10
    _LEVER_TOLERANCE = 0.05  # measured force against F / r: the lever law, not the spec
    _SETTLED_M, _SETTLED_RAD = 1e-4, 1e-3
    _TORQUE, _TORQUE_S = 0.0028, 2.0
    _RAMP, _CAP = 0.05, 10.0  # Nm/s; times the predicted resistance
    _HELD_M, _HELD_RAD = 0.01, 0.1

    def _model(self, package):
        spec = mujoco.MjSpec.from_file(str(package.root / "ada_hand.xml"))
        model, data = _start(package)
        site = model.site(SITE).id
        T = _transform(data.site_xpos[site], data.site_xmat[site]) @ _grasp_in_palm()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        body = spec.worldbody.add_body(name="cylinder", pos=T[:3, 3], quat=quat, gravcomp=1)
        body.add_freejoint(name="cylinder")
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[RADIUS, HEIGHT / 2, 0], pos=[0, 0, HEIGHT / 2], mass=self._MASS
        )
        return spec.compile()

    def _run(self, package, overload: bool) -> dict:
        model = self._model(package)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("resting").id)
        obj, palm = model.body("cylinder").id, model.body(PALM).id
        joint = model.joint("cylinder")
        address, dof = joint.qposadr[0], joint.dofadr[0]
        data.qpos[address : address + 7] = model.qpos0[address : address + 7]
        still = data.qpos[address : address + 7].copy()
        dt = model.opt.timestep
        window = round(self._DWELL_S / dt)
        tendons = [model.tendon(f"finger_{i}_tendon").id for i in (1, 2)]
        actuators = [model.actuator(f"act_finger_{i}").id for i in (1, 2)]
        side = {model.body(b).id: i for i, finger in enumerate(FINGERS) for b in finger}
        names = {model.body(b).id: b for finger in FINGERS for b in finger} | {palm: PALM}

        def contacts():
            out, force = [], np.zeros(6)
            for i in range(data.ncon):
                c = data.contact[i]
                pair = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                if obj in pair:
                    other = pair[1] if pair[0] == obj else pair[0]
                    mujoco.mj_contactForce(model, data, i, force)
                    out.append(
                        (other, c.pos.copy(), c.frame[:3].copy(), float(force[0]), int(c.dim), c.friction.copy())
                    )
            return out

        def touching(found):
            return {name: any(c[0] == body for c in found) for body, name in names.items()}

        def in_palm():
            R = data.xmat[palm].reshape(3, 3)
            return R.T @ data.xmat[obj].reshape(3, 3), R.T @ (data.xpos[obj] - data.xpos[palm])

        def angle(R):
            return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

        # Close, holding the cylinder still as a table would, until the fingers' motion ends.
        for a in actuators:
            data.ctrl[a] = self._STROKE
        history, t = [], 0.0
        while t < self._DEADLINE_S:
            mujoco.mj_step(model, data)
            data.qpos[address : address + 7] = still
            data.qvel[dof : dof + 6] = 0
            t += dt
            history.append(data.ten_length[tendons].copy())
            if len(history) >= window and np.all(np.ptp(history[-window:], axis=0) / self._STROKE <= self._STALL_DELTA):
                break
        closed_s = t
        mujoco.mj_forward(model, data)
        closed = contacts()
        per_finger, arms = [], []
        for i, (proximal, _) in enumerate(FINGERS):
            j = model.joint(proximal.replace("link", "joint")).id
            mine = [c for c in closed if side.get(c[0]) == i]
            normal = sum(c[3] for c in mine)
            moment = sum(c[3] * abs(np.dot(np.cross(c[1] - data.xanchor[j], c[2]), data.xaxis[j])) for c in mine)
            per_finger.append(normal)
            arms.append(moment / normal if normal else 0.0)
        drive = [
            float(-data.actuator_force[a]) if data.actuator_force[a] < 0 else float(data.actuator_force[a])
            for a in actuators
        ]

        # Release: the fingers roll the cylinder into the palm until the hold settles.
        model.body_gravcomp[obj] = 0
        poses, t = [], 0.0
        while t < self._DEADLINE_S:
            mujoco.mj_step(model, data)
            t += dt
            poses.append(in_palm())
            if len(poses) >= window:
                R0, p0 = poses[-window]
                if all(
                    np.linalg.norm(p - p0) <= self._SETTLED_M and angle(R0.T @ R) <= self._SETTLED_RAD
                    for R, p in poses[-window:]
                ):
                    break
        settled_s = t
        held = contacts()
        centroid = []
        for i in range(2):
            mine = [c for c in held if side.get(c[0]) == i]
            centroid.append(sum(c[3] * c[1] for c in mine) / max(sum(c[3] for c in mine), 1e-9))
        axis = centroid[1] - centroid[0]
        axis /= np.linalg.norm(axis)
        torsional = sum(c[3] * c[5][2] for c in held if c[4] >= 4)
        lever = sum(c[3] * c[5][0] * np.linalg.norm(np.cross(c[1] - centroid[0], axis)) for c in held)

        # A torque about the line between the fingers, relative to the palm: constant, or
        # ramped until the cylinder turns past the attachment tolerance or reaches the cap.
        R0, p0 = in_palm()
        cap = self._CAP * (torsional + lever)
        turn = move = 0.0
        torque, breakaway, k = self._TORQUE, None, 0
        while (torque < cap and breakaway is None) if overload else k * dt < self._TORQUE_S:
            k += 1
            if overload:
                torque = self._RAMP * k * dt
            data.xfrc_applied[obj, 3:] = torque * axis
            mujoco.mj_step(model, data)
            R, p = in_palm()
            turn, move = max(turn, angle(R0.T @ R)), max(move, float(np.linalg.norm(p - p0)))
            if overload and turn > self._HELD_RAD:
                breakaway = torque
        applied = (
            {"ramp_Nm_per_s": self._RAMP, "cap_Nm": round(cap, 5), "breakaway_Nm": breakaway and round(breakaway, 5)}
            if overload
            else {"Nm": self._TORQUE, "s": self._TORQUE_S}
        )
        return {
            "close": {
                "after_s": round(closed_s, 3),
                "drive_force": [round(f, 4) for f in drive],
                "pinch_N": [round(f, 3) for f in per_finger],
                "moment_arm_mm": [round(1e3 * r, 2) for r in arms],
                "touching": touching(closed),
            },
            "settled": {
                "after_s": round(settled_s, 3),
                "touching": touching(held),
                "normal_N": {name: round(sum(c[3] for c in held if c[0] == body), 3) for body, name in names.items()},
                "contact_dims": sorted({c[4] for c in held}),
                "predicted_resistance_Nm": {"torsional": round(torsional, 5), "sliding": round(lever, 5)},
            },
            "torque": {**applied, "peak_turn_rad": round(turn, 4), "peak_move_mm": round(1e3 * move, 3)},
        }

    def _record(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        package = ssrobot.load_installed_package("ada_assets.models")
        record = {
            "cylinder": {"radius_m": RADIUS, "height_m": HEIGHT, "mass_kg": self._MASS, "template": TEMPLATE},
            "rated_grip_N_per_finger": self._GRIP_N,
            "held": self._run(package, overload=False),
            "overload": self._run(package, overload=True),
        }
        (directory / "grip.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return record

    def test_the_hand_grips_at_its_rated_force_and_rolls_the_object_into_a_hold(self):
        first = ARTIFACTS / "test_hand_package" / "grip-1"
        record = self._record(first)
        close, settled, torque = record["held"]["close"], record["held"]["settled"], record["held"]["torque"]

        # Closed, before the roll-in: both fingertips press it, the palm does not touch it,
        # each at F / r from its drive, and that is the rated grip.
        assert close["after_s"] < self._DEADLINE_S, record
        assert all(close["touching"][tip] for _, tip in FINGERS), record
        assert not close["touching"][PALM], record
        for force, drive, arm in zip(close["pinch_N"], close["drive_force"], close["moment_arm_mm"]):
            assert force == pytest.approx(drive / (arm / 1e3), rel=self._LEVER_TOLERANCE), record
            assert force == pytest.approx(self._GRIP_N, rel=self._FORCE_TOLERANCE), record

        # Released, it settles into an enveloping hold: the palm and both fingers touch it.
        assert settled["after_s"] < self._DEADLINE_S, f"the hold never settled ({record})"
        assert settled["touching"][PALM], record
        assert all(any(settled["touching"][b] for b in finger) for finger in FINGERS), record

        # The settled hold resists the representative torque, and an overload slips it.
        assert torque["peak_turn_rad"] < self._HELD_RAD and torque["peak_move_mm"] < 1e3 * self._HELD_M, record
        assert record["overload"]["torque"]["breakaway_Nm"] is not None, f"an overload held like a weld ({record})"

        second = self._record(ARTIFACTS / "test_hand_package" / "grip-2")
        assert second == record
        assert (first / "grip.json").read_bytes() == (
            ARTIFACTS / "test_hand_package" / "grip-2" / "grip.json"
        ).read_bytes()
