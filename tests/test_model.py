"""Guards for the model's physical validity (#10).

Each property ADA's models must have is a test. Properties the model does not have yet
are strict expected failures that name the issue fixing them: the fix turns the test
green and removes the mark, and an accidental pass is reported.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from ada_assets import MODELS_DIR
from ada_assets.assembly import assemble_ada

SCENE = MODELS_DIR / "scene.xml"
KEYFRAMES = {"above_plate", "resting", "staging", "stow"}
ARM_CHAIN = [
    "j2n6s200_link_base",
    "j2n6s200_link_1",
    "j2n6s200_link_2",
    "j2n6s200_link_3",
    "j2n6s200_link_4",
    "j2n6s200_link_5",
    "j2n6s200_link_6",
]
ASSEMBLIES = {
    "default": {},
    "spoon": {"tool_tip": "spoon"},
    "forque": {"tool": "forque"},
    "no tool": {"tool": None},
    "no camera": {"with_camera": False},
    "no human": {"with_human": False},
    "no floor": {"with_floor": False},
}


def _issue(number: int, reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(
        strict=True, reason=f"{reason} (https://github.com/personalrobotics/ada_assets/issues/{number})"
    )


@pytest.fixture(scope="module")
def scene():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _name(model, kind, i):
    return mujoco.mj_id2name(model, kind, i)


def _contacts(model, data):
    """The body pairs in contact, by name."""
    return sorted(
        {
            tuple(
                sorted(
                    (
                        _name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]),
                        _name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]),
                    )
                )
            )
            for c in data.contact[: data.ncon]
        }
    )


def _out_of_limits(model, qpos):
    out = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            q = qpos[model.jnt_qposadr[j]]
            low, high = model.jnt_range[j]
            if not low <= q <= high:
                out.append(f"{_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)}={q:.4f} not in [{low:.4f}, {high:.4f}]")
    return out


def _keyframe_contacts(model):
    data = mujoco.MjData(model)
    found = {}
    for key in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        if data.ncon:
            found[_name(model, mujoco.mjtObj.mjOBJ_KEY, key)] = _contacts(model, data)
    return found


# -- Loading -----------------------------------------------------------------------


def test_every_model_file_is_well_formed_xml():
    """Every model parses with a strict XML parser, not only MuJoCo's lenient one, so other
    tools (ssrobot, URDF and MJCF readers) can read it too."""
    malformed = []
    for path in sorted(MODELS_DIR.rglob("*.xml")):
        try:
            ET.parse(path)
        except ET.ParseError as e:
            malformed.append(f"{path.relative_to(MODELS_DIR)}: {e}")
    assert malformed == []


def test_canonical_scene_compiles_with_its_keyframes(scene):
    assert {_name(scene, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(scene.nkey)} == KEYFRAMES


@pytest.mark.parametrize("variant", ASSEMBLIES)
def test_every_assembly_variant_compiles_with_its_keyframes(variant):
    model, _ = assemble_ada(**ASSEMBLIES[variant])
    assert {_name(model, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(model.nkey)} == KEYFRAMES


# -- Keyframes ---------------------------------------------------------------------


def test_every_keyframe_is_within_joint_limits(scene):
    problems = {
        _name(scene, mujoco.mjtObj.mjOBJ_KEY, k): _out_of_limits(scene, scene.key_qpos[k]) for k in range(scene.nkey)
    }
    assert {k: v for k, v in problems.items() if v} == {}


@_issue(16, "scene.xml mounts the arm at the world origin, inside the wheelchair")
def test_canonical_keyframes_are_free_of_contact(scene):
    assert _keyframe_contacts(scene) == {}


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(name, marks=_issue(13, "keyframes leave the attached tool at the world origin"))
        if ASSEMBLIES[name].get("tool", "articutool") is not None
        else name
        for name in ASSEMBLIES
    ],
)
def test_assembled_keyframes_are_free_of_contact(variant):
    model, _ = assemble_ada(**ASSEMBLIES[variant])
    assert _keyframe_contacts(model) == {}


@_issue(13, "every keyframe's ctrl is zero, so the joint servos drive the arm toward zero")
def test_every_keyframe_holds_still(scene):
    """From each keyframe, the joint servos hold the arm still for a second, within
    1 mrad."""
    problems = []
    data = mujoco.MjData(scene)
    for key in range(scene.nkey):
        key_name = _name(scene, mujoco.mjtObj.mjOBJ_KEY, key)
        mujoco.mj_resetDataKeyframe(scene, data, key)
        for _ in range(500):
            mujoco.mj_step(scene, data)
        for a in range(scene.nu):
            if scene.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            address = scene.jnt_qposadr[scene.actuator_trnid[a][0]]
            drift = abs(data.qpos[address] - scene.key_qpos[key][address])
            if drift > 1e-3:
                problems.append(f"{key_name}: {_name(scene, mujoco.mjtObj.mjOBJ_ACTUATOR, a)} drifted {drift:.4g}")
    assert problems == []


@_issue(15, "the default state puts joints 2 and 3 outside their limits, in self-collision")
def test_default_state_is_valid(scene):
    data = mujoco.MjData(scene)
    mujoco.mj_forward(scene, data)
    assert (_out_of_limits(scene, data.qpos), _contacts(scene, data)) == ([], [])


# -- Actuators and contact ---------------------------------------------------------


def test_joint_servo_control_ranges_equal_joint_ranges(scene):
    """A position servo on a limited joint can command exactly the joint's range: never
    less, so no reachable pose is clipped, and never more."""
    mismatched = []
    for a in range(scene.nu):
        if scene.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = scene.actuator_trnid[a][0]
        if not scene.jnt_limited[j]:
            continue
        ctrl = tuple(scene.actuator_ctrlrange[a]) if scene.actuator_ctrllimited[a] else None
        joint_range = tuple(scene.jnt_range[j])
        if ctrl is None or max(abs(c - r) for c, r in zip(ctrl, joint_range)) > 1e-6:
            mismatched.append(f"{_name(scene, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}: {ctrl} vs {joint_range}")
    assert mismatched == []


def test_adjacent_arm_links_are_excluded_from_contact(scene):
    """Neighbouring links touch at their joints by construction, so MuJoCo must not
    count those contacts."""
    excluded = {
        tuple(
            sorted(
                (_name(scene, mujoco.mjtObj.mjOBJ_BODY, s >> 16), _name(scene, mujoco.mjtObj.mjOBJ_BODY, s & 0xFFFF))
            )
        )
        for s in scene.exclude_signature
    }
    pairs = {tuple(sorted(p)) for p in zip(ARM_CHAIN, ARM_CHAIN[1:])}
    pairs |= {tuple(sorted(("j2n6s200_link_6", f"j2n6s200_link_finger_{i}"))) for i in (1, 2)} | {
        tuple(sorted((f"j2n6s200_link_finger_{i}", f"j2n6s200_link_finger_tip_{i}"))) for i in (1, 2)
    }
    assert sorted(pairs - excluded) == []


def test_closed_gripper_touches_only_fingertip_to_fingertip():
    """Closing the empty gripper, with the arm held at a keyframe, makes no contact other
    than the two fingertips meeting, as on the real hand."""
    model, data = assemble_ada(tool=None)
    mujoco.mj_resetDataKeyframe(model, data, model.key("resting").id)
    for a in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name.startswith("act_finger"):
            data.ctrl[a] = model.actuator_ctrlrange[a][1]
        elif model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a][0]]]
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert _contacts(model, data) in (
        [],
        [("j2n6s200_link_finger_tip_1", "j2n6s200_link_finger_tip_2")],
    )
    assert np.all(np.isfinite(data.qpos))
