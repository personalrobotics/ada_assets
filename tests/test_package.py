"""The ssrobot package's declarations agree with the model (M3: #17, #20, #21).

ssrobot's reference gate opens and drives this package; these tests check, with MuJoCo
alone, the facts it relies on.
"""

import mujoco
import numpy as np
import pytest

from ada_assets import MODELS_DIR

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

PACKAGE = MODELS_DIR.parent
CANONICAL = MODELS_DIR / "ada.xml"


@pytest.fixture(scope="module")
def manifest():
    return tomllib.loads((PACKAGE / "ssrobot.toml").read_text())


@pytest.fixture(scope="module")
def profile():
    return tomllib.loads((PACKAGE / "mujoco.toml").read_text())


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(CANONICAL))


def test_named_configurations_equal_their_keyframes(manifest, model):
    """#20: each named pose is its keyframe of the same name, joint for joint."""
    groups = {g["name"]: g["joints"] for g in manifest["semantics"]["groups"]}
    configurations = manifest["semantics"]["configurations"]
    assert {c["name"] for c in configurations} == {model.key(k).name for k in range(model.nkey)}
    for configuration in configurations:
        key = model.key(configuration["name"]).id
        keyframe = [model.key_qpos[key][model.jnt_qposadr[model.joint(j).id]] for j in groups[configuration["group"]]]
        assert keyframe == pytest.approx(configuration["positions"], abs=1e-9), configuration["name"]


def test_profile_starts_at_a_keyframe(profile, model):
    """#17: the start keyframe exists."""
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, profile["keyframe"]) >= 0


def test_profile_gripper_controls_open_and_close_the_hand(profile, model):
    """#17: the profile's open and closed controls are reachable: each finger tendon
    settles at the control, so the opening ssrobot reads spans 0 to 1."""
    (gripper,) = profile["grippers"]
    data = mujoco.MjData(model)
    for end in ("closed", "open"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(profile["keyframe"]).id)
        for actuator in gripper["actuators"]:
            data.ctrl[model.actuator(actuator["actuator"]).id] = actuator[end]
        for _ in range(1500):
            mujoco.mj_step(model, data)
        for actuator in gripper["actuators"]:
            length = data.actuator_length[model.actuator(actuator["actuator"]).id]
            assert length == pytest.approx(actuator[end], abs=0.01), (end, actuator["actuator"])


def test_tool_wrench_reads_the_weight_beyond_the_sensor(manifest, model):
    """#21: at rest, the force-torque sensor carries the weight of everything mounted
    beyond it."""
    (sensor,) = [s for s in manifest["semantics"]["sensors"] if s["kind"] == "force_torque"]
    site = model.site(sensor["frame"]).id
    body = model.site_bodyid[site]

    def ancestors(b):
        while b:
            yield b
            b = model.body_parentid[b]

    beyond = [b for b in range(model.nbody) if body in ancestors(b)]
    weight = sum(model.body_mass[b] for b in beyond) * -model.opt.gravity[2]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("resting").id)
    for _ in range(500):
        mujoco.mj_step(model, data)
    (force,) = [
        s
        for s in range(model.nsensor)
        if model.sensor_type[s] == mujoco.mjtSensor.mjSENS_FORCE and model.sensor_objid[s] == site
    ]
    measured = np.linalg.norm(data.sensordata[model.sensor_adr[force] : model.sensor_adr[force] + 3])
    assert measured == pytest.approx(weight, rel=0.01)
