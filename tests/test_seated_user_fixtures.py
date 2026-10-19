"""The seated user as an ssrobot scene (#37).

``models/seated_user_fixtures.xml`` is composed with the canonical robot as ssrobot
composes a scene: attached at the world origin with no name prefix. The MuJoCo tests
check the composition itself; the ssrobot test, which runs where ssrobot and its MuJoCo
extra are installed and is skipped otherwise, checks the runtime, the scene identity, and
the planning scene.
"""

import shutil

import mujoco
import numpy as np
import pytest

from ada_assets import MODELS_DIR

ROBOT = MODELS_DIR / "ada.xml"
SCENE = MODELS_DIR / "seated_user_fixtures.xml"
FIXTURES = {"user_body", "head"}
ARM = [f"j2n6s200_joint_{i}" for i in range(1, 7)]
NAMED = ("resting", "staging", "above_plate", "stow")


def _composed() -> mujoco.MjModel:
    robot = mujoco.MjSpec.from_file(str(ROBOT))
    robot.attach(mujoco.MjSpec.from_file(str(SCENE)), prefix="", frame=robot.worldbody.add_frame())
    return robot.compile()


@pytest.fixture(scope="module")
def model():
    return _composed()


def _contacts(model, data):
    body = lambda g: model.body(model.geom_bodyid[g]).name  # noqa: E731
    return {tuple(sorted((body(c.geom1), body(c.geom2)))) for c in data.contact[: data.ncon]}


def _fork_tip_at(model, where: str) -> list[float]:
    """An arm configuration with the fork tip at ``where``, reached from ``staging`` by
    damped least squares on the tip position: ``"head"``, the center of the head's
    collision box, or ``"mouth"``, 5 mm in front of the mouth site."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("staging").id)
    mujoco.mj_forward(model, data)
    if where == "head":
        target = data.geom_xpos[model.geom("head_collision").id].copy()
    else:
        mouth = data.site("mouth")
        target = mouth.xpos + 0.005 * mouth.xmat.reshape(3, 3)[:, 0]
    tip = model.site("articutool/fork_tip").id
    joints = [model.joint(j) for j in ARM]
    for _ in range(100):
        error = target - data.site_xpos[tip]
        if np.linalg.norm(error) < 1e-3:
            break
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, None, tip)
        jacobian = jacobian[:, [j.dofadr[0] for j in joints]]
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(3), error)
        for joint, dq in zip(joints, step):
            q = data.qpos[joint.qposadr[0]] + 0.5 * dq
            data.qpos[joint.qposadr[0]] = np.clip(q, *joint.range) if model.jnt_limited[joint.id] else q
        mujoco.mj_forward(model, data)
    assert np.linalg.norm(target - data.site_xpos[tip]) < 1e-3
    return [float(data.qpos[j.qposadr[0]]) for j in joints]


def test_the_user_composes_with_the_robot_as_fixtures(model):
    """Every top-level scene body is a named fixture: no joints, and not a mocap body."""
    scene = mujoco.MjModel.from_xml_path(str(SCENE))
    top = {scene.body(b).name for b in range(1, scene.nbody) if scene.body_parentid[b] == 0}
    assert top == FIXTURES
    for name in FIXTURES:
        body = model.body(name)
        assert body.jntnum[0] == 0 and body.mocapid[0] == -1, name


def test_the_user_is_where_and_what_the_seated_user_scene_has(model):
    """The user's bodies, geoms, and mouth site are those of ``seated_user_scene.xml``:
    same poses, meshes, sizes, groups, and collision bits."""
    reference = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "seated_user_scene.xml"))
    datas = {m: mujoco.MjData(m) for m in (model, reference)}
    for m, d in datas.items():
        mujoco.mj_forward(m, d)

    def describe(m):
        d = datas[m]
        bodies = {n: (d.body(n).xpos.round(9).tolist(), d.body(n).xquat.round(9).tolist()) for n in FIXTURES}
        geoms = {}
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in FIXTURES:
                mesh = m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
                geoms[m.geom(g).name] = (
                    int(m.geom_type[g]),
                    mesh,
                    m.geom_size[g].round(9).tolist(),
                    d.geom_xpos[g].round(9).tolist(),
                    int(m.geom_group[g]),
                    int(m.geom_contype[g]),
                    int(m.geom_conaffinity[g]),
                )
        return bodies, geoms, d.site("mouth").xpos.round(9).tolist()

    assert describe(model) == describe(reference)


def _contacts_at(model, q):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("staging").id)
    for joint, value in zip(ARM, q):
        data.qpos[model.joint(joint).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return _contacts(model, data)


def test_named_configurations_and_the_fork_at_the_mouth_are_free_of_contact(model):
    """The user is no obstacle at the named configurations, nor with the fork tip at the
    mouth: the face wall and body envelope do not collide (see the scene's header), and
    the mouth lies outside the head's collision box."""
    found = {
        name: _contacts_at(model, model.key(name).qpos[[model.joint(j).qposadr[0] for j in ARM]]) for name in NAMED
    }
    found["mouth"] = _contacts_at(model, _fork_tip_at(model, "mouth"))
    assert {name: contacts for name, contacts in found.items() if contacts} == {}


def test_the_fork_through_the_head_is_a_contact_with_the_head(model):
    contacts = _contacts_at(model, _fork_tip_at(model, "head"))
    assert contacts and all("head" in pair for pair in contacts)
    assert ("articutool/fork_tine", "head") in contacts


def test_ssrobot_opens_identifies_and_plans_on_the_scene(model, tmp_path):
    """ssrobot composes the scene with ADA's package: the user's bodies are fixtures, a
    copy of the scene's tree elsewhere has the same identity and materializes as the same
    world, and a planning scene accepts the named configurations and the fork at the mouth,
    and refuses the fork through the head, naming the head."""
    pytest.importorskip("ssrobot.mujoco")
    pytest.importorskip("sscbirrt")
    from ssrobot import RobotContext, load_package
    from ssrobot.mujoco import MujocoRuntime, materialize_world_from

    package = load_package(MODELS_DIR.parent)
    runtime = MujocoRuntime(package, scene=SCENE)
    with RobotContext(package.description, runtime) as ctx:
        assert set(ctx.info.fixtures) == FIXTURES and ctx.info.objects == ()
        identity, world = runtime.mapping.scene_identity, runtime.world()
        files = sorted(f.path for f in runtime.mapping.scene_files)
        planning = ctx.planning_scene(ctx.snapshot(), "arm")
        configurations = {c.name: c.positions for c in package.description.configurations}
        for name in NAMED:
            assert planning.is_valid(configurations[name]), (name, planning.contacts(configurations[name]))
        assert planning.is_valid(_fork_tip_at(model, "mouth"))
        through_head = _fork_tip_at(model, "head")
        assert not planning.is_valid(through_head)
        contacts = planning.contacts(through_head)
        assert contacts and all("head" in (c.first, c.second) for c in contacts)
        assert any({c.first, c.second} == {"head", "articutool/fork_tine"} for c in contacts)

    copy = tmp_path / "elsewhere" / "models"
    for path in files:
        (copy / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODELS_DIR / path, copy / path)
    moved = MujocoRuntime(package, scene=copy / SCENE.name)
    with RobotContext(package.description, moved):
        assert moved.mapping.scene_identity == identity and moved.world() == world
    materialize_world_from(package, world=world, scene=copy / SCENE.name)
