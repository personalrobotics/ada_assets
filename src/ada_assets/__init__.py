# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo models for the ADA robot-assisted feeding system.

Components (each a standalone XML, composable via <include>):
  - jaco2   — Kinova JACO2 j2n6s200 6-DOF arm with underactuated fingers
  - wheelchair — Permobil wheelchair with arm mount
  - seated  — Seated human (tom visual, body collision, mocap head)
  - fork    — Articutool fork end-effector (2-DOF: tilt + roll)

Pre-assembled:
  - ada     — Wheelchair + JACO2 + human (+ optionally fork)
  - scene   — Full scene with floor, lighting, camera

All meshes live in models/assets/ (single flat directory, no duplication).
Component XMLs have no meshdir — the assembly or scene sets it once.

Usage::

    import mujoco
    from ada_assets import MODELS_DIR, ASSETS_DIR, get_model_path

    # Load the full scene
    model = mujoco.MjModel.from_xml_path(str(get_model_path("scene")))

    # Custom assembly via MjSpec
    spec = mujoco.MjSpec.from_file(str(get_model_path("jaco2")))
    spec.meshdir = str(ASSETS_DIR)
"""

from pathlib import Path

__version__ = "0.1.0"

MODELS_DIR = Path(__file__).parent / "models"
ASSETS_DIR = MODELS_DIR / "assets"

# Available component models
COMPONENTS = ["jaco2", "wheelchair", "seated", "articutool"]

# Available assembled models
ASSEMBLIES = ["ada", "scene"]

AVAILABLE_MODELS = COMPONENTS + ASSEMBLIES


def get_model_path(name: str) -> Path:
    """Get the path to a named model XML.

    Args:
        name: Model name — one of the component or assembly names.

    Returns:
        Path to the XML file.

    Raises:
        FileNotFoundError: If the model doesn't exist.

    Example::

        >>> path = get_model_path("scene")
        >>> model = mujoco.MjModel.from_xml_path(str(path))
    """
    path = MODELS_DIR / f"{name}.xml"
    if not path.exists():
        available = sorted(p.stem for p in MODELS_DIR.glob("*.xml"))
        raise FileNotFoundError(f"Model '{name}' not found at {path}. Available: {available}")
    return path


def get_component_path(component: str) -> Path:
    """Get path to a component XML for custom assembly.

    Args:
        component: One of "jaco2", "wheelchair", "seated", "articutool".

    Returns:
        Path to the component XML file.
    """
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component '{component}'. Available: {COMPONENTS}")
    return get_model_path(component)
