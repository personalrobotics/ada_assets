"""Decompose the wheelchair mesh into convex parts for collision (#14).

MuJoCo collides a mesh geom as its convex hull, and the wheelchair's hull fills the
concave space between its seat, back, and armrests. The collision geometry is therefore
a convex decomposition, made with CoACD from the visual mesh. Run it from the
repository root, outside the robot-code workspace:

    uv run --no-project --python 3.12 --with coacd==1.0.14 --with trimesh==5.1.1 \\
        python scripts/decompose_wheelchair.py

It writes models/assets/wheelchair_collision_NN.stl and prints their count. The
parameters below are recorded in models/assets/PROVENANCE.md.
"""

import pathlib

import coacd
import numpy as np
import trimesh

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "src/ada_assets/models/assets"
THRESHOLD = 0.05  # CoACD concavity threshold: smaller follows the shape more closely
SEED = 0


def main() -> None:
    mesh = trimesh.load(ASSETS / "wheelchair.stl", force="mesh")
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=THRESHOLD, seed=SEED)
    for old in ASSETS.glob("wheelchair_collision_*.stl"):
        old.unlink()
    for i, (vertices, faces) in enumerate(parts):
        trimesh.Trimesh(np.asarray(vertices), np.asarray(faces)).export(ASSETS / f"wheelchair_collision_{i:02d}.stl")
    print(f"{len(parts)} convex parts")


if __name__ == "__main__":
    main()
