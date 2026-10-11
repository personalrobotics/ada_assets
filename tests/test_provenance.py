"""Every shipped mesh has recorded provenance and a license (#11)."""

import re

from ada_assets import ASSETS_DIR

PROVENANCE = ASSETS_DIR / "PROVENANCE.md"


def _rows():
    rows = {}
    for line in PROVENANCE.read_text().splitlines():
        match = re.match(r"\| `([^`]+\.stl)` \|", line)
        if match:
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert match.group(1) not in rows, f"{match.group(1)} is listed twice"
            rows[match.group(1)] = cells
    return rows


def test_provenance_lists_exactly_the_shipped_meshes():
    shipped = sorted(p.name for p in ASSETS_DIR.glob("*.stl"))
    assert sorted(_rows()) == shipped


def test_every_mesh_names_a_source_change_origin_and_license():
    incomplete = [name for name, cells in _rows().items() if len(cells) != 5 or not all(cells)]
    assert incomplete == []


def test_every_cited_license_text_ships_beside_the_meshes():
    cited = set(re.findall(r"`(licenses/[^`]+)`", PROVENANCE.read_text()))
    assert cited and sorted(c for c in cited if not (ASSETS_DIR / c).is_file()) == []
