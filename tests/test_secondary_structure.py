from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from CryoNetRefine.data.parse.secondary_structure_restraints import (
    build_default_secondary_structure_restraints,
)
from CryoNetRefine.secondary_structure import detect_secondary_structure


FIXTURE = Path(__file__).parent / "fixtures" / "secondary_structure" / "protein_geometry.pdb"


def _atom_line(serial, atom, residue, chain, resseq, xyz):
    x, y, z = xyz
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def test_public_api_and_detection_modes():
    assert detect_secondary_structure.__module__.startswith("CryoNetRefine.secondary_structure")

    existing = detect_secondary_structure(FIXTURE, mode="existing")
    assert existing.used_existing_protein
    assert len(existing.helices) == 1
    assert len(existing.sheets) == 1

    automatic = detect_secondary_structure(FIXTURE, mode="auto")
    assert automatic.used_existing_protein
    assert automatic.helices[0].source == "existing"

    detected = detect_secondary_structure(FIXTURE, mode="detect")
    assert not detected.used_existing_protein
    assert any(helix.start.chain == "A" for helix in detected.helices)
    assert any(len(sheet.strands) >= 2 for sheet in detected.sheets)


def test_single_beta_strand_option(tmp_path):
    lines = [
        _atom_line(i + 1, "CA", "ALA", "B", i + 1, (i * 3.3, 0.0, 0.0))
        for i in range(6)
    ]
    path = tmp_path / "single_strand.pdb"
    path.write_text("\n".join(lines + ["TER", "END", ""]))

    without_singles = detect_secondary_structure(path, mode="detect")
    with_singles = detect_secondary_structure(
        path,
        mode="detect",
        include_single_strands=True,
    )
    assert without_singles.sheets == []
    assert len(with_singles.sheets) == 1
    assert len(with_singles.sheets[0].strands) == 1


def _write_nucleic_fixture(tmp_path):
    serial = 1
    lines = []

    def add_residue(residue, chain, resseq, atoms):
        nonlocal serial
        for atom, xyz in atoms.items():
            lines.append(_atom_line(serial, atom, residue, chain, resseq, xyz))
            serial += 1
        lines.append("TER")

    cytosine = {
        "N1": (-1.0, 0.0, 0.0),
        "C2": (-1.0, -1.0, 0.0),
        "O2": (0.0, -2.0, 0.0),
        "N3": (0.0, 0.0, 0.0),
        "C4": (0.0, 1.0, 0.0),
        "N4": (0.0, 2.0, 0.0),
        "C5": (1.0, 1.0, 0.0),
        "C6": (1.0, 0.0, 0.0),
    }
    guanine = {
        "N1": (2.89, 0.0, 0.0),
        "C2": (2.77, -1.0, 0.0),
        "N2": (2.77, -2.0, 0.0),
        "N3": (3.8, -1.0, 0.0),
        "C4": (3.8, 0.0, 0.0),
        "C5": (3.8, 1.0, 0.0),
        "C6": (2.96, 1.0, 0.0),
        "O6": (2.96, 2.0, 0.0),
        "N7": (4.8, 1.0, 0.0),
        "C8": (4.8, 0.0, 0.0),
        "N9": (3.8, -0.5, 0.0),
    }
    add_residue("C", "D", 1, cytosine)
    add_residue("G", "E", 1, guanine)

    adenine_ring = {
        "N1": (-1.0, 0.0, 0.0),
        "C2": (-0.5, -0.866, 0.0),
        "N3": (0.5, -0.866, 0.0),
        "C4": (1.0, 0.0, 0.0),
        "C5": (0.5, 0.866, 0.0),
        "C6": (-0.5, 0.866, 0.0),
        "N7": (1.2, 1.5, 0.0),
        "C8": (1.8, 0.8, 0.0),
        "N9": (1.5, 0.0, 0.0),
    }
    add_residue("A", "F", 1, adenine_ring)
    add_residue(
        "A",
        "F",
        2,
        {name: (x, y, z + 3.4) for name, (x, y, z) in adenine_ring.items()},
    )

    path = tmp_path / "nucleic.pdb"
    path.write_text("\n".join(lines + ["END", ""]))
    return path


def test_nucleic_base_pair_and_stacking_detection(tmp_path):
    path = _write_nucleic_fixture(tmp_path)
    result = detect_secondary_structure(path, mode="detect")

    assert any({pair.base1.chain, pair.base2.chain} == {"D", "E"} for pair in result.base_pairs)
    assert any(
        pair.base1.chain == "F" and pair.base2.chain == "F"
        for pair in result.stacking_pairs
    )


def test_protein_and_nucleic_restraint_switches_are_independent(tmp_path):
    protein_only = build_default_secondary_structure_restraints(
        FIXTURE,
        protein_enabled=True,
    )
    assert protein_only["secondary_structure_restraints"]["protein_enabled"]
    assert not protein_only["secondary_structure_restraints"]["nucleic_enabled"]
    assert protein_only["secondary_structure"]["helices"]
    assert protein_only["secondary_structure"]["sheets"]
    assert protein_only["secondary_structure"]["base_pairs"] == []
    assert protein_only["secondary_structure"]["stacking_pairs"] == []

    nucleic_only = build_default_secondary_structure_restraints(
        _write_nucleic_fixture(tmp_path),
        nucleic_enabled=True,
        mode="existing",
    )
    assert not nucleic_only["secondary_structure_restraints"]["protein_enabled"]
    assert nucleic_only["secondary_structure_restraints"]["nucleic_enabled"]
    assert nucleic_only["secondary_structure"]["helices"] == []
    assert nucleic_only["secondary_structure"]["sheets"] == []
    assert nucleic_only["secondary_structure"]["base_pairs"]
    assert nucleic_only["secondary_structure"]["stacking_pairs"]


def test_restraint_builder_and_ramachandran_labels_use_internal_detector():
    from CryoNetRefine.libs.geometry.GeoMetric import GeoMetric

    payload = build_default_secondary_structure_restraints(
        FIXTURE,
        protein_enabled=True,
    )
    assert payload["secondary_structure_restraints"]["enabled"]
    assert payload["secondary_structure"]["helices"]
    assert payload["secondary_structure"]["sheets"]

    metric = GeoMetric.__new__(GeoMetric)
    metric.phi_psi = SimpleNamespace(device="cpu")
    labels = metric.get_secondary_structure_labels(str(FIXTURE))
    assert set(labels.tolist()) == {1, 2}
    assert labels[:12].tolist() == [1] * 12
    assert labels[12:].tolist() == [2] * 12
