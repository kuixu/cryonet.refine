# checked-out cctbx metal-linking parity

This note is the source of truth for CryoNet's `ideal_distance_strategy == "library"` metal restraints.

## Scope

CryoNet is aligned to the checked-out source tree under `/home/huangfuyao/proj/cctbx_project`, not to any newer or runtime-specific Phenix behavior.

## Layer 1: generic automatic linking

This layer mirrors the monomer-library path in:

- `mmtbx/monomer_library/linking_setup.py`
- `mmtbx/monomer_library/linking_utils.py`
- `mmtbx/monomer_library/linking_mixins.py`
- `mmtbx/monomer_library/bondlength_defaults.py`

Key parity points:

- broad metal classification via `ad_hoc_single_metal_residue_element_types`
- donor rejection for non-linking elements and carbon
- HIS atom exclusions `CE1`, `CD2`, `CB`
- class-based link budgets such as `("common_amino_acid", "metal") -> 2`
- one-link-per-donor for amino acid / RNA-DNA / other classes
- distance priors from `bondlength_defaults.run(...)`
- curated tables for `Na Mg K Ca Mn Fe Co Cu Zn`
- oxygen expansion rules for `GLU ASN GLN SER THR TYR WAT TIP`
- carbonyl oxygen special case
- non-protein fallback for `O/N/S`

Important default nuance:

- `linking_setup.py` has a static `metal_coordination_cutoff = 3.5`
- `pdb_interpretation.py` exposes a user-facing default `metal_coordination_cutoff = 3.0`

CryoNet follows the effective runtime path and keeps the external default at `3.0`, while preserving the internal class- and donor-based logic from the monomer-library layer.

## Layer 2: MCL specialization

This layer mirrors:

- `mmtbx/conformation_dependent_library/mcl.py`
- `mmtbx/conformation_dependent_library/metal_coordination_library.py`
- `mmtbx/conformation_dependent_library/mcl_sf4_coordination.py`

In the checked-out tree, this layer covers:

- Zn tetrahedral coordination
- Fe-S cluster coordination

It does not provide a live generic Mg nucleotide specialization in this source tree, because that hook is commented out in `mcl.py`.

## Non-goals for checked-out parity

The following are not part of checked-out cctbx parity and should stay out of the generic layer unless introduced as an explicitly separate compatibility mode:

- Mg-specific phosphate atom-name patches
- ASP/GLU carboxylate anchor bonds like `CD/CG-metal`
- library-distance deviation filters that reject candidates after assignment
- case-by-case fixes reverse engineered from a particular `.metal.edits` output
