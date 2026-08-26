from __future__ import annotations

import argparse
import sys

from .detector import detect_secondary_structure
from .output import as_json, as_phil, as_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect protein and nucleic-acid secondary structure.")
    parser.add_argument("structure", help="Input PDB or mmCIF file")
    parser.add_argument(
        "--mode",
        choices=["auto", "detect", "existing"],
        default="auto",
        help="Protein mode: auto uses HELIX/SHEET records if present, detect forces detection, existing only reads records",
    )
    parser.add_argument(
        "--format",
        choices=["json", "records", "phil"],
        default="json",
        help="Output format",
    )
    parser.add_argument("--no-nucleic", action="store_true", help="Skip nucleic-acid base pair and stacking detection")
    parser.add_argument(
        "--include-single-strands",
        action="store_true",
        help="Include unpaired beta strands as single-strand SHEET records, matching find_ss_from_ca include_single_strands=True",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = detect_secondary_structure(
        args.structure,
        mode=args.mode,
        detect_nucleic=not args.no_nucleic,
        include_single_strands=args.include_single_strands,
    )
    if args.format == "json":
        text = as_json(result)
    elif args.format == "phil":
        text = as_phil(result)
    else:
        text = as_records(result)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
