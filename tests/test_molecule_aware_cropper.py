from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CryoNetRefine.data import const
from CryoNetRefine.data.crop.molecule_aware import MoleculeTypeAwareSlidingWindowCropper


class MoleculeAwareCropperTests(unittest.TestCase):
    def test_crop_sequence_positions_are_local_and_chain_complete(self):
        cropper = MoleculeTypeAwareSlidingWindowCropper(crop_size=3)
        protein = const.chain_type_ids["PROTEIN"]
        batch = {
            "token_pad_mask": torch.ones((1, 6), dtype=torch.bool),
            "token_resolved_mask": torch.ones((1, 6), dtype=torch.bool),
            "mol_type": torch.full((1, 6), protein, dtype=torch.long),
            "asym_id": torch.tensor([[10, 10, 20, 20, 30, 30]], dtype=torch.long),
        }

        crops = cropper.get_molecule_type_aware_crops(batch)

        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0][3]["sequences"], {10: [0, 1], 20: [2]})
        self.assertEqual(crops[1][3]["sequences"], {20: [0], 30: [1, 2]})

    def test_crop_planning_does_not_rescan_every_chain_per_crop(self):
        source = Path(
            "CryoNetRefine/data/crop/molecule_aware.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("for asym_id, positions in sequences.items()", source)


if __name__ == "__main__":
    unittest.main()
