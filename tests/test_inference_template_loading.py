import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "pytorch_lightning" not in sys.modules:
    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningDataModule = object
    sys.modules["pytorch_lightning"] = lightning

from CryoNetRefine.data.module.inference import load_input


class TemplateInfo:
    def __init__(self, name):
        self.name = name


class Record:
    id = "case"
    templates = [TemplateInfo("A"), TemplateInfo("B"), TemplateInfo("C")]


class TemplateLoadingTest(unittest.TestCase):
    def test_shared_template_npz_is_loaded_once(self):
        structure = object()
        with patch(
            "CryoNetRefine.data.module.inference.StructureV2.load",
            return_value=structure,
        ) as load:
            result = load_input(Record(), template_dir=Path("/tmp/templates"))

        load.assert_called_once_with(Path("/tmp/templates/case.npz"))
        self.assertEqual(set(result.templates), {"A", "B", "C"})
        self.assertTrue(all(value is structure for value in result.templates.values()))


if __name__ == "__main__":
    unittest.main()
