import numpy as np
import unittest

from CryoNetRefine.data.feature.featurizer import _copied_long_tensor


class StructuredStrideTests(unittest.TestCase):
    def test_singleton_packed_field_is_copied_before_torch_conversion(self):
        packed = np.zeros(
            1,
            dtype=np.dtype([("prefix", "u1"), ("res_idx", "<i8")], align=False),
        )
        field = packed["res_idx"]

        self.assertEqual(field.strides, (9,))
        self.assertIs(np.ascontiguousarray(field), field)

        tensor = _copied_long_tensor(field)

        self.assertEqual(tensor.tolist(), [0])
        self.assertEqual(tensor.stride(), (1,))


if __name__ == "__main__":
    unittest.main()
