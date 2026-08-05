import importlib
import unittest


class PackagingTests(unittest.TestCase):
    def test_package_can_be_imported(self) -> None:
        module = importlib.import_module("smalloo_ramfd_water_knn")
        self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
