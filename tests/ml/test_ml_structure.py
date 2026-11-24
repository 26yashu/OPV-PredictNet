import os
import unittest
from pathlib import Path


class TestMLStructureAndPolicy(unittest.TestCase):
    """Test suite ensuring ML structure and Phase 1 compliance."""

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent.parent
        self.ml_dir = self.root / "ml"

    def test_ml_directories_exist(self):
        """Verify ML directory layout exists."""
        self.assertTrue((self.ml_dir / "data" / "raw").is_dir())
        self.assertTrue((self.ml_dir / "data" / "processed").is_dir())
        self.assertTrue((self.ml_dir / "models" / "saved_models").is_dir())
        self.assertTrue((self.ml_dir / "src" / "features").is_dir())
        self.assertTrue((self.ml_dir / "src" / "models").is_dir())
        self.assertTrue((self.ml_dir / "src" / "evaluation").is_dir())

    def test_no_fabricated_models_committed(self):
        """Ensure no synthetic or unvalidated binary models exist in Phase 1."""
        saved_models_dir = self.ml_dir / "models" / "saved_models"
        joblib_files = list(saved_models_dir.glob("*.joblib"))
        pkl_files = list(saved_models_dir.glob("*.pkl"))
        self.assertEqual(len(joblib_files), 0, "No joblib models should be present in Phase 1.")
        self.assertEqual(len(pkl_files), 0, "No pickle models should be present in Phase 1.")


if __name__ == "__main__":
    unittest.main()
