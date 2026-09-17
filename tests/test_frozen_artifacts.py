from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_frozen_artifacts.py"

spec = importlib.util.spec_from_file_location("verify_frozen_artifacts", VERIFY)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FrozenArtifactIntegrityTest(unittest.TestCase):
    def test_pre_hidden_manifest_hashes(self) -> None:
        self.assertEqual(module.verify(), [])


if __name__ == "__main__":
    unittest.main()
