"""Run after installing requirements in a supported Python virtual environment."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from kai_local.data import prepare
from kai_local.training import train
from test_local import pose


@unittest.skipUnless((3, 10) <= sys.version_info[:2] <= (3, 13) and importlib.util.find_spec("tensorflow"), "Requiere Python compatible y TensorFlow instalado; los datos son sintéticos.")
class TensorFlowIntegration(unittest.TestCase):
    def test_train_evaluate_and_export_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for group in range(3):
                file = root / f"capture-{group}.json"
                file.write_text(json.dumps({"category": "vocales", "data": [{"label": label, "landmarks": pose(group * 1000 + kind * 100 + index)} for kind, label in enumerate(("E", "Nulo")) for index in range(10)]}))
                paths.append(file)
            labels, parts, audit = prepare(paths, "vocales", ["E", "Nulo"])
            target, report = train("vocales", "smoke-test", labels, parts, audit, root / "output", epochs=1)
            self.assertEqual(json.loads((target / "status.json").read_text())["state"], "complete")
            self.assertTrue((target / "model.keras").is_file())
            self.assertEqual(json.loads((target / "model.kai.json").read_text())["labels"], labels)
            self.assertEqual(len(report["test_confusion_matrix"]), 2)
            self.assertFalse((root / "output" / ".training.lock").exists())


if __name__ == "__main__":
    unittest.main()
