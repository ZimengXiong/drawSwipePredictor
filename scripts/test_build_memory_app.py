"""Check complete model packaging without opening a graphical application."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_memory_app import model_files


class ModelPackagingTests(unittest.TestCase):
    def test_spelling_and_auxiliary_components_are_required_but_training_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["encoder.bin", "words.bin", "auxiliary/encoder.bin", "auxiliary/words.bin",
                     "spelling.bin", "spelling/encoder.bin", "additional1/encoder.bin", "additional1/words.bin",
                     "geometry-ranker.json"]
            for name in files + ["training-checkpoint.pt"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            manifest = {"encoder_sha256": digest("encoder.bin"), "words_sha256": digest("words.bin"),
                        "auxiliary_weight": 0.25, "additional_models": [{"weight": 0.25}], "spelling": {"sha256": digest("spelling.bin"),
                        "context_encoder_sha256": digest("spelling/encoder.bin")},
                        "geometry_ranker": {"sha256": digest("geometry-ranker.json")}}
            (root / "additional1/manifest.json").write_text(json.dumps({"encoder_sha256": digest("additional1/encoder.bin"),
                                                                     "words_sha256": digest("additional1/words.bin")}))
            (root / "manifest.json").write_text(json.dumps(manifest))
            inventory = model_files(root)
            self.assertEqual(set(inventory), {"manifest.json", "additional1/manifest.json", *files})
            self.assertNotIn("training-checkpoint.pt", inventory)
            (root / "spelling/encoder.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                model_files(root)
            (root / "spelling/encoder.bin").write_bytes(b"spelling/encoder.bin")
            (root / "geometry-ranker.json").write_bytes(b"changed correction")
            with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                model_files(root)
            (root / "geometry-ranker.json").unlink()
            with self.assertRaises(FileNotFoundError):
                model_files(root)
            (root / "geometry-ranker.json").write_bytes(b"geometry-ranker.json")
            (root / "additional1/encoder.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                model_files(root)

    def test_changed_declared_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "encoder.bin").write_bytes(b"encoder")
            (root / "words.bin").write_bytes(b"words")
            manifest = {"encoder_sha256": hashlib.sha256(b"encoder").hexdigest(),
                        "words_sha256": hashlib.sha256(b"words").hexdigest()}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "encoder.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                model_files(root)


if __name__ == "__main__":
    unittest.main()
