import base64
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from kai_local.bundle import make_bundle
from kai_local.data import DatasetError, Sample, choose_labels, features_from_landmarks, prepare, split_groups
from kai_local.training import train


def pose(index, two_hands=False):
    return [{"x": 0.0 if point == 0 else (index * 100 + point) / 100_000,
             "y": 0.0 if point == 0 else point / 100,
             "z": .01} for point in range(42 if two_hands else 21)]


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def files(self, groups=3, per_label=10):
        result = []
        for group in range(groups):
            data = [{"label": label, "landmarks": pose(group * 1000 + label_index * 100 + index)} for label_index, label in enumerate(("E", "Nulo")) for index in range(per_label)]
            file = self.root / f"capture-{group}.json"
            file.write_text(json.dumps({"category": "vocales", "data": data}), encoding="utf-8")
            result.append(file)
        return result

    def test_current_download_format_and_group_disjointness(self):
        labels, parts, report = prepare(self.files(), "vocales", ["E", "Nulo"])
        self.assertTrue(report["ready_to_train"])
        self.assertEqual(labels, ["E", "Nulo"])
        group_sets = [{row.group for row in parts[name]} for name in ("train", "validation", "test")]
        for i in range(3):
            for j in range(i):
                self.assertFalse(group_sets[i] & group_sets[j])
        self.assertEqual(report["unique_samples"], 60)

    def test_same_seed_repeats_the_split(self):
        files = self.files(groups=8)
        first = prepare(files, "vocales", ["E", "Nulo"], seed=19)[2]
        second = prepare(files, "vocales", ["E", "Nulo"], seed=19)[2]
        self.assertEqual(first["split"], second["split"])

    def test_exact_duplicate_files_do_not_count_as_new_captures(self):
        file = self.files(groups=1)[0]
        copied = self.root / "copy.json"
        copied.write_bytes(file.read_bytes())
        _, parts, report = prepare([file, copied, file], "vocales", ["E", "Nulo"])
        self.assertIsNone(parts)
        self.assertFalse(report["ready_to_train"])
        self.assertEqual(len(report["files"]), 1)

    def test_duplicates_with_changed_whitespace_are_still_removed(self):
        file = self.files(groups=1)[0]
        copy = self.root / "copy.json"
        copy.write_text(json.dumps(json.loads(file.read_text()), indent=4))
        _, _, report = prepare([file, copy], "vocales", ["E", "Nulo"])
        self.assertEqual(report["unique_samples"], 20)
        self.assertEqual(report["duplicates_removed"], 20)

    def test_conflicting_labels_for_same_pose_fail(self):
        file = self.files(groups=1)[0]
        payload = json.loads(file.read_text())
        payload["data"].append({"label": "Nulo", "landmarks": payload["data"][0]["landmarks"]})
        file.write_text(json.dumps(payload))
        with self.assertRaisesRegex(DatasetError, "contradictorias"):
            prepare([file], "vocales", ["E", "Nulo"])

    def test_invalid_landmarks_and_nonfinite_coordinates_fail(self):
        for value in ([], pose(1)[:20], [{"x": True, "y": 0, "z": 0}] * 21, [{"x": math.nan, "y": 0, "z": 0}] * 21, [{"x": 0, "y": 0, "z": 99}] * 21):
            with self.assertRaises(DatasetError):
                features_from_landmarks(value)

    def test_capture_coordinates_are_not_normalized_twice(self):
        features = features_from_landmarks(pose(4))
        self.assertEqual(features[2], .01)
        self.assertEqual(features[3], .00401)
        self.assertEqual(features[63:], (0.0,) * 63)
        self.assertEqual(len(features_from_landmarks(pose(4, True))), 126)
        bad = pose(4)
        bad[0]["x"] = .5
        with self.assertRaisesRegex(DatasetError, "normalizadas"):
            features_from_landmarks(bad)

    def test_missing_nulo_and_unknown_module_labels_fail(self):
        for category, labels in (("vocales", ["A", "E"]), ("vocales", ["B", "Nulo"]), ("vocales", ["E", "Nulo", "E"])):
            with self.assertRaises(DatasetError):
                choose_labels(category, labels)

    def test_dynamic_letters_cannot_be_published_as_static_models(self):
        for label in ("J", "Z"):
            with self.assertRaisesRegex(DatasetError, "temporal"):
                choose_labels("abecedario", [label, "Nulo"])

    def test_one_file_cannot_train_even_with_many_rows(self):
        _, parts, report = prepare(self.files(groups=1, per_label=40), "vocales", ["E", "Nulo"])
        self.assertIsNone(parts)
        self.assertFalse(report["ready_to_train"])

    def test_foreign_category_fails(self):
        with self.assertRaisesRegex(DatasetError, "otro módulo"):
            prepare(self.files(), "numeros", ["1", "Nulo"])

    def test_legacy_arrays_are_supported_and_audited(self):
        files = self.files()
        for file in files:
            file.write_text(json.dumps(json.loads(file.read_text())["data"]))
        self.assertTrue(prepare(files, "vocales", ["E", "Nulo"])[2]["ready_to_train"])

    def test_little_endian_export_contract(self):
        values = [index / 1000 for index in range(254)]
        bundle = json.loads(make_bundle(["E", "Nulo"], [{"units": 2, "activation": "softmax", "dropout": 0}], values))
        self.assertEqual(bundle["schemaVersion"], 1)
        self.assertEqual(bundle["inputFeatures"], 126)
        unpacked = struct.unpack("<254f", base64.b64decode(bundle["weightData"]))
        self.assertAlmostEqual(unpacked[100], .1)

    def test_export_rejects_mismatched_weights_or_nan(self):
        network = [{"units": 2, "activation": "softmax", "dropout": 0}]
        for weights in ([0] * 253, [math.nan] * 254):
            with self.assertRaises(ValueError):
                make_bundle(["E", "Nulo"], network, weights)

    def test_version_path_cannot_escape_output(self):
        for version in ("../escape", "a/b", "a\\b", "", "v" * 41):
            with self.assertRaises(DatasetError):
                train("vocales", version, [], {}, {"problems": []}, self.root)

    def test_versions_are_never_overwritten(self):
        (self.root / "vocales" / "v1").mkdir(parents=True)
        with self.assertRaisesRegex(DatasetError, "ya existe"):
            train("vocales", "v1", [], {}, {"problems": []}, self.root)
        self.assertTrue((self.root / "vocales" / "v1").is_dir())

    def test_math_can_combine_disjoint_operator_and_number_files(self):
        samples = [Sample(label, (float(index),), f"{label}-{group}") for label in ("1", "Suma", "Nulo") for group in range(3) for index in range(5)]
        parts = split_groups(samples, ["1", "Suma", "Nulo"])
        self.assertTrue(all({row.label for row in part} == {"1", "Suma", "Nulo"} for part in parts.values()))


if __name__ == "__main__":
    unittest.main()
