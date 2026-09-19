import pathlib
import tempfile
import unittest

import sglang_ple_reuse_helper as reuse


class PleReuseHelperTest(unittest.TestCase):
    def test_marker_requires_matching_model_and_table_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model = root / "model"
            table_dir = root / "cache"
            model.mkdir()
            table_dir.mkdir()
            (model / "config.json").write_text('{"model":"test"}\n')
            (model / "model-fp8-mtp-ple.safetensors").write_bytes(b"checkpoint-v1" * 1024)
            table = table_dir / "ple_table_test.bin"
            table.write_bytes(b"table-v1" * 2048)

            fingerprint = reuse.model_fingerprint(model)
            reuse.write_marker(table, fingerprint)
            self.assertTrue(reuse.valid_marker(table, fingerprint))

            with table.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"corrupt!")
            self.assertFalse(reuse.valid_marker(table, fingerprint))
            reuse.prepare(table_dir, fingerprint)
            self.assertFalse(table.exists())

    def test_model_change_invalidates_existing_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model = root / "model"
            table_dir = root / "cache"
            model.mkdir()
            table_dir.mkdir()
            source = model / "model-fp8-mtp-ple.safetensors"
            source.write_bytes(b"checkpoint-v1")
            table = table_dir / "ple_table_test.bin"
            table.write_bytes(b"complete-table")
            first = reuse.model_fingerprint(model)
            reuse.write_marker(table, first)
            source.write_bytes(b"checkpoint-v2")
            second = reuse.model_fingerprint(model)
            self.assertNotEqual(first, second)
            self.assertFalse(reuse.valid_marker(table, second))


if __name__ == "__main__":
    unittest.main()
