from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRO5000_SCRIPTS = REPO_ROOT / "scripts" / "pro5000"


@contextmanager
def pro5000_scripts_on_path():
    sys.path.insert(0, str(PRO5000_SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(PRO5000_SCRIPTS))


def load_script(filename: str):
    path = PRO5000_SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pro5000_scripts_on_path():
        spec.loader.exec_module(module)
    return module


class TestPro5000Stage1(unittest.TestCase):
    def test_flashinfer_scale_copy_plan_preserves_empty_experts(self) -> None:
        smoke = load_script("flashinfer_sm120_fp8_smoke.py")
        self.assertEqual(
            smoke.build_scale_copy_plan([0, 0, 1, 9, 9, 12]),
            [
                (0, 0, 0),
                (0, 1, 0),
                (1, 9, 4),
                (9, 9, 16),
                (9, 12, 20),
            ],
        )


if __name__ == "__main__":
    unittest.main()
