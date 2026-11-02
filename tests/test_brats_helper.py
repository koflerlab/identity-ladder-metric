import json
import os
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
VENV_PY = os.path.join(ROOT, "external", "venv-brats", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))


@pytest.mark.skipif(not os.path.exists(VENV_PY), reason="external/venv-brats not installed")
def test_brats_imports_and_both_c10_models_construct_on_cpu():
    out = subprocess.run([VENV_PY, os.path.join(ROOT, "src", "brats_helper.py"), "check"], capture_output=True,
                         text=True, timeout=600, check=True).stdout.strip().splitlines()[-1]
    assert json.loads(out) == {"BraTS23_1": "Inpainter", "BraTS24_1": "Inpainter"}
