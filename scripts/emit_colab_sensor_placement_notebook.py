"""
Regenerate `notebooks/sensor_placement_opt_colab.ipynb` with:
  - base64-embedded `configs/mujoco_local_smoke.yaml` (as today)
  - base64-embedded zip of `sensor_opt/`, `configs/`, and `requirements.txt`
so Colab can run with no git clone or manual upload.

Run from the repository root:
  python3 scripts/emit_colab_sensor_placement_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_OUT = ROOT / "notebooks" / "sensor_placement_opt_colab.ipynb"
YAML_PATH = ROOT / "configs" / "mujoco_local_smoke.yaml"


def _zip_project_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (ROOT / "sensor_opt").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                arc = p.relative_to(ROOT)
                zf.write(p, arc)
        for p in (ROOT / "configs").rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(ROOT))
        req = ROOT / "requirements.txt"
        if req.is_file():
            zf.write(req, "requirements.txt")
    return buf.getvalue()


def _src_lines(s: str) -> list[str]:
    return [line + "\n" for line in s.splitlines()]


def main() -> int:
    req_path = ROOT / "requirements.txt"
    requirements_txt = req_path.read_text(encoding="utf-8")
    yb64 = base64.b64encode(YAML_PATH.read_bytes()).decode("ascii")
    zb64 = base64.b64encode(_zip_project_bytes()).decode("ascii")
    if len(zb64) > 3_000_000:
        print("Warning: embedded zip b64 is very large; consider trimming.", file=sys.stderr)

    def uid() -> str:
        return uuid.uuid4().hex[:12]

    cells: list[dict] = []
    # --- markdown
    cells.append(
        {
            "cell_type": "markdown",
            "id": uid(),
            "metadata": {},
            "source": _src_lines(
                """# Sensor placement optimization — MuJoCo (Colab, self-contained)

**Requires only this notebook + pip (and a Colab or local kernel).** No separate project folder on your machine. The
optimizer sources ship **inside the notebook** as a base64 zip; the **smoke YAML** is also embedded in base64
(`configs/mujoco_colab_smoke.yaml` is written for you at runtime).

1. **Install** — the full `requirements.txt` from the project (JAX, scikit-learn, MLflow, PyTorch, Matplotlib, pandas, rich, MuJoCo, CMA, …) is embedded at notebook generation time; `pip install -r` in a temp file (same style as the SysID notebook).

2. **Materialize** — default: **decode the embedded project zip** into `/content/sensor_placement_opt_local/`
   (set `USE_EMBEDDED_SOURCE_BUNDLE = False` in that cell to use GitHub download, Colab upload, or a local `cwd` that
   already contains `sensor_opt/` instead).

3. **Smoke config** — writes `mujoco_colab_smoke.yaml` from the embedded copy.

4. **MuJoCo sanity** — minimal MJCF + one `mj_step` (lightweight, SysID-style check).

5. **Run** — `python -m sensor_opt.run_experiment` (short CMA, short MuJoCo rollouts on CPU).

6. **Plots** — CMA-ES generations from `results/.../generations.csv`.

**Refresh the embedded zip:** from the repository root run
`python3 scripts/emit_colab_sensor_placement_notebook.py`
after changing `sensor_opt/` or `configs/`.
"""
            ),
        }
    )
    # --- install (embed full project requirements.txt at generate time; edit that file, then re-run this script)
    install_py = f'''# Pinned in repo `requirements.txt` — full stack (JAX, sklearn, MLflow, torch, matplotlib, …)
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIREMENTS_SENSOR_PLACEMENT_COLAB = {json.dumps(requirements_txt)}
fd, _req_path = tempfile.mkstemp(suffix="-pip-reqs.txt", text=True)
os.close(fd)
try:
    Path(_req_path).write_text(REQUIREMENTS_SENSOR_PLACEMENT_COLAB, encoding="utf-8")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", _req_path])
finally:
    try:
        os.unlink(_req_path)
    except OSError:
        pass

import cma
import jax
import matplotlib
import mlflow
import mujoco
import numpy as np
import pandas as pd
import rich
import scipy
import sklearn  # scikit-learn
import torch
print("python:", sys.executable)
for name, mod in [
    ("cma", cma),
    ("jax", jax),
    ("matplotlib", matplotlib),
    ("mlflow", mlflow),
    ("mujoco", mujoco),
    ("numpy", np),
    ("pandas", pd),
    ("rich", rich),
    ("scipy", scipy),
    ("sklearn", sklearn),
    ("torch", torch),
]:
    v = getattr(mod, "__version__", "") or "?"
    if name == "mujoco":
        v = getattr(mujoco, "__version__", v) or "?"
    print(f"{{name}}: {{v}}")
'''
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(install_py),
        }
    )
    # --- materialize (with embedded b64 in triple quotes - b64 is ascii only)
    mat = fr'''# --- Materialize: expand embedded `sensor_opt/` + `configs/` into WORK, or use fallbacks below ---
from __future__ import annotations

import base64
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
import urllib.request

# Colab uses /content; local Jupyter: use a folder under the current working directory.
if os.path.isdir("/content"):
    WORK = Path("/content/sensor_placement_opt_local")
else:
    WORK = (Path.cwd() / "_sensor_placement_opt_local").resolve()

# Set False to use download / Colab upload / current working tree instead of the bundle below.
USE_EMBEDDED_SOURCE_BUNDLE = True

# Embedded at notebook build time (see `scripts/emit_colab_sensor_placement_notebook.py`).
_PROJECT_ZIP_B64 = {json.dumps(zb64)}

def _find_package_root(p: Path) -> Path | None:
    if (p / "sensor_opt" / "__init__.py").is_file():
        return p
    return None

def _search_tree_for_package(extract_dir: Path) -> Path | None:
    for init in extract_dir.rglob("sensor_opt/__init__.py"):
        return init.parent.parent
    return None

def _activate(project_root: Path) -> None:
    project_root = project_root.resolve()
    os.environ["SENSOR_PLACEMENT_NOTEBOOK_ROOT"] = str(project_root)
    os.chdir(project_root)
    s = str(project_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    print("Using project root:", project_root)
    print("cwd:", os.getcwd())

def _extract_top_level_project_zip_to_work(data: bytes) -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(WORK)
    if not (WORK / "sensor_opt" / "__init__.py").is_file():
        raise FileNotFoundError("Project zip is missing sensor_opt/ at top level.")

def _extract_github_style_zip_to_work(data: bytes) -> None:
    WORK.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORK.parent / f"{{WORK.name}}.tmp_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(tmp)
    found = _search_tree_for_package(tmp)
    if found is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FileNotFoundError("ZIP did not contain sensor_opt/ — use the repository root (Download ZIP from GitHub).")
    if WORK.exists():
        shutil.rmtree(WORK)
    found.rename(WORK)
    shutil.rmtree(tmp, ignore_errors=True)

def materialize() -> Path:
    if USE_EMBEDDED_SOURCE_BUNDLE and _PROJECT_ZIP_B64:
        data = base64.b64decode(_PROJECT_ZIP_B64)
        _extract_top_level_project_zip_to_work(data)
        _activate(WORK)
        return WORK

    r0 = _find_package_root(WORK)
    if r0 is not None:
        _activate(r0)
        return r0

    env_root = (os.environ.get("SENSOR_PLACEMENT_EXTRACTED_ROOT") or "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        r1 = _find_package_root(p)
        if r1 is None:
            raise FileNotFoundError(f"SENSOR_PLACEMENT_EXTRACTED_ROOT is not a project root: {{p}}")
        _activate(r1)
        return r1

    for start in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        r2 = _find_package_root(start)
        if r2 is not None:
            _activate(r2)
            return r2

    url = (os.environ.get("SENSOR_PLACEMENT_ZIP_URL") or "").strip()
    if url:
        print("Downloading:", url)
        req = urllib.request.Request(url, headers={{"User-Agent": "sensor-placement-colab"}})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        _extract_github_style_zip_to_work(data)
        _activate(WORK)
        return WORK

    try:
        from google.colab import files
    except Exception:
        files = None
    if files is not None:
        print("Upload a ZIP of the project root. The archive must contain sensor_opt/.")
        up = files.upload()
        if not up:
            raise SystemExit("No file uploaded.")
        data = next(iter(up.values()))
        _extract_github_style_zip_to_work(data)
        _activate(WORK)
        return WORK

    raise RuntimeError(
        "Set USE_EMBEDDED_SOURCE_BUNDLE = True, or SENSOR_PLACEMENT_ZIP_URL, SENSOR_PLACEMENT_EXTRACTED_ROOT, "
        "or run in Colab and upload a project ZIP, or start Jupyter with cwd at the project root."
    )

PROJECT_ROOT = materialize()
'''
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(mat),
        }
    )
    # --- yaml
    wcell = f'''# --- Write bundled smoke config (base64) + verify import ---
import base64
import importlib
from pathlib import Path

_MUJOCO_LOCAL_SMOKE_YAML_B64 = {json.dumps(yb64)}

def _patch_experiment_name(text: str) -> str:
    return text.replace('  name: "mujoco_local_smoke"', '  name: "mujoco_colab_smoke"', 1)

cfg_path = Path("configs/mujoco_colab_smoke.yaml")
cfg_path.parent.mkdir(parents=True, exist_ok=True)
text = base64.b64decode(_MUJOCO_LOCAL_SMOKE_YAML_B64).decode("utf-8")
# Prefer notebook-embedded snapshot over the zip’s configs/ (same as repo `mujoco_local_smoke`)
cfg_path.write_text(_patch_experiment_name(text), encoding="utf-8")
print("Wrote", cfg_path.resolve())
importlib.import_module("sensor_opt")
print("OK: sensor_opt import")
'''
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(wcell),
        }
    )
    # --- mj sanity
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(
                """# --- MuJoCo sanity: minimal model + one step ---
import numpy as np
import mujoco
from mujoco import MjData, MjModel

_xml = r\"\"\"
<mujoco model="pendulum">
  <option timestep="0.01"/>
  <worldbody>
    <body>
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
\"\"\".strip()

m = MjModel.from_xml_string(_xml)
d = MjData(m)
mujoco.mj_step(m, d)
print("qpos after one step:", np.asarray(d.qpos))
print("MuJoCo OK (minimal MJCF + mj_step).")
"""
            ),
        }
    )
    # --- run
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(
                """# --- CMA-ES + MuJoCo inner loop (smoke) ---
import subprocess
import sys
from pathlib import Path

assert Path("sensor_opt", "__init__.py").is_file(), "Run the materialize + config cells first."

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "sensor_opt.run_experiment",
        "--config",
        "configs/mujoco_colab_smoke.yaml",
        "--no-mlflow",
    ]
)
print("Done. See results/ for generations.csv and final_result.json")
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "markdown",
            "id": uid(),
            "metadata": {},
            "source": _src_lines("## CMA-ES performance plots\n"),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uid(),
            "metadata": {},
            "outputs": [],
            "source": _src_lines(
                """# Matplotlib: loss curves
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from sensor_opt.plotting.cma_matplotlib import plot_cma_generations_matplotlib

mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["axes.grid"] = True
mpl.rcParams["lines.linewidth"] = 2


def resolve_generations_csv(results_dir: str = "results", explicit=None):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(str(p))
    root = Path(results_dir)
    cands = sorted(
        root.glob("*/generations.csv"),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError("No generations.csv — run the experiment cell first (%s)" % root.resolve())
    return cands[0]


RESULTS_CSV = None
csv_path = resolve_generations_csv(explicit=RESULTS_CSV)
print("Using:", csv_path.resolve())
fig = plot_cma_generations_matplotlib(csv_path, title=csv_path.parent.name)
plt.show()
"""
            ),
        }
    )
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Wrote", NB_OUT, "size", NB_OUT.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
