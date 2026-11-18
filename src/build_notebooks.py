"""Convert notebooks/_source/*.py (``# %%`` cell format) into executed .ipynb.

Usage:  python src/build_notebooks.py            # build + execute all
        python src/build_notebooks.py 05         # only notebooks starting 05
"""
import json
import os
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_source"
OUT = ROOT / "notebooks"

# Execute with the interpreter running this script (not whatever the user's
# global "python3" kernelspec points to) via a throwaway local kernelspec.
_KDIR = ROOT / ".build_kernel" / "kernels" / "build-python"
_KDIR.mkdir(parents=True, exist_ok=True)
(_KDIR / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "build", "language": "python"}))
os.environ["JUPYTER_PATH"] = str(ROOT / ".build_kernel")


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                             for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(prefix=""):
    for src in sorted(SRC.glob(f"{prefix}*.py")):
        nb = nbformat.v4.new_notebook()
        nb.cells = parse(src.read_text())
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        print(f"executing {src.name} ...", flush=True)
        ExecutePreprocessor(timeout=1800, kernel_name="build-python").preprocess(
            nb, {"metadata": {"path": str(OUT)}})
        dst = OUT / f"{src.stem}.ipynb"
        nbformat.write(nb, dst)
        print(f"  -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
