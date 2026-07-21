"""Load the model defined in `model_notebook.ipynb` without any Jupyter dependency.

The notebook is the single source of truth for all model code. Every model function, dataclass,
and constant lives in a code cell whose *first line* is exactly `#| export`. This loader parses the
notebook JSON (stdlib only), concatenates the export cells in order, executes them in a fresh
namespace, and returns that namespace. `app.py` uses this so the widget contains zero model math.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

EXPORT_TAG = "#| export"


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src


def _is_export(cell) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = _cell_source(cell)
    for line in src.splitlines():
        if line.strip() == "":
            continue
        return line.strip() == EXPORT_TAG
    return False


def export_sources(path: str | Path) -> list[tuple[int, str]]:
    """Return [(cell_index, source), ...] for every `#| export` code cell, in notebook order.

    `cell_index` is the position of the cell among all cells in the notebook (for display in the
    widget's "Under the hood" tab).
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, cell in enumerate(nb.get("cells", [])):
        if _is_export(cell):
            out.append((i, _cell_source(cell)))
    return out


def export_code(path: str | Path) -> str:
    """The concatenated source of all export cells (what actually gets exec'd)."""
    return "\n\n".join(src for _, src in export_sources(path))


def load_model(path: str | Path = "model_notebook.ipynb") -> SimpleNamespace:
    """Parse the notebook, exec its export cells in a fresh namespace, return it as a namespace.

    The returned object exposes every top-level name defined by the export cells: `Params`,
    `simulate`, `headline`, `delay_comparison`, `monte_carlo`, `PARAM_RANGES`, the component
    functions, etc.
    """
    path = Path(path)
    if not path.is_absolute():
        # resolve relative to this file's directory so callers need not chdir
        here = Path(__file__).resolve().parent
        if (here / path).exists():
            path = here / path
    code = export_code(path)
    import builtins
    ns: dict = {"__builtins__": builtins}  # give the exec'd functions real builtins
    exec(compile(code, str(path), "exec"), ns)  # noqa: S102 - trusted local notebook
    public = {k: v for k, v in ns.items() if not k.startswith("__")}
    return SimpleNamespace(**public)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "model_notebook.ipynb"
    m = load_model(p)
    s = m.simulate(m.Params())
    print("loaded model; simulate() keys:", sorted(s.keys()))
    print("headline:", m.headline(s, m.Params()))
