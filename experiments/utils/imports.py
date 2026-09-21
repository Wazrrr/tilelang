"""Bind experiment workers to this checkout without changing the installation."""

from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


class _LocalTileLang(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "tilelang" and not fullname.startswith("tilelang."):
            return None
        spec = PathFinder.find_spec(fullname, [str(ROOT)] if fullname == "tilelang" else path)
        if spec is None:
            raise ModuleNotFoundError(f"{fullname} is missing from the experiment checkout", name=fullname)
        return spec


def use_local_tilelang():
    """Override an editable-install redirector only for TileLang in this process."""
    loaded = sys.modules.get("tilelang")
    if loaded is not None and Path(loaded.__file__).resolve() != ROOT / "tilelang/__init__.py":
        raise RuntimeError("TileLang was already imported from another checkout; start a fresh experiment worker")
    if not any(isinstance(finder, _LocalTileLang) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LocalTileLang())
