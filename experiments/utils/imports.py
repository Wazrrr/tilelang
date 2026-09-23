"""Bind experiment workers to this checkout without changing the installation."""

from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ("tilelang", "tiletune_core")


class _LocalTileLang(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        package = next((name for name in LOCAL_PACKAGES if fullname == name or fullname.startswith(name + ".")), None)
        if package is None:
            return None
        spec = PathFinder.find_spec(fullname, [str(ROOT)] if fullname == package else path)
        if spec is None:
            raise ModuleNotFoundError(f"{fullname} is missing from the experiment checkout", name=fullname)
        return spec


def use_local_tilelang():
    """Override editable-install redirectors for local TileLang packages."""
    for package in LOCAL_PACKAGES:
        loaded = sys.modules.get(package)
        if loaded is not None and Path(loaded.__file__).resolve() != ROOT / package / "__init__.py":
            raise RuntimeError(f"{package} was already imported from another checkout; start a fresh experiment worker")
    if not any(isinstance(finder, _LocalTileLang) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LocalTileLang())
