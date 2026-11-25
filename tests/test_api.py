"""The public API names no decision model: models sit side by side in dmeans.deciders."""

import re
import tomllib
from pathlib import Path

import pytest

import dmeans
from dmeans import deciders

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Jev", "LLM")


def test_top_level_names_no_model():
    for name in MODELS:
        assert name not in dmeans.__all__ and not hasattr(dmeans, name), name
    assert not (ROOT / "dmeans" / "jev.py").exists()
    with pytest.raises(ImportError):
        exec("from dmeans import DMeans, Jev", {})
    with pytest.raises(ModuleNotFoundError):
        exec("import dmeans.jev", {})


def test_models_sit_side_by_side_in_deciders():
    for name in MODELS:
        cls = getattr(deciders, name)
        assert name in deciders.__all__ and issubclass(cls, deciders.CachedDecider)
        assert isinstance(cls.__new__(cls), deciders.Decider)


def test_docs_import_no_model_from_top_level():
    readme = (ROOT / "README.md").read_text()
    tops = re.findall(r"^from dmeans import (.+)$", readme, re.M)
    assert tops and all(not set(MODELS) & {n.strip() for n in t.split(",")} for t in tops), tops
    assert "from dmeans.deciders import" in readme
    desc = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["description"]
    assert desc.startswith("d-means: k-means clustering steered by any typed decision model"), desc
