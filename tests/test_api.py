"""The public API names no decision model: models sit side by side in dmeans.deciders."""

import re
import tomllib
from pathlib import Path

import pytest

import dmeans
from dmeans import deciders

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Decisions", "LLM")  # the decider classes; each takes its model as a string
MODEL_NAMES = ("Jev", "Clef")   # model names never become importable names


def test_top_level_names_no_model():
    for name in MODELS:
        assert name not in dmeans.__all__ and not hasattr(dmeans, name), name
    for name in MODEL_NAMES:
        assert not hasattr(dmeans, name) and not hasattr(deciders, name), name
        assert name not in deciders.__all__, name
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
    imported = {n.strip().split(" ")[0] for t in re.findall(r"^\s*from dmeans(?:\.\w+)? import (.+?)(?:\s+#.*)?$", readme, re.M) for n in t.split(",")}
    assert imported and not imported & set(MODEL_NAMES), imported
    assert not re.search(r"\bimport\s+(?:jev|clef)\b", readme, re.I)
    desc = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["description"]
    assert desc.startswith("d-means: k-means clustering steered by any typed decision model"), desc
