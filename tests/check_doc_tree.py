"""Check DM-40: d-means docs filed under the benchmark results document, nothing lost."""
import hashlib, json, os, sys, unicodedata
from pathlib import Path

DOCS = Path(os.environ.get("PQ_DOCS", Path.home() / ".preconquer/hub/d-means-26bce00f/docs"))
PARENT = "d-means 조사와 벤치마크 결과"
CHILDREN = {  # sha256 of each child before DM-40; moving must not change a byte
    "DM-32 채점 결과": "7e9b857b08a1fdd177857521c1c6d4f7dd915160a01b970581f5637f1d373c16",
    "DM-30 채점 결과": "b76082a671a71402981ab098a94d752d4e0e5cb4c1f70af19a3631554a76879b",
    "DM-28 채점 결과": "42753835419260ba11ad6e64a14586e01a84c0364af1fa7253477316c11f6474",
    "DM-22와 DM-25 채점 결과": "200c27b1c01476cd74e5fbfb5a5ab45fb8509684fabeb5b6db2c4aee5c816af1",
    "d-means 의사결정 모델 3종 벤치마크": "9169cdf80474fa7b463aa5e1fa8db43f4fae6c88645db643fe8eac060091763e",
}
PARENT_ORIG = "2de47d6d828085465826ba010747ce2a7af7ebc9bf89f18718c8b5e333b4c2c1"
TOC = "\n\n## 하위 문서\n"


def fail(msg):
    print("FAIL:", msg); sys.exit(1)


def tree():
    f = DOCS / ".tree.json"
    if not f.exists():
        fail("no .tree.json")
    parents = {unicodedata.normalize("NFC", k): unicodedata.normalize("NFC", v)
               for k, v in json.loads(f.read_text()).get("parents", {}).items()}
    for name in CHILDREN:
        if parents.get(name) != PARENT:
            fail(f"{name!r} is under {parents.get(name)!r}, not {PARENT!r}")
    tops = [n for n in (unicodedata.normalize("NFC", p.stem) for p in DOCS.glob("*.md")) if n not in parents]
    if tops != [PARENT]:
        fail(f"top-level docs: {tops}")


def intact():
    for name, digest in CHILDREN.items():
        got = hashlib.sha256((DOCS / f"{name}.md").read_bytes()).hexdigest()
        if got != digest:
            fail(f"{name!r} content changed")
    text = (DOCS / f"{PARENT}.md").read_text()
    if TOC not in text:
        fail("parent has no 하위 문서 section")
    base = text.split(TOC)[0].rstrip("\n")
    for cand in (base, base + "\n"):
        if hashlib.sha256(cand.encode()).hexdigest() == PARENT_ORIG:
            return
    fail("parent's original text changed")


def toc():
    text = (DOCS / f"{PARENT}.md").read_text()
    section = text.split(TOC, 1)[-1] if TOC in text else ""
    for name in CHILDREN:
        if f"[[{name}]]" not in section:
            fail(f"TOC lacks [[{name}]]")


{"tree": tree, "intact": intact, "toc": toc}[sys.argv[1]]()
print("ok")
