"""The pinned stacks under launch/ agree with pyproject.toml.

A pin outside a bound the package declares makes the documented install
unresolvable (`uv pip install -e ".[...]" -c <file>` refuses it), and nothing
else fails until someone builds the environment from nothing. Each pin is
checked against every requirement of the same package in the dependencies
and the extras, whatever its platform marker.
"""
import re

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from conftest import ROOT

tomllib = pytest.importorskip("tomllib")

PIN_FILES = ["launch/constraints-gpu.txt", "launch/constraints-mac.txt"]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")


def declared() -> dict[str, list[Requirement]]:
    """Every requirement in pyproject.toml's dependencies and extras, by package."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lines = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        lines += extra
    out: dict[str, list[Requirement]] = {}
    for line in lines:
        req = Requirement(line)
        out.setdefault(canonicalize_name(req.name), []).append(req)
    return out


def pins(path) -> dict[str, str]:
    """A constraints file's pins, by package; every other line is a comment."""
    out = {}
    for n, line in enumerate((ROOT / path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PIN.match(line)
        assert m, f"{path}:{n}: {line!r} is not a name==version pin"
        out[canonicalize_name(m[1])] = m[2]
    return out


@pytest.mark.parametrize("path", PIN_FILES)
def test_every_pin_satisfies_the_bounds_pyproject_declares(path):
    wanted = declared()
    pinned = pins(path)
    checked = [name for name in pinned if name in wanted]
    assert "torch" in checked, f"{path} shares no package with pyproject.toml"
    bad = [f"{name}=={version} against {req}"
           for name, version in pinned.items() for req in wanted.get(name, [])
           if not req.specifier.contains(Version(version), prereleases=True)]
    assert not bad, f"{path} pins outside pyproject.toml's bounds: {bad}"
