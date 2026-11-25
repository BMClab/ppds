"""Keep the software citation in step with the package versions."""

import re
import tomllib

from packaging.version import Version

from ppds.paths import ROOT


def field(text, name):
    match = re.search(rf'^{name}: "([^"]+)"$', text, re.MULTILINE)
    assert match, f"CITATION.cff needs a quoted {name}"
    return match[1]


def test_citation_matches_the_package_release():
    toolkit = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reader = tomllib.loads((ROOT / "framework/pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert toolkit["version"] == reader["version"]
    package = Version(toolkit["version"])
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cited = Version(field(cff, "version"))
    if package.is_devrelease:
        # Development versions still cite the latest published release.
        assert cited < Version(package.base_version)
    else:
        assert cited == package, "Update version and date-released in CITATION.cff"
    doi = field(cff, "doi")
    assert toolkit["urls"]["Software"] == f"https://doi.org/{doi}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    software = next(line for line in readme.splitlines() if line.startswith("- **Software**"))
    assert f"(v{cited})" in software and f"[{doi}](https://doi.org/{doi})" in software
