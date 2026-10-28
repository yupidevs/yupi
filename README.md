<div align="center">

<img src="logo.png" alt="yupi logo">

# yupi

**A package for tracking and analysing objects trajectories**

[![QA](https://github.com/yupidevs/yupi/actions/workflows/qa.yml/badge.svg)](https://github.com/yupidevs/yupi/actions/workflows/qa.yml)
[![Release](https://img.shields.io/github/v/release/yupidevs/yupi?label=release)](https://github.com/yupidevs/yupi/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/yupi)](https://pypi.org/project/yupi/)
[![Docs](https://readthedocs.org/projects/yupi/badge/?version=latest)](https://yupi.readthedocs.io)
[![DOI](https://zenodo.org/badge/304602979.svg)](https://zenodo.org/badge/latestdoi/304602979)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

[Quick start](#quick-start) · [Usage](#usage) · [Development](#development)

</div>

---

Standing for *Yet Underused Path Instruments*, **yupi** is a set of tools designed
for collecting, generating and processing trajectory data.

## Features

- **Convert raw data to trajectories** ... *different input manners*
- **I/O operations with trajectories** ... *json and csv serializers*
- **Trajectory extraction from video inputs** ... *even with moving camera*
- **Artificial trajectory generation** ... *several models implemented*
- **Trajectory basic operations** ... *rotation, shift, scaling, ...*
- **Trajectory transformations** ... *filters, resamplers, ...*
- **Statistical analysis from trajectories ensembles** ... *turning angles histogram, velocity autocorrelation function, power spectral density, and much more ...*
- **Results visualization** ... *each statistical observable has a related plot function*
- **Spacial projection of trajectories** ... *for 2D and 3D trajectories*

## Quick start

Current recommended installation method is via PyPI:

```cmd
pip install yupi
```

### Compatibility

- Python 3.11 or later
- Ubuntu 18.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later.

## Usage

In the [official documentation](https://yupi.readthedocs.io/en/latest/) there
are some resources to start using the library: Tutorials, Examples  and a
detailed description of the API.

### Examples

Code examples (with additional multimedia resources) can be found in
[this repository](https://github.com/yupidevs/yupi_examples). Additionally, in
the [Examples section](https://yupi.readthedocs.io/en/latest/examples/examples.html)
of the documentation, you can find the same examples with additional comments
and expected execution results in order to inspect the examples without actually
executing them.

## Development

Setup, QA, docs and releases: [DEVELOPMENT.md](DEVELOPMENT.md).

## Contributing

Issues and pull requests are welcome; `make qa` must pass.

## License

MIT: see [LICENSE](LICENSE).

---

<sub>🚀 Project starter provided by [Cookie Pyrate v1.0.3](https://github.com/gvieralopez/cookie-pyrate/tree/v1.0.3)</sub>
