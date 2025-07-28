<img src="docs/assets/nomad-plugin-logo.png" alt="NOMAD Plugin Logo" width="200">

# NOMAD UIBK Plugin

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://python.org)
![GitHub Issues](https://img.shields.io/github/issues/Thin-Layers-Technology-Innsbruck/nomad-UIBK-plugin)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This software is a plugin for the [NOMAD](https://nomad-lab.eu/nomad-lab/) research data management system.
It provides a collection of schemas and parsers tailored to measurement methods around fundamental solar cell research.
It is intended to run on a self-hosted version of NOMAD called [Oasis](https://nomad-lab.eu/nomad-lab/nomad-oasis.html).

## Installation

### For Usage in a NOMAD Distribution Image

To use this plugin in an NOMAD distribution image [include it in its pyproject.toml](https://github.com/Thin-Layers-Technology-Innsbruck/nomad-UIBK-image?tab=readme-ov-file#adding-a-plugin) and generate a new image.

### Stand alone installation and/or to customize the plugin

To install this package, it is recommended to create and activate a virtual environment.
This practice isolates the package dependencies and prevents conflicts with other globally installed Python packages.
Since the codebase of NOMAD is currently based on Python 3.9, using that version is recommended.

```sh
python3.9 -m venv .pyenv
source .pyenv/bin/activate
```

The `nomad-lab` package is needed to use and test the plugin functionalities:

```sh
pip install --upgrade pip
pip install -e '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

> **Note:** As of now, there is no official pypi NOMAD release with the plugin functionality. Therefore, make sure to include internal package registry of NOMAD (e.g. via `--index-url`).

#### Testing

You can run automated tests with `pytest`:

```sh
pytest -svx tests
```

#### Run linting

```sh
ruff check .
```

#### Run auto-formatting

This is entirely optional. To add this as a check in github actions pipeline, uncomment the `ruff-formatting` step in `./github/workflows/actions.yaml`.

```sh
ruff format .
```

### Developing a NOMAD plugin

Follow the [guide](https://nomad-lab.eu/prod/v1/staging/docs/howto/plugins/plugins.html) on how to develop NOMAD plugins.

### Build the python package

The `pyproject.toml` file contains everything that is necessary to turn the project
into a pip installable python package. Run the python build tool to create a package distribution:

```
pip install build
python -m build --sdist
```

You can install the package with pip:

```
pip install dist/nomad-uibk-plugin-0.1.0
```

Read more about python packages, `pyproject.toml`, and how to upload packages to PyPI
on the [PyPI documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Technical details

This plugin was generated with `Cookiecutter` along with NOMAD's [cookiecutter-nomad-plugin template](https://github.com/blueraft/cookiecutter-nomad-plugin).

### Template update

We use cruft to update the project based on template changes. A `cruft-update.yml` is included in Github workflows to automatically check for updates and create pull requests to apply updates.
Follow the [instructions](https://github.blog/changelog/2022-05-03-github-actions-prevent-github-actions-from-creating-and-approving-pull-requests/) on how to enable Github Actions to create pull requests.

To run the check for updates locally, follow the instructions on [`cruft` website](https://cruft.github.io/cruft/#updating-a-project).

### Documentation on Github pages

This project uses `mkdocs` and a automatic Github Workflow to deploy its documentation.
Its content is entirely stored within markdown files (.md) in the `docs/` directory.

To view the documentation locally, install the documentation related packages and run the documentation server:
```sh
pip install -r requirements_docs.txt
mkdocs serve
```

### License
Distributed under the terms of the [Apache Software License 2.0](LICENSE).
`nomad-uibk-plugin` is free and open source software.

## Acknowledgments

Funding for this work has been provided by the European Union as part of the SolMates project (Project Nr. 101122288).

<img src="docs/assets/eu_funding_logo.png" alt="EU Funding Logo" width="300">
<img src="docs/assets/solmates_logo.png" alt="SolMates Logo" width="300">
