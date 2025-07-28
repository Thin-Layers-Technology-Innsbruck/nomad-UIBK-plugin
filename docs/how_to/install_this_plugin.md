# Install This Plugin

## For Usage in an NOMAD Distribution Image

To use this plugin in an NOMAD distribution image such as [this one](https://github.com/Thin-Layers-Technology-Innsbruck/nomad-uibk-image) simply list it in the `plugin.txt`.

## Stand alone

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

!!! info
    As of now, there is no official pypi NOMAD release with the plugin functionality. Therefore, make sure to include internal package registry of NOMAD (e.g. via `--index-url`).