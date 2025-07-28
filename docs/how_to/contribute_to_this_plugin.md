# Contribute to This Plugin

This code is currently under development. To contribute to this project, you should clone the respective repository, [install it](install_this_plugin.md) and create a pull request following the standard procedure.:

```sh
git clone git@github.com:Thin-Layers-Technology-Innsbruck/nomad-UIBK-plugin.git
cd nomad-UIBK-plugin
```

## Contribute to the Documentation

This project uses `mkdocs` to publish its documentation.
Its content is entirely stored within markdown files (`.md`) in the `docs/` directory.
A Github workflow deploys automatically updated documentation after commits to the `main` branch.
See the official [mkdocs documentation](https://squidfunk.github.io/mkdocs-material/reference/) for more details.

To view the documentation locally, install the documentation related packages and run the documentation server:
```sh
pip install -r requirements_docs.txt
mkdocs serve
```
