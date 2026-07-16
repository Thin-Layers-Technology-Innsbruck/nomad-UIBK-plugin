#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def _provider_versions(providers: list[str]) -> str:
    versions = []
    for provider in dict.fromkeys(providers):
        try:
            dist = importlib.metadata.distribution(provider)
        except importlib.metadata.PackageNotFoundError:
            versions.append(provider)
        else:
            versions.append(f'{provider} {dist.version}')
    return ', '.join(versions)


def _origin_text(origin: str | None) -> str:
    if origin is None:
        return 'namespace package or unknown origin'

    try:
        origin_path = Path(origin).resolve()
    except OSError:
        return origin

    try:
        return str(origin_path.relative_to(Path.cwd()))
    except ValueError:
        return str(origin_path)


def explain_module(module_name: str) -> str:
    top_level_name = module_name.split('.', maxsplit=1)[0]
    providers_by_package = importlib.metadata.packages_distributions()
    providers = providers_by_package.get(top_level_name, [])

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        spec = None
        import_error = f'{type(exc).__name__}: {exc}'
    else:
        import_error = None

    lines = [module_name]
    if providers:
        lines.append(f'  provided by: {_provider_versions(providers)}')
    elif top_level_name in sys.stdlib_module_names:
        lines.append('  provided by: Python standard library')
    else:
        lines.append('  provided by: no installed distribution metadata found')

    if spec is None:
        lines.append('  importable: no')
        if import_error:
            lines.append(f'  import check: {import_error}')
    else:
        lines.append('  importable: yes')
        lines.append(f'  origin: {_origin_text(spec.origin)}')

    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Show which installed distribution provides an import module.'
    )
    parser.add_argument(
        'modules',
        nargs='*',
        default=['nomad'],
        help='Import module names, e.g. nomad, nomad.datamodel, pandas, locale.',
    )
    args = parser.parse_args()

    print('\n\n'.join(explain_module(module_name) for module_name in args.modules))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
