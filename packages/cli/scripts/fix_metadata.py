"""Runtime hook: shim importlib.metadata for frozen interactive dependencies."""

from importlib import metadata as md

_VERSIONS = {
    'readchar': '4.2.2',
    'inquirer': '3.4.1',
    'blessed': '1.52.0',
    'jinxed': '1.3.0',
}

_orig_distribution = md.distribution


def _distribution(distname):
    try:
        return _orig_distribution(distname)
    except md.PackageNotFoundError:
        if distname in _VERSIONS:
            return type('_Dist', (), {'version': _VERSIONS[distname]})()
        raise


md.distribution = _distribution
