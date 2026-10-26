"""Sphinx configuration file for Deepwave documentation."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import re
import sys
from typing import Any

from sphinx.errors import ConfigError

# -- Project information --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #project-information

project = "Deepwave"
project_copyright = "2026, Alan Richardson"
author = "Alan Richardson"

# Single-source the version from src/deepwave/_version.py rather than
# hardcoding it here (which previously had to be hand-synced each release).
_version_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "deepwave"
    / "_version.py"
)
_version_match = re.search(
    r"""__version__ = ['"]([^'"]+)['"]""", _version_path.read_text()
)
if _version_match is None:
    raise RuntimeError("Could not parse the version from _version.py")
version = _version_match.group(1)
release = "v" + version

# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# The four ``*_derivation.md`` developer-derivation documents are plain
# Markdown (they predate their inclusion in the rendered docs); myst_parser
# lets Sphinx render them directly alongside the ``.rst`` pages rather than
# requiring a lossy hand-conversion of ~2,200 lines of technical prose,
# unicode maths, and tables.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Resolve cross-references to Python and PyTorch objects (e.g. the many
# torch.Tensor references in the API docs that were previously dead links).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Every cross-reference must resolve. Without this, a cross-reference to a
# name that does not exist renders as plain text and no warning is emitted,
# so a reference can rot without the ``-W`` build noticing.
nitpicky = True

# The only references allowed not to resolve, each because the target is
# genuinely absent from the inventories intersphinx has rather than because
# the reference is wrong.
nitpick_ignore = [
    # Typing constructs that Sphinx resolves as classes but that Python's
    # own inventory does not publish under these names.
    ("py:class", "optional"),
    ("py:class", "Any"),
    ("py:class", "Callable"),
    ("py:class", "Iterable"),
    ("py:class", "Mapping"),
    ("py:class", "Sequence"),
    # PyTorch's inventory publishes these as data/attributes or not at all,
    # so the class reference autodoc generates from the annotation cannot be
    # resolved even though the link text is correct.
    ("py:class", "torch.dtype"),
    ("py:class", "torch.device"),
    ("py:class", "torch.Size"),
    # Type aliases and unions that autodoc renders from signature
    # annotations. Sphinx emits them as class references with no source
    # location (``<unknown>:1``), and a subscripted alias is not a class
    # anyone can document; the surrounding prose names the real types.
    ("py:class", "StoragePath"),
    ("py:class", "_StorageModeStr"),
    ("py:class", "tuple[torch.Tensor, LocationsLike]"),
    (
        "py:class",
        "tuple[torch.Tensor, LocationsLike, Optional[torch.Tensor]]",
    ),
    ("py:class", "tuple[LocationsLike, Optional[torch.Tensor]]"),
]

# Merge the class docstring with __init__'s docstring when rendering a class
# (the default, "class", renders only the class docstring and ignores
# __init__'s entirely). Deepwave's propagator Module classes document their
# constructor arguments in one of the two docstrings depending on the class
# (see each class), so both need to be visible in the rendered docs.
autoclass_content = "both"

# ``docs/dev/*.md`` are explicitly developer notes, not rendered docs (see
# their own header). Enabling myst_parser's ``.md`` source suffix above
# makes every Markdown file under ``docs/`` a candidate Sphinx source
# document, so exclude that directory explicitly rather than relying on it
# staying un-toctree'd (which would otherwise raise ``toc.not_included``
# warnings under ``-W``).
exclude_patterns += [
    "dev/README.md",
    "dev/api_conventions.md",
    "dev/boundary_conditions/CAO_CHEN_PM_SUMMARY.md",
    "dev/boundary_conditions/ROBERTSSON_FS_SUMMARY.md",
    "dev/location_coordinate_semantics.md",
]

# -- Options for the linkcheck builder -------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #options-for-the-linkcheck-builder

# docs.pytorch.org's API pages resolve fine (verified manually) but render
# their in-page anchors client-side, so linkcheck's static HTML anchor scan
# always reports them missing; skip anchor checking for that host while
# still checking the page itself loads.
linkcheck_anchors_ignore_for_url = [
    r"https://docs\.pytorch\.org/.*",
    r"https://pytorch\.org/docs/.*",
]

# Example pages whose "view source on GitHub"/"full recipe code" links point
# at a path on the `master` branch that a given documentation build may be
# ahead of: the page exists here, but GitHub's `master` has not caught up
# with it yet, so linkcheck reports a 404 that resolves by itself on the
# next push. Listed by name rather than as full URLs so the list stays
# readable and each line stays inside the line-length limit.
_AHEAD_OF_MASTER_EXAMPLES = [
    "example_acoustic_born",
    "example_boundary_faces",
    "example_elastic_born_recipe",
    "example_flagship_3d",
    "example_fwi_generated",
    "example_fwi_recipes",
    "example_moment_tensor",
    "example_parameterisation",
    "example_photoacoustic",
    "example_source_encoding",
    "example_time_dispersion",
    "example_topography",
    "example_ultrasound",
    "example_viscoacoustic",
    "example_vti",
]
_GITHUB_BLOB = r"^https://github\.com/ar4/deepwave/blob/master/docs/"

linkcheck_ignore = [
    # geoscienceworld.org blocks linkcheck's request (403) even though the
    # DOIs resolve fine in a normal browser.
    r"^https://doi\.org/10\.1190/1\.3513453$",
    r"^https://doi\.org/10\.1190/geo2018-0098\.1$",
    r"^https://doi\.org/10\.1190/1\.3059320$",
    # Crossref confirms these DOIs; the publisher pages reject automated
    # requests with HTTP 403 (Oxford Academic and International Press).
    r"^https://doi\.org/10\.1111/j\.1365-246X\.2006\.02978\.x$",
    r"^https://doi\.org/10\.4310/CMS\.2014\.v12\.n5\.a7$",
    *[_GITHUB_BLOB + name + r"\.py$" for name in _AHEAD_OF_MASTER_EXAMPLES],
]


# -- Options for HTML output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #options-for-html-output

html_theme = "alabaster"

html_theme_options = {
    "show_relbars": True,
}

# -- Generated content -------------------------------------------------


def _write_capabilities_table(app: Any) -> None:
    """Regenerates ``capabilities.rst``'s included table from live code.

    Runs on the ``builder-inited`` event, before any source file is read, so
    the ``.. include::`` in ``docs/capabilities.rst`` always picks up a
    freshly probed table -- there is no separate "remember to regenerate the
    docs" step for CI, or a contributor, to forget. The probing logic lives
    in :mod:`deepwave._capabilities`, imported here rather than at module
    level so a plain ``import conf`` (e.g. ``sphinx-build -M help``, which
    never fires ``builder-inited``) does not pay the cost of importing
    ``torch``/``deepwave``. ``tests/test_capabilities.py`` independently
    re-derives a subset of the same facts through separate calls, so this
    table cannot silently drift from what the propagators actually do.
    """
    import deepwave._capabilities as capabilities

    generated_dir = pathlib.Path(__file__).resolve().parent / "_generated"
    generated_dir.mkdir(exist_ok=True)
    (generated_dir / "capabilities_table.rst").write_text(
        capabilities.capabilities_rst()
    )


def _write_callback_names_table(app: Any) -> None:
    """Regenerates ``callback_names.rst``'s internal-name table.

    The same ``builder-inited`` pattern as ``_write_capabilities_table``
    above, and for the same reason: every name in that table is one the
    propagators' own ``PropagatorSpec`` declares and that
    ``deepwave._engine`` hands a :class:`deepwave.CallbackState`, so the
    page and the specs must be one thing rather than two that can drift.
    Unlike the capability matrix this one runs no propagators, so it is
    cheap.
    ``tests/test_callback_names.py`` independently re-derives it by running
    real callbacks and asking them which names they were given.
    """
    import deepwave._capabilities as capabilities

    generated_dir = pathlib.Path(__file__).resolve().parent / "_generated"
    generated_dir.mkdir(exist_ok=True)
    (generated_dir / "callback_names_table.rst").write_text(
        capabilities.callback_names_rst()
    )


def _write_points_per_wavelength_table(app: Any) -> None:
    """Regenerates ``discretisation.rst``'s sampling table from live code.

    The same ``builder-inited`` pattern as ``_write_capabilities_table``
    above, and for the same reason: the minimum points-per-wavelength per
    accuracy order are what
    :func:`deepwave.discretisation.suggest_discretisation` designs a grid
    from, so the page and the constant it reads must be one thing rather
    than two that can drift.
    """
    import deepwave.discretisation as discretisation

    generated_dir = pathlib.Path(__file__).resolve().parent / "_generated"
    generated_dir.mkdir(exist_ok=True)
    # Private on purpose: rendering the table is a docs-build detail, not
    # part of the module's public API, and this is the docs build.
    (generated_dir / "points_per_wavelength_table.rst").write_text(
        discretisation._points_per_wavelength_rst()  # noqa: SLF001
    )


def setup(app: Any) -> None:
    """Registers the generated-content hooks with the Sphinx application."""
    import deepwave

    if deepwave.__version__ != version:
        raise ConfigError(
            f"These docs require Deepwave {version}, but Python imported "
            f"Deepwave {deepwave.__version__} from {deepwave.__file__}. "
            "From the repository root, "
            "run `python -m pip install -e '.[docs]' -Cbuild-dir=build`, "
            "then build with the same interpreter: "
            "`python -m sphinx -M html docs docs/_build`. "
            "See CONTRIBUTING.md for documentation build instructions."
        )

    # Matching version strings do not identify a checkout: multiple local
    # candidates can all say 0.1.0. Autodoc and generated declarations must
    # read this candidate's Python files, whether installed or editable.
    source_package = _version_path.parent
    installed_package = pathlib.Path(deepwave.__file__).resolve().parent
    expected = {
        path.name: path.read_bytes() for path in source_package.glob("*.py")
    }
    actual = {
        path.name: path.read_bytes()
        for path in installed_package.glob("*.py")
    }
    differing = sorted(
        name
        for name in expected.keys() | actual.keys()
        if expected.get(name) != actual.get(name)
    )
    if differing:
        raise ConfigError(
            f"Deepwave {version} imported from {deepwave.__file__} "
            "has Python "
            "sources that differ from this documentation checkout: "
            f"{', '.join(differing)}. From the repository root, run "
            "`python -m pip install -e '.[docs]' -Cbuild-dir=build`, then "
            "build the docs with the same interpreter."
        )

    app.connect("builder-inited", _write_capabilities_table)
    app.connect("builder-inited", _write_callback_names_table)
    app.connect("builder-inited", _write_points_per_wavelength_table)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from geoazur import check_marmousi_link

    app.connect("builder-inited", check_marmousi_link)
