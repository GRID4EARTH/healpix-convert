# -- Project information -----------------------------------------------------
import datetime as dt
import json
import pathlib

project = "healpix-convert"
author = f"{project} developers"
initial_year = "2026"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

# The root toctree document.
root_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_nb",
]

extlinks = {
    "issue": ("https://github.com/GRID4EARTH/healpix-convert/issues/%s", "GH%s"),
    "pull": ("https://github.com/GRID4EARTH/healpix-convert/pull/%s", "PR%s"),
}

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# The tutorial notebooks need data access (CDS, ADS, Polytope credentials or
# local input files) and large downloads: render them as they are, without
# executing them.
nb_execution_mode = "off"

# The tutorial notebooks are kept in the top-level ``notebooks/`` folder, which
# is outside of the Sphinx source directory: copy them into
# ``tutorials/notebooks/`` (ignored by git).
#
# Most notebooks have no kernel metadata: without it, myst-nb finds no lexer
# for the code cells (no syntax highlighting, and a warning for each cell), so
# the copies get the Python language metadata if they have none. Only write the
# notebooks that changed, so that ``sphinx-autobuild`` doesn't rebuild in a
# loop.
docs_dir = pathlib.Path(__file__).parent
notebooks_dir = docs_dir / "tutorials" / "notebooks"
notebooks_dir.mkdir(exist_ok=True)
for notebook in sorted((docs_dir.parent / "notebooks").glob("*.ipynb")):
    content = json.loads(notebook.read_text(encoding="utf-8"))
    content.setdefault("metadata", {}).setdefault(
        "language_info", {"name": "python", "pygments_lexer": "ipython3"}
    )
    text = json.dumps(content, indent=1, ensure_ascii=False) + "\n"
    copied = notebooks_dir / notebook.name
    if not copied.exists() or copied.read_text(encoding="utf-8") != text:
        copied.write_text(text, encoding="utf-8")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "directory", ".ipynb_checkpoints", "jupyter_execute"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GRID4EARTH/healpix-convert",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/healpix-convert/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}

# -- Options for the intersphinx extension -----------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "distributed": ("https://distributed.dask.org/en/stable/", None),
}

# Suppress warnings from napoleon's type preprocessing trying (and failing) to
# cross-reference the freeform type descriptions used in the docstrings, e.g.,
# "list of paths" or "iterable of ints": these are not Python types.
nitpick_ignore = [
    ("py:class", "callable"),  # ConvertSettings.filter_and_replace
    ("py:class", "iterable"),  # convert_group_to_healpix (chunk_indices)
    ("py:class", "ints"),  # convert_group_to_healpix (chunk_indices)
    ("py:class", "paths"),  # create_healpix_dataset, create_staging_cache
]

autosummary_generate = True
autodoc_typehints = "none"

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
