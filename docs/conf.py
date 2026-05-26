import os
import sys

# -- Path setup --------------------------------------------------------------
# docs/conf.py is at <repo_root>/docs/conf.py
# One level up is the repo root, where the seisgo package directory lives.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'SeisGo'
copyright = '2024, Xiaotao Yang'
author = 'Xiaotao Yang'
release = '0.9'

# -- Mock heavy C/Fortran/optional dependencies ------------------------------
autodoc_mock_imports = [
    'obspy',
    'pyasdf',
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'tslearn',
    'minisom',
    'kneed',
    'pycwt',
    'pysurf96',
    'utm',
    'cartopy',
    'pygmt',
    'plotly',
    'numba',
    'netCDF4',
    'shapely',
    'stockwell',
    'h5py',
]

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# -- autodoc -----------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Napoleon ----------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- MyST --------------------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]
