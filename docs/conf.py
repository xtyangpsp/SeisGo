import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SeisGo'
copyright = '2024, Xiaotao Yang'
author = 'Xiaotao Yang'
release = '0.9'

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

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'obspy': ('https://docs.obspy.org', None),
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
