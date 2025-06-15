import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'well-read'
copyright = '2024, Rounak Sen'
author = 'Rounak Sen'

release = '0.1.0'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # For Google-style docstrings
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for autodoc
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
