# Building the Documentation

Agent-Tunix documentation is built using Sphinx and hosted using Read the Docs theme.

## Prerequisites

Install documentation dependencies using `uv`:

```bash
uv pip install -e ".[docs]"
```

Or using `pip`:

```bash
pip install -e ".[docs]"
```

Or install individual packages:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## Building the Documentation

### Build HTML Documentation

From the `docs/` directory:

```bash
make html
```

The built documentation will be in `_build/html/`.

### View Documentation

Open the built documentation in a browser:

```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

Or serve locally with Python:

```bash
make serve
```

Then open http://localhost:8000 in your browser.

### Clean Build

Remove previous build artifacts:

```bash
make clean
```

Then rebuild:

```bash
make html
```

## Build Targets

The Makefile supports several output formats:

- `make html` - HTML output (recommended)
- `make dirhtml` - HTML with directory indices
- `make singlehtml` - Single-page HTML
- `make pdf` - PDF output (requires LaTeX)
- `make epub` - EPUB ebook format
- `make man` - Man page output
- `make text` - Plain text output

## Documentation Structure

```
docs/
├── conf.py                      # Sphinx configuration
├── index.rst                    # Main documentation index
├── getting_started/             # Getting started guides
│   ├── installation.rst
│   ├── quick_start.rst
│   └── configuration.rst
├── guide/                       # User guides
│   ├── training.rst
│   ├── evaluation.rst
│   ├── hyperparameter_tuning.rst
│   └── experiments.rst
├── config/                      # Configuration reference
│   ├── overview.rst
│   ├── model.rst
│   ├── optimizer.rst
│   └── training.rst
├── api/                         # API reference
│   ├── train.rst
│   ├── evaluate.rst
│   ├── models.rst
│   ├── data.rst
│   └── rewards.rst
├── advanced/                    # Advanced topics
│   ├── distributed_training.rst
│   ├── custom_rewards.rst
│   └── troubleshooting.rst
├── references/                  # Reference material
│   ├── faq.rst
│   └── glossary.rst
├── _static/                     # Static assets (CSS, JS, images)
├── _templates/                  # Custom Sphinx templates
├── _build/                      # Build output (generated)
└── Makefile                     # Build automation
```

## Sphinx Extensions

The documentation uses these Sphinx extensions:

- **sphinx.ext.autodoc** - Extract documentation from docstrings
- **sphinx.ext.napoleon** - Support for Google-style docstrings
- **sphinx.ext.intersphinx** - Cross-reference external documentation
- **sphinx.ext.viewcode** - Link to source code from API docs
- **sphinx_rtd_theme** - Read the Docs theme

## Configuration

Sphinx configuration is in `conf.py`. Key settings:

- **project**: Project name
- **extensions**: Enabled extensions
- **html_theme**: HTML theme (RTD)
- **autodoc_default_options**: autodoc behavior
- **napoleon_google_docstring**: Support Google-style docstrings

## Adding New Documentation

To add a new documentation page:

1. Create `.rst` file in appropriate directory
2. Add entry to `index.rst` toctree
3. Rebuild with `make clean && make html`

Example `.rst` file:

```rst
My New Page
===========

Introduction paragraph.

Section Title
-------------

Content with **bold** and *italic* text.

Code Example
~~~~~~~~~~~~

Code blocks with syntax highlighting::

    python code here

Lists
~~~~~

- Item 1
- Item 2

Links to other docs:

:doc:`../other_page`
:ref:`section-label`

Next Steps
----------

- :doc:`next_page` - Description
```

## Troubleshooting

**sphinx-build command not found**

Install Sphinx:

```bash
pip install sphinx sphinx-rtd-theme
```

**Theme not found**

Install Read the Docs theme:

```bash
pip install sphinx-rtd-theme
```

**Build errors about missing modules**

Ensure the source directory is in Python path. Check `conf.py`:

```python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**Autodoc not pulling docstrings**

1. Ensure source modules are importable (in `src/`)
2. Check that functions/classes have docstrings
3. Verify `sphinx.ext.autodoc` is in extensions list

## CI/CD Integration

To build documentation in CI/CD:

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
```

The built documentation can then be deployed to Read the Docs or any static hosting.

## Read the Docs Integration

This documentation is set up for Read the Docs hosting:

1. Connect GitHub repository to Read the Docs
2. Add `requirements-docs.txt` to project root
3. Read the Docs will automatically build on commits

See https://docs.readthedocs.io/ for setup instructions.

## Version Tracking

The documentation version is set in `conf.py`:

```python
release = "0.1.0"
```

Update this when releasing new versions.

## License

Documentation is included with the main project license (see LICENSE in project root).
