# Changelog

## [0.1.6] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.1.4 (2026-03-22) [unreleased]
- Remove emoji from discussion CTA
- fix: reconcile README benchmark numbers with actual benchmark script

## v0.1.4 (2026-03-21)
- Add blog post link and community CTA to README
- Add distillation benchmark comparing SurrogateGLM vs direct GLM
- Add Databricks test notebook for v0.1.4
- Bump to 0.1.4: sync __version__ with pyproject.toml
- Fix __version__ in __init__.py to match pyproject.toml 0.1.3
- Fix pyproject.toml license format for hatchling compatibility
- Add LassoGuidedGLM: partial-dependence-guided binning with lasso selection
- Add pdoc API documentation with GitHub Pages
- Add Google Colab quickstart notebook and Open-in-Colab badge
- docs: add benchmark table, performance, limitations and references to README
- Add quickstart notebook
- fix: README technical errors from quality review
- Add MIT license
- Fix CI: use uv sync instead of uv pip install
- Fix test import error and publish v0.1.2 to PyPI
- Add GitHub Actions CI workflow for Python 3.11 and 3.12
- Fix numpy 2.x compatibility and update install instructions
- Fix tree binner sentinel value and Gini test expectations; all 61 tests pass
- Initial commit: insurance-distill v0.1.0

