# Tests and Continuous Integration

## Local checks

```bash
pip install -e ".[test]"
python -m pytest -q
python -m src.ecg_sqi_inference verify-bundles
mkdocs build --strict
```

## Code CI

The GitHub test job performs five project-facing steps:

1. check out the repository;
2. configure Python 3.11 and dependency caching;
3. install CPU PyTorch, project requirements, and the editable package;
4. run the complete pytest suite;
5. verify every inference artifact against its manifest hash.

Tests cover inference shapes and output contracts, metric conventions,
reproduction manifests, and repository-relative paths. Data-heavy research
runs remain explicit reproduction targets rather than unit tests.

## Documentation CI

Pull requests must pass `mkdocs build --strict`. A push to GitHub `main`
uploads the same generated `site/` directory and deploys it to GitHub Pages.
The assessed GitLab snapshot is locked and is not updated by this documentation
workflow.

