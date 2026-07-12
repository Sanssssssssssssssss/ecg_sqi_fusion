# Tests and CI

Run the same checks locally:

```bash
pip install -e ".[test]"
python -m pytest -q
python -m src.ecg_sqi_inference verify-bundles
```

The main CI job has five steps:

1. check out the repository;
2. configure Python 3.11 and pip caching;
3. install CPU PyTorch, project dependencies, and the package;
4. run the complete pytest suite;
5. verify all four inference bundle hashes.

The documentation workflow separately builds this MkDocs site on pull
requests. Pushes to `main` also publish the built artifact to GitHub Pages.

