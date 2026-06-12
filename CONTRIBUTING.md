## Developing

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/dek3rr/hamon.git
cd hamon
```

Then install with development dependencies:

```bash
# Using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[development,testing,examples]"

# Or using uv
uv venv venv
source venv/bin/activate
uv pip install -e ".[development,testing,examples]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

The pre-commit hooks will automatically run `ruff` (formatting + linting) and `pyright` (type checking) on every commit.

To skip hooks for a WIP commit:

```bash
git commit --no-verify -m "wip: your message"
```

## Running tests

```bash
# All fast tests
pytest -v -m "not slow" tests/

# Everything including the slow MNIST test
pytest -v tests/
```

The suite pins JAX's default device to the CPU (`tests/conftest.py`): test
models are tiny, so GPU runs are dominated by compilation and kernel-launch
overhead, not compute. GPU code paths are covered by a small smoke subset:

```bash
pytest -v -m gpu tests/          # auto-skipped when no GPU is visible
HAMON_TEST_DEVICE=gpu pytest -v -m "not slow" tests/   # whole suite on GPU
```

To calibrate hamon's automatic CPU/GPU routing threshold for your machine,
run `python benchmarks/device_crossover.py` and export the recommended
`HAMON_DEVICE_THRESHOLD`.

## Branching

- `main` — stable, always passing CI
- `dev` — active development, merge into main when ready
- Feature branches off `dev`: `feat/your-feature`, `fix/your-fix`

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new sampling strategy
fix: correct ragged block handling in hinton_init
perf: thread global state through scan carry
test: add vmap correctness tests for parallel tempering
docs: update README installation instructions
chore: bump ruff to v0.11.0
```

## Releasing

Releases are automated via GitHub Actions. To publish a new version:

1. Update `version` in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Merge to `main`
4. Tag and push:
   ```bash
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin v0.1.0
   ```
5. Create a GitHub Release from the tag
6. The publish workflow handles PyPI automatically
