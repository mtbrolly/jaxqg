# Contributing to jaxqg

Thanks for your interest in improving `jaxqg`.

## Development install

```bash
git clone https://github.com/mtbrolly/jaxqg.git
cd jaxqg
python -m pip install -e ".[dev]"
```

Optional extras:

- `notebooks` — matplotlib and jupyter notebook support
- `reference` — pyqg for equivalence tests (Python < 3.12 only;
  upstream does not build on 3.12+)

## Running the tests

The suite must pass in **both** precision modes:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 pytest
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest
```

Force CPU so results do not depend on the local accelerator (Metal, for
example, does not support float64).

Tests are also marked with pytest markers registered in
`pyproject.toml`:

- `slow` — longer-running tests
- `reference` — tests requiring the pyqg reference package
- `sharding` — tests requiring more than one visible device

## Linting

```bash
ruff check src/jaxqg tests
```

must pass before any commit.

## Building and checking the distribution

```bash
python -m build
```

Then install the built sdist/wheel into a throwaway environment and
import it; CI performs this check on every push.

## Commit conventions

- One commit per concern: separate structural moves, behaviour changes,
  and documentation.
- Imperative subject line ("Add ...", "Fix ...", not "Added ...").
- Keep generated bulk outputs (run artefacts, profiles, figures) out of
  commits; `runs/` and `dist/` are gitignored for this reason.

## Pull requests

- Branch from `main`.
- Keep PRs focused; describe the physics implications of any change to
  domain-sensitive behaviour (gradients, normalisation, units, spectral
  conventions).
- New features need tests passing in both precision modes.
- CI runs the precision matrix, pyqg equivalence (3.11), sharding, and
  packaging checks; all must be green.

## Notebooks

The maintained notebooks are `notebooks/quickstart.ipynb` (end-to-end
walkthrough) and `notebooks/ad_timing_demo.ipynb` (autodiff cost/memory
guide). If you change the public API in a way that affects them,
re-execute the affected notebook end-to-end with clean outputs before
committing:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/quickstart.ipynb
```

## Scientific behaviour

Treat domain-sensitive code (PV inversion, mean-flow gradients, spectral
normalisation, diagnostics denominators, units) as contract, not
incidental detail. Semantic changes must be explicit, justified, and
separately validated — never folded silently into a refactor.
