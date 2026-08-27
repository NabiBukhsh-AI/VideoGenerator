# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
videogen doctor
```

You also need `ffmpeg` on PATH to run the render tests. See the README for install
commands.

## Before opening a PR

```bash
make check      # ruff + mypy + pytest
```

Or individually:

```bash
ruff check videogen tests
ruff format videogen tests
mypy videogen
pytest
```

## Tests

The suite runs **entirely offline** — no API keys, no network, no spend. That is a hard
requirement, not a convenience: every provider has a deterministic offline stub
(`echo`, `placeholder`, `silent`) and tests use those.

- Tests needing ffmpeg are marked `@pytest.mark.ffmpeg` and skip when it is absent.
- `pytest -m "not ffmpeg"` skips them explicitly.
- If you add a provider, add its stub behaviour to the tests rather than mocking the
  vendor SDK.

## Adding a provider

1. Implement the relevant ABC from `videogen/providers/base.py`
   (`LLMProvider`, `ImageProvider`, or `TTSProvider`).
2. Put it under `videogen/providers/<kind>/`.
3. Add a branch in `videogen/providers/registry.py` and the name to `KNOWN`.
4. Import the vendor SDK **inside** the method that uses it, and raise
   `DependencyError` with a `pip install` hint when it is missing.
5. Add the extra to `pyproject.toml` under `[project.optional-dependencies]`.

Nothing in `pipeline.py` should need to change.

## Conventions

- Errors are `VideoGenError` subclasses and carry a `hint` telling the user how to fix
  the problem.
- Never import an optional dependency at module top level.
- Secrets are `SecretStr` and only ever come from settings. No key literals, in source
  or in tests.
- Comments explain *why*, not *what*.

## Reporting bugs

Include the output of `videogen doctor` and the command you ran. If a render failed,
`--log-level DEBUG` includes the ffmpeg stderr.
