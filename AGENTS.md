# yupi

A package for tracking and analysing objects trajectories

Style, conventions and tooling for this project.

## Layout

1. `src/yupi/{models,errors,main}.py`; sub-domains as sub-packages (e.g. `validation/`).
2. Data structures live in `models.py`, error types in `errors.py`, unless the framework in use recommends a different file structure.
3. Order functions top-down — callers before callees, private helpers (`_name`) last — so the most important read first.
4. Do not write docstrings or descriptive comments unless the user asks for them or they provide context that cannot be inferred by reading the code.

## Design

1. Prefer immutable dataclasses for plain data: `@dataclass(frozen=True)` (add `slots=True` when it fits).
2. Where a framework owns the type — ORM entities, pydantic models, settings objects — follow its idiom instead of forcing a dataclass.
3. Avoid default arguments in functions.
4. If a function takes a bool that drives an `if`, prefer splitting into two functions (one delegating to the other, or factoring out shared logic when it saves ≥2 LoC).

## Hygiene

1. No commented-out code.
2. Every `# noqa` must specify a rule code, e.g. `# noqa: E501` and should be avoided unless the user asks for it.
3. Serialise JSON with `indent=2`.
4. Wrap any disk I/O in `TemporaryDirectory()` and operate via `Path` objects.

## Python

1. Always use f-strings, `pathlib`, and idiomatic constructs.
2. Avoid `open()` for reading or writing — use `Path.read_text()` / `Path.write_text()` / `Path.read_bytes()` / `Path.write_bytes()` (ruff `PTH`).
3. Never use `print` outside `scripts/`. Use `logger = logging.getLogger(__name__)` at module level (ruff `T20`).
4. Annotate everything; mypy runs with `disallow_untyped_defs = true`. Use modern typing (`list[...]`, `dict[...]`, `X | None`, `collections.abc.MutableMapping`, etc.).

## Tests

1. Function-based pytest with `@pytest.mark.parametrize`. No test classes.
2. The test tree mirrors the package tree: a module at `validation/rules.py` is tested by `tests/validation/test_rules.py`.
3. Shared fixtures, env dicts and fixture-path constants live in `tests/conftest.py`. Test data under `tests/data/` or `tests/fixtures/`.
4. `.env.test` holds the suite's baseline environment; `pytest-env` loads it before collection. Override a single variable in a single test with `monkeypatch.setenv` / `monkeypatch.delenv` — never assign to `os.environ` directly, so nothing leaks between tests.
5. Don't lower the `--cov-fail-under` threshold configured in `pyproject.toml`; add tests instead.

## Tooling

1. Use **ruff + mypy + pytest** with **uv**.
2. Standard QA invocation: `make qa` — it covers lint, format, type checks and tests. `make test` runs the suite on its own.
3. Never call `python3` directly in agent scripts; use `uv run` so inline script metadata can declare dependencies.
