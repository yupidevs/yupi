# Makefile for QA checks (lint, format, type-check, tests) and tasks (build, clean, run)

.PHONY: qa lint format typecheck test clean build repo worktree  

# Default target
qa: lint format typecheck test
	@printf "\033[92m[QA] All checks passed successfully.\033[0m\n"

lint:
	@printf "\n\033[1;34mRunning Ruff Linter\033[0m\n"
	uv run ruff check --fix

format:
	@printf "\n\033[1;34mRunning Ruff Format\033[0m\n"
	uv run ruff format

typecheck:
	@printf "\n\033[1;34mRunning Mypy\033[0m\n"
	uv run mypy

test:
	@printf "\n\033[1;34mRunning Pytest\033[0m\n"
	uv run pytest

repo:
	@printf "\n\033[1;34mCreating the git repository\033[0m\n"
	@command -v git >/dev/null || { printf "error: git is required; install it from https://git-scm.com/downloads\n" >&2; exit 1; }
	@if [ -e .git ]; then echo "· local repository already initialised"; \
	else git init -q -b main && echo "· initialised the local repository"; fi
	@if git rev-parse --verify -q HEAD >/dev/null; then echo "· initial commit already present"; \
	else git add --all && git commit -q -m "Initial commit" && echo "· created the initial commit"; fi
	uv run python scripts/create_remote.py $(REPO_ARGS)


worktree:
	@bash scripts/worktree.sh

	
build: clean 
	uv build



clean:
	@printf "\n\033[1;34mCleaning build and cache artifacts\033[0m\n"
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +

