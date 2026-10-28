# Development Guide

## Prerequisites

Before developing this project, ensure you have the following installed:

### [uv](https://docs.astral.sh/uv/)  
Python project and environment management.  
Install: [uv docs](https://docs.astral.sh/uv/getting-started/installation/)

### [make](https://www.gnu.org/software/make/)  
Run common project tasks via the `Makefile`.

#### macOS  
```sh
xcode-select --install
```

#### Windows  
```powershell
choco install make
```

#### Linux
Usually pre-installed. If not:
```sh
sudo apt install build-essential   # Debian/Ubuntu
sudo dnf groupinstall "Development Tools"   # Fedora
```

### [git](https://git-scm.com/downloads)  
Version control, and what `make repo` uses to initialise the repository.

### [GitHub CLI](https://cli.github.com)  
Lets `make repo` create the GitHub remote and apply the branch policies.  
Authenticate once with `gh auth login`.


## Installation in development mode

Install all dependencies including development tools:

```bash
uv sync
```

This creates a `.venv` folder and installs the project and default development
dependencies from `pyproject.toml` and the committed `uv.lock` file. After changing
dependencies, use `uv add`/`uv remove`, then run `uv lock` and commit the lockfile.

Activate the environment:

* **Linux / macOS (bash/zsh):**
  ```bash
  source .venv/bin/activate
  ```

* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

* **Windows (cmd.exe):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

## Development Tasks

All tasks are defined in the `Makefile` for convenience.

### Running the Full Quality Gate

```bash
make qa
```

Runs **Ruff** for linting and formatting, **Mypy** for type checking, and **Pytest** for
the test suite. Each stage is also available on its own: `make lint`, `make format`,
`make typecheck` and `make test`.

### Running Unit Tests

Before running tests, configure environment variables in `.env.test` if needed.

```bash
make test
```

Executes the test suite using **Pytest**, without the lint, format and type checks.

### Creating the Repository

Initialise the local repository and create its first commit on `main`:

```bash
make repo
```

The same command then creates the GitHub remote, pushes `main`, and applies the branch
policies. The remote is private by default; to make it public:

```bash
make repo REPO_ARGS=--public
```

To create it under an organization instead of your own account:

```bash
make repo REPO_ARGS="--owner my-org"
```

The README's badges link to your own account; for an organization, replace your username with
the organization in those links.

`main` is protected: changes land through pull requests, merged with a merge commit, once all
review conversations are resolved, the branch is up to date with `main`, and the `qa / Run QA`
check is green. No approval is required, except a code owner's when the project has a
`CODEOWNERS` file; new pushes dismiss earlier approvals. Force pushes and branch deletion are
rejected. Repository admins and the release deploy key can bypass these rules; the key is how
the release workflow pushes its version bumps.

Branch rulesets are a paid GitHub feature on private repositories. On a free plan the
command says so, leaves `main` unprotected and finishes the rest; make the repository
public or upgrade the plan, then re-run `make repo` to apply the policies.

It also turns on Dependabot alerts and security updates: when a version pinned in `uv.lock`
has a published advisory, Dependabot opens a pull request that moves it to a fixed version.
Routine version updates stay off, since every merge to `main` cuts a release.

The local steps need only git; the remote ones need the
[GitHub CLI](https://cli.github.com) authenticated with `gh auth login`. Without it the
local repository is still created and nothing is pushed.

The command is safe to re-run: each step is skipped if it is already done, so an
interrupted run can be finished by running it again.

### Migrating to Git Worktrees

Migrate the repository to a shared bare-repository layout:

```bash
make worktree
```

This creates `.bare` for the shared Git history and a directory for the current
branch. If the current branch is not the remote default branch, a second directory
is created for that default branch. The converted repository root contains only
`.bare`, the `.git` pointer, and the worktree directories. Before converting, it
copies the complete original repository, including `.git`, to a sibling directory
named `<repo-name>-legacy-bak`, restores the original files to the current branch
worktree, and refuses to overwrite an existing backup or worktree layout. If the
migration fails, the original repository is restored and the temporary backup is
removed.

### Building the Project

```bash
make build
```

Generates a distribution package in the `dist/` directory.

### Releasing

Every push to `main` that passes QA cuts a release: the version is finalized,
tagged, and `main` is bumped to the next prerelease. You can control the release
process with the following options on the `.github/workflows/release.yml` workflow:

- `create-github-release` (on by default) — cuts a GitHub release for the new tag.
- `attach-distribution` (off by default) — builds the distribution and attaches it to the
  GitHub release as downloadable files. Requires `create-github-release` to also be on.
- `prepare-next-version` (on by default) — bumps `main` to the next prerelease once the
  release is out.

#### Publishing to PyPI

Set up Trusted Publishing so no API token has to be stored as a secret:

1. Go to <https://pypi.org/manage/account/publishing/> (for a project already on PyPI, its own
   "Publishing" settings)
2. Add a GitHub publisher with this repository, the workflow name `release.yml` and the
   environment name `pypi`
3. Save. PyPI will now trust publish requests coming from that workflow.

Then add a `pypi` job to `.github/workflows/release.yml`, next to the `release` job:

```yaml
  pypi:
    needs: release
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/yupi/
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: gvieralopez/cookie-pyrate/.github/actions/publish-to-pypi@v1.0.3
        with:
          sha: ${{ needs.release.outputs.sha }}
```

The job has to live in this file: PyPI only trusts the workflow file that defines the uploading
job, so the upload can't happen inside the reusable release. The action builds the released
commit and uploads it. It runs once the release is out, so if the upload fails the GitHub release
is already published: re-run the failed `pypi` job; files PyPI already has are skipped.

To try it on TestPyPI first, register the publisher on <https://test.pypi.org> instead and add
`index: testpypi` under `with:`. A build that needs `system-dependencies` or a `setup-command`
takes them under `with:` too; secrets the setup command needs go in the step's `env:`.

#### Extra release files

Files the reusable release doesn't build (executables, wheels compiled per OS, PDFs, archives)
come from jobs you add to `.github/workflows/release.yml`, after the `release` job. It hands them
the released `version`, `tag` and `sha`, so they build exactly the released commit and attach to
exactly that release:

```yaml
  windows-exe:
    needs: release
    runs-on: windows-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v7
        with:
          ref: ${{ needs.release.outputs.sha }}
      # ...build the files into dist/...
      - uses: softprops/action-gh-release@v3
        with:
          tag_name: ${{ needs.release.outputs.tag }}
          files: dist/*.exe
```

The release is already published when these jobs start, so its extra files appear a few
minutes later.

### Extra CI steps

The reusable workflows take three optional inputs for what lint, types and tests don't cover.
Add them under `with:` in `.github/workflows/qa.yml` (`setup-command` and `secret-env` also in
`release.yml`):

- `setup-command`: bash run before the dependencies are installed, for files the install needs,
  such as the wheels of a private dependency. In releases it also runs before the distribution
  and the Docker image are built.
- `extra-checks`: bash run after the tests; a failure fails QA. For example import contracts
  (`uv run lint-imports`) or a strict docs build.
- `secret-env`: names of repository secrets to export as environment variables for those steps.
  The job calling the reusable workflow needs `secrets: inherit`: `release.yml` already passes
  it (that's how the release key reaches the release), `qa.yml` needs it added.
  Dependabot's pull requests only get Dependabot secrets, so a secret QA needs must also be added
  as a Dependabot secret (`gh secret set NAME --app dependabot`), or its security fixes fail QA.
  In an organization, set it once as an organization secret for both Actions and Dependabot.

```yaml
jobs:
  qa:
    uses: gvieralopez/cookie-pyrate/.github/workflows/reusable-qa.yml@v1.0.3
    secrets: inherit
    with:
      cookie-pyrate-ref: v1.0.3
      python-version: "3.11"
      secret-env: WHEELS_TOKEN
      setup-command: uv run --no-project --with requests scripts/download_wheels.py
      extra-checks: |
        uv run lint-imports
```

Commands run with `bash -eo pipefail`, on Windows runners too, and stop at the first failure.

### Cleaning Up

```bash
make clean
```

Removes build artifacts, caches, and temporary files.






## Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run code quality checks automatically before each commit.

### Setup

Install the hooks:

```bash
make repo  # if you haven't already
uv run pre-commit install
```

Checks will now run automatically on every commit.

### Manual Execution

Trigger all hooks manually:

```bash
uv run pre-commit run --all-files
```

This runs the same lint, format and type checks as `make qa`, without the tests.

