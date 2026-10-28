import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_NAME = "yupi"
PYPROJECT_FILE = Path("pyproject.toml")
CODEOWNERS_FILE = Path("CODEOWNERS")
QA_STATUS_CHECK_NAME = "qa / Run QA"
DEFAULT_BRANCH = "main"
BRANCH_RULESET_NAME = "protect main"
RELEASE_DEPLOY_KEY_TITLE = "cookie-pyrate release"
RELEASE_KEY_SECRET_NAME = "RELEASE_SSH_KEY"  # noqa: S105
REPO_ADMIN_ROLE_ID = 5
GIT_DOWNLOAD_URL = "https://git-scm.com/downloads"
GH_DOWNLOAD_URL = "https://cli.github.com"
INSTALL_URLS = {"gh": GH_DOWNLOAD_URL, "git": GIT_DOWNLOAD_URL, "ssh-keygen": GIT_DOWNLOAD_URL}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and configure this project's remote.")
    parser.add_argument("--public", action="store_true", help="make the remote repository public")
    parser.add_argument(
        "--owner", default="", help="create the repository under this organization instead"
    )
    args = parser.parse_args()
    create_remote("public" if args.public else "private", repository_name(args.owner))


def create_remote(visibility: str, repository: str) -> None:
    check_github_access()
    create_remote_repository(visibility, repository)
    push_branch()
    generate_and_upload_release_key()
    try_protect_default_branch()
    try_enable_security_updates()
    print(f"\nRepository ready on branch '{DEFAULT_BRANCH}'.")


def check_github_access() -> None:
    if not _succeeds("gh", "auth", "status"):
        raise SystemExit("error: the GitHub CLI is not authenticated; run `gh auth login`")


def repository_name(owner: str) -> str:
    return f"{owner}/{REPO_NAME}" if owner else REPO_NAME


def create_remote_repository(visibility: str, repository: str) -> None:
    if _succeeds("git", "remote", "get-url", "origin"):
        print("· remote 'origin' already configured")
        return

    if _succeeds("gh", "repo", "view", repository):
        url = _existing_remote_url(repository)
        _output("git", "remote", "add", "origin", url)
        print(f"· linked the existing remote {url}")
        return

    description = tomllib.loads(PYPROJECT_FILE.read_text())["project"]["description"]
    creation = ["repo", "create", repository, f"--{visibility}", "--description", description]
    _output("gh", *creation, "--source", ".", "--remote", "origin")
    print(f"· created the {visibility} repository {repository}")


def push_branch() -> None:
    if _succeeds("git", "rev-parse", "--abbrev-ref", DEFAULT_BRANCH + "@{upstream}"):
        print(f"· branch '{DEFAULT_BRANCH}' already tracks a remote")
        return
    _output("git", "push", "--set-upstream", "origin", DEFAULT_BRANCH)
    print(f"· pushed '{DEFAULT_BRANCH}' and set it as upstream")


def generate_and_upload_release_key() -> None:
    repository = _get_repository_id()
    for key_id in _get_release_key_ids(repository):
        _output("gh", "api", "--method", "DELETE", f"repos/{repository}/keys/{key_id}")

    with TemporaryDirectory() as directory:
        key = Path(directory) / "release_key"
        keygen = ["-t", "ed25519", "-N", "", "-C", RELEASE_DEPLOY_KEY_TITLE, "-f", str(key)]
        _output("ssh-keygen", *keygen)
        add = ["repo", "deploy-key", "add", f"{key}.pub", "--title", RELEASE_DEPLOY_KEY_TITLE]
        _output("gh", *add, "--allow-write")
        _output("gh", "secret", "set", RELEASE_KEY_SECRET_NAME, stdin=key.read_text())

    print(f"· added the release deploy key and the {RELEASE_KEY_SECRET_NAME} secret")


def try_protect_default_branch() -> None:
    repository = _get_repository_id()
    query = f'.[] | select(.name=="{BRANCH_RULESET_NAME}") | .id'
    listing = _run("gh", "api", f"repos/{repository}/rulesets", "--jq", query)
    if listing.returncode != 0:
        _warn_unprotected(listing)
        return

    ruleset_id = _first_line(listing.stdout)
    method, endpoint = ("PUT", f"repos/{repository}/rulesets/{ruleset_id}")
    if ruleset_id is None:
        method, endpoint = ("POST", f"repos/{repository}/rulesets")

    with TemporaryDirectory() as directory:
        ruleset_file = Path(directory) / "ruleset.json"
        ruleset_file.write_text(json.dumps(_get_ruleset_config(), indent=2))
        result = _run("gh", "api", "--method", method, endpoint, "--input", str(ruleset_file))

    if result.returncode != 0:
        _warn_unprotected(result)
        return
    print(f"· protected '{DEFAULT_BRANCH}'")


def try_enable_security_updates() -> None:
    repository = _get_repository_id()
    # Security updates need the alerts first; both are idempotent.
    for feature in ("vulnerability-alerts", "automated-security-fixes"):
        result = _run("gh", "api", "--method", "PUT", f"repos/{repository}/{feature}")
        if result.returncode != 0:
            print(
                "warning: left Dependabot security updates off. "
                f"GitHub said: {_details(result)}\n"
                "The repository is otherwise ready: resolve that and re-run `make repo`.",
                file=sys.stderr,
            )
            return
    print("· turned on Dependabot alerts and security updates")


def _warn_unprotected(result: subprocess.CompletedProcess[str]) -> None:
    print(
        f"warning: left '{DEFAULT_BRANCH}' unprotected. GitHub said: {_details(result)}\n"
        "The repository is otherwise ready: resolve that and re-run `make repo`.",
        file=sys.stderr,
    )


def _get_ruleset_config() -> dict[str, Any]:
    return {
        "name": BRANCH_RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [
            {"actor_id": None, "actor_type": "DeployKey", "bypass_mode": "always"},
            {
                "actor_id": REPO_ADMIN_ROLE_ID,
                "actor_type": "RepositoryRole",
                "bypass_mode": "always",
            },
        ],
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "dismiss_stale_reviews_on_push": True,
                    "require_code_owner_review": CODEOWNERS_FILE.is_file(),
                    "require_last_push_approval": False,
                    "required_review_thread_resolution": True,
                    "allowed_merge_methods": ["merge"],
                },
            },
            {
                "type": "required_status_checks",
                "parameters": {
                    "required_status_checks": [{"context": QA_STATUS_CHECK_NAME}],
                    "strict_required_status_checks_policy": True,
                },
            },
        ],
    }


def _get_release_key_ids(repository: str) -> list[str]:
    query = f'.[] | select(.title=="{RELEASE_DEPLOY_KEY_TITLE}") | .id'
    return _output("gh", "api", f"repos/{repository}/keys", "--jq", query).split()


def _first_line(output: str) -> str | None:
    lines = output.strip().splitlines()
    return lines[0] if lines else None


def _get_repository_id() -> str:
    return _output("gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")


def _existing_remote_url(repository: str) -> str:
    if _output("gh", "config", "get", "git_protocol") == "ssh":
        return _output("gh", "repo", "view", repository, "--json", "sshUrl", "--jq", ".sshUrl")
    return _output("gh", "repo", "view", repository, "--json", "url", "--jq", ".url") + ".git"


def _executable(program: str) -> str:
    path = shutil.which(program)
    if path is None:
        raise SystemExit(f"error: {program} is required; install it from {INSTALL_URLS[program]}")
    return path


def _run(program: str, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    command = [_executable(program), *args]
    return subprocess.run(command, check=False, capture_output=True, text=True, input=stdin)


def _succeeds(program: str, *args: str) -> bool:
    return _run(program, *args).returncode == 0


def _output(program: str, *args: str, stdin: str | None = None) -> str:
    result = _run(program, *args, stdin=stdin)
    if result.returncode != 0:
        raise SystemExit(f"error: `{program} {' '.join(args)}` failed: {_details(result)}")
    return result.stdout.strip()


def _details(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout).strip()


if __name__ == "__main__":
    main()
