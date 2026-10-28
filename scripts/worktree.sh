#!/usr/bin/env bash

set -euo pipefail

backup_dir=''
backup_created=false
migration_started=false
migration_succeeded=false

error() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

log() {
    printf '==> %s\n' "$1"
}

write_readme() {
    cat <<'EOF'
# Worktrees layout

Each folder (beside `.bare`) holds an **independent** state of the whole project
checked out at a specific branch, even with different uncommitted changes on
each of them; this is called a worktree. No more `git stash` or fake commits to
switch branches when you have uncommitted changes. Just `cd` into the folder for
the branch you want to work on.

## Create a worktree

From the repository root (the folder containing `.bare` and `.git`), create a
new worktree with:

```bash
git worktree add branch-name
```

This checks out `branch-name` into a folder of the same name, creating the
branch first if it doesn't exist yet.

For branch names that contain `/` (e.g. `feat/example`), pass the name twice
so the folder and branch stay in sync:

```bash
git worktree add -b feat/example feat/example
```

## Working inside a worktree

Inside a worktree, you can run all the usual Git commands (e.g. `git status`,
`git add`, `git commit`, etc.).

## Remove a worktree

Simply run this from the repository root:

```bash
git worktree remove branch-name
```
EOF
}

add_worktree() {
    local branch=$1 out
    out=$(git worktree add "$branch" "$branch" 2>&1) \
        || error "unable to create worktree for '$branch': $out"
    created_worktrees+=("$branch")
}

created_worktrees=()

rollback() {
    if [ "$migration_succeeded" = true ] || [ "$backup_created" = false ]; then
        return
    fi

    set +e

    if [ "$migration_started" = true ]; then
        # Restore the original repository files from the backup
        find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        find "$backup_dir" -mindepth 1 -maxdepth 1 -exec cp -a {} . \;
    fi

    rm -rf -- "$backup_dir"
}

trap rollback EXIT

# ------------------------------------------------------------------------------------
# Let's first validate the preconditions for running the migration
# ------------------------------------------------------------------------------------
repo_root=$(git rev-parse --show-toplevel 2>/dev/null) \
    || error 'worktree must run inside a git repository'
[ "$(pwd -P)" = "$(cd "$repo_root" && pwd -P)" ] \
    || error 'worktree must run from the repository root'
[ -d .git ] \
    || error 'repository is already converted or has an unsupported .git file'
[ ! -e .bare ] \
    || error '.bare already exists; refusing to overwrite it'

# ------------------------------------------------------------------------------------
# Next we keep track of the current branch as it may be different from the default
# branch. If that is the case, we will create later a worktree for the current
# branch as well.
# ------------------------------------------------------------------------------------
current_branch=$(git symbolic-ref --quiet --short HEAD) \
    || error 'detached HEAD is not supported'

# ------------------------------------------------------------------------------------
# Refuse to migrate if there's state that would be silently lost. The restore
# step later only copies working-tree files back into the current branch's
# worktree — it can't carry over the index (staged changes) or anything living
# inside .git (in-progress merge/rebase/cherry-pick/bisect), because that .git
# gets renamed to .bare and shared across all worktrees.
# ------------------------------------------------------------------------------------
for state_file in MERGE_HEAD CHERRY_PICK_HEAD BISECT_LOG rebase-merge rebase-apply; do
    [ ! -e ".git/$state_file" ] \
        || error "an operation is in progress (.git/$state_file exists); finish or abort it first"
done
git diff --cached --quiet \
    || error 'you have staged changes; commit or unstage them before running this script'

# ------------------------------------------------------------------------------------
# Now we create a backup of the original repository so that we can restore it in case
# of any failure during the migration.
# ------------------------------------------------------------------------------------
backup_dir=$(dirname "$repo_root")/$(basename "$repo_root")-legacy-bak
[ ! -e "$backup_dir" ] \
    || error "backup path already exists: $backup_dir"
log "Creating backup: $backup_dir"
mkdir "$backup_dir" || error "unable to create backup directory: $backup_dir"
backup_created=true
cp -a "$repo_root"/. "$backup_dir"/ || error "unable to copy the repository to: $backup_dir"
migration_started=true
log 'Backup created'

# ------------------------------------------------------------------------------------
# Remove the original repository files from the root.
# ------------------------------------------------------------------------------------
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf -- {} +

# ------------------------------------------------------------------------------------
# Next we fetch all remotes and determine the default branch of the origin remote.
# ------------------------------------------------------------------------------------
log 'Fetching remotes'
fetch_out=$(git fetch --all 2>&1) || error "unable to fetch all remotes: $fetch_out"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-head origin -a >/dev/null 2>&1 \
        || error "unable to determine origin's default branch"
    default_ref=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD) \
        || error 'origin has no default branch'
    default_branch=${default_ref#origin/}
else
    default_branch=$current_branch
fi

# ------------------------------------------------------------------------------------
# Branch names containing '/' become nested directories when used as worktree
# paths (e.g. `release/1.0` -> ./release/1.0). If one of our two branch names
# is a path-prefix of the other, the second `git worktree add` would try to
# create a directory inside a path that's already a worktree.
# ------------------------------------------------------------------------------------
if [ "$current_branch" != "$default_branch" ]; then
    case "$current_branch" in
        "$default_branch"/*) error "branch '$current_branch' is nested under '$default_branch'; their worktree paths would collide" ;;
    esac
    case "$default_branch" in
        "$current_branch"/*) error "branch '$default_branch' is nested under '$current_branch'; their worktree paths would collide" ;;
    esac
fi

# ------------------------------------------------------------------------------------
# Convert the repository to a bare repository
# ------------------------------------------------------------------------------------
mv .git .bare
printf 'gitdir: ./.bare\n' > .git
git config --file .bare/config core.bare true

# ------------------------------------------------------------------------------------
# Now we create a worktree for the default branch
# ------------------------------------------------------------------------------------
log "Creating worktree: $default_branch"
if git show-ref --verify --quiet "refs/heads/$default_branch"; then
    # Default branch already exists locally
    add_worktree "$default_branch"
else
    # Default branch does not exist locally, we need to create it and track the remote
    out=$(git worktree add --track -b "$default_branch" "$default_branch" "origin/$default_branch" 2>&1) \
        || error "unable to create worktree for '$default_branch': $out"
    created_worktrees+=("$default_branch")
fi

# ------------------------------------------------------------------------------------
# If the current branch is different from the default branch, we create a worktree
# for it as well and restore the original project files into it.
# ------------------------------------------------------------------------------------
if [ "$current_branch" != "$default_branch" ]; then
    log "Creating worktree: $current_branch"
    add_worktree "$current_branch"
fi

current_worktree="$repo_root/$current_branch"
log "Restoring repo state to worktree '$current_branch' as it was before the migration"
if ! find "$backup_dir" -mindepth 1 -maxdepth 1 ! -name .git \
    -exec cp -a {} "$current_worktree"/ \;; then
    error "unable to restore the original files to: $current_worktree"
fi

# ------------------------------------------------------------------------------------
# Finally, we write a README.md file to the repository root explaining the new
# worktrees layout and how to use it.
# ------------------------------------------------------------------------------------
if ! write_readme > README.md; then
    error 'unable to write README.md to the repository root'
fi

printf '\nWorktree migration complete.\n'
printf 'Default branch: %s\n' "$default_branch"
[ "$current_branch" = "$default_branch" ] \
    || printf 'Current branch: %s\n' "$current_branch"

printf 'Backup retained at sibling path: %s\n' "$backup_dir"
migration_succeeded=true