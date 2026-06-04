---
name: commit-push-pr
description: Use when the user wants to commit changes, push to remote, and open a pull request in one step. Triggers on requests like "commit and push", "open a PR", "create pull request", or "ship this".
---

# Commit, Push, and Open a PR

## Overview

Stages all changes, commits with an appropriate message, pushes the branch, and opens a pull request — all in one sequence.

## Steps

1. **Check status** — run `git status` and `git diff HEAD` to understand what changed
2. **Branch** — if on `main`/`master`, create a new branch first
3. **Commit** — stage relevant files and commit with a descriptive message
4. **Push** — push the branch to `origin`
5. **PR** — create a pull request with `gh pr create`

Do all steps in a single response with parallel tool calls where possible.

## Quick Reference

```bash
# Check state
git status
git diff HEAD
git branch --show-current

# If on main, branch first
git checkout -b <feature-branch>

# Stage and commit
git add <files>
git commit -m "feat: <summary>"

# Push and open PR
git push -u origin HEAD
gh pr create --title "<title>" --body "<body>"
```

## PR Body Template

```
## Summary
- <bullet point changes>

## Test plan
- [ ] <what to verify>
```

## Common Mistakes

- Committing on `main` directly — always branch first
- Using `git add -A` without checking for secrets/large files — prefer named files
- Empty PR body — always include a summary and test plan
