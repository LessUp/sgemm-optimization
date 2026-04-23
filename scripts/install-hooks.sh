#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
git config core.hooksPath "$repo_root/.githooks"
echo "Configured core.hooksPath to $repo_root/.githooks"
