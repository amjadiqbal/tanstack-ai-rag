#!/usr/bin/env bash
# Local CI: GitHub workflows never run on this repo, so every change runs this and pastes the summary
# into the PR or commit notes. Typecheck, lint, tests, build, publint, then a dry-run pack.
# Set PGVECTOR_URL (e.g. postgres://postgres:rag@127.0.0.1:5544/rag, pgvector/pgvector:pg17 in Docker) to also run
# the Postgres suites through pg, postgres.js and Drizzle; without it they are skipped and PGlite covers pgvector.
set -uo pipefail
cd "$(dirname "$0")/.."

RESULTS=(); FAILED=0
step() { local name=$1; shift; printf '\n== %s\n' "$name"; if "$@"; then RESULTS+=("pass  $name"); else RESULTS+=("FAIL  $name"); FAILED=1; fi; }

[ -f package.json ] || { echo "no package.json yet"; exit 1; }
step "install"   npm ci --no-audit --no-fund
step "typecheck" npm run --silent typecheck
step "lint"      npm run --silent lint
step "test${PGVECTOR_URL:+ (+ Docker pgvector)}" npm test --silent
step "build"     npm run --silent build
step "publint"   npx --no-install publint
step "pack"      npm pack --dry-run

printf '\n== summary\n'; printf '%s\n' "${RESULTS[@]}"
exit $FAILED
