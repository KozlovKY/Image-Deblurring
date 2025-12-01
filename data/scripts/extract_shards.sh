#!/usr/bin/env bash
set -euo pipefail


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PART_DIR="${1:-"$PROJECT_ROOT/data/zip"}"
TARGET_DIR="${2:-"$PROJECT_ROOT/data/datasets"}"

if [[ ! -d "$PART_DIR" ]]; then
  echo "Parts directory not found: $PART_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

shopt import_expand_aliases || true
shopt -s nullglob

shards=()
for file in "$PART_DIR"/*.tar.*; do
  if [[ -f "$file" && "$file" != *.dvc ]]; then
    shards+=("$file")
  fi
done

if [[ ${#shards[@]} -eq 0 ]]; then
  echo "No shard archives (*.tar.*) found in: $PART_DIR" >&2
  exit 1
fi

for shard in "${shards[@]}"; do
  tar -xJvf "$shard" -C "$TARGET_DIR" || true
done
