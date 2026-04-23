#!/usr/bin/env bash
set -euo pipefail

preset="${1:-default}"
source_path="build/${preset}/compile_commands.json"
target_path="compile_commands.json"

if [[ ! -f "${source_path}" ]]; then
    echo "Missing ${source_path}. Run 'cmake --preset ${preset}' first." >&2
    exit 1
fi

ln -sfn "${source_path}" "${target_path}"
echo "Linked ${target_path} -> ${source_path}"
