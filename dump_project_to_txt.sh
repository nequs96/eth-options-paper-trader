#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
OUT="$ROOT/project_code_dump.txt"

# Remove previous dump so it does not include itself
rm -f "$OUT"

echo "Creating project dump..."
echo "Project root: $ROOT"
echo "Output file: $OUT"

{
  echo "PROJECT CODE DUMP"
  echo "================="
  echo ""
  echo "Project root:"
  echo "$ROOT"
  echo ""
  echo "Generated at:"
  date
  echo ""
  echo "================================================================================"
  echo "INCLUDED FILE TREE"
  echo "================================================================================"
  echo ""

  find "$ROOT" \
    \( \
      -path "$ROOT/.git" -o \
      -path "$ROOT/.idea" -o \
      -path "$ROOT/.vscode" -o \
      -path "$ROOT/.pytest_cache" -o \
      -path "$ROOT/.mypy_cache" -o \
      -path "$ROOT/.ruff_cache" -o \
      -path "$ROOT/__pycache__" -o \
      -path "$ROOT/venv" -o \
      -path "$ROOT/.venv" -o \
      -path "$ROOT/env" -o \
      -path "$ROOT/node_modules" -o \
      -path "$ROOT/outputs" -o \
      -path "$ROOT/logs" -o \
      -path "$ROOT/data" -o \
      -path "$ROOT/storage" -o \
      -path "$ROOT/audit_bundle" -o \
      -path "$ROOT/q32_profitability_robustness_patch" \
    \) -prune -o \
    -type f \
    \( \
      -name "*.py" -o \
      -name "*.md" -o \
      -name "*.txt" -o \
      -name "*.json" -o \
      -name "*.yaml" -o \
      -name "*.yml" -o \
      -name "*.toml" -o \
      -name "*.ini" -o \
      -name "*.cfg" -o \
      -name "*.sh" -o \
      -name "*.env.example" \
    \) \
    ! -name "project_code_dump.txt" \
    ! -name "code_review_dump.txt" \
    ! -name "audit_bundle_loss_review_combined.txt" \
    ! -name "q32_outputs_combined.txt" \
    ! -name ".env" \
    ! -name ".DS_Store" \
    ! -name "*.pyc" \
    ! -name "*.pyo" \
    ! -name "*.zip" \
    ! -name "*.gz" \
    ! -name "*.tar" \
    ! -name "*.db" \
    ! -name "*.sqlite" \
    ! -name "*.sqlite3" \
    ! -name "*.csv" \
    ! -name "*.log" \
    ! -name "*.png" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.pdf" \
    ! -name "*.html" \
    -size -300k \
    | sort \
    | sed "s#^$ROOT/##"

  echo ""
  echo "================================================================================"
  echo "FILE CONTENTS"
  echo "================================================================================"
  echo ""

} >> "$OUT"

find "$ROOT" \
  \( \
    -path "$ROOT/.git" -o \
    -path "$ROOT/.idea" -o \
    -path "$ROOT/.vscode" -o \
    -path "$ROOT/.pytest_cache" -o \
    -path "$ROOT/.mypy_cache" -o \
    -path "$ROOT/.ruff_cache" -o \
    -path "$ROOT/__pycache__" -o \
    -path "$ROOT/venv" -o \
    -path "$ROOT/.venv" -o \
    -path "$ROOT/env" -o \
    -path "$ROOT/node_modules" -o \
    -path "$ROOT/outputs" -o \
    -path "$ROOT/logs" -o \
    -path "$ROOT/data" -o \
    -path "$ROOT/storage" -o \
    -path "$ROOT/audit_bundle" -o \
    -path "$ROOT/q32_profitability_robustness_patch" \
  \) -prune -o \
  -type f \
  \( \
    -name "*.py" -o \
    -name "*.md" -o \
    -name "*.txt" -o \
    -name "*.json" -o \
    -name "*.yaml" -o \
    -name "*.yml" -o \
    -name "*.toml" -o \
    -name "*.ini" -o \
    -name "*.cfg" -o \
    -name "*.sh" -o \
    -name "*.env.example" \
  \) \
  ! -name "project_code_dump.txt" \
  ! -name "code_review_dump.txt" \
  ! -name "audit_bundle_loss_review_combined.txt" \
  ! -name "q32_outputs_combined.txt" \
  ! -name ".env" \
  ! -name ".DS_Store" \
  ! -name "*.pyc" \
  ! -name "*.pyo" \
  ! -name "*.zip" \
  ! -name "*.gz" \
  ! -name "*.tar" \
  ! -name "*.db" \
  ! -name "*.sqlite" \
  ! -name "*.sqlite3" \
  ! -name "*.csv" \
  ! -name "*.log" \
  ! -name "*.png" \ccccccccccc
  ! -name "*.jpg" \
  ! -name "*.jpeg" \
  ! -name "*.pdf" \
  ! -name "*.html" \
  -size -300k \
  | sort \
  | while IFS= read -r file; do
      rel="${file#$ROOT/}"

      {
        echo ""
        echo "================================================================================"
        echo "FILE: $rel"
        echo "================================================================================"
        echo ""

        # Skip binary files defensively
        if LC_ALL=C grep -q $'\000' "$file"; then
          echo "[SKIPPED BINARY FILE]"
        else
          cat "$file"
        fi

        echo ""
      } >> "$OUT"
    done

{
  echo ""
  echo "================================================================================"
  echo "DONE"
  echo "================================================================================"
  echo ""
  echo "Output file:"
  echo "$OUT"
  echo ""
  echo "File size:"
  ls -lh "$OUT"
} >> "$OUT"

echo "Done."
ls -lh "$OUT"
