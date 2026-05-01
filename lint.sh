#!/bin/bash
# Quick local linting script that mirrors CI behavior

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔍 Running local lint checks..."

# Determine scope
if [ "$1" = "--full" ]; then
    echo "📋 Full repository scan"
    PY_FILES=$(git ls-files '*.py')
else
    echo "📋 Changed files only (use --full for repo-wide scan)"
    BRANCH=${2:-origin/main}
    PY_FILES=$(git diff --name-only "$BRANCH" -- '*.py' || git ls-files '*.py')
fi

if [ -z "$PY_FILES" ]; then
    echo "✅ No Python files to lint"
    exit 0
fi

# Black check
echo ""
echo "🎨 Checking code formatting with Black..."
black --check --line-length=100 $PY_FILES || {
    BLACK_FILES=$(printf '%s' "$PY_FILES" | tr '\n' ' ')
    BLACK_CMD="black --line-length=100 $BLACK_FILES"
    echo "⚠️  Black formatting issues found. Run: $BLACK_CMD"
}

# Pylint
echo ""
echo "🐍 Running Pylint..."
mkdir -p lint-reports

pylint --jobs=0 --rcfile=.pylintrc \
    --output-format=json \
    $PY_FILES > lint-reports/pylint.json 2>/dev/null || true

pylint --rcfile=.pylintrc --output-format=text \
    $PY_FILES > lint-reports/pylint.txt 2>&1 || true

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "⚠️  No python interpreter found; skipping lint summary generation"
    echo "    Install Python: https://www.python.org/downloads/"
    echo "    Or on Ubuntu: sudo apt-get update && sudo apt-get install -y python3 python3-pip"
else
        "$PYTHON_BIN" << 'EOF'
import json
import os
import sys

report_file = "lint-reports/pylint.json"

if not os.path.exists(report_file):
    print("No lint report generated")
    sys.exit(0)

try:
    with open(report_file) as f:
        results = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    print("Could not parse lint report")
    sys.exit(0)

stats = {}
for msg in results:
    msg_type = msg.get("type", "unknown")
    stats[msg_type] = stats.get(msg_type, 0) + 1

total = len(results)
summary_lines = [
    "## Pylint Summary",
    f"**Scope:** Local run",
    f"**Total Issues:** {total}",
    "",
]

if stats:
    summary_lines.append("### Breakdown by severity:")
    for msg_type in sorted(stats.keys()):
        summary_lines.append(f"- **{msg_type}**: {stats[msg_type]}")
else:
    summary_lines.append("No issues found!")

summary = "\n".join(summary_lines)
print(summary)

with open("lint-reports/summary.md", "w") as f:
    f.write(summary)
EOF
fi

echo ""
echo "✅ Lint check complete"
echo "Results saved to: lint-reports/pylint.txt, lint-reports/pylint.json, lint-reports/summary.md"
