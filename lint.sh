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
    echo "⚠️  Black formatting issues found. Run: black --line-length=100 $PY_FILES"
}

# Pylint
echo ""
echo "🐍 Running Pylint..."
pylint --jobs=0 --rcfile=.pylintrc $PY_FILES || true

echo ""
echo "✅ Lint check complete"
