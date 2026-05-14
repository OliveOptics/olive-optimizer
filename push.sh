#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Staging changes..."
git add -A
echo "Committing..."
git commit -m "${1:-Update}"
echo "Pushing..."
git push
echo "Done!"
