#!/bin/bash
# QMD Installer — Copy full QMD system to target agent
# Usage: bash qmd_install.sh [target_hermes_dir]
# Example: bash qmd_install.sh ~/.hermes
#          bash qmd_install.sh /home/otheruser/.hermes

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Source (current installation)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_MEMORY="$(dirname "$SCRIPT_DIR")"  # ~/.hermes/memory
SOURCE_SKILL="$(dirname "$(dirname "$SOURCE_MEMORY")")/skills/productivity/qmd"

# Target
TARGET="${1:-$HOME/.hermes}"
TARGET_MEMORY="$TARGET/memory"
TARGET_SKILL="$TARGET/skills/productivity/qmd"

echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  QMD Installer — Memory Upgrade System${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo "Source:  $SOURCE_MEMORY"
echo "Target:  $TARGET"
echo ""

# Check source exists
if [ ! -d "$SOURCE_MEMORY/scripts" ]; then
    echo -e "${RED}ERROR: Source scripts not found at $SOURCE_MEMORY/scripts${NC}"
    exit 1
fi

# Create directories
echo -e "${YELLOW}[1/6] Creating directories...${NC}"
mkdir -p "$TARGET_MEMORY"/{warm,deep,scripts,logs}
mkdir -p "$TARGET_SKILL"

# Copy scripts
echo -e "${YELLOW}[2/6] Copying scripts...${NC}"
for f in "$SOURCE_MEMORY"/scripts/qmd_*.py "$SOURCE_MEMORY"/scripts/qmd_schema.sql; do
    [ -f "$f" ] && cp -v --no-clobber "$f" "$TARGET_MEMORY/scripts/" 2>/dev/null || true
done

# Copy config
echo -e "${YELLOW}[3/6] Copying config...${NC}"
if [ -f "$TARGET_MEMORY/config.json" ]; then
    echo "  config.json already exists — skipping (keeping existing config)"
else
    cp -v "$SOURCE_MEMORY/config.json" "$TARGET_MEMORY/config.json"
fi

# Copy skill
echo -e "${YELLOW}[4/6] Copying skill...${NC}"
cp -v "$SOURCE_SKILL/SKILL.md" "$TARGET_SKILL/SKILL.md"

# Init database
echo -e "${YELLOW}[5/6] Initializing SQLite database...${NC}"
if [ -f "$TARGET_MEMORY/warm/memories.db" ]; then
    echo "  memories.db already exists — skipping"
else
    sqlite3 "$TARGET_MEMORY/warm/memories.db" < "$TARGET_MEMORY/scripts/qmd_schema.sql"
    echo "  Database created."
fi

# Check dependencies
echo -e "${YELLOW}[6/6] Checking Python dependencies...${NC}"
python3 -c "import chromadb" 2>/dev/null && echo "  ✅ chromadb" || echo "  ❌ chromadb — run: pip install chromadb"
python3 -c "import sentence_transformers" 2>/dev/null && echo "  ✅ sentence-transformers" || echo "  ❌ sentence-transformers — run: pip install sentence-transformers"
python3 -c "import sqlite3" 2>/dev/null && echo "  ✅ sqlite3" || echo "  ❌ sqlite3"
python3 -c "import onnxruntime" 2>/dev/null && echo "  ✅ onnxruntime" || echo "  ⚠️ onnxruntime — optional, for GPU acceleration: pip install onnxruntime-gpu optimum"
python3 -c "import optimum" 2>/dev/null && echo "  ✅ optimum" || echo "  ⚠️ optimum — optional, for GPU acceleration: pip install optimum[onnxruntime]"

echo ""
echo "Optional GPU acceleration:"
echo "  NVIDIA CUDA:  pip install onnxruntime-gpu optimum[onnxruntime]"
echo "  Apple CoreML: pip install onnxruntime optimum[onnxruntime]"
echo "  Windows DirectML: pip install onnxruntime-directml optimum[onnxruntime]"

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo "Files installed:"
echo "  Skill:   $TARGET_SKILL/SKILL.md"
echo "  Config:  $TARGET_MEMORY/config.json"
echo "  Scripts: $TARGET_MEMORY/scripts/"
echo "  Database: $TARGET_MEMORY/warm/memories.db"
echo ""
echo "Next steps:"
echo "  1. pip install chromadb sentence-transformers  (if missing)"
echo "  2. Migrate existing memory: python3 $TARGET_MEMORY/scripts/qmd_migrate.py"
echo "  3. Verify: python3 $TARGET_MEMORY/scripts/qmd_integration_test.py"
echo "  4. Setup decay cron:"
echo "     cronjob(action='create', name='qmd-decay-daily', schedule='0 3 * * *',"
echo "             prompt='Run python3 $TARGET_MEMORY/scripts/qmd_decay.py')"
echo ""
