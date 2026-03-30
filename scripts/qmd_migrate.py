#!/usr/bin/env python3
"""
QMD Migration Script
Migrates flat MEMORY.md + USER.md into structured SQLite + ChromaDB storage.
"""

import os
import re
import sqlite3
import uuid
import json
from datetime import datetime

DB_PATH = os.path.expanduser("~/.hermes/memory/warm/memories.db")
CHROMA_PATH = os.path.expanduser("~/.hermes/memory/deep/chroma")
MEMORY_MD = os.path.expanduser("~/.hermes/memories/MEMORY.md")
USER_MD = os.path.expanduser("~/.hermes/memories/USER.md")

# Category mapping from flat file prefixes
CATEGORY_MAP = {
    "About Kawa": "relationship",
    "About Mas Wahyu": "user_preference",
    "Key Principles": "lesson_learned",
    "Blog": "project_context",
    "Image Rules": "tool_config",
    "Social": "credential",
    "Hermes Agent": "workflow",
    "Facebook": "project_context",
    "Runware": "tool_config",
    "Gold Tracker": "tool_config",
    "Name": "user_preference",
    "What to call": "user_preference",
    "Pronouns": "user_preference",
    "Timezone": "user_preference",
    "Notes": "user_preference",
    "Context": "user_preference",
    "User": "user_preference",
    "Wahyu": "user_preference",
}


def parse_entries(content: str, source: str) -> list[dict]:
    """Parse flat memory file into structured entries."""
    entries = []

    # Split by § marker
    raw_entries = content.split("§")

    for raw in raw_entries:
        text = raw.strip()

        # Skip headers and separators
        if not text or text.startswith("_") or text == "---":
            continue

        # Clean up the text
        text = text.strip()

        # Detect category from prefix
        category = "general"
        for prefix, cat in CATEGORY_MAP.items():
            if text.startswith(prefix):
                category = cat
                break

        # Detect tags from content
        tags = []
        tag_patterns = {
            "blog": ["blog", "astro", "cloudflare", "post"],
            "social": ["threads", "linkedin", "facebook", "repliz"],
            "ai": ["runware", "embedding", "model", "ai"],
            "automation": ["cron", "automation", "workflow", "hermes"],
            "business": ["qawwa", "ukm", "bisnis", "enterprise"],
        }

        text_lower = text.lower()
        for tag, keywords in tag_patterns.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)

        entries.append({
            "id": str(uuid.uuid4()),
            "content": text,
            "category": category,
            "tags": tags,
            "source": source,
            "confidence": 0.9,
            "created_at": datetime.utcnow().isoformat(),
        })

    return entries


def migrate_to_sqlite(entries: list[dict], conn: sqlite3.Connection):
    """Insert entries into SQLite."""
    cursor = conn.cursor()

    for entry in entries:
        # Insert memory
        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (id, content, category, created_at, last_accessed,
             access_count, confidence, source, tier, decay_score)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, 'warm', 1.0)
        """, (
            entry["id"],
            entry["content"],
            entry["category"],
            entry["created_at"],
            entry["created_at"],
            entry["confidence"],
            entry["source"],
        ))

        # Insert tags
        for tag in entry["tags"]:
            cursor.execute("""
                INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                VALUES (?, ?)
            """, (entry["id"], tag))

    conn.commit()
    print(f"  SQLite: {len(entries)} records inserted")


def migrate_to_chroma(entries: list[dict]):
    """Insert entries into ChromaDB with embeddings."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="hermes_memory",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')

    ids = []
    documents = []
    metadatas = []

    for entry in entries:
        ids.append(entry["id"])
        documents.append(entry["content"])
        metadatas.append({
            "category": entry["category"],
            "source": entry["source"],
            "confidence": entry["confidence"],
            "tags": ",".join(entry["tags"]),
            "created_at": entry["created_at"],
        })

    # Batch embed and add
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  ChromaDB: {len(entries)} records with embeddings")


def main():
    print("=" * 50)
    print("QMD Migration: Flat Files → Structured Storage")
    print("=" * 50)

    # Parse MEMORY.md
    print(f"\n1. Parsing {MEMORY_MD}")
    with open(MEMORY_MD, 'r') as f:
        memory_content = f.read()
    memory_entries = parse_entries(memory_content, "memory_md")
    print(f"   Found {len(memory_entries)} entries")

    # Parse USER.md
    print(f"\n2. Parsing {USER_MD}")
    with open(USER_MD, 'r') as f:
        user_content = f.read()
    user_entries = parse_entries(user_content, "user_md")
    print(f"   Found {len(user_entries)} entries")

    all_entries = memory_entries + user_entries
    print(f"\n3. Total entries to migrate: {len(all_entries)}")

    # Migrate to SQLite
    print("\n4. Migrating to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    migrate_to_sqlite(all_entries, conn)
    conn.close()

    # Migrate to ChromaDB
    print("\n5. Migrating to ChromaDB...")
    migrate_to_chroma(all_entries)

    # Summary
    print("\n" + "=" * 50)
    print("Migration complete!")
    print(f"  SQLite: {DB_PATH}")
    print(f"  ChromaDB: {CHROMA_PATH}")
    print(f"  Total records: {len(all_entries)}")

    # List categories
    cats = {}
    for e in all_entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
