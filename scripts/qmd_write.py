#!/usr/bin/env python3
"""
QMD Write Engine
Store, update, and manage memory records in SQLite + ChromaDB.
"""

import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

# Paths
CONFIG_PATH = os.path.expanduser("~/.hermes/memory/config.json")
DB_PATH = os.path.expanduser("~/.hermes/memory/warm/memories.db")
CHROMA_PATH = os.path.expanduser("~/.hermes/memory/deep/chroma")
LOG_PATH = os.path.expanduser("~/.hermes/memory/logs/qmd.log")

# Valid categories
VALID_CATEGORIES = [
    "user_preference",
    "project_context",
    "tool_config",
    "lesson_learned",
    "credential",
    "relationship",
    "workflow",
    "general",
]

# Logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("qmd_write")


class QMDWrite:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.db_path = os.path.expanduser(self.config["storage"]["sqlite_path"])
        self.chroma_path = os.path.expanduser(self.config["storage"]["chroma_path"])

        # Lazy init
        self._model = None
        self._chroma_collection = None

        logger.info("QMDWrite initialized")

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    @property
    def chroma_collection(self):
        if self._chroma_collection is None:
            client = chromadb.PersistentClient(path=self.chroma_path)
            self._chroma_collection = client.get_or_create_collection(
                name="hermes_memory",
                metadata={"hnsw:space": "cosine"}
            )
        return self._chroma_collection

    def store(self, content: str, category: str = "general",
              tags: list[str] = None, source: str = "extracted",
              confidence: float = 0.8) -> str:
        """
        Store new memory record.
        Returns the memory ID.
        """
        # Validate category
        if category not in VALID_CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'general'")
            category = "general"

        memory_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        tags = tags or []

        logger.info(f"Storing memory: {memory_id} [{category}] {content[:50]}...")

        # 1. SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO memories
                (id, content, category, created_at, last_accessed,
                 access_count, confidence, source, tier, decay_score)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, 'warm', 1.0)
            """, (memory_id, content, category, now, now, confidence, source))

            # Insert tags
            for tag in tags:
                cursor.execute("""
                    INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                    VALUES (?, ?)
                """, (memory_id, tag))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"SQLite store error: {e}")
            return None

        # 2. ChromaDB
        try:
            embedding = self.model.encode(content).tolist()
            metadata = {
                "category": category,
                "source": source,
                "confidence": confidence,
                "tags": ",".join(tags),
                "created_at": now,
            }

            self.chroma_collection.add(
                ids=[memory_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
            )

        except Exception as e:
            logger.error(f"ChromaDB store error: {e}")
            # Don't fail if ChromaDB fails — SQLite is primary

        logger.info(f"Memory stored: {memory_id}")
        return memory_id

    def update(self, memory_id: str, content: str = None,
               category: str = None, tags: list[str] = None,
               confidence: float = None) -> bool:
        """Update an existing memory record."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build update query dynamically
            updates = []
            params = []

            if content is not None:
                updates.append("content = ?")
                params.append(content)
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            if confidence is not None:
                updates.append("confidence = ?")
                params.append(confidence)

            if updates:
                params.append(memory_id)
                cursor.execute(
                    f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                    params
                )

            # Update tags if provided
            if tags is not None:
                cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
                for tag in tags:
                    cursor.execute(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        (memory_id, tag)
                    )

            conn.commit()
            conn.close()

            # Update ChromaDB if content changed
            if content is not None:
                try:
                    embedding = self.model.encode(content).tolist()
                    self.chroma_collection.update(
                        ids=[memory_id],
                        documents=[content],
                        embeddings=[embedding],
                    )
                except Exception as e:
                    logger.error(f"ChromaDB update error: {e}")

            logger.info(f"Memory updated: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Update error: {e}")
            return False

    def delete(self, memory_id: str) -> bool:
        """Delete a memory record."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete tags first (FK constraint)
            cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

            conn.commit()
            conn.close()

            # Delete from ChromaDB
            try:
                self.chroma_collection.delete(ids=[memory_id])
            except Exception as e:
                logger.error(f"ChromaDB delete error: {e}")

            logger.info(f"Memory deleted: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def list_by_category(self, category: str, limit: int = 20) -> list[dict]:
        """List memories by category."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM memories
            WHERE category = ?
            ORDER BY last_accessed DESC
            LIMIT ?
        """, (category, limit))

        rows = cursor.fetchall()
        conn.close()

        return [dict(r) for r in rows]

    def list_all(self, limit: int = 50) -> list[dict]:
        """List all memories."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM memories
            ORDER BY last_accessed DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Get storage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
        by_category = dict(cursor.fetchall())

        cursor.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier")
        by_tier = dict(cursor.fetchall())

        conn.close()

        return {
            "total": total,
            "by_category": by_category,
            "by_tier": by_tier,
            "chromadb": self.chroma_collection.count(),
        }


def main():
    """CLI interface for testing."""
    import sys

    writer = QMDWrite()

    if len(sys.argv) < 2:
        print("Usage: qmd_write.py <command> [args]")
        print("Commands:")
        print("  store <content> [category] [tags]")
        print("  list [category]")
        print("  stats")
        return

    cmd = sys.argv[1]

    if cmd == "store":
        content = sys.argv[2] if len(sys.argv) > 2 else "Test memory"
        category = sys.argv[3] if len(sys.argv) > 3 else "general"
        tags = sys.argv[4].split(",") if len(sys.argv) > 4 else []

        memory_id = writer.store(content, category, tags)
        print(f"Stored: {memory_id}")

    elif cmd == "list":
        category = sys.argv[2] if len(sys.argv) > 2 else None

        if category:
            memories = writer.list_by_category(category)
        else:
            memories = writer.list_all()

        for m in memories:
            print(f"  [{m['category']}] {m['content'][:80]}...")

    elif cmd == "stats":
        print(json.dumps(writer.get_stats(), indent=2))


if __name__ == "__main__":
    main()
