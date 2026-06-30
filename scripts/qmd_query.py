#!/usr/bin/env python3
"""
QMD Query Engine
3-tier cascade retrieval: Hot → Warm (SQLite FTS5) → Deep (ChromaDB)
"""

import os
import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from qmd_embeddings import AcceleratedEmbedder, get_embedder

# Paths
CONFIG_PATH = os.path.expanduser("~/.hermes/memory/config.json")
DB_PATH = os.path.expanduser("~/.hermes/memory/warm/memories.db")
CHROMA_PATH = os.path.expanduser("~/.hermes/memory/deep/chroma")
HOT_MEMORY = os.path.expanduser("~/.hermes/memories/MEMORY.md")
HOT_USER = os.path.expanduser("~/.hermes/memories/USER.md")
LOG_PATH = os.path.expanduser("~/.hermes/memory/logs/qmd.log")

# Logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("qmd")


@dataclass
class MemoryRecord:
    id: str
    content: str
    category: str
    tier: str
    confidence: float
    decay_score: float
    similarity: float = 0.0
    tags: list = None
    source: str = ""
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0


class QMDQuery:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.db_path = os.path.expanduser(self.config["storage"]["sqlite_path"])
        self.chroma_path = os.path.expanduser(self.config["storage"]["chroma_path"])

        # Initialize embedding model (lazy load)
        self._model = None

        # Initialize ChromaDB client
        self._chroma_client = None
        self._chroma_collection = None
        # Initialize embedding model (lazy load)
        self._embedder = None
        self._embedding_backend = None

    @property
    def model(self):
        """Backward-compatible property returning sentence-transformers model."""
        if self._embedder is None:
            self._embedder = get_embedder(CONFIG_PATH)
        return self._embedder

    @property
    def embedder(self) -> AcceleratedEmbedder:
        """Return accelerated embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(CONFIG_PATH)
        return self._embedder

    @property
    def embedding_backend(self) -> Optional[str]:
        """Return active embedding backend name."""
        if self._embedding_backend is None:
            self._embedding_backend = self.embedder.backend
        return self._embedding_backend

    @property
    def chroma_collection(self):
        if self._chroma_collection is None:
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="hermes_memory",
                metadata={"hnsw:space": "cosine"}
            )
        return self._chroma_collection

    def _extract_essential(self, content: str, max_chars: int) -> str:
        """Extract essential lines from a memory file, respecting char budget."""
        lines = content.split("§")
        essential = []
        current_chars = 0

        for line in lines:
            line = line.strip()
            if not line or line == "---" or line.startswith("_"):
                continue

            # Prioritize identity/preference lines
            is_essential = any(kw in line.lower() for kw in [
                "name:", "role:", "timezone:", "vibe:", "vision:",
                "communication", "principle", "style:", "target:"
            ])

            if is_essential or current_chars < max_chars * 0.7:
                if current_chars + len(line) + 2 <= max_chars:
                    essential.append(line)
                    current_chars += len(line) + 2

        return "§".join(essential)

    def _query_hot(self) -> list[MemoryRecord]:
        """Hot tier: Always include condensed core identity."""
        max_chars = self.config["retrieval"]["hot_max_chars"]
        half_chars = max_chars // 2
        records = []

        # Read and condense MEMORY.md
        if os.path.exists(HOT_MEMORY):
            with open(HOT_MEMORY, 'r') as f:
                content = f.read().strip()
            condensed = self._extract_essential(content, half_chars)
            records.append(MemoryRecord(
                id="hot_memory_md",
                content=condensed,
                category="hot",
                tier="hot",
                confidence=1.0,
                decay_score=1.0,
                similarity=1.0,
                source="hot_memory"
            ))

        # Read and condense USER.md
        if os.path.exists(HOT_USER):
            with open(HOT_USER, 'r') as f:
                content = f.read().strip()
            condensed = self._extract_essential(content, half_chars)
            records.append(MemoryRecord(
                id="hot_user_md",
                content=condensed,
                category="hot",
                tier="hot",
                confidence=1.0,
                decay_score=1.0,
                similarity=1.0,
                source="hot_user"
            ))

        return records

    def _query_warm(self, query_text: str, top_k: int = 5) -> list[MemoryRecord]:
        """Warm tier: SQLite FTS5 keyword search."""
        records = []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # FTS5 search with ranking
            cursor.execute("""
                SELECT m.*, rank
                FROM memories_fts fts
                JOIN memories m ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query_text, top_k))

            rows = cursor.fetchall()

            for row in rows:
                # Get tags
                cursor.execute(
                    "SELECT tag FROM memory_tags WHERE memory_id = ?",
                    (row["id"],)
                )
                tags = [r[0] for r in cursor.fetchall()]

                records.append(MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    category=row["category"],
                    tier="warm",
                    confidence=row["confidence"],
                    decay_score=row["decay_score"],
                    similarity=1.0 / (1.0 + abs(row["rank"])),
                    tags=tags,
                    source=row["source"],
                    created_at=row["created_at"],
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                ))

            conn.close()

        except Exception as e:
            logger.error(f"Warm query error: {e}")

        return records

    def _query_deep(self, query_text: str, top_k: int = 3,
                    min_similarity: float = 0.3) -> list[MemoryRecord]:
        """Deep tier: ChromaDB semantic search with accelerated embeddings."""
        records = []

        try:
            collection = self.chroma_collection

            if collection.count() == 0:
                return records

            # Use accelerated embedder for query embedding
            query_embedding = self.embedder.encode_single(query_text).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # cosine distance → similarity

                    if similarity < min_similarity:
                        continue

                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    records.append(MemoryRecord(
                        id=doc_id,
                        content=results["documents"][0][i],
                        category=metadata.get("category", "unknown"),
                        tier="deep",
                        confidence=metadata.get("confidence", 0.5),
                        decay_score=0.5,
                        similarity=similarity,
                        tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                        source=metadata.get("source", ""),
                        created_at=metadata.get("created_at", ""),
                    ))

        except Exception as e:
            logger.error(f"Deep query error: {e}")

        return records

    def _deduplicate(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Remove duplicate records, keep highest relevance."""
        seen = {}
        for r in records:
            key = r.id
            if key not in seen or r.similarity > seen[key].similarity:
                seen[key] = r
        return list(seen.values())

    def _rank(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Rank records by relevance score."""
        for r in records:
            # Composite score: similarity * decay * confidence
            r._rank_score = r.similarity * r.decay_score * r.confidence

        return sorted(records, key=lambda r: r._rank_score, reverse=True)

    def query(self, message: str, context: list = None) -> list[MemoryRecord]:
        """
        Main query method. Returns relevant memories.
        1. Hot tier (always)
        2. Warm tier (FTS5 keyword)
        3. Deep tier (ChromaDB semantic)
        4. Merge, deduplicate, rank
        """
        logger.info(f"Query: {message[:100]}...")

        # 1. Hot tier - always included
        hot = self._query_hot()

        # 2. Warm tier - keyword search
        warm = self._query_warm(
            message,
            top_k=self.config["retrieval"]["warm_top_k"]
        )

        # 3. Deep tier - semantic search
        deep = self._query_deep(
            message,
            top_k=self.config["retrieval"]["deep_top_k"],
            min_similarity=self.config["retrieval"]["min_similarity"]
        )

        # 4. Merge and deduplicate
        all_records = hot + warm + deep
        unique = self._deduplicate(all_records)

        # 5. Rank
        ranked = self._rank(unique)

        logger.info(f"Results: {len(hot)} hot + {len(warm)} warm + {len(deep)} deep = {len(ranked)} total")

        # Update access counts for returned records
        self._update_access_batch([r for r in ranked if r.tier != "hot"])

        return ranked

    def _update_access_batch(self, records: list[MemoryRecord]):
        """Update last_accessed and access_count for retrieved records."""
        if not records:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            for r in records:
                if r.tier == "warm":
                    cursor.execute("""
                        UPDATE memories
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE id = ?
                    """, (now, r.id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Access update error: {e}")

    def format_for_injection(self, records: list[MemoryRecord],
                             max_chars: int = None) -> str:
        """Format records as compact text for system prompt injection."""
        if max_chars is None:
            max_chars = self.config["retrieval"]["max_total_chars"]

        lines = []
        current_chars = 0

        for r in records:
            # Format entry
            if r.tier == "hot":
                entry = r.content
            else:
                # Structured format for warm/deep
                tags_str = f" [{','.join(r.tags)}]" if r.tags else ""
                entry = f"[{r.category}{tags_str}] {r.content}"

            entry_chars = len(entry)

            # Check budget
            if current_chars + entry_chars > max_chars:
                # Try to fit a truncated version
                remaining = max_chars - current_chars
                if remaining > 50:
                    entry = entry[:remaining - 3] + "..."
                    lines.append(entry)
                break

            lines.append(entry)
            current_chars += entry_chars

        return "\n§\n".join(lines)

    def stats(self) -> dict:
        """Get memory system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total records by tier
        cursor.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier")
        tier_counts = dict(cursor.fetchall())

        # Total records by category
        cursor.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
        category_counts = dict(cursor.fetchall())

        # Average decay
        cursor.execute("SELECT AVG(decay_score) FROM memories")
        avg_decay = cursor.fetchone()[0]

        # ChromaDB count
        chroma_count = self.chroma_collection.count()

        conn.close()

        return {
            "sqlite": tier_counts,
            "chromadb": chroma_count,
            "by_category": category_counts,
            "avg_decay_score": round(avg_decay, 3) if avg_decay else 0,
        }


def main():
    """CLI interface for testing."""
    import sys

    qmd = QMDQuery()

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "blog astro cloudflare"

    print(f"Query: {query_text}\n")

    results = qmd.query(query_text)

    print(f"Found {len(results)} records:\n")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r.tier}] {r.category} (sim={r.similarity:.2f}, decay={r.decay_score:.2f})")
        print(f"     {r.content[:100]}...")
        print()

    formatted = qmd.format_for_injection(results)
    print(f"\nFormatted for injection ({len(formatted)} chars):\n")
    print(formatted)

    print(f"\nStats: {qmd.stats()}")


if __name__ == "__main__":
    main()
