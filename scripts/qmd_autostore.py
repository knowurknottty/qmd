#!/usr/bin/env python3
"""
QMD Auto-Store
Extract and store new facts from conversation context.

Usage:
  python3 qmd_autostore.py "fact content" [category] [tag1,tag2]
  python3 qmd_autostore.py --detect "user message" "assistant response"
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmd_write import QMDWrite

# Patterns that indicate storable facts
FACT_PATTERNS = [
    # Direct statements
    (r"(?:ingat|remember|catat|note|simpan)[:\s]+(.+)", "general"),
    # Config/setup info — domain/repo changes
    (r"(?:domain|url|repo|github)\S*\s+.*?(?:ganti|pindah|pake|pakai)\s+(?:ke\s+)?(\S+\.\S+)", "tool_config"),
    (r"(?:domain|url|repo|github)\S*\s+.*?(?:pindah|baru|:)\s+(?:ke\s+)?(\S+/\S+)", "tool_config"),
    (r"(?:api|token|key)\S*\s+.*?(?:baru|:)\s+(\S+)", "tool_config"),
    # Schedules/times
    (r"(?:jadwal|schedule|cron|setiap|tiap|daily|weekly)[\s:]+(.+)", "workflow"),
    # Preferences
    (r"(?:prefer|suka|mau|lebih baik|jangan)[\s:]+(.+)", "user_preference"),
    # Decisions
    (r"(?:keputusan|decided|pilih|pilihan)[:\s]+(.+)", "lesson_learned"),
]

# Keywords that suggest a new fact
SIGNAL_KEYWORDS = [
    "baru", "new", "update", "ubah", "change", "ganti", "replace",
    "tambah", "add", "hapus", "remove", "jangan", "don't",
    "ingat", "remember", "catat", "note", "save", "simpan",
    "domain", "url", "api", "key", "token", "password",
    "cron", "schedule", "jadwal",
]


def detect_facts(user_msg: str, assistant_msg: str = "") -> list[dict]:
    """
    Detect storable facts from conversation exchange.
    Returns list of {content, category, tags, confidence}.
    """
    facts = []
    combined = f"{user_msg} {assistant_msg}".lower()

    # Check if message contains signal keywords
    has_signal = any(kw in combined for kw in SIGNAL_KEYWORDS)

    # Pattern-based extraction (user message only for content)
    for pattern, category in FACT_PATTERNS:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for match in matches:
            if len(match) > 10:  # Filter noise
                facts.append({
                    "content": match.strip(),
                    "category": category,
                    "tags": extract_tags(match),
                    "confidence": 0.8 if has_signal else 0.6,
                })

    # Extract explicit "remember this" facts
    remember_patterns = [
        r"ingat(?:lah)?[:\s]+(.+?)(?:\.|$)",
        r"catat[:\s]+(.+?)(?:\.|$)",
        r"simpan[:\s]+(.+?)(?:\.|$)",
        r"remember[:\s]+(.+?)(?:\.|$)",
        r"note[:\s]+(.+?)(?:\.|$)",
    ]
    for pattern in remember_patterns:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for match in matches:
            if len(match) > 5:
                facts.append({
                    "content": match.strip(),
                    "category": "general",
                    "tags": extract_tags(match),
                    "confidence": 0.9,
                })

    # Deduplicate
    seen = set()
    unique = []
    for f in facts:
        key = f["content"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def extract_tags(text: str) -> list[str]:
    """Extract relevant tags from text."""
    tag_keywords = {
        "blog": ["blog", "artikel", "post", "tulisan"],
        "social": ["threads", "linkedin", "facebook", "instagram", "twitter"],
        "deploy": ["deploy", "publish", "push", "cloudflare", "vercel"],
        "api": ["api", "endpoint", "token", "key", "credential"],
        "cron": ["cron", "schedule", "jadwal", "otomatis", "daily", "weekly"],
        "design": ["gambar", "image", "design", "logo", "banner"],
        "ai": ["ai", "llm", "model", "gpt", "claude", "agent"],
        "automation": ["automation", "automasi", "otomatis", "workflow"],
        "ukm": ["ukm", "bisnis", "business", "toko"],
    }

    tags = []
    text_lower = text.lower()
    for tag, keywords in tag_keywords.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)

    return tags


def store_fact(content: str, category: str = "general",
               tags: list[str] = None, confidence: float = 0.8) -> str:
    """Store a single fact to QMD."""
    writer = QMDWrite()

    # Check for duplicates (similar content already exists)
    import sqlite3
    db_path = os.path.expanduser("~/.hermes/memory/warm/memories.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Simple duplicate check: exact or very similar content
    c.execute("SELECT id, content FROM memories")
    existing = c.fetchall()

    content_lower = content.lower().strip()
    for eid, econtent in existing:
        # Exact match
        if econtent.lower().strip() == content_lower:
            conn.close()
            return f"SKIP (duplicate): {eid}"
        # High overlap (>80% of shorter string)
        if len(content_lower) > 20:
            words_new = set(content_lower.split())
            words_exist = set(econtent.lower().split())
            if words_new and words_exist:
                overlap = len(words_new & words_exist) / min(len(words_new), len(words_exist))
                if overlap > 0.8:
                    conn.close()
                    return f"SKIP (similar): {eid}"

    conn.close()

    # Store
    memory_id = writer.store(content, category, tags or [], confidence=confidence)
    return f"STORED: {memory_id} [{category}]"


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  qmd_autostore.py store <content> [category] [tag1,tag2]")
        print("  qmd_autostore.py detect <user_msg> [assistant_msg]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "store":
        content = sys.argv[2] if len(sys.argv) > 2 else ""
        category = sys.argv[3] if len(sys.argv) > 3 else "general"
        tags = sys.argv[4].split(",") if len(sys.argv) > 4 else []
        print(store_fact(content, category, tags))

    elif cmd == "detect":
        user_msg = sys.argv[2] if len(sys.argv) > 2 else ""
        assistant_msg = sys.argv[3] if len(sys.argv) > 3 else ""

        facts = detect_facts(user_msg, assistant_msg)
        if not facts:
            print("No facts detected.")
        else:
            print(f"Detected {len(facts)} fact(s):")
            for f in facts:
                result = store_fact(f["content"], f["category"], f["tags"], f["confidence"])
                print(f"  {result}")
                print(f"    Content: {f['content'][:60]}...")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
