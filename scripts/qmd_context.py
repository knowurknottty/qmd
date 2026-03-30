#!/usr/bin/env python3
"""
QMD Context Generator
Lightweight wrapper untuk auto-query sebelum respond.
Output: formatted context string siap inject.

Usage:
  python3 qmd_context.py "user message here"
  python3 qmd_context.py --test
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmd_query import QMDQuery


def get_context(message: str, max_chars: int = 1500) -> str:
    """Query QMD and return formatted context for injection."""
    start = time.time()

    try:
        qmd = QMDQuery()
        records = qmd.query(message)

        # Filter out hot tier (already injected by platform)
        warm_deep = [r for r in records if r.tier != "hot"]

        if not warm_deep:
            return ""

        formatted = qmd.format_for_injection(warm_deep, max_chars=max_chars)
        latency = (time.time() - start) * 1000

        if formatted:
            return f"[QMD Context — {len(warm_deep)} records, {latency:.0f}ms]\n{formatted}"
        return ""

    except Exception as e:
        return f"[QMD Error: {e}]"


def main():
    if len(sys.argv) < 2:
        print("Usage: qmd_context.py <message>")
        print("       qmd_context.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        test_queries = [
            "blog astro deploy",
            "runware image generation",
            "facebook social media posting",
            "gold tracker harga emas",
            "siapa mas wahyu",
        ]
        for q in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            print('='*60)
            ctx = get_context(q)
            print(ctx if ctx else "(no relevant context)")
    else:
        message = " ".join(sys.argv[1:])
        print(get_context(message))


if __name__ == "__main__":
    main()
