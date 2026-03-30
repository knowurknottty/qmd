#!/usr/bin/env python3
"""
QMD Test Suite
Test cases untuk validasi QMD system.
Output: concrete metrics untuk review konten social media.
"""

import os
import sys
import json
import time
import sqlite3

# Add scripts dir to path
sys.path.insert(0, os.path.expanduser("~/.hermes/memory/scripts"))
from qmd_query import QMDQuery

DB_PATH = os.path.expanduser("~/.hermes/memory/warm/memories.db")
CONFIG_PATH = os.path.expanduser("~/.hermes/memory/config.json")

# ========== TEST CASES ==========

TEST_QUERIES = [
    {
        "id": "TC01",
        "name": "Blog Astro Cloudflare",
        "query": "blog astro cloudflare pages",
        "expected_categories": ["project_context"],
        "expected_keywords": ["astro", "cloudflare", "blog"],
    },
    {
        "id": "TC02",
        "name": "Threads Social Media Posting",
        "query": "posting ke threads social media repliz",
        "expected_categories": ["credential", "project_context"],
        "expected_keywords": ["threads", "repliz", "social"],
    },
    {
        "id": "TC03",
        "name": "Runware Image Generation",
        "query": "runware image generation API gambar",
        "expected_categories": ["tool_config", "workflow"],
        "expected_keywords": ["runware", "image"],
    },
    {
        "id": "TC04",
        "name": "Gold Tracker Price",
        "query": "gold tracker harga emas logammulia scraper",
        "expected_categories": ["tool_config"],
        "expected_keywords": ["gold", "logammulia", "tracker"],
    },
    {
        "id": "TC05",
        "name": "User Communication Style",
        "query": "komunikasi user preference style langsung",
        "expected_categories": ["user_preference"],
        "expected_keywords": ["komunikasi", "langsung"],
    },
    {
        "id": "TC06",
        "name": "Facebook Page Cron",
        "query": "facebook page smart people cron posting",
        "expected_categories": ["project_context", "credential"],
        "expected_keywords": ["facebook", "smart people"],
    },
    {
        "id": "TC07",
        "name": "Hermes Agent Migration",
        "query": "hermes agent migration openclaw workspace",
        "expected_categories": ["user_preference", "workflow"],
        "expected_keywords": ["hermes", "migration", "openclaw"],
    },
    {
        "id": "TC08",
        "name": "Image Rules Design",
        "query": "aturan gambar desain stickman flat minimal",
        "expected_categories": ["tool_config"],
        "expected_keywords": ["stickman", "flat", "minimal"],
    },
    {
        "id": "TC09",
        "name": "Irrelevant Query (Negative Test)",
        "query": "resep masakan nasi goreng bumbu",
        "expected_categories": [],
        "expected_keywords": [],
    },
    {
        "id": "TC10",
        "name": "Generic Identity Query",
        "query": "siapa saya apa role saya",
        "expected_categories": ["relationship", "user_preference"],
        "expected_keywords": ["kawa", "wahyu", "role"],
    },
]


def measure_relevance(record, expected_categories, expected_keywords):
    """Score 0-1 for relevance of a record to expected results."""
    score = 0.0

    # Category match (0.5 weight)
    if record.category in expected_categories:
        score += 0.5

    # Keyword match (0.5 weight)
    content_lower = record.content.lower()
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
    if expected_keywords:
        score += 0.5 * (keyword_hits / len(expected_keywords))

    return round(score, 2)


def run_tests():
    print("=" * 70)
    print("QMD TEST SUITE")
    print("=" * 70)

    qmd = QMDQuery(CONFIG_PATH)

    results = []
    total_latency = 0
    total_relevant = 0
    total_records = 0
    total_injected_chars = 0

    for tc in TEST_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"[{tc['id']}] {tc['name']}")
        print(f"Query: \"{tc['query']}\"")
        print(f"Expected categories: {tc['expected_categories']}")

        # Measure latency
        start = time.time()
        records = qmd.query(tc["query"])
        latency_ms = (time.time() - start) * 1000
        total_latency += latency_ms

        # Analyze results
        warm_deep = [r for r in records if r.tier != "hot"]
        relevant_count = 0
        details = []

        for r in warm_deep:
            rel_score = measure_relevance(r, tc["expected_categories"], tc["expected_keywords"])
            is_relevant = rel_score >= 0.3
            if is_relevant:
                relevant_count += 1

            details.append({
                "tier": r.tier,
                "category": r.category,
                "similarity": round(r.similarity, 3),
                "decay": round(r.decay_score, 3),
                "relevance": rel_score,
                "relevant": is_relevant,
                "content_preview": r.content[:80],
            })

        # Formatted injection measurement
        formatted = qmd.format_for_injection(records)
        injected_chars = len(formatted)

        # Calculate metrics for this test case
        if warm_deep:
            hit_rate = relevant_count / len(warm_deep)
        else:
            hit_rate = 0

        total_relevant += relevant_count
        total_records += len(warm_deep)
        total_injected_chars += injected_chars

        # Print results
        print(f"Latency: {latency_ms:.0f}ms")
        print(f"Results: {len(warm_deep)} warm/deep records, {relevant_count} relevant")
        print(f"Hit rate: {hit_rate:.0%}")
        print(f"Injected chars: {injected_chars}")

        for d in details:
            mark = "✓" if d["relevant"] else "✗"
            print(f"  {mark} [{d['tier']}] {d['category']} "
                  f"sim={d['similarity']} rel={d['relevance']} "
                  f"| {d['content_preview']}...")

        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "query": tc["query"],
            "latency_ms": round(latency_ms, 0),
            "total_results": len(warm_deep),
            "relevant_results": relevant_count,
            "hit_rate": round(hit_rate, 2),
            "injected_chars": injected_chars,
            "details": details,
        })

    # ========== SUMMARY ==========
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    avg_latency = total_latency / len(TEST_QUERIES)
    overall_hit_rate = total_relevant / total_records if total_records > 0 else 0
    avg_injected = total_injected_chars / len(TEST_QUERIES)

    # Old system baseline
    old_chars = 3525  # MEMORY.md + USER.md full injection
    token_reduction = (old_chars - avg_injected) / old_chars * 100

    print(f"\nTest cases executed: {len(TEST_QUERIES)}")
    print(f"\n--- Latency ---")
    print(f"Average: {avg_latency:.0f}ms")
    print(f"Target: <200ms")
    print(f"Status: {'PASS' if avg_latency < 200 else 'FAIL'}")

    print(f"\n--- Relevance ---")
    print(f"Total warm/deep records returned: {total_records}")
    print(f"Total relevant: {total_relevant}")
    print(f"Overall hit rate: {overall_hit_rate:.0%}")
    print(f"Target: >80%")
    print(f"Status: {'PASS' if overall_hit_rate > 0.8 else 'NEEDS WORK'}")

    print(f"\n--- Token Reduction ---")
    print(f"Old system (flat injection): {old_chars} chars/turn")
    print(f"New system (QMD average): {avg_injected:.0f} chars/turn")
    print(f"Reduction: {token_reduction:.1f}%")
    print(f"Target: 40-60%")
    print(f"Status: {'PASS' if 40 <= token_reduction <= 70 else 'CHECK'}")

    print(f"\n--- Per-Test Hit Rates ---")
    for r in results:
        status = "✓" if r["hit_rate"] >= 0.5 else "✗" if r["total_results"] > 0 else "○"
        print(f"  {status} [{r['id']}] {r['name']}: "
              f"{r['hit_rate']:.0%} ({r['relevant_results']}/{r['total_results']}) "
              f"| {r['injected_chars']} chars | {r['latency_ms']:.0f}ms")

    # Save results to JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "test_cases": len(TEST_QUERIES),
            "avg_latency_ms": round(avg_latency, 0),
            "total_records_returned": total_records,
            "total_relevant": total_relevant,
            "overall_hit_rate": round(overall_hit_rate, 2),
            "old_system_chars": old_chars,
            "avg_injected_chars": round(avg_injected, 0),
            "token_reduction_pct": round(token_reduction, 1),
        },
        "test_results": results,
    }

    output_path = os.path.expanduser("~/.hermes/memory/logs/qmd_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_tests()
