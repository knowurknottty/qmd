#!/usr/bin/env python3
"""
QMD Integration Test Suite — 20 Test Cases
Tests the full Opsi B workflow: hot tier, query, autostore, dedup.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.expanduser("~/.hermes/memory/scripts"))
from qmd_query import QMDQuery
from qmd_context import get_context
from qmd_autostore import detect_facts, store_fact

# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    # --- GROUP 1: HOT TIER (Platform Injection) ---
    {
        "id": "TC01",
        "group": "Hot Tier",
        "name": "Core identity in hot tier",
        "type": "query",
        "query": "siapa kawa",
        "expect": {
            "hot_contains": ["Kawa", "CEO"],
            "min_hot_records": 1,
        },
    },
    {
        "id": "TC02",
        "group": "Hot Tier",
        "name": "User profile in hot tier",
        "type": "query",
        "query": "siapa mas wahyu",
        "expect": {
            "hot_contains": ["Wahyu", "Qawwa"],
            "min_hot_records": 1,
        },
    },
    {
        "id": "TC03",
        "group": "Hot Tier",
        "name": "Hot tier under 500 chars each",
        "type": "hot_budget",
        "query": "",
        "expect": {
            "max_chars_per_record": 500,
        },
    },

    # --- GROUP 2: WARM TIER (SQLite FTS5) ---
    {
        "id": "TC04",
        "group": "Warm Tier",
        "name": "Blog project context via keyword",
        "type": "query",
        "query": "blog astro cloudflare",
        "expect": {
            "min_warm_results": 1,
            "categories_include": ["project_context"],
        },
    },
    {
        "id": "TC05",
        "group": "Warm Tier",
        "name": "Image rules via semantic",
        "type": "query",
        "query": "image rules stickman flat design",
        "expect": {
            "min_deep_results": 1,
            "content_contains_any": ["stickman", "flat", "Image", "minimal"],
        },
    },
    {
        "id": "TC06",
        "group": "Warm Tier",
        "name": "Social media accounts via semantic",
        "type": "query",
        "query": "social media threads linkedin accounts",
        "expect": {
            "min_deep_results": 1,
            "content_contains_any": ["Threads", "LinkedIn", "social", "Facebook"],
        },
    },

    # --- GROUP 3: DEEP TIER (ChromaDB Semantic) ---
    {
        "id": "TC07",
        "group": "Deep Tier",
        "name": "Semantic match: deployment process",
        "type": "query",
        "query": "deploy blog cloudflare github push",
        "expect": {
            "min_deep_results": 1,
            "categories_include": ["lesson_learned"],
        },
    },
    {
        "id": "TC08",
        "group": "Deep Tier",
        "name": "Semantic match: visual content",
        "type": "query",
        "query": "cara buat visual untuk konten blog",
        "expect": {
            "min_deep_results": 1,
            "categories_include_any": ["tool_config", "project_context"],
        },
    },
    {
        "id": "TC09",
        "group": "Deep Tier",
        "name": "Irrelevant query returns minimal results",
        "type": "query",
        "query": "resep masakan nasi goreng bumbu tradisional jawa",
        "expect": {
            "max_total_results": 3,
            "no_categories": ["project_context", "tool_config", "workflow"],
        },
    },

    # --- GROUP 4: AUTO-STORE — Pattern Detection ---
    {
        "id": "TC10",
        "group": "Auto-Store",
        "name": "Detect domain change",
        "type": "autostore",
        "user_msg": "Domain blog ganti ke maswahyu.com",
        "expect": {
            "min_facts": 1,
            "category": "tool_config",
            "content_contains": "maswahyu.com",
        },
    },
    {
        "id": "TC11",
        "group": "Auto-Store",
        "name": "Detect GitHub repo change",
        "type": "autostore",
        "user_msg": "GitHub repo pindah ke QawwaTech/new-blog",
        "expect": {
            "min_facts": 1,
            "category": "tool_config",
            "content_contains": "QawwaTech/new-blog",
        },
    },
    {
        "id": "TC12",
        "group": "Auto-Store",
        "name": "Detect explicit remember command",
        "type": "autostore",
        "user_msg": "Ingat: password database baru adalah db_2026_xyz",
        "expect": {
            "min_facts": 1,
            "category": "general",
            "content_contains": "db_2026_xyz",
        },
    },
    {
        "id": "TC13",
        "group": "Auto-Store",
        "name": "Detect schedule/cron change",
        "type": "autostore",
        "user_msg": "Tambahin cron untuk posting Instagram setiap Rabu jam 10 pagi",
        "expect": {
            "min_facts": 1,
            "category": "workflow",
            "content_contains": "Instagram",
        },
    },
    {
        "id": "TC14",
        "group": "Auto-Store",
        "name": "Detect design preference",
        "type": "autostore",
        "user_msg": "Jangan pakai font serif lagi, ganti ke sans-serif",
        "expect": {
            "min_facts": 1,
            "category": "user_preference",
            "content_contains": "sans-serif",
        },
    },
    {
        "id": "TC15",
        "group": "Auto-Store",
        "name": "Casual chat returns no facts",
        "type": "autostore",
        "user_msg": "Hari ini cuaca bagus ya, mau makan apa?",
        "expect": {
            "max_facts": 0,
        },
    },

    # --- GROUP 5: DEDUP ---
    {
        "id": "TC16",
        "group": "Dedup",
        "name": "Duplicate content not stored twice",
        "type": "dedup",
        "user_msg": f"Ingat: test dedup marker {int(time.time())}",
        "expect": {
            "first_store": True,
            "second_store": "SKIP",
        },
    },

    # --- GROUP 6: CONTEXT FORMATTING ---
    {
        "id": "TC17",
        "group": "Context Format",
        "name": "Formatted output under 1500 chars",
        "type": "context_format",
        "query": "blog deploy social media",
        "expect": {
            "max_chars": 1500,
            "not_empty": True,
        },
    },
    {
        "id": "TC18",
        "group": "Context Format",
        "name": "No hot tier in context output",
        "type": "context_format",
        "query": "siapa kawa",
        "expect": {
            "no_hot_in_output": True,
        },
    },

    # --- GROUP 7: END-TO-END WORKFLOW ---
    {
        "id": "TC19",
        "group": "E2E Workflow",
        "name": "Store then query retrieves it",
        "type": "e2e_store_query",
        "store_content": f"Dashboard API endpoint adalah api.qawwa.id/v2/{int(time.time())}",
        "store_category": "tool_config",
        "store_tags": ["api", "dashboard", "qawwa"],
        "query": "api endpoint dashboard qawwa",
        "expect": {
            "store_success": True,
            "query_finds": True,
        },
    },
    {
        "id": "TC20",
        "group": "E2E Workflow",
        "name": "Full workflow: user msg → detect → store → query",
        "type": "e2e_full",
        "user_msg": f"Ingat: Telegram bot token CS baru adalah bot_{int(time.time())}",
        "query": "telegram bot token customer service",
        "expect": {
            "fact_detected": True,
            "stored": True,
            "queryable": True,
        },
    },
]


# ============================================================
# TEST RUNNER
# ============================================================

class TestRunner:
    def __init__(self):
        self.qmd = QMDQuery()
        self.results = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run_all(self):
        print("=" * 70)
        print("QMD INTEGRATION TEST SUITE — 20 Test Cases")
        print("=" * 70)

        for tc in TEST_CASES:
            self.run_test(tc)

        self.print_summary()

    def run_test(self, tc):
        print(f"\n{'─' * 70}")
        print(f"[{tc['id']}] {tc['group']} — {tc['name']}")

        start = time.time()

        try:
            if tc["type"] == "query":
                result = self.test_query(tc)
            elif tc["type"] == "hot_budget":
                result = self.test_hot_budget(tc)
            elif tc["type"] == "autostore":
                result = self.test_autostore(tc)
            elif tc["type"] == "dedup":
                result = self.test_dedup(tc)
            elif tc["type"] == "context_format":
                result = self.test_context_format(tc)
            elif tc["type"] == "e2e_store_query":
                result = self.test_e2e_store_query(tc)
            elif tc["type"] == "e2e_full":
                result = self.test_e2e_full(tc)
            else:
                result = {"status": "SKIP", "reason": f"Unknown type: {tc['type']}"}

        except Exception as e:
            result = {"status": "FAIL", "error": str(e)}

        latency = (time.time() - start) * 1000
        result["latency_ms"] = round(latency, 0)

        status = result.get("status", "FAIL")
        if status == "PASS":
            self.passed += 1
            print(f"  ✅ PASS ({latency:.0f}ms)")
        elif status == "SKIP":
            self.skipped += 1
            print(f"  ⏭️ SKIP — {result.get('reason', '')}")
        else:
            self.failed += 1
            print(f"  ❌ FAIL ({latency:.0f}ms)")
            for k, v in result.items():
                if k != "status":
                    print(f"     {k}: {v}")

        self.results.append({"id": tc["id"], "name": tc["name"], **result})

    # --- Individual test types ---

    def test_query(self, tc):
        records = self.qmd.query(tc["query"])
        warm_deep = [r for r in records if r.tier != "hot"]
        hot = [r for r in records if r.tier == "hot"]
        expect = tc["expect"]

        # Hot tier checks
        if "min_hot_records" in expect:
            if len(hot) < expect["min_hot_records"]:
                return {"status": "FAIL", "reason": f"Hot records: {len(hot)} < {expect['min_hot_records']}"}

        if "hot_contains" in expect:
            hot_text = " ".join(r.content for r in hot)
            for kw in expect["hot_contains"]:
                if kw.lower() not in hot_text.lower():
                    return {"status": "FAIL", "reason": f"Hot missing keyword: {kw}"}

        # Warm/Deep checks
        if "min_warm_results" in expect:
            warm = [r for r in warm_deep if r.tier == "warm"]
            if len(warm) < expect["min_warm_results"]:
                return {"status": "FAIL", "reason": f"Warm results: {len(warm)} < {expect['min_warm_results']}"}

        if "min_deep_results" in expect:
            deep = [r for r in warm_deep if r.tier == "deep"]
            if len(deep) < expect["min_deep_results"]:
                return {"status": "FAIL", "reason": f"Deep results: {len(deep)} < {expect['min_deep_results']}"}

        if "categories_include" in expect:
            found_cats = set(r.category for r in warm_deep)
            for cat in expect["categories_include"]:
                if cat not in found_cats:
                    return {"status": "FAIL", "reason": f"Missing category: {cat}. Found: {found_cats}"}

        if "categories_include_any" in expect:
            found_cats = set(r.category for r in warm_deep)
            if not any(cat in found_cats for cat in expect["categories_include_any"]):
                return {"status": "FAIL", "reason": f"None of {expect['categories_include_any']} in {found_cats}"}

        if "content_contains" in expect:
            combined = " ".join(r.content for r in warm_deep).lower()
            for kw in expect["content_contains"]:
                if kw.lower() not in combined:
                    return {"status": "FAIL", "reason": f"Missing content keyword: {kw}"}

        if "content_contains_any" in expect:
            combined = " ".join(r.content for r in warm_deep).lower()
            if not any(kw.lower() in combined for kw in expect["content_contains_any"]):
                return {"status": "FAIL", "reason": f"None of {expect['content_contains_any']} found"}

        if "max_total_results" in expect:
            if len(warm_deep) > expect["max_total_results"]:
                return {"status": "FAIL", "reason": f"Too many results: {len(warm_deep)} > {expect['max_total_results']}"}

        if "no_categories" in expect:
            found_cats = set(r.category for r in warm_deep)
            for cat in expect["no_categories"]:
                if cat in found_cats:
                    return {"status": "FAIL", "reason": f"Unexpected category found: {cat}"}

        print(f"  Results: {len(hot)} hot + {len(warm_deep)} warm/deep")
        return {"status": "PASS", "total_results": len(records)}

    def test_hot_budget(self, tc):
        hot = self.qmd._query_hot()
        expect = tc["expect"]
        for r in hot:
            if len(r.content) > expect["max_chars_per_record"]:
                return {"status": "FAIL", "reason": f"{r.id}: {len(r.content)} chars > {expect['max_chars_per_record']}"}
        print(f"  Hot records: {[(r.id, len(r.content)) for r in hot]}")
        return {"status": "PASS"}

    def test_autostore(self, tc):
        facts = detect_facts(tc["user_msg"], "")
        expect = tc["expect"]

        if "max_facts" in expect and expect["max_facts"] == 0:
            if len(facts) > 0:
                return {"status": "FAIL", "reason": f"Expected no facts, got {len(facts)}"}
            print(f"  No facts detected (correct)")
            return {"status": "PASS"}

        if len(facts) < expect.get("min_facts", 1):
            return {"status": "FAIL", "reason": f"Detected {len(facts)} facts, expected >= {expect['min_facts']}"}

        if "category" in expect:
            if not any(f["category"] == expect["category"] for f in facts):
                cats = [f["category"] for f in facts]
                return {"status": "FAIL", "reason": f"Expected category {expect['category']}, got {cats}"}

        if "content_contains" in expect:
            combined = " ".join(f["content"] for f in facts).lower()
            if expect["content_contains"].lower() not in combined:
                return {"status": "FAIL", "reason": f"Content missing: {expect['content_contains']}"}

        print(f"  Detected: {[(f['category'], f['content'][:40]) for f in facts]}")
        return {"status": "PASS", "facts_count": len(facts)}

    def test_dedup(self, tc):
        expect = tc["expect"]

        # First store
        mid1 = store_fact(tc["user_msg"], "general", ["test", "dedup"])
        first_ok = mid1 and mid1.startswith("STORED")
        print(f"  First store: {mid1}")

        # Second store (should skip)
        mid2 = store_fact(tc["user_msg"], "general", ["test", "dedup"])
        second_skipped = mid2 and "SKIP" in mid2
        print(f"  Second store: {mid2}")

        # Cleanup
        if first_ok:
            import sqlite3
            db = os.path.expanduser("~/.hermes/memory/warm/memories.db")
            conn = sqlite3.connect(db)
            c = conn.cursor()
            mem_id = mid1.split(": ")[1] if ": " in mid1 else ""
            if mem_id:
                c.execute("DELETE FROM memory_tags WHERE memory_id = ?", (mem_id,))
                c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                conn.commit()
            conn.close()

        if not first_ok:
            return {"status": "FAIL", "reason": "First store failed"}
        if not second_skipped:
            return {"status": "FAIL", "reason": "Second store should be skipped (duplicate)"}

        return {"status": "PASS"}

    def test_context_format(self, tc):
        ctx = get_context(tc["query"])
        expect = tc["expect"]

        if expect.get("not_empty") and not ctx:
            return {"status": "FAIL", "reason": "Context is empty"}

        if expect.get("max_chars") and len(ctx) > expect["max_chars"]:
            return {"status": "FAIL", "reason": f"Context too long: {len(ctx)} > {expect['max_chars']}"}

        if expect.get("no_hot_in_output"):
            if "[hot]" in ctx.lower() or "hot_memory" in ctx.lower():
                return {"status": "FAIL", "reason": "Hot tier content found in context output"}

        print(f"  Context length: {len(ctx)} chars")
        return {"status": "PASS", "chars": len(ctx)}

    def test_e2e_store_query(self, tc):
        # Store
        mid = store_fact(tc["store_content"], tc["store_category"], tc["store_tags"])
        stored = mid and mid.startswith("STORED")
        print(f"  Store: {mid}")

        # Wait for index
        time.sleep(0.5)

        # Query
        results = self.qmd.query(tc["query"])
        warm_deep = [r for r in results if r.tier != "hot"]
        found = any(tc["store_content"][:20].lower() in r.content.lower() for r in warm_deep)
        print(f"  Query found: {found} ({len(warm_deep)} results)")

        # Cleanup
        if stored:
            import sqlite3
            db = os.path.expanduser("~/.hermes/memory/warm/memories.db")
            conn = sqlite3.connect(db)
            c = conn.cursor()
            mem_id = mid.split(": ")[1]
            c.execute("DELETE FROM memory_tags WHERE memory_id = ?", (mem_id,))
            c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            conn.commit()
            conn.close()

        if not stored:
            return {"status": "FAIL", "reason": "Store failed"}
        if not found:
            return {"status": "FAIL", "reason": "Stored fact not found in query"}

        return {"status": "PASS"}

    def test_e2e_full(self, tc):
        # Step 1: Detect
        facts = detect_facts(tc["user_msg"], "")
        detected = len(facts) > 0
        print(f"  Detect: {len(facts)} facts")

        if not detected:
            return {"status": "FAIL", "reason": "No facts detected"}

        # Step 2: Store
        stored_ids = []
        for f in facts:
            mid = store_fact(f["content"], f["category"], f["tags"], f["confidence"])
            if mid and mid.startswith("STORED"):
                stored_ids.append(mid.split(": ")[1])
            print(f"  Store: {mid}")

        time.sleep(0.5)

        # Step 3: Query
        results = self.qmd.query(tc["query"])
        warm_deep = [r for r in results if r.tier != "hot"]
        queryable = any(
            any(kw.lower() in r.content.lower() for kw in ["telegram", "bot", "token", "customer"])
            for r in warm_deep
        )
        print(f"  Query: {queryable} ({len(warm_deep)} results)")

        # Cleanup
        import sqlite3
        db = os.path.expanduser("~/.hermes/memory/warm/memories.db")
        conn = sqlite3.connect(db)
        c = conn.cursor()
        for mid in stored_ids:
            c.execute("DELETE FROM memory_tags WHERE memory_id = ?", (mid,))
            c.execute("DELETE FROM memories WHERE id = ?", (mid,))
        conn.commit()
        conn.close()

        if not queryable:
            return {"status": "FAIL", "reason": "Stored fact not queryable"}

        return {"status": "PASS"}

    def print_summary(self):
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)

        total = len(self.results)
        print(f"\nTotal test cases: {total}")
        print(f"  ✅ Passed:  {self.passed}")
        print(f"  ❌ Failed:  {self.failed}")
        print(f"  ⏭️ Skipped: {self.skipped}")

        avg_latency = sum(r.get("latency_ms", 0) for r in self.results) / total if total else 0
        print(f"\nAverage latency: {avg_latency:.0f}ms")

        pass_rate = (self.passed / total * 100) if total else 0
        print(f"Pass rate: {pass_rate:.0f}%")

        if pass_rate >= 90:
            print("\n🟢 SYSTEM STATUS: READY FOR PRODUCTION")
        elif pass_rate >= 70:
            print("\n🟡 SYSTEM STATUS: NEEDS MINOR FIXES")
        else:
            print("\n🔴 SYSTEM STATUS: NEEDS MAJOR WORK")

        # Save results
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": round(pass_rate, 1),
            "avg_latency_ms": round(avg_latency, 0),
            "results": self.results,
        }
        output_path = os.path.expanduser("~/.hermes/memory/logs/qmd_integration_test.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved: {output_path}")

        # Failed tests detail
        if self.failed > 0:
            print(f"\n{'─' * 70}")
            print("FAILED TESTS:")
            for r in self.results:
                if r.get("status") == "FAIL":
                    print(f"  [{r['id']}] {r['name']}")
                    for k, v in r.items():
                        if k not in ("id", "name", "status", "latency_ms"):
                            print(f"    {k}: {v}")


if __name__ == "__main__":
    runner = TestRunner()
    runner.run_all()
