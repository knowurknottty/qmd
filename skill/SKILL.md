---
name: qmd
description: Query Memory Database — 3-tier semantic memory system. Replaces flat MEMORY.md injection with queryable semantic search using SQLite (keyword) + ChromaDB (vector).
version: 1.0.0
metadata:
  hermes:
    tags: [memory, semantic-search, chromadb, sqlite, rag]
---

# QMD — Query Memory Database

3-tier memory system for intelligent context injection:

- **Hot Tier:** MEMORY.md + USER.md (always injected, <500 chars)
- **Warm Tier:** SQLite + FTS5 (keyword search)
- **Deep Tier:** ChromaDB + all-MiniLM-L6-v2 (semantic search)

## When to use

- **query** — Search relevant memories before responding to user
- **store** — Save new facts, preferences, lessons learned
- **autostore** — Auto-detect and store facts from conversation
- **list** — Browse memories by category
- **stats** — Check memory system health

## Integration (Opsi B — Split Responsibility) — PRODUCTION

Platform injects MEMORY.md + USER.md (~400 chars) as "always-on" hot context.
QMD warm/deep tiers provide on-demand relevant context via query.

### Workflow per message:
1. Platform auto-injects MEMORY.md + USER.md (hot tier, free)
2. Call `qmd_context.py "<message>"` → get relevant warm/deep context (<1500 chars)
3. Merge and respond
4. After response, call `qmd_autostore.py detect "<user_msg>" "<assistant_msg>"` → store new facts

### Files:
- `qmd_context.py` — Auto-query wrapper. Returns formatted context string.
- `qmd_autostore.py` — Auto-detect facts from conversation. Patterns: domain/repo changes, API keys, cron schedules, preferences, explicit "ingat/catat/simpan".
- `qmd_integration_test.py` — 20 test cases covering all tiers + autostore + dedup + E2E.

### Auto-store detects:
- Domain/repo changes: "domain ganti ke X.com", "github pindah ke org/repo"
- API/credentials: "API key baru adalah X"
- Schedules: "cron untuk LinkedIn setiap Senin"
- Preferences: "jangan pakai biru, ganti orange"
- Explicit: "ingat: X", "catat: X", "simpan: X"
- Casual chat → no false positives

### Config tuning:
- `min_similarity`: 0.4 (was 0.3 — reduced noise from irrelevant queries)
- `hot_max_chars`: 500 per record
- `max_total_chars`: 1500 for injection

### Pitfalls (updated):
7. Hot tier path: `~/.hermes/memories/MEMORY.md` (NOT `~/.hermes/MEMORY.md`)
8. Test data cleanup: use unique timestamps (`int(time.time())`) to avoid dedup collisions across test runs
9. Semantic search: query wording matters — use terms similar to stored content for best results (e.g. "image rules stickman" works, "aturan gambar desain" doesn't)
10. Autostore regex: domain/repo patterns use `.*?` (non-greedy) to handle variable filler words. Pattern: `(?:domain|url|repo|github)\S*\s+.*?(?:ganti|pindah|pake|pakai)\s+(?:ke\s+)?(\S+\.\S+)`
11. Auto-store content: only extract from `user_msg`, never include `assistant_msg` in stored content

## Scripts

All scripts: `~/.hermes/memory/scripts/`

### Script Inventory

| File | Lines | Purpose |
|---|---|---|
| `qmd_schema.sql` | 51 | SQLite schema (tables, FTS5, triggers, indexes) |
| `qmd_query.py` | 418 | 3-tier query engine (Hot→Warm→Deep→Merge→Rank) |
| `qmd_write.py` | 328 | Store, update, delete memories in SQLite + ChromaDB |
| `qmd_decay.py` | 158 | Daily decay maintenance (recalculate scores, archive) |
| `qmd_migrate.py` | — | Migrate from flat MEMORY.md to QMD |
| `qmd_context.py` | 54 | Auto-query wrapper (returns formatted context string) |
| `qmd_autostore.py` | 180 | Auto-detect & store facts from conversation |
| `qmd_test.py` | 264 | Basic test suite (10 test cases) |
| `qmd_integration_test.py` | 500+ | Full integration test (20 test cases, 7 groups) |

### qmd_schema.sql

SQLite schema untuk Warm Tier:
- `memories` table: id, content, category, created_at, last_accessed, access_count, confidence, source, tier, decay_score
- `memory_tags` table: memory_id, tag (FK)
- `memories_fts`: FTS5 virtual table untuk full-text search
- Triggers: auto-sync FTS on INSERT/DELETE/UPDATE
- Indexes: category, tier, last_accessed, decay_score, created_at

### qmd_query.py

3-tier cascade retrieval engine.

Class `QMDQuery`:
- `query(message)` → returns `list[MemoryRecord]` ranked by relevance
- `format_for_injection(records, max_chars)` → returns compact string untuk prompt injection
- `stats()` → returns dict dengan tier counts, category counts, avg decay

Flow:
1. `_query_hot()` → read MEMORY.md + USER.md, extract essential lines (<500 chars each)
2. `_query_warm(query_text, top_k)` → SQLite FTS5 keyword search
3. `_query_deep(query_text, top_k, min_similarity)` → ChromaDB cosine similarity
4. `_deduplicate(records)` → remove duplicates, keep highest relevance
5. `_rank(records)` → sort by `similarity × decay_score × confidence`

Key: `_extract_essential(content, max_chars)` — parses §-delimited content, prioritizes identity keywords (name, role, timezone, vibe, vision).

### qmd_write.py

Store/update/delete engine. Class `QMDWrite`:
- `store(content, category, tags, source, confidence)` → UUID, writes to SQLite + ChromaDB
- `update(memory_id, content, category, tags, confidence)` → update existing record
- `delete(memory_id)` → delete from both SQLite + ChromaDB
- `list_by_category(category, limit)` → list records by category
- `list_all(limit)` → list all records
- `get_stats()` → storage statistics

Valid categories: user_preference, project_context, tool_config, lesson_learned, credential, relationship, workflow, general

Dual-write: SQLite first (primary), then ChromaDB (best-effort). If ChromaDB fails, SQLite still succeeds.

### qmd_decay.py

Daily maintenance. Function `run_decay()`:
1. Read all memories from SQLite
2. Calculate new decay score per record:
   - `recency = exp(-days_since_access / half_life_days)`
   - `frequency = 0.5 + 0.5 * log(1 + access_count) / log(1 + max_access)`
   - `decay = recency * max(0.1, frequency)`
3. If decay < archive_threshold (0.3): demote tier warm → deep
4. Update decay_score and tier in SQLite

CLI: `python3 qmd_decay.py` (run) or `python3 qmd_decay.py --dry-run` (preview)

### qmd_context.py

Auto-query wrapper untuk integration.

Function `get_context(message, max_chars)` → str:
1. Init QMDQuery
2. Query with message
3. Filter out hot tier (already injected by platform)
4. Format remaining records as context string
5. Return `[QMD Context — N records, Xms]\nformatted_content`

CLI: `python3 qmd_context.py "user message"` atau `python3 qmd_context.py --test`

### qmd_autostore.py

Auto-detect dan store fakta dari percakapan.

Function `detect_facts(user_msg, assistant_msg)` → list[dict]:
- Regex pattern matching untuk domain, repo, API, cron, preferences
- Explicit commands: "ingat:", "catat:", "simpan:"
- Signal keywords detection (baru, ganti, update, hapus, dll)
- Dedup sebelum return

Function `store_fact(content, category, tags, confidence)` → str:
- Check duplicate (exact match + similarity >80% word overlap)
- If new: store to SQLite + ChromaDB, return "STORED: {id}"
- If duplicate: return "SKIP (duplicate)"

Function `extract_tags(text)` → list[str]:
- Auto-detect tags dari keywords: blog, social, deploy, api, cron, design, ai, automation, ukm

CLI:
- `python3 qmd_autostore.py store "content" "category" "tag1,tag2"`
- `python3 qmd_autostore.py detect "user message" "assistant response"`

### qmd_integration_test.py

20 test cases across 7 groups. Class `TestRunner`:
- `test_query(tc)` — verify warm/deep tier results, categories, content keywords
- `test_hot_budget(tc)` — verify each hot record <500 chars
- `test_autostore(tc)` — verify pattern detection (domain, repo, remember, cron, preference, casual)
- `test_dedup(tc)` — verify first store OK, second store SKIP
- `test_context_format(tc)` — verify output <1500 chars, no hot tier included
- `test_e2e_store_query(tc)` — store fact → query finds it
- `test_e2e_full(tc)` — full pipeline: user msg → detect → store → query

Auto-cleanup: test records removed after each run using unique timestamps.
Results saved to: `~/.hermes/memory/logs/qmd_integration_test.json`

## Quick Install (portable)

Copy full QMD system ke agent lain:

```bash
# Same machine, default target
bash ~/.hermes/memory/scripts/qmd_install.sh

# Different target
bash ~/.hermes/memory/scripts/qmd_install.sh /path/to/other/.hermes
```

Script otomatis:
1. Create directories (memory/{warm,deep,scripts,logs})
2. Copy 9 scripts + schema
3. Copy config.json (skip if exists)
4. Copy SKILL.md
5. Init SQLite database
6. Check Python dependencies

## Setup

Python dependencies (already installed):
- chromadb
- sentence-transformers

Database initialized at:
- SQLite: `~/.hermes/memory/warm/memories.db`
- ChromaDB: `~/.hermes/memory/deep/chroma/`

## Usage

### Query (search memories)

```bash
python3 ~/.hermes/memory/scripts/qmd_query.py "search terms here"
```

Returns ranked memories by relevance, filtered by:
- Keyword match (FTS5)
- Semantic similarity (ChromaDB cosine)
- Decay score (recency × frequency)

### Store (add new memory)

```bash
python3 ~/.hermes/memory/scripts/qmd_write.py store "content" "category" "tag1,tag2"
```

Categories: user_preference, project_context, tool_config, lesson_learned, credential, relationship, workflow, general

### Auto-query context

```bash
python3 ~/.hermes/memory/scripts/qmd_context.py "user message"
```

### Auto-store facts

```bash
python3 ~/.hermes/memory/scripts/qmd_autostore.py detect "user msg" "assistant msg"
```

### List memories

```bash
python3 ~/.hermes/memory/scripts/qmd_write.py list [category]
```

### Stats

```bash
python3 ~/.hermes/memory/scripts/qmd_write.py stats
```

### Decay maintenance (run daily via cron)

```bash
python3 ~/.hermes/memory/scripts/qmd_decay.py
```

### Run integration tests

```bash
python3 ~/.hermes/memory/scripts/qmd_integration_test.py
```

## Implementation Guide

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers
```

If no pip available:
```bash
python -m ensurepip --upgrade
python -m pip install chromadb sentence-transformers
```

### 2. Create directory structure

```bash
mkdir -p ~/.hermes/memory/{warm,deep,scripts,logs}
```

### 3. Create config.json

```json
{
    "version": "1.0",
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384
    },
    "retrieval": {
        "hot_max_chars": 500,
        "warm_top_k": 5,
        "deep_top_k": 3,
        "min_similarity": 0.4,
        "max_total_chars": 1500
    },
    "decay": {
        "half_life_days": 30,
        "archive_threshold": 0.3,
        "promote_threshold": 0.8
    },
    "storage": {
        "sqlite_path": "~/.hermes/memory/warm/memories.db",
        "chroma_path": "~/.hermes/memory/deep/chroma"
    }
}
```

### 4. Init database

```bash
sqlite3 ~/.hermes/memory/warm/memories.db < ~/.hermes/memory/scripts/qmd_schema.sql
```

### 5. Register skill + setup cron

```python
# Skill
skill_manage(action='create', name='qmd', category='productivity')

# Decay cron (daily 3 AM)
cronjob(action='create', name='qmd-decay-daily',
        schedule='0 3 * * *',
        prompt='Run python3 ~/.hermes/memory/scripts/qmd_decay.py')
```

### 6. Migrate existing memory

```bash
python3 ~/.hermes/memory/scripts/qmd_migrate.py
```

### 7. Verify

```bash
python3 ~/.hermes/memory/scripts/qmd_query.py "test query"
python3 ~/.hermes/memory/scripts/qmd_write.py stats
python3 ~/.hermes/memory/scripts/qmd_decay.py --dry-run
python3 ~/.hermes/memory/scripts/qmd_integration_test.py
```

## Tuning

| Parameter | Default | Increase when | Decrease when |
|-----------|---------|---------------|---------------|
| hot_max_chars | 500 | Identity is complex | Need more room for search |
| warm_top_k | 5 | Broad topics | Narrow, precise queries |
| deep_top_k | 3 | Conceptual recall needed | Strict keyword matching OK |
| min_similarity | 0.4 | Too few results | Too many irrelevant results |
| max_total_chars | 1500 | Have token budget | Tight context window |
| half_life_days | 30 | Want persistent memory | Want aggressive pruning |

## Pitfalls (learned during implementation)

1. **Decay formula too aggressive:** `log(1+count)/log(1+100)` ≈ 0.01 for count=1 → all records archive immediately. Fix: floor frequency at 0.5: `0.5 + 0.5 * log(1+count)/log(1+100)`

2. **Hot tier budget overflow:** Full MEMORY.md (2154) + USER.md (1371) = 3525 chars exceeds 1500 budget. Fix: extract only essential lines (identity keywords) in hot tier.

3. **Migration access_count=0:** Fresh records have no access → frequency=0 → instant decay. Fix: set access_count=1 post-migration.

4. **Repliz API needs page param:** `/public/account` returns 400 without `?page=1`. Always include page param.

5. **Env vars in ~/.hermes/.env:** Platform doesn't inject REPLIZ_* to env. Must `source ~/.hermes/.env` in subprocess calls.

6. **Package installation:** venv has no pip. Use `python -m ensurepip --upgrade` first, then `python -m pip install`.

7. **Integration test design:** 20 test cases across 7 groups (Hot, Warm/Deep, Auto-Store, Dedup, Context Format, E2E Store+Query, E2E Full). Use `int(time.time())` for unique markers to avoid dedup collisions. Clean test data after runs. See `qmd_integration_test.py`.

8. **Semantic search query wording:** Queries must use terms similar to stored content. "image rules stickman" finds Image Rules record (sim=0.63), but "aturan gambar desain untuk blog" doesn't. Test with actual terms from records.

9. **Deep tier irrelevant noise:** min_similarity=0.3 returned too many false positives from irrelevant queries. Increased to 0.4. "resep masakan" now returns max 3 results instead of 5+.

## Configuration

Edit `~/.hermes/memory/config.json`:

```json
{
  "retrieval": {
    "hot_max_chars": 500,
    "warm_top_k": 5,
    "deep_top_k": 3,
    "min_similarity": 0.4,
    "max_total_chars": 1500
  },
  "decay": {
    "half_life_days": 30,
    "archive_threshold": 0.3,
    "promote_threshold": 0.8
  }
}
```

## Architecture

```
User Message → Query Router
  ├── Hot Tier (always) → core facts
  ├── Warm Tier (keyword) → FTS5 match
  └── Deep Tier (semantic) → ChromaDB similarity
         ↓
    Merge → Deduplicate → Rank by (relevance × decay)
         ↓
    Injected Context (<1500 chars)
```

## Integration Test Suite — 20 Test Cases

File: `~/.hermes/memory/scripts/qmd_integration_test.py`
Run: `python3 ~/.hermes/memory/scripts/qmd_integration_test.py`

### Test Groups

**TC01-TC03: Hot Tier**
| TC | Name | Check |
|---|---|---|
| TC01 | Core identity in hot tier | "siapa kawa" → returns Kawa, CEO |
| TC02 | User profile in hot tier | "siapa mas wahyu" → returns Wahyu, Qawwa |
| TC03 | Hot tier under 500 chars | Each record <500 chars budget |

**TC04-TC09: Warm/Deep Tier**
| TC | Name | Query | Check |
|---|---|---|---|
| TC04 | Blog project keyword | "blog astro cloudflare" | warm >=1, category=project_context |
| TC05 | Image rules semantic | "image rules stickman flat design" | deep >=1, content contains stickman/flat |
| TC06 | Social media semantic | "social media threads linkedin accounts" | deep >=1, content contains Threads/LinkedIn |
| TC07 | Deployment semantic | "deploy blog cloudflare github push" | deep >=1, category=lesson_learned |
| TC08 | Visual content semantic | "cara buat visual untuk konten blog" | deep >=1, category=tool_config/project_context |
| TC09 | Irrelevant filter | "resep masakan nasi goreng bumbu tradisional jawa" | max 3 results, no project/tool/workflow |

**TC10-TC15: Auto-Store**
| TC | Name | Input | Expected |
|---|---|---|---|
| TC10 | Detect domain change | "Domain blog ganti ke maswahyu.com" | tool_config, contains maswahyu.com |
| TC11 | Detect repo change | "GitHub repo pindah ke QawwaTech/new-blog" | tool_config, contains QawwaTech/new-blog |
| TC12 | Detect remember | "Ingat: password database baru adalah db_2026_xyz" | general, contains db_2026_xyz |
| TC13 | Detect cron | "Tambahin cron untuk posting Instagram setiap Rabu" | workflow, contains Instagram |
| TC14 | Detect preference | "Jangan pakai font serif lagi, ganti ke sans-serif" | user_preference, contains sans-serif |
| TC15 | Casual chat filter | "Hari ini cuaca bagus ya, mau makan apa?" | 0 facts |

**TC16: Dedup**
| TC | Name | Check |
|---|---|---|
| TC16 | Duplicate not stored twice | First store = STORED, second = SKIP |

**TC17-TC18: Context Format**
| TC | Name | Check |
|---|---|---|
| TC17 | Output under budget | Context <1500 chars (actual: ~388 avg) |
| TC18 | No hot tier in output | Hot tier excluded from context string |

**TC19-TC20: E2E Workflow**
| TC | Name | Check |
|---|---|---|
| TC19 | Store then query | Store fact → query finds it |
| TC20 | Full workflow | User msg → detect → store → queryable |

### Test Results (2026-03-29)

```
======================================================================
QMD INTEGRATION TEST SUITE — 20 Test Cases
======================================================================

[TC01-TC03] Hot Tier .................. ✅ 3/3
[TC04-TC09] Warm/Deep Tier ............ ✅ 6/6
[TC10-TC15] Auto-Store ................ ✅ 6/6
[TC16]      Dedup ..................... ✅ 1/1
[TC17-TC18] Context Format ............ ✅ 2/2
[TC19-TC20] E2E Workflow .............. ✅ 2/2

Total: 20/20 PASS
Average latency: 558ms
Status: 🟢 READY FOR PRODUCTION
```

### Metrics

| Metric | Value |
|---|---|
| Hot tier chars | 177 + 225 = 402 chars |
| Context output avg | 388 chars |
| Token reduction vs flat | 84% (3525 → ~566) |
| Avg query latency | 558ms |
| Auto-store patterns | 6 (domain, repo, remember, cron, preference, filter) |
| Dedup accuracy | 100% (exact + similarity) |
| SQLite records | 33 |
| ChromaDB vectors | 33 |
