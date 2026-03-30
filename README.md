# QMD — Query Memory Database

3-tier semantic memory system for AI agents.

## Install

```bash
git clone https://github.com/PseudoPort/qmd.git ~/.hermes/memory
cd ~/.hermes/memory
bash install.sh
```

## Components

- **Hot Tier**: Core identity (<500 chars, always injected)
- **Warm Tier**: SQLite + FTS5 (keyword search)
- **Deep Tier**: ChromaDB + all-MiniLM-L6-v2 (semantic search)
- **Auto-Store**: Detect & save facts from conversation
- **Decay**: Automatic memory archival based on recency + frequency

## Scripts

| Script | Purpose |
|--------|--------|
| qmd_query.py | 3-tier retrieval engine |
| qmd_write.py | Store/update/delete memories |
| qmd_context.py | Auto-query wrapper |
| qmd_autostore.py | Auto-detect facts from conversation |
| qmd_decay.py | Daily decay maintenance |
| qmd_integration_test.py | 20 test cases |

## Test Results

20/20 PASS — Ready for Production

## License

MIT
