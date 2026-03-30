#!/usr/bin/env python3
"""
QMD Decay Maintenance
Recalculate decay scores, promote/demote tiers.
Run daily via cron.
"""

import os
import json
import sqlite3
import math
import logging
from datetime import datetime, timedelta

# Paths
CONFIG_PATH = os.path.expanduser("~/.hermes/memory/config.json")
DB_PATH = os.path.expanduser("~/.hermes/memory/warm/memories.db")
LOG_PATH = os.path.expanduser("~/.hermes/memory/logs/qmd.log")

# Logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("qmd_decay")


def calculate_decay(last_accessed: str, access_count: int,
                    half_life_days: int = 30, max_access: int = 100) -> float:
    """
    Calculate decay score.

    decay = base * recency_weight * frequency_weight

    recency_weight = exp(-days_since_access / half_life)
    frequency_weight = log(1 + access_count) / log(1 + max_access)
    """
    try:
        last_access = datetime.fromisoformat(last_accessed)
    except (ValueError, TypeError):
        return 0.5  # Default if date is invalid

    days_since = (datetime.utcnow() - last_access).days

    # Recency weight (exponential decay)
    recency = math.exp(-days_since / half_life_days)

    # Frequency weight (logarithmic, floor at 0.5 for single access)
    frequency = 0.5 + 0.5 * (math.log(1 + access_count) / math.log(1 + max_access))

    # Combined score
    decay = 1.0 * recency * max(0.1, frequency)  # Min 0.1 to avoid zero

    return round(min(1.0, max(0.0, decay)), 4)


def run_decay(config_path: str = CONFIG_PATH):
    """Run decay maintenance."""
    logger.info("Starting decay maintenance")

    with open(config_path, 'r') as f:
        config = json.load(f)

    half_life = config["decay"]["half_life_days"]
    archive_threshold = config["decay"]["archive_threshold"]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all memories
    cursor.execute("SELECT * FROM memories")
    memories = cursor.fetchall()

    updated = 0
    archived = 0

    for mem in memories:
        # Calculate new decay score
        new_decay = calculate_decay(
            mem["last_accessed"],
            mem["access_count"],
            half_life_days=half_life
        )

        # Determine new tier
        new_tier = mem["tier"]
        if new_decay < archive_threshold and mem["tier"] == "warm":
            new_tier = "deep"
            archived += 1

        # Update record
        cursor.execute("""
            UPDATE memories
            SET decay_score = ?, tier = ?
            WHERE id = ?
        """, (new_decay, new_tier, mem["id"]))

        updated += 1

    conn.commit()

    # Summary stats
    cursor.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier")
    tier_counts = dict(cursor.fetchall())

    cursor.execute("SELECT AVG(decay_score) FROM memories")
    avg_decay = cursor.fetchone()[0]

    conn.close()

    logger.info(f"Decay complete: {updated} records updated, {archived} archived")
    logger.info(f"Tier distribution: {tier_counts}")
    logger.info(f"Average decay: {avg_decay:.4f}")

    return {
        "updated": updated,
        "archived": archived,
        "tier_counts": tier_counts,
        "avg_decay": round(avg_decay, 4) if avg_decay else 0,
    }


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Dry run mode - no changes")

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memories")
        memories = cursor.fetchall()

        print(f"\n{'ID':<38} {'Category':<18} {'Decay':>6} → {'New':>6} {'Tier':<6}")
        print("-" * 80)

        for mem in memories:
            new_decay = calculate_decay(
                mem["last_accessed"],
                mem["access_count"],
                half_life_days=config["decay"]["half_life_days"]
            )
            print(f"{mem['id']:<38} {mem['category']:<18} {mem['decay_score']:>6.3f} → {new_decay:>6.3f} {mem['tier']:<6}")

        conn.close()
    else:
        result = run_decay()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
